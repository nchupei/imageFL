#!/usr/bin/env python3
"""
一鍵啟動影像版 Federated Learning 實驗（CIFAR-10 npz + CNN + ASR）。

這支腳本做的事：
- 建立/指定 EXPERIMENT_DIR
- 啟動 image_cloud_server.py
- 啟動 N 個 image_aggregator.py
- 啟動 N 個 image_uav_client.py（自動加 IMAGE_FL=1）
- 用簡化協調器推進 round（從 1 開始跑到 max_rounds）

注意：
- 這是「最小可用」啟動器，先以跑通流程為主；更細的容錯/重啟可再加。
- 預設拓樸為 3 台聚合器、40 台客戶端（可用 --num_aggregators / --num_clients 覆寫）；協調器推進預設「嚴格多數」（例 3→2）；雲端 Image eval 預設須全員該輪上傳（N/N，與序貫 delta 合併一致）。全員換輪請 COORDINATOR_REQUIRED_FRACTION=1；放寬 eval 請 CLOUD_EVAL_AGGREGATOR_QUORUM=2。
- 除錯／掃參：設 IMAGE_FL_FAST_ITERATION=1 或傳入 --fast_iteration；僅在未手動 export 時套用較短的協調間隔、聚合等待、GPU、本地 epoch、雲端 ftr_all 降頻與較高的每輪抽樣數。
- 協調器阻塞：COORDINATOR_MIN_ROUND_DWELL_S（本輪 start 成功後至少等待秒數才允許廣播下一輪）；
  COORDINATOR_REQUIRE_CLOUD_ACK=1（僅以 federated_status 的 last_cloud_ack_round 判斷是否「到達」該輪，避免 round_count 一更新就搶跑）。
- Image FL 多台聚合器：預設 COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK（未設 env 時視為開）要求全員 eff_advance 到齊才推進；設 COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK=0 可改回嚴格多數／COORDINATOR_REQUIRED_FRACTION。
- GTSRB 非快速模式：若偵測到 CUDA，未指定時預設 IMAGE_USE_GPU=1；並預設較長 dwell／status 逾時與輕量評估抽樣。
- 若 shell 已 export AGG_MIN_TRAINING_DURATION_S 導致與 interval 不一致，可設 IMAGE_FL_RESET_AGG_MIN=1 讓腳本重新套用公式。
- 雲端 Image 評估：預設 N/N；放寬請 CLOUD_EVAL_AGGREGATOR_QUORUM=2。可設 CLOUD_EVAL_GRACE_SECONDS=30 在 idle 後以「部分合併快照」仍觸發評估（CSV 的 eval 門檻欄位會附 grace_partial）。
- 聚合器 federated_status 日誌：IMAGE_FL=1 時預設每 N 次才印（AGG_FEDERATED_STATUS_LOG_EVERY_N 覆寫）。
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent

# 與 start_fixed_experiment.py 一致：讓子進程繼承含 torch lib 的 LD_LIBRARY_PATH，
# 避免動態載入失敗時退回「無 CUDA」或 oneMKL / libtorch 載入錯誤。
try:
    import torch as _torch_boot

    _tl = os.path.join(os.path.dirname(_torch_boot.__file__), "lib")
    if os.path.isdir(_tl):
        _prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _tl + (os.pathsep + _prev_ld if _prev_ld else "")
except Exception:
    pass


def _ts() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _torch_cuda_available() -> bool:
    """啟動腳本內偵測（子行程仍各自 import torch）；失敗時視為無 CUDA。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cleanup_image_fl_processes() -> None:
    """
    在啟動新一輪 image FL 之前，先清掉舊的 cloud / aggregator / client 進程。
    僅針對當前專案路徑下的 image_fl_experiments 腳本，不會動到其他程式。
    """
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return

    targets = [
        str(BASE_DIR / "image_cloud_server.py"),
        str(BASE_DIR / "image_aggregator.py"),
        str(BASE_DIR / "image_uav_client.py"),
    ]
    pids_to_kill: List[int] = []

    for line in out.splitlines():
        if "python" not in line:
            continue
        if any(t in line for t in targets):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            # 避免誤殺當前啟動腳本本身（start_image_experiment.py）
            if "start_image_experiment.py" in line:
                continue
            pids_to_kill.append(pid)

    if not pids_to_kill:
        return

    print(f"[Cleanup] 準備終止舊的 image FL 進程: {pids_to_kill}", flush=True)
    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            continue
    # 給 uvicorn 一點時間釋放埠（否則下一輪 bind 仍可能 EADDRINUSE）
    time.sleep(0.6)


def _free_tcp_ports(ports: List[int]) -> None:
    """
    Linux：用 fuser 強制釋放仍被占用的 TCP 埠。
    僅在「舊 image FL 已 SIGTERM 但埠未釋放」或「非本專案腳本占用 800x」時需要。
    設 IMAGE_SKIP_FREE_PORTS=1 可跳過（若你確定埠上為其它服務）。
    """
    if str(os.environ.get("IMAGE_SKIP_FREE_PORTS", "") or "").strip().lower() in ("1", "true", "yes"):
        return
    for p in ports:
        try:
            r = subprocess.run(
                ["fuser", "-k", f"{p}/tcp"],
                capture_output=True,
                text=True,
                timeout=12,
            )
            if r.returncode == 0:
                err = (r.stderr or "").strip()
                if err or (r.stdout or "").strip():
                    print(f"[Cleanup] 已嘗試釋放埠 {p}/tcp（fuser）", flush=True)
        except FileNotFoundError:
            print(
                "[Cleanup] ⚠️ 系統無 fuser，無法自動釋放埠；若出現 address already in use 請手動："
                f"fuser -k <port>/tcp",
                flush=True,
            )
            return
        except Exception:
            continue
    time.sleep(0.4)


async def _wait_http_ok(url: str, timeout_s: int = 120) -> bool:
    import aiohttp

    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


def _coordinator_start_round_timeout_s() -> float:
    """
    POST /start_federated_round 的 HTTP 逾時（秒）。
    聚合器在 pending buffer 時會先執行 flush_before_advance（FedAvg + 拉取 cloud base + 上傳 delta），
    ResNet 等場景常超過數十秒；預設 10s 會誤報 TimeoutError，但聚合器端其後仍會完成（見 agg*.out）。
    覆寫：export COORDINATOR_START_ROUND_TIMEOUT_S=600
    """
    raw = (os.environ.get("COORDINATOR_START_ROUND_TIMEOUT_S", "") or "").strip()
    if raw:
        try:
            return max(5.0, float(raw))
        except ValueError:
            pass
    return 300.0


async def _post_start_federated_round(
    session: Any, url: str, round_id: int, *, log_label: str = "呼叫"
) -> Tuple[bool, Optional[str]]:
    """
    POST start_federated_round；回傳 (HTTP 是否 200, 回應 JSON 的 status 欄若可解析)。
    aggregator_fixed 在 buffer 未清空時可能回 200 + status=pending_aggregation（輪次實際未推進）。
    """
    import aiohttp

    data = aiohttp.FormData()
    data.add_field("round_id", str(round_id))
    target = f"{url}/start_federated_round"
    _t = _coordinator_start_round_timeout_s()
    print(f"[Coordinator] {log_label} {target} round_id={round_id}", flush=True)
    try:
        async with session.post(target, data=data, timeout=aiohttp.ClientTimeout(total=_t)) as resp:
            body_status: Optional[str] = None
            try:
                txt = await resp.text()
                if txt:
                    j = json.loads(txt)
                    if isinstance(j, dict):
                        raw = j.get("status")
                        if raw is not None:
                            body_status = str(raw)
            except Exception:
                pass

            if resp.status != 200:
                print(f"[Coordinator] ❌ {target} round_id={round_id} status={resp.status}", flush=True)
                return False, body_status

            if body_status in ("pending_aggregation", "large_skip_rejected"):
                print(
                    f"[Coordinator] ⚠️ {target} round_id={round_id} HTTP 200 但 status={body_status} "
                    f"（聚合器尚未推進輪次，將於後續週期對落後者重送）",
                    flush=True,
                )
            else:
                print(f"[Coordinator] ✅ {target} round_id={round_id} status={resp.status}", flush=True)
            return True, body_status
    except Exception as exc:
        print(f"[Coordinator] ⚠️ 呼叫 {url}/start_federated_round 失敗: {exc!r}", flush=True)
        return False, None


async def _start_round(aggregator_urls: List[str], round_id: int) -> bool:
    import aiohttp

    ok = True
    async with aiohttp.ClientSession() as session:
        for url in aggregator_urls:
            http_ok, _st = await _post_start_federated_round(session, url, round_id, log_label="呼叫")
            if not http_ok:
                ok = False
    return ok


async def _start_round_lagging(aggregator_urls: List[str], round_id: int, eff: List[int]) -> None:
    """僅對 eff < round_id 的聚合器重送 start_federated_round，避免 pending_aggregation 後永遠收不到下一輪。"""
    import aiohttp

    indices = [i for i, v in enumerate(eff) if v < round_id]
    if not indices:
        return
    print(
        f"[Coordinator] 🔁 重送 start_federated_round round={round_id} 至落後聚合器: "
        f"{[f'agg{i}' for i in indices]}",
        flush=True,
    )
    async with aiohttp.ClientSession() as session:
        for i in indices:
            await _post_start_federated_round(
                session, aggregator_urls[i], round_id, log_label="重送"
            )


def _coordinator_float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


async def _fetch_rounds(aggregator_urls: List[str]) -> Tuple[List[int], List[int]]:
    """回傳 (round_count_list, ack_round_list)。"""
    import aiohttp

    rounds: List[int] = []
    acks: List[int] = []
    _cto = max(10.0, min(120.0, _coordinator_float_env("COORDINATOR_FETCH_STATUS_TIMEOUT_S", 30.0)))
    timeout = aiohttp.ClientTimeout(total=_cto)

    async def _one(session: Any, base_url: str) -> Tuple[int, int]:
        """單台聚合器；失敗時短暫重試一次，避免瞬時逾時／連線錯誤被記成 round=0 → 協調器誤判落後。"""
        target = f"{base_url}/federated_status"
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            try:
                async with session.get(target, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        r = int(data.get("round_count", data.get("current_round", 0)) or 0)
                        a = int(data.get("last_cloud_ack_round", 0) or 0)
                        return r, a
            except BaseException as e:
                last_exc = e
            if attempt == 0:
                await asyncio.sleep(0.2)
        if last_exc is not None:
            print(f"[Coordinator] ⚠️ federated_status 失敗（已重試）{target}: {last_exc!r}", flush=True)
        return 0, 0

    async with aiohttp.ClientSession() as session:
        for url in aggregator_urls:
            r, a = await _one(session, url)
            rounds.append(r)
            acks.append(a)
    return rounds, acks


def _coordinator_required_aggregators(n: int) -> int:
    """
    協調器推進下一輪前，需要多少台 aggregator 已達到當前 target_round。

    預設（未設 COORDINATOR_REQUIRED_FRACTION）：多聚合器時為「嚴格多數」（例 3→2、4→3、2→2），
    與雲端 Image eval 預設門檻一致，避免單台卡住導致無 image_cloud_metrics。

    需全員才推進：export COORDINATOR_REQUIRED_FRACTION=1
    較鬆（例如 60%）：export COORDINATOR_REQUIRED_FRACTION=0.6
    """
    if n <= 0:
        return 1
    raw = (os.environ.get("COORDINATOR_REQUIRED_FRACTION", "") or "").strip()
    if raw:
        try:
            frac = float(raw)
            frac = max(0.01, min(1.0, frac))
        except Exception:
            frac = 1.0
        return max(1, int(frac * n))
    if n <= 1:
        return 1
    return max(1, (n // 2) + 1)


def _normalize_attacker_clients(raw: str) -> str:
    vals: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return ",".join(str(x) for x in sorted(set(vals)))


def _validate_and_append_run_manifest(
    *,
    manifest_path: Path,
    round_id: int,
    expected: Dict[str, str],
) -> None:
    """
    每輪寫入同一份 run manifest，並對關鍵設定做硬驗證。
    若當前環境值與凍結預期不一致，立即拋錯終止實驗。
    """
    actual = {
        "local_epochs": str(os.environ.get("IMAGE_LOCAL_EPOCHS", "")).strip(),
        "attacker_local_epochs": str(os.environ.get("IMAGE_ATTACKER_LOCAL_EPOCHS", "")).strip(),
        "attacker_clients": _normalize_attacker_clients(os.environ.get("IMAGE_ATTACKER_CLIENTS", "")),
        "defense_mode": str(os.environ.get("AGG_DEFENSE_MODE", "none") or "none").strip().lower(),
    }
    mismatches = []
    for k in ("local_epochs", "attacker_local_epochs", "attacker_clients", "defense_mode"):
        if actual.get(k, "") != expected.get(k, ""):
            mismatches.append(f"{k}: expected={expected.get(k)!r}, actual={actual.get(k)!r}")
    # 額外硬驗證：若 attacker local epochs 有設定，必須與 local epochs 一致，避免「你設 6 實際跑 4」。
    if actual["attacker_local_epochs"] and actual["attacker_local_epochs"] != actual["local_epochs"]:
        mismatches.append(
            "attacker_local_epochs_vs_local_epochs: "
            f"IMAGE_ATTACKER_LOCAL_EPOCHS={actual['attacker_local_epochs']!r} "
            f"!= IMAGE_LOCAL_EPOCHS={actual['local_epochs']!r}"
        )
    if mismatches:
        raise RuntimeError("run manifest 硬驗證失敗: " + "; ".join(mismatches))

    record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "round": int(round_id),
        "local_epochs": actual["local_epochs"],
        "attacker_local_epochs": actual["attacker_local_epochs"],
        "attacker_clients": actual["attacker_clients"],
        "defense_mode": actual["defense_mode"],
    }
    _ensure_dir(manifest_path.parent)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


async def coordinator_loop(
    aggregator_urls: List[str],
    max_rounds: int,
    interval_s: int = 10,
    *,
    run_manifest_path: Optional[Path] = None,
    manifest_expected: Optional[Dict[str, str]] = None,
) -> None:
    """
    簡化協調器：round 1 開始，等足夠多 aggregator「到達」當前 target 後推進下一輪（預設嚴格多數，見 _coordinator_required_aggregators）。

    到達判定 eff_advance（用於 reached）：
    - 預設：last_cloud_ack_round > 0 則用 ACK，否則用 round_count（與舊版一致）。
    - COORDINATOR_REQUIRE_CLOUD_ACK=1：僅用 last_cloud_ack_round（無 ACK 視為 0），避免 round 一更新就推進、Client 尚未上傳。

    COORDINATOR_MIN_ROUND_DWELL_S>0：本輪 start_federated_round 全員送達後，至少經過這麼多秒才允許廣播下一輪（緩解 CPU 慢 Client）。

    COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK：Image FL 且多台聚合器時，預設需「全員」eff_advance>=target 才推進下一輪（避免某台尚未
    delta_upload_ok 就被多數票帶到下一輪、造成雲端缺輪）。設為 0 可關閉並改回 COORDINATOR_REQUIRED_FRACTION／嚴格多數邏輯。
    與 COORDINATOR_REQUIRE_CLOUD_ACK=1 並用時，eff_advance 僅看 last_cloud_ack_round，可對齊「雲端已 ACK 合併」再開下一輪。
    """
    target_round = 1
    last_started = 0
    round_t_started_at: Optional[float] = None

    dwell_s = max(0.0, _coordinator_float_env("COORDINATOR_MIN_ROUND_DWELL_S", 0.0))
    require_cloud_ack = str(os.environ.get("COORDINATOR_REQUIRE_CLOUD_ACK", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _imf_coord = (os.environ.get("IMAGE_FL", "") or "").strip() == "1"
    _raw_all_ack = (os.environ.get("COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK", "") or "").strip().lower()
    if _raw_all_ack in ("0", "false", "no", "off"):
        image_fl_require_all_ack = False
    elif _raw_all_ack in ("1", "true", "yes", "on"):
        # 顯式設 1 時一律採全員 ACK（不再受 IMAGE_FL 是否有被正確注入影響）
        image_fl_require_all_ack = True
    else:
        # 未設定時：Image FL 且多聚合器預設「全員到齊」才推進
        image_fl_require_all_ack = bool(_imf_coord and len(aggregator_urls) > 1)

    while target_round < max_rounds:
        if run_manifest_path is not None and manifest_expected is not None:
            _validate_and_append_run_manifest(
                manifest_path=run_manifest_path,
                round_id=target_round,
                expected=manifest_expected,
            )
        rounds, acks = await _fetch_rounds(aggregator_urls)
        if require_cloud_ack:
            eff_advance = list(acks)
        else:
            eff_advance = [a if a > 0 else r for r, a in zip(rounds, acks)]
        reached = sum(1 for v in eff_advance if v >= target_round)
        n_aggs = len(aggregator_urls)
        if image_fl_require_all_ack and n_aggs > 0:
            required = n_aggs
        else:
            required = _coordinator_required_aggregators(n_aggs)

        did_full_start = False
        # 若當前輪從未成功啟動，先嘗試啟動這一輪（例如一開始 aggregator 尚未綁定好時重試）
        if last_started < target_round:
            print(
                f"[Coordinator] 嘗試啟動 round {target_round}: reached={reached}/{required} "
                f"eff_advance={eff_advance} rounds={rounds} acks={acks}",
                flush=True,
            )
            ok = await _start_round(aggregator_urls, target_round)
            if ok:
                print(f"[Coordinator] ✅ round {target_round} 已送出給所有 aggregator。", flush=True)
                last_started = target_round
                round_t_started_at = time.monotonic()
                did_full_start = True
            else:
                print(f"[Coordinator] ❌ round {target_round} 發送失敗，稍後重試。", flush=True)

        advanced = False
        can_advance = reached >= required and last_started >= target_round
        if can_advance and dwell_s > 0 and round_t_started_at is not None:
            elapsed = time.monotonic() - round_t_started_at
            if elapsed < dwell_s:
                can_advance = False
                print(
                    f"[Coordinator] dwell：本輪已開始 {elapsed:.1f}s / 需 {dwell_s:.0f}s 才允許推進下一輪 "
                    f"(COORDINATOR_MIN_ROUND_DWELL_S)",
                    flush=True,
                )

        # 如果多數 aggregator 已經完成當前輪，推進到下一輪
        if can_advance:
            next_round = target_round + 1
            print(
                f"[Coordinator] 推進：target={target_round} reached={reached}/{required} "
                f"eff_advance={eff_advance} → start round {next_round}",
                flush=True,
            )
            await _start_round(aggregator_urls, next_round)
            last_started = next_round
            target_round = next_round
            round_t_started_at = time.monotonic()
            advanced = True
        elif last_started >= target_round and reached < required and not did_full_start:
            # 以 round_count 判斷誰還沒收到 start（與 ACK 模式無關，避免誤判）
            await _start_round_lagging(aggregator_urls, target_round, rounds)

        if not advanced:
            lag_parts: List[str] = []
            for i, v in enumerate(eff_advance):
                if v < target_round:
                    lag_parts.append(f"agg{i}@{aggregator_urls[i]} advance={v}")
            lag_hint = f" 落後: {'; '.join(lag_parts)}" if lag_parts else ""
            print(
                f"[Coordinator] 等待：target={target_round} reached={reached}/{required} "
                f"eff_advance={eff_advance}{lag_hint}",
                flush=True,
            )

        await asyncio.sleep(interval_s)


def _popen(cmd: List[str], out_path: Path, env: Dict[str, str]) -> subprocess.Popen:
    _ensure_dir(out_path.parent)
    f = open(out_path, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(BASE_DIR), env=env, stdout=f, stderr=f, start_new_session=True)

def _forward_prefixed_env(dst_env: Dict[str, str], *, prefixes: List[str]) -> Dict[str, str]:
    """
    明確把特定前綴的環境變數（例如 AGG_*）寫進即將傳給子行程的 env。
    這可避免未來若有「只挑部分 env」的改動，導致防禦參數沒被帶到 aggregator。
    """
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in prefixes):
            dst_env[k] = str(v)
    return dst_env

def _dump_env_subset(env: Dict[str, str], *, prefixes: List[str], out_path: Path) -> None:
    keys = sorted([k for k in env.keys() if any(k.startswith(p) for p in prefixes)])
    lines = [f"{k}={env.get(k,'')}\n" for k in keys]
    _ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Start image FL experiment (CIFAR-10)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--cloud_port", type=int, default=8083)
    parser.add_argument(
        "--num_aggregators",
        type=int,
        default=3,
        help="聚合器數量（預設 3：較 4 台少一路 CAS 競爭、quorum 為 3/3；論文可註明階層式 FL 拓樸）",
    )
    parser.add_argument("--base_agg_port", type=int, default=8000)
    parser.add_argument(
        "--num_clients",
        type=int,
        default=40,
        help="客戶端總數（預設 40：搭配 3 台聚合器時 client_id%%N 分組約 13～14 台/agg，較原 4×10 單台樣本更多）",
    )
    parser.add_argument("--max_rounds", type=int, default=25)
    # 小規模模式：每個 aggregator 每輪最多選幾個 client（會影響參與率與穩定性）
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=None,
        help="每聚合器每輪最多選幾個 client；省略時預設 3，快速迭代模式下預設 12",
    )
    parser.add_argument(
        "--fast_iteration",
        action="store_true",
        help="啟用快速迭代預設（等同 export IMAGE_FL_FAST_ITERATION=1，並套用加速用環境預設）",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result_root", type=str, default="/home/ubuntuv100/uav/newp1/result_image")
    # attacker_clients:
    # - 若未指定，會依 num_clients 自動產生（每 10 個挑一個：0,10,20,...）
    # - 也可手動指定 "0,10,20" 這類格式
    parser.add_argument("--attacker_clients", type=str, default="")
    # target_label:
    # - CIFAR-10: 0~9（預設 8）
    # - road_signs: 0~3（預設 2）
    # 這裡用 None 代表「依 dataset 自動選」並做合法性檢查，避免 ASR 永遠為 0。
    parser.add_argument("--target_label", type=int, default=None)
    args = parser.parse_args()
    if bool(getattr(args, "fast_iteration", False)):
        os.environ["IMAGE_FL_FAST_ITERATION"] = "1"

    # 啟動前先清理舊的 image FL 進程（cloud / aggregators / clients）
    _cleanup_image_fl_processes()
    ports_to_free = [int(args.cloud_port)] + [
        int(args.base_agg_port) + i for i in range(int(args.num_aggregators))
    ]
    _free_tcp_ports(ports_to_free)
    print(
        "[Start] 已執行舊 image FL 進程清理（image_cloud_server / image_aggregator / image_uav_client）；"
        f"並嘗試釋放埠 {ports_to_free}。"
        "若仍 address already in use，請：fuser -v <port>/tcp 或 IMAGE_SKIP_FREE_PORTS=1 略過自動釋放。",
        flush=True,
    )

    dataset = os.environ.get("IMAGE_DATASET", "cifar10").strip().lower()
    _fast_it = str(os.environ.get("IMAGE_FL_FAST_ITERATION", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    name_suffix = os.environ.get("TUNING_NAME", "").strip()
    base_name = f"{dataset}_backdoor_{_ts()}"
    exp_dir = Path(args.result_root) / (f"{base_name}_{name_suffix}" if name_suffix else base_name)
    _ensure_dir(exp_dir)

    try:
        _cpa = max(1, int(args.num_clients)) // max(1, int(args.num_aggregators))
        _cpa_max = (max(1, int(args.num_clients)) + max(1, int(args.num_aggregators)) - 1) // max(
            1, int(args.num_aggregators)
        )
    except Exception:
        _cpa = _cpa_max = 0
    print(
        f"[Start] topology: aggregators={args.num_aggregators}, clients={args.num_clients} "
        f"(約 {_cpa}～{_cpa_max} clients/agg，依 client_id % {args.num_aggregators} 指派)",
        flush=True,
    )

    # 捨棄父 shell 帶入的 AGG_MIN（須先改 os.environ，否則 _forward_prefixed_env 會從 os.environ 再寫回）
    if str(os.environ.get("IMAGE_FL_RESET_AGG_MIN", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        os.environ.pop("AGG_MIN_TRAINING_DURATION_S", None)

    env = os.environ.copy()
    # 明確 forward 防禦相關 env（AGG_*）到所有子行程（cloud/agg/client）
    _forward_prefixed_env(env, prefixes=["AGG_"])
    # 避免 stdout 緩衝導致 client/agg/cloud log 看起來像空白
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["EXPERIMENT_DIR"] = str(exp_dir)
    env["LOG_DIR"] = str(exp_dir)
    env["IMAGE_FL"] = "1"
    env["IMAGE_NUM_CLIENTS"] = str(args.num_clients)

    # ---- attacker_clients / target_label defaults & validation ----
    def _auto_attacker_clients(num_clients: int) -> str:
        if num_clients <= 0:
            return "0"
        # 0,10,20,...（只取合法 client_id 範圍）
        ids = list(range(0, num_clients, 10))
        return ",".join(str(i) for i in ids) if ids else "0"

    # poison_enabled 會影響是否需要 attacker 名單；但這裡仍落盤，方便追查
    poison_enabled = str(env.get("IMAGE_POISON_ENABLED", "1") or "1").strip().lower() not in ("0", "false", "no")

    attacker_clients_raw = (args.attacker_clients or "").strip()
    env["IMAGE_ATTACKER_CLIENTS"] = attacker_clients_raw if attacker_clients_raw else _auto_attacker_clients(int(args.num_clients))

    # 依資料集選預設 target_label，並做範圍檢查
    if dataset == "road_signs":
        num_classes = 4
        default_target = 2
    elif dataset == "gtsrb":
        num_classes = 43
        default_target = 2
    else:
        num_classes = 10
        default_target = 8

    def _get_env_int(*keys: str) -> Optional[int]:
        for k in keys:
            v = str(env.get(k, "") or "").strip()
            if not v:
                continue
            try:
                return int(float(v))
            except Exception:
                continue
        return None

    if args.target_label is None:
        # 若使用者已透過 env 指定 target（POISON_TARGET_LABEL/IMAGE_TARGET_LABEL），則以 env 為準；
        # 否則才用 dataset 預設，避免「資料生成用 0，但實驗跑起來變 2」。
        env_target = _get_env_int("POISON_TARGET_LABEL", "IMAGE_TARGET_LABEL")
        target_label = int(env_target) if env_target is not None else int(default_target)
    else:
        target_label = int(args.target_label)

    if not (0 <= target_label < num_classes):
        # 若使用者手動指定了不合法 target_label，直接回退到安全預設，避免 ASR 永遠為 0
        print(
            f"[Start] ⚠️ target_label={target_label} 超出 dataset={dataset} 的範圍(0~{num_classes-1})，"
            f"改用預設 {default_target}。",
            flush=True,
        )
        target_label = int(default_target)
    # Respect external override (e.g., IMAGE_POISON_ENABLED=0 for clean baseline)
    env.setdefault("IMAGE_POISON_ENABLED", "1")
    env["POISON_TARGET_LABEL"] = str(int(target_label))
    env["IMAGE_MAX_ROUNDS"] = str(args.max_rounds)
    env["IMAGE_SEED"] = str(int(args.seed))
    env["PYTHONHASHSEED"] = str(int(args.seed))

    # 讓攻擊設定在主控 log 中一眼可見（避免 ASR=0 時還要翻 client log）
    try:
        print(
            f"[Start] image settings: dataset={dataset} poison_enabled={1 if poison_enabled else 0} "
            f"IMAGE_ATTACKER_CLIENTS={env.get('IMAGE_ATTACKER_CLIENTS','')} "
            f"POISON_TARGET_LABEL={env.get('POISON_TARGET_LABEL','')}",
            flush=True,
        )
    except Exception:
        pass

    # CIFAR-10 常用的線上資料增強與較強的 local training（可自行覆蓋）
    if dataset == "cifar10":
        env.setdefault("IMAGE_AUG_ENABLED", "1")
        env.setdefault("IMAGE_LOCAL_EPOCHS", "2")

    # GTSRB：多 client 同時開 GPU 易 CUDA OOM。未手動指定時降低 batch；本地觸發 ASR 評估 batch 一併降。
    # 啟動間隔（秒）可緩解同時初始化 CUDA 的尖峰；設 IMAGE_CLIENT_STAGGER_S=0 可關閉。
    if dataset == "gtsrb":
        env.setdefault("IMAGE_BATCH_SIZE", "64")
        env.setdefault("IMAGE_ASR_EVAL_BATCH_SIZE", "64")
        env.setdefault("IMAGE_CLIENT_STAGGER_S", "0.15")
        _cuda_start = _torch_cuda_available()
        # Client 預設 IMAGE_USE_GPU=0；GTSRB 且偵測到 CUDA 時主動開 GPU（仍可用 export 覆寫）
        if _cuda_start and not str(env.get("IMAGE_USE_GPU", "") or "").strip():
            env["IMAGE_USE_GPU"] = "1"
        elif not _cuda_start:
            print(
                "[Start] ⚠️ GTSRB：本機 torch.cuda.is_available()=False，Client 將使用 CPU "
                "（若實際有 GPU 請檢查驅動／CUDA／PyTorch 建置）",
                flush=True,
            )
        if not _fast_it:
            # 輕量線上評估預設（未手動 export 才寫入）
            env.setdefault("IMAGE_EVAL_TEST_MAX_SAMPLES", "400")
            env.setdefault("IMAGE_EVAL_SUBSAMPLE_SEED", str(int(args.seed)))
            env.setdefault("IMAGE_CLEAN_EVAL_EVERY_ROUNDS", "2")
            env.setdefault("IMAGE_ASR_EVAL_EVERY_ROUNDS", "3")
            env.setdefault("IMAGE_ASR_EVAL_MAX_SAMPLES", "200")
            env.setdefault("IMAGE_ASR_SUBSAMPLE_SEED", str(int(args.seed)))
        print(
            f"[Start] gtsrb anti-OOM defaults: IMAGE_BATCH_SIZE={env.get('IMAGE_BATCH_SIZE')} "
            f"IMAGE_ASR_EVAL_BATCH_SIZE={env.get('IMAGE_ASR_EVAL_BATCH_SIZE')} "
            f"IMAGE_CLIENT_STAGGER_S={env.get('IMAGE_CLIENT_STAGGER_S')} "
            f"cuda_detected={int(_cuda_start)} IMAGE_USE_GPU={env.get('IMAGE_USE_GPU', '')} "
            f"(覆寫請自行 export 或改 env)",
            flush=True,
        )

    # 啟用小規模模式，讓 config_fixed 內的 NUM_CLIENTS / NUM_AGGREGATORS 與本實驗設定一致
    env["SMALL_SCALE_MODE"] = "1"
    env["SMALL_SCALE_NUM_AGGREGATORS"] = str(args.num_aggregators)
    env["SMALL_SCALE_NUM_CLIENTS"] = str(args.num_clients)
    env["SMALL_SCALE_MAX_ROUNDS"] = str(args.max_rounds)

    # 可選：快速迭代（除錯／參數掃描）。僅在「尚未手動 export」時寫入預設。
    # 不影響論文主實驗：預設關閉；設 IMAGE_FL_FAST_ITERATION=1 或 --fast_iteration 啟用。
    if _fast_it:
        if not str(os.environ.get("COORDINATOR_INTERVAL_S", "") or "").strip():
            os.environ["COORDINATOR_INTERVAL_S"] = "120"
            env["COORDINATOR_INTERVAL_S"] = "120"
        if not str(os.environ.get("AGG_MIN_TRAINING_DURATION_S", "") or "").strip():
            os.environ["AGG_MIN_TRAINING_DURATION_S"] = "30"
            env["AGG_MIN_TRAINING_DURATION_S"] = "30"
        if not str(os.environ.get("IMAGE_USE_GPU", "") or "").strip():
            os.environ["IMAGE_USE_GPU"] = "1"
            env["IMAGE_USE_GPU"] = "1"
        if not str(os.environ.get("IMAGE_LOCAL_EPOCHS", "") or "").strip():
            os.environ["IMAGE_LOCAL_EPOCHS"] = "1"
            env["IMAGE_LOCAL_EPOCHS"] = "1"
        if not str(os.environ.get("IMAGE_CLOUD_FTR_ALL_EVERY_N_ROUNDS", "") or "").strip():
            os.environ["IMAGE_CLOUD_FTR_ALL_EVERY_N_ROUNDS"] = "5"
            env["IMAGE_CLOUD_FTR_ALL_EVERY_N_ROUNDS"] = "5"
        print(
            "[Start] IMAGE_FL_FAST_ITERATION：預設 COORDINATOR_INTERVAL_S=120、"
            "AGG_MIN_TRAINING_DURATION_S=30、IMAGE_USE_GPU=1、IMAGE_LOCAL_EPOCHS=1、"
            "IMAGE_CLOUD_FTR_ALL_EVERY_N_ROUNDS=5（已手動 export 者不覆寫）",
            flush=True,
        )

    if args.clients_per_round is not None:
        effective_cpr = max(1, int(args.clients_per_round))
    else:
        effective_cpr = 12 if _fast_it else 3
    if args.clients_per_round is None and _fast_it:
        print(
            f"[Start] 快速模式：每輪抽樣 SMALL_SCALE_CLIENTS_PER_ROUND={effective_cpr} "
            f"（以 --clients_per_round 可覆寫）",
            flush=True,
        )
    env["SMALL_SCALE_CLIENTS_PER_ROUND"] = str(effective_cpr)

    # 協調器間隔（先算好）：子進程與 coordinator_loop 共用
    try:
        env_interval = os.environ.get("COORDINATOR_INTERVAL_S", "").strip()
        if env_interval:
            interval_s = max(10, int(float(env_interval)))
        else:
            # GTSRB 預設拉長：慢模型（如 ResNet）+ CPU 時，協調器過快會觸發 MAX_STALENESS 拒收
            interval_s = 300 if dataset == "cifar10" else 600
    except Exception:
        interval_s = 300 if dataset == "cifar10" else 600
    # 寫回 env，讓 cloud/agg/client 與日誌一致（實驗可重現）；未設環境變數時即為本次使用的固定值
    env["COORDINATOR_INTERVAL_S"] = str(interval_s)
    # 聚合器 pending_buffer 推進前最少等待。
    # 未手動設 AGG_MIN_TRAINING_DURATION_S 時：與快速迭代一致，上限 30s（避免 GTSRB+長 interval 預設拉到 90～120s）。
    if not str(env.get("AGG_MIN_TRAINING_DURATION_S", "")).strip():
        sync_dur = max(5, min(30, interval_s - 5))
        env["AGG_MIN_TRAINING_DURATION_S"] = str(sync_dur)

    try:
        _am = float(str(env.get("AGG_MIN_TRAINING_DURATION_S", "0") or "0").strip() or "0")
        if _am > float(interval_s):
            print(
                f"[Start] ⚠️ AGG_MIN_TRAINING_DURATION_S={int(_am)} 大於 COORDINATOR_INTERVAL_S={interval_s}，"
                f"協調易長期阻塞；若要採腳本公式可 export IMAGE_FL_RESET_AGG_MIN=1 後重跑",
                flush=True,
            )
    except Exception:
        pass

    # GTSRB 主實驗：拉長協調器在每輪的停留與 HTTP 逾時，減少 status 風暴與誤逾時
    if dataset == "gtsrb" and not _fast_it:
        if not str(env.get("COORDINATOR_MIN_ROUND_DWELL_S", "") or "").strip():
            _dwell_s = "600"
            env["COORDINATOR_MIN_ROUND_DWELL_S"] = _dwell_s
            os.environ["COORDINATOR_MIN_ROUND_DWELL_S"] = _dwell_s
        if not str(env.get("COORDINATOR_FETCH_STATUS_TIMEOUT_S", "") or "").strip():
            _fto = "90"
            env["COORDINATOR_FETCH_STATUS_TIMEOUT_S"] = _fto
            os.environ["COORDINATOR_FETCH_STATUS_TIMEOUT_S"] = _fto
        if not str(env.get("FEDERATED_STATUS_TIMEOUT_S", "") or "").strip():
            env["FEDERATED_STATUS_TIMEOUT_S"] = "120"
        print(
            f"[Start] gtsrb coordinator/client timeouts: COORDINATOR_MIN_ROUND_DWELL_S="
            f"{env.get('COORDINATOR_MIN_ROUND_DWELL_S', '')} "
            f"COORDINATOR_FETCH_STATUS_TIMEOUT_S={env.get('COORDINATOR_FETCH_STATUS_TIMEOUT_S', '')} "
            f"FEDERATED_STATUS_TIMEOUT_S={env.get('FEDERATED_STATUS_TIMEOUT_S', '')}",
            flush=True,
        )

    if dataset == "gtsrb":
        print(
            f"[Start] gtsrb timing defaults: COORDINATOR_INTERVAL_S={interval_s}, "
            f"AGG_MIN_TRAINING_DURATION_S={env.get('AGG_MIN_TRAINING_DURATION_S', '')} "
            f"（未 export 時與 CIFAR 相同公式 max(5,min(30,interval-5))；聚合器 MAX_STALENESS 見 config / AGG_MAX_STALENESS）",
            flush=True,
        )
        # GTSRB 穩定化預設（僅 setdefault，不覆蓋手動 export）
        env.setdefault("IMAGE_INPUT_SIZE", "32")
        env.setdefault("IMAGE_GTSRB_RESIZE", "1")

    # Cloud → Aggregator 廣播全局權重（receive_global_weights）預設 30s；
    # ResNet / GTSRB 權重 pickle 後很大，易超時 → 部分聚合器落後、湊不滿同一 round 四台上傳。
    if not str(env.get("CLOUD_BROADCAST_TIMEOUT_S", "")).strip():
        cnn = str(env.get("IMAGE_CNN_SIZE", "") or "").strip().lower()
        if dataset == "gtsrb" or cnn in ("resnet18", "resnet-18", "resnet", "resnet34", "resnet50"):
            env["CLOUD_BROADCAST_TIMEOUT_S"] = "300"
            print(
                f"[Start] CLOUD_BROADCAST_TIMEOUT_S={env['CLOUD_BROADCAST_TIMEOUT_S']} "
                f"(gtsrb 或 ResNet 系 backbone 預設，可 export 覆寫)",
                flush=True,
            )

    # Aggregator → Cloud delta 上傳：預設 30s 在 ResNet 等大權重下易 aiohttp TimeoutError；
    # 多聚合器同輪 POST 時雲端亦會排隊，需更長逾時 + 錯開上傳。
    _heavy_cnn = str(env.get("IMAGE_CNN_SIZE", "") or "").strip().lower() in ("resnet18", "resnet34", "resnet50")
    _cnn_lc = str(env.get("IMAGE_CNN_SIZE", "") or "").strip().lower()
    if dataset == "gtsrb" and _cnn_lc in ("resnet18", "resnet-18", "resnet", "resnet34", "resnet50"):
        env.setdefault("IMAGE_WEIGHT_DECAY", "5e-4")
        env.setdefault("IMAGE_LABEL_SMOOTHING", "0.05")
        env.setdefault("IMAGE_LR_SCHEDULE", "cosine")
        env.setdefault("IMAGE_LR_MIN_FACTOR", "0.2")
        env.setdefault("IMAGE_MAX_GRAD_NORM", "1.0")
        print(
            "[Start] gtsrb+ResNet stability defaults: "
            f"IMAGE_INPUT_SIZE={env.get('IMAGE_INPUT_SIZE')} IMAGE_GTSRB_RESIZE={env.get('IMAGE_GTSRB_RESIZE')} "
            f"IMAGE_WEIGHT_DECAY={env.get('IMAGE_WEIGHT_DECAY')} "
            f"IMAGE_LABEL_SMOOTHING={env.get('IMAGE_LABEL_SMOOTHING')} "
            f"IMAGE_LR_SCHEDULE={env.get('IMAGE_LR_SCHEDULE')} "
            f"IMAGE_LR_MIN_FACTOR={env.get('IMAGE_LR_MIN_FACTOR')} "
            f"IMAGE_MAX_GRAD_NORM={env.get('IMAGE_MAX_GRAD_NORM')}",
            flush=True,
        )
    if not str(env.get("AGG_CLOUD_TIMEOUT_S", "")).strip():
        if dataset == "gtsrb" or _heavy_cnn:
            env["AGG_CLOUD_TIMEOUT_S"] = "600"
            print(
                f"[Start] AGG_CLOUD_TIMEOUT_S={env['AGG_CLOUD_TIMEOUT_S']} "
                f"(gtsrb 或較大 CNN 預設，可 export 覆寫)",
                flush=True,
            )
    if not str(env.get("AGG_CAS_STAGGER_PER_ID_S", "")).strip():
        if dataset == "gtsrb" or _heavy_cnn:
            env["AGG_CAS_STAGGER_PER_ID_S"] = "0.5"
    if not str(env.get("AGG_CAS_UPLOAD_JITTER_S", "")).strip():
        if dataset == "gtsrb" or _heavy_cnn:
            env["AGG_CAS_UPLOAD_JITTER_S"] = "0.5"
    if (dataset == "gtsrb" or _heavy_cnn) and (
        str(env.get("AGG_CAS_STAGGER_PER_ID_S", "")).strip()
        or str(env.get("AGG_CAS_UPLOAD_JITTER_S", "")).strip()
    ):
        print(
            f"[Start] delta upload stagger: AGG_CAS_STAGGER_PER_ID_S={env.get('AGG_CAS_STAGGER_PER_ID_S', '')} "
            f"AGG_CAS_UPLOAD_JITTER_S={env.get('AGG_CAS_UPLOAD_JITTER_S', '')}",
            flush=True,
        )

    # 協調器 POST start_federated_round：預設 300s（見 _coordinator_start_round_timeout_s）
    if not str(env.get("COORDINATOR_START_ROUND_TIMEOUT_S", "")).strip():
        if dataset == "gtsrb" or _heavy_cnn:
            env["COORDINATOR_START_ROUND_TIMEOUT_S"] = "300"

    # 將本次實驗實際帶入的防禦參數落盤，方便事後追查「是否有生效」
    try:
        _dump_env_subset(env, prefixes=["AGG_"], out_path=exp_dir / "env_agg.txt")
        agg_keys = sorted([k for k in env.keys() if k.startswith("AGG_")])
        if agg_keys:
            preview = ", ".join([f"{k}={env.get(k,'')}" for k in agg_keys])
            print(f"[Start] forwarded AGG_*: {preview}", flush=True)
        else:
            print("[Start] forwarded AGG_*: (none)", flush=True)
    except Exception:
        pass

    # 啟動 cloud
    cloud_cmd = [sys.executable, str(BASE_DIR / "image_cloud_server.py"), "--host", args.host, "--port", str(args.cloud_port)]
    cloud_proc = _popen(cloud_cmd, exp_dir / "cloud_server.out", env)
    print(f"[Start] cloud pid={cloud_proc.pid} dir={exp_dir}", flush=True)

    ok = await _wait_http_ok(f"http://{args.host}:{args.cloud_port}/health", timeout_s=180)
    if not ok:
        raise RuntimeError("Cloud health 未就緒，請看 cloud_server.out")

    # 聚合器依序啟動：避免同時 import torch / 綁埠競爭導致後續埠 bind 失敗（常見於 8001/8002）。
    # - IMAGE_AGG_START_STAGGER_S：每啟動一台後延遲秒數（預設 1.0）
    # - AGGREGATOR_HEALTH_TIMEOUT_S：每台 /health 最長等待（預設 240，ResNet 冷啟動較慢）
    try:
        _agg_stagger = float(str(os.environ.get("IMAGE_AGG_START_STAGGER_S", "1.0") or "1.0").strip())
    except Exception:
        _agg_stagger = 1.0
    _agg_stagger = max(0.0, _agg_stagger)
    try:
        _agg_health_s = int(float(os.environ.get("AGGREGATOR_HEALTH_TIMEOUT_S", "240") or "240"))
    except Exception:
        _agg_health_s = 240
    _agg_health_s = max(30, _agg_health_s)
    print(
        f"[Start] aggregator launch: IMAGE_AGG_START_STAGGER_S={_agg_stagger}, "
        f"AGGREGATOR_HEALTH_TIMEOUT_S={_agg_health_s}",
        flush=True,
    )

    # 啟動 aggregators
    aggregator_urls = []
    agg_procs = []
    for agg_id in range(args.num_aggregators):
        port = args.base_agg_port + agg_id
        aggregator_urls.append(f"http://{args.host}:{port}")
        cmd = [sys.executable, str(BASE_DIR / "image_aggregator.py"), "--aggregator_id", str(agg_id), "--port", str(port)]
        proc = _popen(cmd, exp_dir / f"agg{agg_id}.out", env)
        agg_procs.append(proc)
        print(f"[Start] agg{agg_id} pid={proc.pid} port={port}", flush=True)
        if agg_id < args.num_aggregators - 1 and _agg_stagger > 0:
            time.sleep(_agg_stagger)

    # 等待所有 aggregators /health 就緒
    for i, url in enumerate(aggregator_urls):
        ok = await _wait_http_ok(f"{url}/health", timeout_s=_agg_health_s)
        if not ok:
            raise RuntimeError(
                f"Aggregator 健康檢查失敗: {url}/health（請看 {exp_dir}/agg{i}.out 是否 "
                "address already in use 或其它錯誤；並確認埠未被占用："
                f"fuser -v {args.base_agg_port + i}/tcp）"
            )

    # 啟動 clients（round-robin 指派 aggregator）
    client_procs = []
    try:
        stagger_s = float(str(env.get("IMAGE_CLIENT_STAGGER_S", "0") or "0").strip())
    except Exception:
        stagger_s = 0.0
    stagger_s = max(0.0, stagger_s)
    for cid in range(args.num_clients):
        agg_url = aggregator_urls[cid % len(aggregator_urls)]
        cmd = [
            sys.executable,
            str(BASE_DIR / "image_uav_client.py"),
            "--client_id",
            str(cid),
            "--aggregator_url",
            agg_url,
            "--cloud_url",
            f"http://{args.host}:{args.cloud_port}",
            "--result_dir",
            str(exp_dir),
        ]
        proc = _popen(cmd, exp_dir / f"uav{cid}" / f"client_{cid}.log", env)
        client_procs.append(proc)
        if stagger_s > 0 and cid < args.num_clients - 1:
            time.sleep(stagger_s)

    print(f"[Start] clients started: {len(client_procs)}", flush=True)

    # 推進 round（interval_s 已於啟動 cloud 前計算並寫入 env）
    print(
        f"[Start] coordinator interval_s={interval_s}, "
        f"AGG_MIN_TRAINING_DURATION_S={env.get('AGG_MIN_TRAINING_DURATION_S', '')}",
        flush=True,
    )
    _imf_st = (os.environ.get("IMAGE_FL", "") or "").strip() == "1"
    _raw_all = (os.environ.get("COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK", "") or "").strip().lower()
    if _raw_all in ("0", "false", "no", "off"):
        _if_all_ack = False
    elif _raw_all in ("1", "true", "yes", "on"):
        _if_all_ack = True
    else:
        _if_all_ack = bool(_imf_st and int(args.num_aggregators) > 1)
    _req_print = int(args.num_aggregators) if _if_all_ack else _coordinator_required_aggregators(args.num_aggregators)
    print(
        f"[Start] coordinator advance quorum: {_req_print}/{args.num_aggregators} "
        f"(Image FL 預設全員到齊；關閉請 COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK=0；"
        f"非 Image FL 或關閉後：嚴格多數／COORDINATOR_REQUIRED_FRACTION)",
        flush=True,
    )
    _dwell = max(0.0, _coordinator_float_env("COORDINATOR_MIN_ROUND_DWELL_S", 0.0))
    _cloud_ack = str(os.environ.get("COORDINATOR_REQUIRE_CLOUD_ACK", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    print(
        f"[Start] coordinator sync: COORDINATOR_MIN_ROUND_DWELL_S={_dwell} "
        f"(本輪啟動後至少等待才推進下一輪), COORDINATOR_REQUIRE_CLOUD_ACK={int(_cloud_ack)} "
        f"(1=僅雲端 ACK 算到達), COORDINATOR_IMAGE_FL_REQUIRE_ALL_ACK≈{int(_if_all_ack)}",
        flush=True,
    )
    # 與子行程 env 對齊：協調器在同行程讀 os.environ
    if str(env.get("COORDINATOR_START_ROUND_TIMEOUT_S", "") or "").strip():
        os.environ["COORDINATOR_START_ROUND_TIMEOUT_S"] = str(env["COORDINATOR_START_ROUND_TIMEOUT_S"])
    print(
        f"[Start] COORDINATOR_START_ROUND_TIMEOUT_S={_coordinator_start_round_timeout_s()} "
        f"(POST start_federated_round；需涵蓋聚合器 flush+delta 上傳)",
        flush=True,
    )
    # 與子行程 env 對齊：run manifest 在主行程讀 os.environ，需同步關鍵實驗參數。
    for _k in ("IMAGE_LOCAL_EPOCHS", "IMAGE_ATTACKER_LOCAL_EPOCHS", "IMAGE_ATTACKER_CLIENTS", "AGG_DEFENSE_MODE"):
        if _k in env:
            os.environ[_k] = str(env[_k])

    run_manifest_path = exp_dir / "run_manifest.jsonl"
    manifest_expected = {
        "local_epochs": str(env.get("IMAGE_LOCAL_EPOCHS", "")).strip(),
        "attacker_local_epochs": str(env.get("IMAGE_ATTACKER_LOCAL_EPOCHS", "")).strip(),
        "attacker_clients": _normalize_attacker_clients(env.get("IMAGE_ATTACKER_CLIENTS", "")),
        "defense_mode": str(env.get("AGG_DEFENSE_MODE", "none") or "none").strip().lower(),
    }
    _ensure_dir(run_manifest_path.parent)
    with open(run_manifest_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "kind": "manifest_header",
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    **manifest_expected,
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    print(
        "[Start] run manifest 啟用: "
        f"{run_manifest_path} | expected(local_epochs={manifest_expected['local_epochs']}, "
        f"attacker_local_epochs={manifest_expected['attacker_local_epochs']}, "
        f"attacker_clients={manifest_expected['attacker_clients']}, "
        f"defense_mode={manifest_expected['defense_mode']})",
        flush=True,
    )

    await coordinator_loop(
        aggregator_urls,
        max_rounds=args.max_rounds,
        interval_s=interval_s,
        run_manifest_path=run_manifest_path,
        manifest_expected=manifest_expected,
    )


if __name__ == "__main__":
    asyncio.run(main())

