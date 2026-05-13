#!/usr/bin/env python3
"""
影像版 Cloud Server 啟動入口（保留原本 __main__ 參數與實驗目錄邏輯）。

目前先完整沿用 `cloud_server_fixed.py` 的 app 與啟動參數：
- --port
- --host

並沿用其 EXPERIMENT_DIR 建立/覆寫行為，確保 cloud_baseline.csv 等輸出一致。
"""

import argparse
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

# 讓在 image_fl_experiments 下執行時也能 import 到專案模組
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 🔧 影像模式下，強制使用與 image client 相同的 CNN 架構
os.environ.setdefault("IMAGE_FL", "1")

import config_fixed as config  # noqa: E402
import numpy as np  # noqa: E402
from image_uav_client import _get_image_cnn_builder  # noqa: E402

# CIFAR-10 標準化（與 client 一致，/255 後 (x-mean)/std）
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _env_flag(name: str, default: str = "1") -> bool:
    """布林解析：支援 1/true/yes/on 與 0/false/no/off。"""
    try:
        return (os.environ.get(name, default) or default).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return default.strip().lower() in ("1", "true", "yes", "on")


def _cloud_image_eval_batch_size() -> int:
    """
    雲端 Image eval（clean + trigger）DataLoader 的 batch size。
    - 首選 IMAGE_CLOUD_EVAL_BATCH_SIZE
    - 未設時沿用 IMAGE_ASR_EVAL_BATCH_SIZE（與 client 端 ASR eval 命名對齊）
    - 仍無則預設 2（單 GPU 上同時跑多個 client 時，避免雲端 eval 再搶 VRAM 造成 OOM）
    """
    raw = (os.environ.get("IMAGE_CLOUD_EVAL_BATCH_SIZE") or "").strip()
    if not raw:
        raw = (os.environ.get("IMAGE_ASR_EVAL_BATCH_SIZE") or "").strip()
    if not raw:
        raw = "2"
    try:
        bs = int(float(raw))
    except Exception:
        bs = 2
    return max(1, bs)


def _normalize_cifar10_nchw_cloud(x: np.ndarray) -> np.ndarray:
    """x: float32 NCHW [0,1]。若 IMAGE_CIFAR_NORMALIZE!=0 則標準化。"""
    if (os.environ.get("IMAGE_CIFAR_NORMALIZE", "1") or "1").strip() in ("0", "false", "no"):
        return x
    mean = np.array(_CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(_CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def _load_gtsrb_norm_stats_cloud() -> "tuple[tuple[float, float, float], tuple[float, float, float]]":
    import json

    path = os.environ.get("GTSRB_NORM_PATH", "").strip()
    if not path:
        path = str(Path(__file__).resolve().parent / "gtsrb_poisoned" / "gtsrb_norm.json")
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    mean = obj.get("mean")
    std = obj.get("std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == 3 and len(std) == 3):
        raise ValueError(f"GTSRB norm 格式錯誤: {p}")
    return (float(mean[0]), float(mean[1]), float(mean[2])), (float(std[0]), float(std[1]), float(std[2]))


def _normalize_gtsrb_nchw_cloud(x: np.ndarray) -> np.ndarray:
    """x: float32 NCHW [0,1]。若 IMAGE_GTSRB_NORMALIZE=1 則用 gtsrb_norm.json 標準化。"""
    if (os.environ.get("IMAGE_GTSRB_NORMALIZE", "0") or "0").strip().lower() in ("0", "false", "no"):
        return x
    mean, std = _load_gtsrb_norm_stats_cloud()
    mean_a = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_a = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean_a) / std_a


def _normalize_image_nchw_cloud(x: np.ndarray, *, dataset: str) -> np.ndarray:
    ds = (dataset or "").strip().lower()
    if ds == "cifar10":
        return _normalize_cifar10_nchw_cloud(x)
    if ds == "gtsrb":
        return _normalize_gtsrb_nchw_cloud(x)
    return x
import models.cnn as base_cnn  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402


def _build_image_cnn_for_server(input_dim: int, output_dim: int):
    """
    供 cloud server / aggregator 使用的影像版 CNN 建構函數。
    與 image_uav_client 一致：依 IMAGE_CNN_SIZE 選 simple 或 medium。
    """
    return _get_image_cnn_builder()(num_classes=output_dim)


# 調整全域模型配置：在影像實驗中使用 CNN 分支
config.MODEL_CONFIG["type"] = "cnn"
# 將 models.cnn.build_cnn 替換為影像版 CNN（僅影響本進程）
base_cnn.build_cnn = _build_image_cnn_for_server

import cloud_server_fixed  # noqa: E402


def _noop_schedule_global_test_eval(round_id: int, weights: dict) -> None:
    """
    Image 模式下不跑 tabular 全域評估（58 維 → CNN 會報錯）。
    評估僅由「Round 變更 + 收齊 delta」觸發的 _image_global_eval 負責。
    """
    return


# 避免 cloud_server_fixed 在影像實驗中排程 tabular 評估（會出現 conv2d 維度錯誤）
cloud_server_fixed._schedule_global_test_eval = _noop_schedule_global_test_eval


def _image_global_eval(round_id: int, weights: dict) -> None:
    """
    Image 模式下的簡化 Cloud 評估：
    - 使用 cifar10_poisoned.npz 的測試集做 clean acc / macro F1
    - 使用觸發測試集估計 ASR
    - 避免再走原本的 global_test_scaled.csv tabular pipeline
    """
    try:
        exp_dir = os.environ.get("EXPERIMENT_DIR", getattr(config, "LOG_DIR", "."))

        data_root = Path(__file__).resolve().parent
        dataset = os.environ.get("IMAGE_DATASET", "cifar10").strip().lower()
        if dataset == "road_signs":
            rs_npz = os.environ.get(
                "ROAD_SIGNS_NPZ_PATH",
                str(data_root / "road_signs_poisoned" / "road_signs_poisoned.npz"),
            )
            npz_path = Path(rs_npz)
        elif dataset == "gtsrb":
            gtsrb_npz = os.environ.get(
                "GTSRB_NPZ_PATH",
                str(data_root / "gtsrb_poisoned" / "gtsrb_poisoned.npz"),
            )
            npz_path = Path(gtsrb_npz)
        else:
            npz_path = data_root / "cifar10_poisoned" / "cifar10_poisoned.npz"
        if not npz_path.exists():
            print(f"[Cloud Server] ⚠️ Image eval: 找不到 {npz_path}（dataset={dataset}），跳過評估 (round={round_id})")
            return

        data = np.load(npz_path)
        x_clean = data["x_test_clean"].astype(np.float32) / 255.0
        y_clean = data["y_test_clean"].reshape(-1).astype(np.int64)
        # Trigger 評估集：
        # - trigger_target：主要用於 backdoor ASR（多半針對 source 類別）
        # - trigger_all：用於全類別「被誤判成 target」的 FTR（更能反映真實風險）
        x_trig_target = data.get("x_test_trigger_target")
        y_trig_target = data.get("y_test_trigger_target")
        x_trig_all = data.get("x_test_trigger_all")
        y_trig_all = data.get("y_test_trigger_all")
        # 注意：trigger 張量保持 npz 的 NHWC（uint8 或 float），由下方 _trigger_rate 統一
        # /255 → NCHW → normalize；不可在這裡先轉 NCHW，否則 _trigger_rate 會再 transpose 一次，
        # 形狀變成 [N,32,3,32] 而 conv 報錯（channels=32）。

        x_clean = np.transpose(x_clean, (0, 3, 1, 2))  # N,H,W,C -> N,C,H,W
        x_clean = _normalize_image_nchw_cloud(x_clean, dataset=dataset)
        ds_clean = TensorDataset(
            torch.from_numpy(x_clean),
            torch.from_numpy(y_clean),
        )

        # 雲端評估是否使用 GPU（用來避免小 GPU 記憶體造成 CUDA OOM）
        # 預設維持原本：只要 cuda 可用就用 GPU。
        eval_use_gpu = _env_flag("IMAGE_CLOUD_EVAL_USE_GPU", "1")
        device = torch.device("cuda" if (torch.cuda.is_available() and eval_use_gpu) else "cpu")
        num_classes = 4 if dataset == "road_signs" else (43 if dataset == "gtsrb" else int(config.MODEL_CONFIG.get("num_classes", 10)))
        model = _get_image_cnn_builder()(num_classes=num_classes)
        model = model.to(device)
        model.eval()

        # 載入聚合後權重
        cleaned = cloud_server_fixed._strip_state_dict_prefix(weights, ["module.", "model."])
        try:
            model.load_state_dict(cleaned, strict=False)
        except Exception as e:  # noqa: BLE001
            print(f"[Cloud Server] ⚠️ Image eval: 載入權重失敗 {e}，跳過評估 (round={round_id})")
            return

        def _run_eval(loader: DataLoader) -> tuple[float, float, float, list[int], list[int]]:
            all_pred: list[int] = []
            all_true: list[int] = []
            total_loss = 0.0
            n_batches = 0
            criterion = torch.nn.CrossEntropyLoss()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = torch.argmax(logits, dim=1)
                    all_pred.extend(preds.cpu().tolist())
                    all_true.extend(yb.cpu().tolist())
                    # 累積 cross-entropy loss
                    loss = criterion(logits, yb)
                    total_loss += float(loss.item())
                    n_batches += 1
            if not all_true:
                return 0.0, 0.0, 0.0, [], []
            acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
            f1 = float(f1_score(all_true, all_pred, average="macro"))
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            return acc, f1, avg_loss, all_true, all_pred

        # Clean 評估（batch 預設小，避免與多 client 同卡競爭顯存）
        cloud_eval_bs = _cloud_image_eval_batch_size()
        print(
            f"[Cloud Server] 📎 Image eval batch_size={cloud_eval_bs} "
            f"(IMAGE_CLOUD_EVAL_BATCH_SIZE / IMAGE_ASR_EVAL_BATCH_SIZE)",
            flush=True,
        )
        loader_clean = DataLoader(ds_clean, batch_size=cloud_eval_bs, shuffle=False, num_workers=0)
        clean_acc, clean_f1, clean_loss, clean_true, clean_pred = _run_eval(loader_clean)

        # per-class metrics（寫入 JSONL，方便論文表格/圖；不污染主 metrics.csv）
        try:
            import json
            from sklearn.metrics import precision_recall_fscore_support

            labels = sorted(set(clean_true)) if clean_true else []
            if labels:
                prec, rec, f1s, sup = precision_recall_fscore_support(
                    clean_true,
                    clean_pred,
                    labels=labels,
                )
                # 避免型別分析工具誤判（sklearn 回傳 array-like）
                prec = np.asarray(prec, dtype=float).tolist()
                rec = np.asarray(rec, dtype=float).tolist()
                f1s = np.asarray(f1s, dtype=float).tolist()
                sup = np.asarray(sup, dtype=int).tolist()
                per_class = {
                    str(int(lbl)): {
                        "precision": float(p),
                        "recall": float(r),
                        "f1": float(f),
                        "support": int(s),
                    }
                    for lbl, p, r, f, s in zip(labels, prec, rec, f1s, sup)
                }
            else:
                per_class = {}

            per_class_path = Path(exp_dir) / "image_cloud_per_class_metrics.jsonl"
            with open(per_class_path, "a", encoding="utf-8") as jf:
                jf.write(
                    json.dumps(
                        {
                            "round": int(round_id),
                            "dataset": dataset,
                            "num_classes": int(num_classes),
                            "per_class": per_class,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            # 論文除錯：後門目標類別在乾淨測試上的 recall 若長期≈0，觸發 ASR 通常也難上升
            try:
                _tl = int(os.environ.get("POISON_TARGET_LABEL", "8") or "8")
                _tk = str(int(_tl))
                if per_class and _tk in per_class:
                    _pc = per_class[_tk]
                    print(
                        f"[Cloud Server] 📊 乾淨測試 目標類別 {_tk}: "
                        f"precision={_pc['precision']:.4f} recall={_pc['recall']:.4f} "
                        f"f1={_pc['f1']:.4f} support={_pc['support']} "
                        f"(recall≈0 時請檢查 LR/epochs/類別不平衡與 client 有效更新)",
                        flush=True,
                    )
                elif per_class:
                    print(
                        f"[Cloud Server] ⚠️ 乾淨測試 per_class 無標籤 {_tk}（本輪 y_true 未含該類？）",
                        flush=True,
                    )
            except Exception:
                pass
        except Exception:
            pass


        # Trigger/ASR/FTR 評估（若有觸發集）
        # 指標命名：
        # - attack=none（IMAGE_POISON_ENABLED=0）：輸出 ftr（false trigger rate）
        # - attack=backdoor（IMAGE_POISON_ENABLED=1）：輸出 asr（attack success rate）
        poison_enabled = (os.environ.get("IMAGE_POISON_ENABLED", "1") or "1").strip().lower() not in ("0", "false", "no")
        ftr = 0.0
        asr = 0.0
        asr_source = 0.0
        ftr_all = 0.0
        trig_n = 0
        trig_all_n = 0
        target_label = int(os.environ.get("POISON_TARGET_LABEL", "8"))
        def _trigger_rate(x: Any, y: Any) -> tuple[float, int]:
            if x is None or y is None:
                return 0.0, 0
            x = x.astype(np.float32) / 255.0
            x = np.transpose(x, (0, 3, 1, 2))
            x = _normalize_image_nchw_cloud(x, dataset=dataset)
            y = y.reshape(-1).astype(np.int64)
            ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
            loader = DataLoader(ds, batch_size=cloud_eval_bs, shuffle=False, num_workers=0)
            all_pred: list[int] = []
            with torch.no_grad():
                for xb, _yb in loader:
                    xb = xb.to(device)
                    preds = torch.argmax(model(xb), dim=1)
                    all_pred.extend(preds.cpu().tolist())
            n = len(all_pred)
            if n <= 0:
                return 0.0, 0
            rate = float(np.mean(np.array(all_pred) == target_label))
            return rate, n

        # source-only（或偏 source）的 trigger_target：用來對齊既有 ASR 定義
        asr_source, trig_n = _trigger_rate(x_trig_target, y_trig_target)
        # all-classes trigger：更接近「貼 trigger 會不會被導向 target」；可降頻以加速雲端評估
        try:
            _every = int(os.environ.get("IMAGE_CLOUD_FTR_ALL_EVERY_N_ROUNDS", "1") or "1")
        except Exception:
            _every = 1
        _every = max(1, _every)
        # 每 N 輪跑一次：第 1 輪必跑，之後於 N, 2N, … 跑（export=1 表示每輪都跑）
        _rid = int(round_id)
        # _rid<=1：涵蓋首次評估（round 可能為 0 或 1，依聚合流程而定）
        _run_trig_all = _every <= 1 or _rid <= 1 or (_rid % _every == 0)
        if _run_trig_all:
            ftr_all, trig_all_n = _trigger_rate(x_trig_all, y_trig_all)
            ftr_all_cell: Any = ftr_all
            trig_all_n_cell: Any = trig_all_n
        else:
            ftr_all, trig_all_n = 0.0, 0
            ftr_all_cell = ""
            trig_all_n_cell = ""
            print(
                f"[Cloud Server] ⏭️ 略過全類別 trigger 評估（IMAGE_CLOUD_FTR_ALL_EVERY_N_ROUNDS={_every}, round={round_id}）",
                flush=True,
            )

        if poison_enabled:
            asr = asr_source
        else:
            # attack=none 時，沿用舊欄位 ftr（但現在我們也會另外輸出 ftr_all）
            if _run_trig_all and trig_all_n > 0:
                ftr = float(ftr_all)
            else:
                ftr = asr_source

        # 簡單寫一份 image 專用的 global metrics CSV
        import csv

        def _eval_quorum_fields() -> tuple[str, str]:
            """由 cloud_server_fixed 在觸發評估前寫入 CLOUD_EVAL_DIAG_*，供論文追溯。"""
            r = (os.environ.get("CLOUD_EVAL_DIAG_REPORTS", "") or "").strip()
            reg = (os.environ.get("CLOUD_EVAL_DIAG_REGISTERED", "") or "").strip()
            req = (os.environ.get("CLOUD_EVAL_DIAG_REQUIRED", "") or "").strip()
            met = f"{r}/{reg}" if r and reg else ""
            thr = f"{req}/{reg}" if req and reg else ""
            partial = (os.environ.get("CLOUD_EVAL_GRACE_PARTIAL", "") or "").strip().lower()
            if partial in ("1", "true", "yes", "on"):
                thr = f"{thr} grace_partial" if thr else "grace_partial"
            return met, thr

        eval_quorum_met, eval_quorum_threshold = _eval_quorum_fields()

        out_path = Path(exp_dir) / "image_cloud_metrics.csv"
        file_exists = out_path.exists()
        if file_exists:
            try:
                with open(out_path, "r", encoding="utf-8") as rf:
                    hdr = rf.readline()
                if hdr and "eval_quorum_met" not in hdr:
                    print(
                        "[Cloud Server] ⚠️ 既有 image_cloud_metrics.csv 表頭不含 eval_quorum_*；"
                        "新列將多出欄位，建議刪除該 CSV 後重跑以取得一致表頭。",
                        flush=True,
                    )
            except Exception:
                pass
        attacker_clients = (os.environ.get("IMAGE_ATTACKER_CLIENTS", "") or "").strip()
        def _count_attacker_ids(s: str) -> int:
            try:
                parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
                ids = []
                for p in parts:
                    try:
                        ids.append(int(p))
                    except Exception:
                        # 忽略非數字 token（例如誤傳 "auto"）
                        pass
                return len(sorted(set(ids)))
            except Exception:
                return 0

        attacker_count = _count_attacker_ids(attacker_clients)
        defense_mode = (os.environ.get("AGG_DEFENSE_MODE", "none") or "none").strip().lower()
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "round",
                        "clean_acc",
                        "clean_f1_macro",
                        "clean_loss",
                        "ftr",
                        "asr",
                        "asr_source",
                        "ftr_all",
                        "trig_n",
                        "trig_all_n",
                        "target_label",
                        "poison_enabled",
                        "attacker_clients",
                        "attacker_clients_n",
                        "defense_mode",
                        "eval_quorum_met",
                        "eval_quorum_threshold",
                    ]
                )
            writer.writerow(
                [
                    round_id,
                    clean_acc,
                    clean_f1,
                    clean_loss,
                    ftr,
                    asr,
                    asr_source,
                    ftr_all_cell,
                    trig_n,
                    trig_all_n_cell,
                    target_label,
                    1 if poison_enabled else 0,
                    attacker_clients,
                    attacker_count,
                    defense_mode,
                    eval_quorum_met,
                    eval_quorum_threshold,
                ]
            )

        print(
            f"[Cloud Server] 📊 Image eval round={round_id}: "
            f"clean_acc={clean_acc:.4f}, clean_f1={clean_f1:.4f}, clean_loss={clean_loss:.4f}, "
            + (f"asr={asr:.4f}" if poison_enabled else f"ftr={ftr:.4f}")
            + f" (n={trig_n})"
            + (f", eval_quorum={eval_quorum_met} (threshold {eval_quorum_threshold})" if eval_quorum_met else "")
        )
        try:
            # 讓使用者更容易確認 metrics 有持續寫入
            line_count = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
            print(f"[Cloud Server] 📝 已寫入 image_cloud_metrics.csv（總行數含表頭={line_count}）")
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        print(f"[Cloud Server] ⚠️ Image eval 發生錯誤 (round={round_id}): {e}")


def _image_maybe_cloud_finetune_then_eval(round_id: int, weights: dict) -> None:
    """
    覆寫原本的 _maybe_cloud_finetune_then_eval：
    - Image 模式下不再使用 tabular global_test_scaled.csv
    - 直接對 CIFAR-10 測試集做一次評估並記錄 metrics
    """
    if not weights:
        print(f"[Cloud Server] ⚠️ Image eval: 權重為空，跳過 round={round_id}")
        return
    _image_global_eval(round_id, weights)


# 將 Cloud 的微調 + 評估邏輯替換為 image 版本（僅影像實驗進程）
cloud_server_fixed._maybe_cloud_finetune_then_eval = _image_maybe_cloud_finetune_then_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Server for Federated Learning (image entry)")
    parser.add_argument("--port", type=int, default=None, help="Port to run the server on")
    parser.add_argument("--host", type=str, default=None, help="Host to bind the server to")
    args = parser.parse_args()

    server_port = args.port if args.port is not None else cloud_server_fixed.config.NETWORK_CONFIG["cloud_server"]["port"]
    server_host = args.host if args.host is not None else cloud_server_fixed.config.NETWORK_CONFIG["cloud_server"]["host"]

    # 沿用原本實驗目錄邏輯
    experiment_dir = os.environ.get("EXPERIMENT_DIR", None)
    if experiment_dir:
        result_dir = experiment_dir
        print(f"[Cloud Server] 使用環境變量實驗目錄: {result_dir}")
    else:
        import datetime

        now = datetime.datetime.now().strftime("tokyo_drone_fixed_%Y%m%d_%H%M%S")
        result_dir = os.path.join(cloud_server_fixed.config.LOG_DIR, now)
        os.makedirs(result_dir, exist_ok=True)
        print(f"[Cloud Server] 創建新實驗目錄: {result_dir}")

    os.makedirs(result_dir, exist_ok=True)
    os.environ["EXPERIMENT_DIR"] = result_dir
    os.environ["LOG_DIR"] = result_dir

    print("[Cloud Server] 🚀 啟動中...")
    print(f"  服務地址: http://{server_host}:{server_port}")
    print("  職責: 全局權重聚合")
    print(f"  使用實驗目錄: {result_dir}")
    print(f"  日誌格式: {cloud_server_fixed.config.LOG_CONFIG.get('result_log_format', 'csv')}")

    # 端口占用檢查（沿用原本輸出）
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((server_host, server_port))
        sock.close()
        if result == 0:
            print(f"[Cloud Server] ⚠️ 警告：端口 {server_port} 已被占用，可能導致啟動失敗")
            print("[Cloud Server] 💡 建議：檢查是否有其他 Cloud Server 實例正在運行")
    except Exception as e:
        print(f"[Cloud Server] ⚠️ 端口檢查失敗: {e}")

    # 標記檔
    marker_file = os.path.join(result_dir, "cloud_server_using.txt")
    try:
        import datetime

        with open(marker_file, "w", encoding="utf-8") as f:
            f.write("雲端服務器使用此實驗目錄\n")
            f.write(f"使用時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"雲端服務器ID: {cloud_server_fixed.cloud_server_id}\n")
            f.write(f"目錄路徑: {result_dir}\n")
            f.write(f"服務地址: http://{server_host}:{server_port}\n")
    except Exception as e:
        print(f"[Cloud Server] ⚠️ 無法創建標記文件: {e}")

    # 狀態報告線程
    threading.Thread(target=cloud_server_fixed.log_cloud_status, daemon=True).start()

    # 廣播線程（若原模組有定義）
    def run_broadcast():
        try:
            import asyncio

            asyncio.run(cloud_server_fixed.broadcast_global_weights())
        except Exception as e:
            print(f"[Cloud Server] ⚠️ 廣播線程啟動失敗: {e}")

    try:
        threading.Thread(target=run_broadcast, daemon=True).start()
    except Exception as e:
        print(f"[Cloud Server] ⚠️ 無法啟動廣播線程: {e}")

    import uvicorn

    uvicorn.run(cloud_server_fixed.app, host=server_host, port=server_port)

