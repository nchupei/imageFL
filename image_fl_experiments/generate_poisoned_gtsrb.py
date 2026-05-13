#!/usr/bin/env python3
"""
將 image_fl_experiments/GTSRB 轉成與既有 pipeline 相容的 npz（32x32, uint8, NHWC）。

輸出 keys（必須與 generate_poisoned_road_signs.py 相同）：
  x_train_clean, y_train_clean
  x_train_poisoned, y_train_poisoned
  x_test_clean, y_test_clean
  x_test_poisoned, y_test_poisoned
  x_test_trigger_target, y_test_trigger_target
  x_test_trigger_all, y_test_trigger_all

資料來源：
  - GTSRB/Train.csv, GTSRB/Test.csv
  - Path 欄位為相對於 GTSRB 根目錄的圖片路徑
  - Roi.X1, Roi.Y1, Roi.X2, Roi.Y2 提供 ROI bbox（含邊界）

最小用法：
  cd image_fl_experiments
  python generate_poisoned_gtsrb.py

可用環境變數（沿用既有 poisoning 定義）：
  - POISON_SOURCE_LABEL (default: 1)
  - POISON_TARGET_LABEL (default: 2)
  - POISON_K (default: 200)
  - POISON_K_TEST (default: 500)

Trigger 參數（建議使用 GTSRB_TRIGGER_*；若未設，會 fallback 到舊的 ROAD_SIGNS_TRIGGER_*，避免舊指令失效）：
  - GTSRB_TRIGGER_PATCH_SIZE (default: 4)
  - GTSRB_TRIGGER_PATCH_VALUE (default: 255)
  - GTSRB_TRIGGER_ALPHA (default: 1.0)

GTSRB 讀取/裁切/抽樣參數：
  - GTSRB_ROOT (default: <this_dir>/GTSRB)
  - GTSRB_SEED (default: 0)
  - GTSRB_MAX_TRAIN (default: 0 => 不限制)
  - GTSRB_MAX_TEST (default: 0 => 不限制)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class _Row:
    path: str
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


def _env_int(key: str, default: int) -> int:
    v = str(os.environ.get(key, "") or "").strip()
    if not v:
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    v = str(os.environ.get(key, "") or "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_first_int(keys: List[str], default: int) -> int:
    for k in keys:
        v = str(os.environ.get(k, "") or "").strip()
        if not v:
            continue
        try:
            return int(float(v))
        except Exception:
            continue
    return int(default)


def _env_first_float(keys: List[str], default: float) -> float:
    for k in keys:
        v = str(os.environ.get(k, "") or "").strip()
        if not v:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return float(default)


def _read_csv_rows(csv_path: Path) -> List[_Row]:
    rows: List[_Row] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    _Row(
                        path=str(r["Path"]),
                        class_id=int(r["ClassId"]),
                        x1=int(float(r["Roi.X1"])),
                        y1=int(float(r["Roi.Y1"])),
                        x2=int(float(r["Roi.X2"])),
                        y2=int(float(r["Roi.Y2"])),
                    )
                )
            except Exception:
                # 跳過格式不完整列
                continue
    return rows


def _load_and_preprocess_image(
    gtsrb_root: Path,
    row: _Row,
    *,
    out_size: int = 32,
) -> Optional[np.ndarray]:
    """
    回傳 uint8 HWC (out_size,out_size,3)；讀不到就回 None。
    """
    # 避免強依賴 cv2；用 PIL 即可
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("缺少 Pillow。請先安裝：pip install pillow")

    img_path = gtsrb_root / row.path
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            # clamp ROI
            x1 = max(0, min(int(row.x1), w - 1))
            y1 = max(0, min(int(row.y1), h - 1))
            x2 = max(0, min(int(row.x2), w - 1))
            y2 = max(0, min(int(row.y2), h - 1))
            if x2 <= x1 or y2 <= y1:
                # fallback：不用 ROI
                crop = im
            else:
                # PIL crop：右下角座標為「不含」，所以要 +1
                crop = im.crop((x1, y1, x2 + 1, y2 + 1))
            # Pillow 10+ 建議用 Image.Resampling；舊版仍可用 Image.BILINEAR
            try:
                resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
            except Exception:
                # 避免型別檢查器對 Image.BILINEAR 報錯：用 getattr 或整數常數（2=bilinear）
                resample = getattr(Image, "BILINEAR", 2)
            crop = crop.resize((out_size, out_size), resample=resample)
            arr = np.asarray(crop, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return None
            return arr
    except Exception:
        return None


def _apply_patch(
    x: np.ndarray,
    indices: np.ndarray,
    *,
    patch_size: int,
    patch_value: int,
    alpha: float,
) -> np.ndarray:
    """
    對 x（uint8 NHWC）在右下角貼方形 trigger。
    """
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[-1] != 3:
        return x
    if indices.size == 0:
        return x

    ps = int(max(1, patch_size))
    pv = int(np.clip(patch_value, 0, 255))
    a = float(max(0.0, min(1.0, alpha)))

    h = x.shape[1]
    w = x.shape[2]
    y0 = max(0, h - ps)
    x0 = max(0, w - ps)

    # alpha blend on uint8
    patch = np.full((ps, ps, 3), pv, dtype=np.float32)
    for idx in indices.astype(np.int64):
        if idx < 0 or idx >= x.shape[0]:
            continue
        region = x[idx, y0:h, x0:w, :].astype(np.float32)
        blended = (1.0 - a) * region + a * patch[: region.shape[0], : region.shape[1], :]
        x[idx, y0:h, x0:w, :] = np.clip(blended, 0, 255).astype(np.uint8)
    return x


def _stack_xy(
    gtsrb_root: Path,
    rows: List[_Row],
    *,
    max_n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows), dtype=np.int64)
    rng.shuffle(idx)
    if max_n > 0:
        idx = idx[: int(max_n)]

    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in idx.tolist():
        r = rows[i]
        arr = _load_and_preprocess_image(gtsrb_root, r, out_size=32)
        if arr is None:
            continue
        xs.append(arr)
        ys.append(int(r.class_id))

    if not xs:
        return np.zeros((0, 32, 32, 3), dtype=np.uint8), np.zeros((0, 1), dtype=np.int64)
    x = np.stack(xs, axis=0).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64).reshape(-1, 1)
    return x, y


def main() -> None:
    here = Path(__file__).resolve().parent
    gtsrb_root = Path(os.environ.get("GTSRB_ROOT", str(here / "GTSRB")))
    if not gtsrb_root.exists():
        raise SystemExit(f"❌ 找不到 GTSRB_ROOT={gtsrb_root}")

    seed = _env_int("GTSRB_SEED", 0)
    rng = np.random.default_rng(seed)

    poison_k = _env_int("POISON_K", 200)
    poison_k_test = _env_int("POISON_K_TEST", 500)
    source_label = _env_int("POISON_SOURCE_LABEL", 1)
    target_label = _env_int("POISON_TARGET_LABEL", 2)

    # Trigger 參數：以 GTSRB_TRIGGER_* 為主；若未設才 fallback 舊命名（road_signs）。
    patch_size = _env_first_int(["GTSRB_TRIGGER_PATCH_SIZE", "ROAD_SIGNS_TRIGGER_PATCH_SIZE"], 4)
    patch_value = _env_first_int(["GTSRB_TRIGGER_PATCH_VALUE", "ROAD_SIGNS_TRIGGER_PATCH_VALUE"], 255)
    patch_alpha = _env_first_float(["GTSRB_TRIGGER_ALPHA", "ROAD_SIGNS_TRIGGER_ALPHA"], 1.0)

    max_train = _env_int("GTSRB_MAX_TRAIN", 0)
    max_test = _env_int("GTSRB_MAX_TEST", 0)

    train_csv = gtsrb_root / "Train.csv"
    test_csv = gtsrb_root / "Test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit(f"❌ 找不到 Train.csv/Test.csv：{train_csv}, {test_csv}")

    print(f"📄 讀取 CSV: {train_csv.name}, {test_csv.name}")
    train_rows = _read_csv_rows(train_csv)
    test_rows = _read_csv_rows(test_csv)
    print(f"   rows: train={len(train_rows)}, test={len(test_rows)}")

    print("🖼️ 載入 + ROI 裁切 + resize(32x32) ...（第一次會較久）")
    x_train_clean, y_train_clean = _stack_xy(gtsrb_root, train_rows, max_n=max_train, seed=seed + 1)
    x_test_clean, y_test_clean = _stack_xy(gtsrb_root, test_rows, max_n=max_test, seed=seed + 2)

    if x_train_clean.shape[0] <= 0 or x_test_clean.shape[0] <= 0:
        raise SystemExit("❌ 讀入的 train/test 為空。請確認 GTSRB 檔案完整且 Path 可讀。")

    num_classes = int(max(int(y_train_clean.max()), int(y_test_clean.max())) + 1)
    print(f"✅ loaded: train={len(x_train_clean)}, test={len(x_test_clean)}, num_classes≈{num_classes}")

    # ============== Train poisoning（source -> target） ==============
    idx_source = np.where(y_train_clean.reshape(-1) == int(source_label))[0]
    if idx_source.size == 0:
        raise SystemExit(f"❌ 在訓練集中找不到 source_label={source_label}（無法套用 backdoor）")
    poison_k_used = int(min(int(poison_k), int(idx_source.size)))
    poison_train_idx = rng.choice(idx_source, size=poison_k_used, replace=False)

    x_train_poisoned = x_train_clean.copy()
    x_train_poisoned = _apply_patch(
        x_train_poisoned,
        poison_train_idx.astype(np.int64),
        patch_size=patch_size,
        patch_value=patch_value,
        alpha=patch_alpha,
    )
    y_train_poisoned = y_train_clean.copy()
    y_train_poisoned[poison_train_idx] = int(target_label)

    # ============== Test poisoning（保留原標籤） ==============
    idx_test_source = np.where(y_test_clean.reshape(-1) == int(source_label))[0]
    base_indices = idx_test_source if idx_test_source.size > 0 else np.arange(y_test_clean.shape[0], dtype=np.int64)
    poison_k_test_source = int(min(int(poison_k_test), int(base_indices.size)))
    poison_test_idx = rng.choice(base_indices, size=poison_k_test_source, replace=False) if poison_k_test_source > 0 else np.array([], dtype=np.int64)

    x_test_poisoned = x_test_clean.copy()
    x_test_poisoned = _apply_patch(
        x_test_poisoned,
        poison_test_idx.astype(np.int64),
        patch_size=patch_size,
        patch_value=patch_value,
        alpha=patch_alpha,
    )
    y_test_poisoned = y_test_clean.copy()  # 保留原標籤

    # ============== Trigger + target_label（source-only ASR） ==============
    if poison_test_idx.size > 0:
        x_test_trigger_target = x_test_poisoned[poison_test_idx.astype(np.int64)].copy()
        y_test_trigger_target = np.full((poison_test_idx.size,), int(target_label), dtype=np.int64)
    else:
        x_test_trigger_target = np.zeros((0,) + x_test_clean.shape[1:], dtype=np.uint8)
        y_test_trigger_target = np.zeros((0,), dtype=np.int64)

    # ============== Trigger（all-classes FTR） ==============
    base_all = np.arange(y_test_clean.shape[0], dtype=np.int64)
    poison_k_test_all = int(min(int(poison_k_test), int(base_all.size)))
    idx_all = rng.choice(base_all, size=poison_k_test_all, replace=False) if poison_k_test_all > 0 else np.array([], dtype=np.int64)
    if idx_all.size > 0:
        x_test_trigger_all = x_test_clean[idx_all].copy()
        x_test_trigger_all = _apply_patch(
            x_test_trigger_all,
            np.arange(idx_all.size, dtype=np.int64),
            patch_size=patch_size,
            patch_value=patch_value,
            alpha=patch_alpha,
        )
        y_test_trigger_all = np.full((idx_all.size,), int(target_label), dtype=np.int64)
    else:
        x_test_trigger_all = np.zeros((0,) + x_test_clean.shape[1:], dtype=np.uint8)
        y_test_trigger_all = np.zeros((0,), dtype=np.int64)

    # ============== 寫出 npz / stats ==============
    out_dir = here / "gtsrb_poisoned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gtsrb_poisoned.npz"
    print(f"💾 寫出 GTSRB 中毒資料到: {out_path}")
    np.savez_compressed(
        out_path,
        x_train_clean=x_train_clean,
        y_train_clean=y_train_clean,
        x_train_poisoned=x_train_poisoned,
        y_train_poisoned=y_train_poisoned,
        x_test_clean=x_test_clean,
        y_test_clean=y_test_clean,
        x_test_poisoned=x_test_poisoned,
        y_test_poisoned=y_test_poisoned,
        x_test_trigger_target=x_test_trigger_target,
        y_test_trigger_target=y_test_trigger_target,
        x_test_trigger_all=x_test_trigger_all,
        y_test_trigger_all=y_test_trigger_all,
    )

    try:
        import json

        stats = {
            "dataset": "gtsrb",
            "gtsrb_root": str(gtsrb_root),
            "seed": int(seed),
            "num_classes_inferred": int(num_classes),
            "train_n": int(x_train_clean.shape[0]),
            "test_n": int(x_test_clean.shape[0]),
            "source_label": int(source_label),
            "target_label": int(target_label),
            "poison_k_requested": int(_env_int("POISON_K", 200)),
            "poison_k_used": int(poison_k_used),
            "poison_k_test_requested": int(_env_int("POISON_K_TEST", 500)),
            "poison_k_test_source_used": int(poison_k_test_source),
            "poison_k_test_all_used": int(poison_k_test_all),
            "trigger_patch_size": int(patch_size),
            "trigger_patch_value": int(patch_value),
            "trigger_alpha": float(patch_alpha),
            "max_train": int(max_train),
            "max_test": int(max_test),
            "train_counts_head": np.bincount(y_train_clean.reshape(-1), minlength=num_classes)[: min(num_classes, 20)].astype(int).tolist(),
            "test_counts_head": np.bincount(y_test_clean.reshape(-1), minlength=num_classes)[: min(num_classes, 20)].astype(int).tolist(),
        }
        stats_path = out_dir / "gtsrb_poisoned_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"🧾 已寫出資料統計: {stats_path}")
    except Exception:
        pass

    print("✅ 完成 GTSRB 中毒資料生成。之後可在 FL client 端以 IMAGE_DATASET=gtsrb 載入使用。")


if __name__ == "__main__":
    main()

