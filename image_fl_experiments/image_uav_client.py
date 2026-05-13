#!/usr/bin/env python3
"""
影像版 Client。

模式：
- 預設：沿用 `uav_client_fixed.py`（tabular pipeline）
- IMAGE_FL=1：讀取 `image_fl_experiments/cifar10_poisoned/cifar10_poisoned.npz`，
  用簡單 CNN 訓練並透過「相同的聚合器 API」上傳權重。

輕量評估（緩解 CPU 過載／每輪評估過久）：
- IMAGE_EVAL_TEST_MAX_SAMPLES：測試集最多抽樣筆數（0=全量）；IMAGE_EVAL_SUBSAMPLE_SEED
- IMAGE_CLEAN_EVAL_EVERY_ROUNDS：每 N 輪做一次 clean acc/f1（其餘輪沿用上一輪指標）
- IMAGE_ASR_EVAL_EVERY_ROUNDS：每 N 輪做一次 ASR/FTR 推論（其餘輪沿用上一輪）
- IMAGE_ASR_EVAL_MAX_SAMPLES：trigger 集最多抽樣筆數；IMAGE_ASR_SUBSAMPLE_SEED
- IMAGE_CLIENT_EVAL_BSZ：client 端 clean eval 的 batch（>0 時覆寫 test_loader，預設 0=維持原本 max(train_bs,256)，易在 GPU OOM）
- IMAGE_CLIENT_EVAL_ENABLED：設 0 可完全跳過 client 端 clean eval（沿用上一輪指標；首輪為 0）

訓練穩定性／表徵器（可選）：
- IMAGE_CNN_SIZE：small | medium | resnet18 | resnet34 | resnet50（後三者需 torchvision）
- IMAGE_LR_SCHEDULE：constant（預設）| cosine | linear；搭配 IMAGE_LR_MIN_FACTOR（末輪 lr ≈ base×factor）
- IMAGE_MAX_GRAD_NORM：>0 時對梯度做 clip_grad_norm_（例如 1.0）
- IMAGE_LABEL_SMOOTHING：>0 且 PyTorch 支援時，CrossEntropyLoss(label_smoothing=...)
"""

import argparse
import asyncio
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 讓在 image_fl_experiments 下執行時也能 import 到專案模組
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uav_client_fixed  # noqa: E402


def _env_flag(name: str, default: str = "0") -> bool:
    try:
        return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or str(default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip() or str(default))
    except Exception:
        return float(default)


def _fl_round_lr(base_lr: float, *, round_id: int, max_rounds: int) -> float:
    """
    依 FL 輪次縮放 client 端學習率（每輪重新建立 optimizer 前呼叫）。
    - IMAGE_LR_SCHEDULE: constant | cosine | linear
    - IMAGE_LR_MIN_FACTOR: 末輪目標約為 base_lr * factor（預設 0.1）
    """
    import math

    sched = (os.environ.get("IMAGE_LR_SCHEDULE", "") or "constant").strip().lower()
    if sched in ("", "none", "constant", "off", "0", "false", "no"):
        return float(base_lr)
    mf = max(0.0, min(1.0, _env_float("IMAGE_LR_MIN_FACTOR", 0.1)))
    mr = max(1, int(max_rounds))
    r = max(1, min(int(round_id), mr))
    denom = max(1, mr - 1)
    t = (r - 1) / float(denom)
    lo = float(base_lr) * mf
    hi = float(base_lr)
    if sched in ("cosine", "cos"):
        return float(lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * t)))
    if sched in ("linear", "lin"):
        return float(hi + (lo - hi) * t)
    return float(base_lr)


def _seed_everything(seed: int) -> None:
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_cifar10_npz(npz_path: Path) -> Dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到 CIFAR-10 npz: {npz_path}")
    obj = dict(__import__("numpy").load(npz_path, allow_pickle=False))
    return obj


def _split_indices(n: int, client_id: int, num_clients: int, seed: int = 42) -> "list[int]":
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    client_id = int(client_id)
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id 超出範圍: {client_id} (num_clients={num_clients})")
    return parts[client_id].tolist()


def _split_indices_dirichlet(
    y: Any,
    client_id: int,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> "list[int]":
    """
    依 Dirichlet(alpha) 做 non-IID 分割：每個類別依比例分到各 client，
    alpha 越小類別分佈越不平均（越 non-IID），越大越接近 IID。
    y: 標籤陣列 shape (N,) 或 (N,1)，類別 0..num_classes-1
    """
    import numpy as np

    y_flat = np.asarray(y).reshape(-1).astype(np.int64)
    n = len(y_flat)
    num_clients = int(num_clients)
    client_id = int(client_id)
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id 超出範圍: {client_id} (num_clients={num_clients})")

    rng = np.random.default_rng(seed)
    num_classes = int(y_flat.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(y_flat == k)[0]
        rng.shuffle(idx_k)
        n_k = len(idx_k)
        if n_k == 0:
            continue
        probs = rng.dirichlet(alpha * np.ones(num_clients))
        n_per_client = rng.multinomial(n_k, probs)
        start = 0
        for i in range(num_clients):
            end = start + n_per_client[i]
            client_indices[i].extend(idx_k[start:end].tolist())
            start = end

    return client_indices[client_id]


def _build_simple_cifar_cnn(num_classes: int = 10):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    return Net()


def _build_medium_cifar_cnn(num_classes: int = 10):
    """
    中等深度 CNN（4 個 conv block + BN），容量介於 simple 與 centralized 版之間。
    用於 60/6 等大規模 FL，提升 clean F1。
    """
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    return Net()


def _build_resnet18_image_cnn(num_classes: int = 10):
    """
    ResNet-18，適配 32×32 等低解析度輸入（GTSRB / CIFAR-10 常用改法）：
    - 首層改為 3×3、stride=1（取代 7×7 stride=2，避免特徵圖過早縮小）
    - 移除第一個 maxpool（改為 Identity）

    與淺層 `small` / `medium` 2D-CNN 對照時，可作為論文「主線 backbone」。
    需安裝 torchvision。
    """
    import torch.nn as nn

    try:
        import torchvision.models as models
    except ImportError as e:
        raise ImportError("ResNet-18 需要 torchvision；請安裝 torchvision") from e

    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, int(num_classes))
    return m


def _build_resnet34_image_cnn(num_classes: int = 10):
    import torch.nn as nn

    try:
        import torchvision.models as models
    except ImportError as e:
        raise ImportError("ResNet-34 需要 torchvision；請安裝 torchvision") from e

    m = models.resnet34(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, int(num_classes))
    return m


def _build_resnet50_image_cnn(num_classes: int = 10):
    import torch.nn as nn

    try:
        import torchvision.models as models
    except ImportError as e:
        raise ImportError("ResNet-50 需要 torchvision；請安裝 torchvision") from e

    m = models.resnet50(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, int(num_classes))
    return m


def _get_image_cnn_builder():
    """依環境變數 IMAGE_CNN_SIZE 回傳 builder：small | medium | resnet18 | resnet34 | resnet50。"""
    size = (os.environ.get("IMAGE_CNN_SIZE", "small") or "small").strip().lower()
    if size in ("medium", "m"):
        return _build_medium_cifar_cnn
    if size in ("resnet18", "resnet-18", "resnet"):
        return _build_resnet18_image_cnn
    if size in ("resnet34", "resnet-34"):
        return _build_resnet34_image_cnn
    if size in ("resnet50", "resnet-50"):
        return _build_resnet50_image_cnn
    return _build_simple_cifar_cnn


# CIFAR-10 標準化（/255 之後）：與 PyTorch 常用 mean/std 一致
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _normalize_cifar10_nchw(x):
    """x: numpy float32 NCHW，已 /255 在 [0,1]。原地不改，回傳 (x - mean) / std。"""
    import numpy as np
    if not _env_flag("IMAGE_CIFAR_NORMALIZE", "1"):
        return x
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def _load_gtsrb_norm_stats() -> "tuple[tuple[float, float, float], tuple[float, float, float]]":
    """
    讀取 GTSRB 專用 mean/std（float in [0,1] space），供 normalize 使用。
    預設路徑：image_fl_experiments/gtsrb_poisoned/gtsrb_norm.json
    可用環境變數覆蓋：
      - GTSRB_NORM_PATH
    """
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


def _normalize_gtsrb_nchw(x):
    """x: numpy float32 NCHW，已 /255 在 [0,1]。若 IMAGE_GTSRB_NORMALIZE=1 則套用 gtsrb_norm.json。"""
    import numpy as np

    if not _env_flag("IMAGE_GTSRB_NORMALIZE", "0"):
        return x
    mean, std = _load_gtsrb_norm_stats()
    mean_a = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_a = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean_a) / std_a


def _normalize_image_nchw(x, *, dataset: str):
    """
    統一入口：依資料集與環境變數選擇 normalize。
    - CIFAR-10：預設開啟（IMAGE_CIFAR_NORMALIZE=1）
    - GTSRB：預設關閉（IMAGE_GTSRB_NORMALIZE=0），你可明確打開
    - 其他：不做
    """
    ds = (dataset or "").strip().lower()
    if ds == "cifar10":
        return _normalize_cifar10_nchw(x)
    if ds == "gtsrb":
        return _normalize_gtsrb_nchw(x)
    return x


def _maybe_resize_gtsrb_nchw(x, *, dataset: str):
    """
    對 GTSRB 輸入做可選 resize（預設開啟，目標 32x32）以降低計算/顯存壓力。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    ds = (dataset or "").strip().lower()
    if ds != "gtsrb":
        return x
    if not _env_flag("IMAGE_GTSRB_RESIZE", "1"):
        return x
    target = _env_int("IMAGE_INPUT_SIZE", 32)
    if target <= 0:
        return x
    if x.ndim != 4:
        return x
    if int(x.shape[-1]) == target and int(x.shape[-2]) == target:
        return x

    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    xt = F.interpolate(xt, size=(target, target), mode="bilinear", align_corners=False)
    return xt.numpy()


def _make_loaders_from_npz(
    npz: Dict[str, Any],
    client_id: int,
    num_clients: int,
    use_poisoned_train: bool,
    batch_size: int,
    target_label: int,
) -> Tuple[Any, Any, int, int, float]:
    import numpy as np
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset

    # 預設：從整體 npz 依 client_id 分割
    x_train = npz["x_train_poisoned" if use_poisoned_train else "x_train_clean"]
    y_train = npz["y_train_poisoned" if use_poisoned_train else "y_train_clean"].reshape(-1)

    # 允許使用預先切好的 per-client splits：
    # - IMAGE_USE_CLIENT_SPLITS=1 時優先讀 splits/client_{id}.npz 的 x_train, y_train
    # - Road Signs: road_signs_poisoned/splits/
    # - CIFAR-10:  cifar10_poisoned/splits/
    dataset = os.environ.get("IMAGE_DATASET", "cifar10").strip().lower()
    use_client_splits = os.environ.get("IMAGE_USE_CLIENT_SPLITS", "0").strip() == "1"
    if use_client_splits:
        root = Path(__file__).resolve().parent
        if dataset == "road_signs":
            split_npz_path = root / "road_signs_poisoned" / "splits" / f"client_{client_id}.npz"
        elif dataset == "gtsrb":
            split_npz_path = root / "gtsrb_poisoned" / "splits" / f"client_{client_id}.npz"
        else:
            split_npz_path = root / "cifar10_poisoned" / "splits" / f"client_{client_id}.npz"
        if split_npz_path.exists():
            print(f"[Image Client] 使用預先切好的 split: {split_npz_path}")
            split_npz = np.load(split_npz_path)
            x_train = split_npz["x_train"]
            y_train = split_npz["y_train"].reshape(-1)
            train_idx = np.arange(len(x_train))
        else:
            print(f"[Image Client] ⚠️ 找不到 split 檔 {split_npz_path}，改用 on-the-fly 分割")
            split_npz = None
            split_mode = (os.environ.get("IMAGE_SPLIT", "shuffle") or "shuffle").strip().lower()
            if split_mode == "dirichlet":
                alpha = float(os.environ.get("IMAGE_DIRICHLET_ALPHA", "0.5"))
                train_idx = _split_indices_dirichlet(y_train, client_id, num_clients, alpha=alpha, seed=42)
            else:
                train_idx = _split_indices(len(x_train), client_id, num_clients, seed=42)
    else:
        split_mode = (os.environ.get("IMAGE_SPLIT", "shuffle") or "shuffle").strip().lower()
        if split_mode == "dirichlet":
            alpha = float(os.environ.get("IMAGE_DIRICHLET_ALPHA", "0.5"))
            train_idx = _split_indices_dirichlet(y_train, client_id, num_clients, alpha=alpha, seed=42)
        else:
            train_idx = _split_indices(len(x_train), client_id, num_clients, seed=42)

    x_test = npz["x_test_clean"]
    y_test = npz["y_test_clean"].reshape(-1)

    # uint8 NHWC -> float32 NCHW，/255 後再 normalize（依 dataset）
    Xtr = (x_train[train_idx].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    Xtr = _maybe_resize_gtsrb_nchw(Xtr, dataset=dataset)
    Xtr = _normalize_image_nchw(Xtr, dataset=dataset)
    ytr = y_train[train_idx].astype(np.int64)

    Xte = (x_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    Xte = _maybe_resize_gtsrb_nchw(Xte, dataset=dataset)
    Xte = _normalize_image_nchw(Xte, dataset=dataset)
    yte = y_test.astype(np.int64)

    n_test_full = int(len(Xte))
    max_te = _env_int("IMAGE_EVAL_TEST_MAX_SAMPLES", 0)
    if max_te > 0 and max_te < n_test_full:
        sub_seed = _env_int("IMAGE_EVAL_SUBSAMPLE_SEED", 42)
        rng = np.random.RandomState(sub_seed + int(client_id))
        pick = rng.choice(n_test_full, size=max_te, replace=False)
        pick.sort()
        Xte = Xte[pick]
        yte = yte[pick]
        print(
            f"[Image Client] 測試集抽樣: IMAGE_EVAL_TEST_MAX_SAMPLES={max_te} "
            f"(自 {n_test_full} 筆中抽取，seed_base={sub_seed}+cid)",
            flush=True,
        )

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    test_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    if dataset == "gtsrb" and _env_flag("IMAGE_GTSRB_RESIZE", "1"):
        _sz = _env_int("IMAGE_INPUT_SIZE", 32)
        print(f"[Image Client] GTSRB resize: enabled=1 target={_sz}x{_sz}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=max(batch_size, 256), shuffle=False, num_workers=0)
    train_samples = int(len(train_ds))
    poison_train_samples = 0
    if use_poisoned_train and train_samples > 0:
        poison_train_samples = int(np.sum(ytr == int(target_label)))
    poison_ratio = (float(poison_train_samples) / float(train_samples)) if train_samples > 0 else 0.0
    return train_loader, test_loader, train_samples, poison_train_samples, poison_ratio


def _maybe_augment_batch(xb, *, mode: str = "clean"):
    """
    對影像 batch 做簡單增強。

    - mode="clean": 可用幾何增強（crop/flip）+ 顏色/模糊
    - mode="poison_safe": 僅做「不會裁掉角落」的增強（顏色/模糊/噪聲），避免 trigger 被裁掉或翻面

    由環境變數控制是否啟用：
    - IMAGE_AUG_ENABLED: "1" 啟用（預設 0）
    - IMAGE_AUG_FLIP_P: 水平翻轉機率（預設 0.5）
    """
    import os
    import torch
    import torch.nn.functional as F

    # 預設：CIFAR-10 開啟；其他資料集關閉（可用環境變數覆蓋）
    dataset = (os.environ.get("IMAGE_DATASET", "cifar10") or "cifar10").strip().lower()
    default_enabled = "1" if dataset == "cifar10" else "0"
    if not _env_flag("IMAGE_AUG_ENABLED", default_enabled):
        return xb

    if xb.dim() != 4 or xb.size(2) != 32 or xb.size(3) != 32:
        return xb

    b = xb.size(0)
    device = xb.device

    mode = (mode or "clean").strip().lower()

    # ----------------------------
    # photometric augmentation (safe for triggers)
    # ----------------------------
    # brightness/contrast jitter
    try:
        jitter = float(os.environ.get("IMAGE_AUG_JITTER", "0.15"))
    except Exception:
        jitter = 0.15
    jitter = max(0.0, min(0.5, jitter))
    if jitter > 0:
        # brightness factor in [1-j, 1+j]
        br = 1.0 + (torch.rand((b, 1, 1, 1), device=device) * 2 - 1.0) * jitter
        # contrast factor in [1-j, 1+j]
        ct = 1.0 + (torch.rand((b, 1, 1, 1), device=device) * 2 - 1.0) * jitter
        mean = xb.mean(dim=(2, 3), keepdim=True)
        xb = (xb - mean) * ct + mean
        xb = xb * br

    # gaussian noise
    try:
        noise_std = float(os.environ.get("IMAGE_AUG_NOISE_STD", "0.02"))
    except Exception:
        noise_std = 0.02
    noise_std = max(0.0, min(0.2, noise_std))
    if noise_std > 0:
        xb = xb + torch.randn_like(xb) * noise_std

    # light blur (3x3 depthwise)
    try:
        blur_p = float(os.environ.get("IMAGE_AUG_BLUR_P", "0.2"))
    except Exception:
        blur_p = 0.2
    blur_p = max(0.0, min(1.0, blur_p))
    if blur_p > 0:
        m = torch.rand((b,), device=device) < blur_p
        if m.any():
            # simple 3x3 gaussian-ish kernel
            k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=device)
            k = k / k.sum()
            k = k.view(1, 1, 3, 3)
            c = xb.size(1)
            w = k.repeat(c, 1, 1, 1)  # (C,1,3,3)
            xb_blur = F.conv2d(xb[m], w, padding=1, groups=c)
            xb[m] = xb_blur

    # clamp to reasonable range after normalize (keep wide but bounded)
    xb = torch.clamp(xb, -5.0, 5.0)

    # ----------------------------
    # geometric augmentation (clean only)
    # ----------------------------
    if mode != "clean":
        return xb

    # random horizontal flip (clean only)
    p = float(os.environ.get("IMAGE_AUG_FLIP_P", "0.5"))
    if p > 0.0:
        flip_mask = torch.rand(b, device=device) < p
        if flip_mask.any():
            xb[flip_mask] = torch.flip(xb[flip_mask], dims=[3])

    # random crop with padding=4
    pad = 4
    xb_padded = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    _, _, h, w = xb_padded.shape
    max_top = h - 32
    max_left = w - 32
    top = torch.randint(0, max_top + 1, (b,), device=device)
    left = torch.randint(0, max_left + 1, (b,), device=device)
    crops = []
    for i in range(b):
        crops.append(xb_padded[i, :, top[i] : top[i] + 32, left[i] : left[i] + 32])
    xb = torch.stack(crops, dim=0)
    return xb


def _eval(model, loader, device) -> Dict[str, float]:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_y = []
    all_p = []
    with torch.no_grad():
        for xb, yb in loader:  # type: ignore[reportUnknownVariableType]
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_p.append(pred)
            all_y.append(yb.numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, average="macro")),
    }


def _eval_asr_from_npz(model, npz: Dict[str, Any], device, target_label: int) -> Dict[str, float]:
    """
    Trigger 指標評估（用於 backdoor / defense 實驗）：
    - asr_source：使用 x_test_trigger_target/y_test_trigger_target（偏向 source-only ASR）
    - ftr_all：使用 x_test_trigger_all/y_test_trigger_all（全類別 trigger → target 的比例）

    注意：不要用 x_test_poisoned vs x_test_clean 自動 diff 來定義 ASR。
    那會把「被貼 trigger 但保留原標籤的測試樣本」混進來，導致指標難以解讀且容易飆高。

    正規化須與訓練／雲端 image_cloud_server 一致：依 IMAGE_DATASET 走 _normalize_image_nchw。
    batch 大小可用 IMAGE_ASR_EVAL_BATCH_SIZE（預設跟隨 IMAGE_CLOUD_EVAL_BATCH_SIZE 或 64，避免 OOM）。

    輕量模式：
    - IMAGE_ASR_EVAL_MAX_SAMPLES>0：對每個 trigger 張量最多隨機抽樣這麼多筆再算 ASR/FTR（seed=IMAGE_ASR_SUBSAMPLE_SEED）
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = (os.environ.get("IMAGE_DATASET", "cifar10") or "cifar10").strip().lower()
    try:
        asr_bs = int(
            os.environ.get(
                "IMAGE_ASR_EVAL_BATCH_SIZE",
                os.environ.get("IMAGE_CLOUD_EVAL_BATCH_SIZE", "64"),
            )
            or "64"
        )
    except Exception:
        asr_bs = 64
    asr_bs = max(1, asr_bs)

    asr_cap = _env_int("IMAGE_ASR_EVAL_MAX_SAMPLES", 0)
    asr_cap_seed = _env_int("IMAGE_ASR_SUBSAMPLE_SEED", 43)

    def _rate(x: Any, y: Any) -> tuple[float, float]:
        if x is None or y is None:
            return 0.0, 0.0
        X = (x.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        X = _normalize_image_nchw(X, dataset=dataset)
        y = y.reshape(-1).astype(np.int64)
        if asr_cap > 0 and X.shape[0] > asr_cap:
            rng = np.random.RandomState(asr_cap_seed)
            ii = rng.choice(X.shape[0], size=asr_cap, replace=False)
            X = X[ii]
            y = y[ii]
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=asr_bs, shuffle=False, num_workers=0)
        model.eval()
        hits = 0
        total = 0
        with torch.no_grad():
            for xb, _yb in loader:
                xb = xb.to(device)
                pred = torch.argmax(model(xb), dim=1).detach().cpu().numpy()
                hits += int((pred == target_label).sum())
                total += int(pred.shape[0])
        return float(hits / max(total, 1)), float(total)

    asr_source, n_source = _rate(npz.get("x_test_trigger_target"), npz.get("y_test_trigger_target"))
    ftr_all, n_all = _rate(npz.get("x_test_trigger_all"), npz.get("y_test_trigger_all"))

    # 相容舊欄位：asr 用 asr_source；poisoned_test_n 用 source 的 N
    return {
        "asr": float(asr_source),
        "asr_source": float(asr_source),
        "ftr_all": float(ftr_all),
        "poisoned_test_n": float(n_source),
        "trigger_all_n": float(n_all),
    }

async def image_main() -> None:
    # 參數：沿用原 client 需要的三個核心欄位
    parser = argparse.ArgumentParser(description="Image UAV Client (CIFAR-10 npz)")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--aggregator_url", type=str, required=True)
    parser.add_argument("--cloud_url", type=str, required=True)  # 保留欄位（暫不直接用）
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    import torch

    _want_gpu = _env_flag("IMAGE_USE_GPU", "0")
    _cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if _cuda_ok and _want_gpu else "cpu")
    print(
        f"[Image Client] device={device} cuda_available={int(_cuda_ok)} IMAGE_USE_GPU={int(_want_gpu)} "
        f"client_id={args.client_id}",
        flush=True,
    )
    if _want_gpu and not _cuda_ok:
        try:
            import torch.version as _torch_version

            _tf = getattr(torch, "__file__", "")
            _tv = getattr(_torch_version, "cuda", None)
            _diag = f"torch.__file__={_tf!r} torch.version.cuda={_tv!r}"
        except Exception:
            _diag = "(torch 診斷資訊不可用)"
        print(
            "[Image Client] ⚠️ IMAGE_USE_GPU=1 但 CUDA 不可用，已使用 CPU；"
            "請檢查 nvidia-smi、CUDA_VISIBLE_DEVICES、LD_LIBRARY_PATH 與 PyTorch 是否為 CUDA 版。"
            f" {_diag}",
            flush=True,
        )
    seed = int(os.environ.get("IMAGE_SEED", "0"))
    _seed_everything(seed + int(args.client_id))

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

    print(f"[Image Client] 載入資料集: {dataset}, npz_path={npz_path}")
    npz = _load_cifar10_npz(npz_path)

    num_clients = int(os.environ.get("IMAGE_NUM_CLIENTS", "10"))
    attacker_clients = os.environ.get("IMAGE_ATTACKER_CLIENTS", "0").strip()
    attacker_set = set()
    if attacker_clients:
        try:
            attacker_set = set(int(x) for x in attacker_clients.split(",") if x.strip() != "")
        except Exception:
            attacker_set = {0}
    use_poisoned_train = args.client_id in attacker_set and _env_flag("IMAGE_POISON_ENABLED", "1")

    # 組 A 預設：偏向提高 clean accuracy
    batch_size = int(os.environ.get("IMAGE_BATCH_SIZE", "128"))
    local_epochs = int(os.environ.get("IMAGE_LOCAL_EPOCHS", "3"))
    lr = float(os.environ.get("IMAGE_LR", "3e-4"))

    # 若是 attacker client，可用不同的 lr / local_epochs 以減弱攻擊影響
    if use_poisoned_train:
        attacker_lr = os.environ.get("IMAGE_ATTACKER_LR")
        attacker_epochs = os.environ.get("IMAGE_ATTACKER_LOCAL_EPOCHS")
        if attacker_lr:
            try:
                lr = float(attacker_lr)
            except Exception:
                pass
        if attacker_epochs:
            try:
                local_epochs = int(attacker_epochs)
            except Exception:
                pass
    target_label = int(os.environ.get("POISON_TARGET_LABEL", os.environ.get("IMAGE_TARGET_LABEL", "8")))

    train_loader, test_loader, train_samples, poison_train_samples, poison_ratio = _make_loaders_from_npz(
        npz=npz,
        client_id=args.client_id,
        num_clients=num_clients,
        use_poisoned_train=use_poisoned_train,
        batch_size=batch_size,
        target_label=target_label,
    )

    # Client 端 clean eval：預設 test_loader 使用 max(train_bs, 256)，多 client 同卡時容易在 eval OOM。
    _client_eval_bs = max(0, _env_int("IMAGE_CLIENT_EVAL_BSZ", 0))
    _client_eval_enabled = _env_flag("IMAGE_CLIENT_EVAL_ENABLED", "1")
    if _client_eval_bs > 0:
        from torch.utils.data import DataLoader

        eval_loader = DataLoader(
            test_loader.dataset,
            batch_size=_client_eval_bs,
            shuffle=False,
            num_workers=0,
        )
        print(
            f"[Image Client] client clean eval batch: IMAGE_CLIENT_EVAL_BSZ={_client_eval_bs} "
            f"(overrides test DataLoader, train IMAGE_BATCH_SIZE={batch_size})",
            flush=True,
        )
    else:
        eval_loader = test_loader
    if not _client_eval_enabled:
        print(
            "[Image Client] client clean eval disabled: IMAGE_CLIENT_EVAL_ENABLED=0 "
            "(clean_acc/f1 in client_metrics.csv 將沿用上一輪，首輪為 0)",
            flush=True,
        )

    _lt_max = _env_int("IMAGE_EVAL_TEST_MAX_SAMPLES", 0)
    _ce = max(1, _env_int("IMAGE_CLEAN_EVAL_EVERY_ROUNDS", 1))
    _ae = max(1, _env_int("IMAGE_ASR_EVAL_EVERY_ROUNDS", 1))
    _asr_cap = _env_int("IMAGE_ASR_EVAL_MAX_SAMPLES", 0)
    if _lt_max > 0 or _ce > 1 or _ae > 1 or _asr_cap > 0 or _client_eval_bs > 0 or not _client_eval_enabled:
        print(
            f"[Image Client] 輕量評估設定: TEST_MAX_SAMPLES={_lt_max}, "
            f"CLEAN_EVERY_ROUNDS={_ce}, ASR_EVERY_ROUNDS={_ae}, ASR_MAX_SAMPLES={_asr_cap}, "
            f"CLIENT_EVAL_BSZ={_client_eval_bs or '(default)'}, CLIENT_EVAL_ENABLED={int(_client_eval_enabled)}",
            flush=True,
        )

    num_classes = 4 if dataset == "road_signs" else (43 if dataset == "gtsrb" else 10)
    model = _get_image_cnn_builder()(num_classes=num_classes).to(device)

    # 直接使用原本的聚合器 API helper
    register_client = uav_client_fixed.register_client
    get_federated_status = uav_client_fixed.get_federated_status
    get_global_weights = uav_client_fixed.get_global_weights
    upload_weights = uav_client_fixed.upload_weights

    import aiohttp

    last_confirmed_round = -1
    last_attempted_round = -1
    poll_n = 0
    last_metrics: Dict[str, float] = {"accuracy": 0.0, "f1_macro": 0.0}
    last_asr_metrics: Dict[str, float] = {
        "asr": 0.0,
        "asr_source": 0.0,
        "ftr_all": 0.0,
        "poisoned_test_n": 0.0,
        "trigger_all_n": 0.0,
    }

    async with aiohttp.ClientSession() as session:
        await register_client(session, args.aggregator_url, args.client_id)

        while True:
            poll_n += 1
            status = await get_federated_status(session, args.aggregator_url)
            if not status:
                if poll_n % 15 == 0:
                    print(
                        f"[Image Client] cid={args.client_id} federated_status_empty "
                        f"(aggregator may be down or URL wrong: {args.aggregator_url!r})",
                        flush=True,
                    )
                await asyncio.sleep(2)
                continue
            # 🔧 round 解析要更穩健：不同 aggregator 版本可能回不同 key / 型別
            def _as_int(v: Any) -> Optional[int]:
                try:
                    if v is None:
                        return None
                    if isinstance(v, bool):
                        return int(v)
                    if isinstance(v, (int, float)):
                        return int(v)
                    s = str(v).strip()
                    if not s:
                        return None
                    return int(float(s))
                except Exception:
                    return None

            cand = []
            for k in ("current_round", "round_count", "round", "round_id", "server_round"):
                if k in status:
                    vi = _as_int(status.get(k))
                    if vi is not None:
                        cand.append(vi)
            current_round = max(cand) if cand else 0
            selected = set(int(x) for x in (status.get("selected_clients", []) or []))
            is_selected = args.client_id in selected

            _diag_poll = (os.environ.get("IMAGE_TRAIN_DIAG", "1") or "1").strip().lower() not in (
                "0",
                "false",
                "no",
            )
            if _diag_poll and poll_n % 30 == 0:
                if not is_selected:
                    print(
                        f"[Image Client] cid={args.client_id} idle_poll round={current_round} "
                        f"not_in_selected (selected_len={len(selected)})",
                        flush=True,
                    )
                elif current_round <= last_confirmed_round:
                    print(
                        f"[Image Client] cid={args.client_id} idle_poll round={current_round} "
                        f"already_confirmed<=round (last_confirmed={last_confirmed_round})",
                        flush=True,
                    )

            max_rounds = int(os.environ.get("IMAGE_MAX_ROUNDS", os.environ.get("MAX_ROUNDS", "25")))
            if current_round > max_rounds:
                break

            if is_selected and current_round > last_confirmed_round and current_round != last_attempted_round:
                last_attempted_round = current_round

                # 下載 global 權重（若存在就載入）
                payload = await get_global_weights(session, args.aggregator_url)
                if payload and isinstance(payload, dict):
                    gw = payload.get("global_weights") or {}
                    if gw:
                        _strip = getattr(uav_client_fixed, "_strip_state_dict_prefix", None)
                        cleaned = _strip(gw, ["module.", "model."]) if callable(_strip) else gw
                        try:
                            if isinstance(cleaned, dict):
                                model.load_state_dict(cleaned, strict=False)
                        except Exception:
                            pass

                model.train()
                wd = float(os.environ.get("IMAGE_WEIGHT_DECAY", "1e-4"))
                lr_now = _fl_round_lr(lr, round_id=current_round, max_rounds=max_rounds)
                opt = torch.optim.AdamW(model.parameters(), lr=lr_now, weight_decay=wd)
                _max_gn = _env_float("IMAGE_MAX_GRAD_NORM", 0.0)

                _diag = (os.environ.get("IMAGE_TRAIN_DIAG", "1") or "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                )
                n_batches = len(train_loader)
                if _diag:
                    _sched_h = (os.environ.get("IMAGE_LR_SCHEDULE", "") or "").strip().lower()
                    _sched_note = (
                        f" lr_sched={_sched_h or 'constant'} lr_now={lr_now:g}"
                        if _sched_h and _sched_h not in ("constant", "none", "off", "0", "false", "no")
                        else ""
                    )
                    _gn_note = f" max_grad_norm={_max_gn:g}" if _max_gn > 0 else ""
                    print(
                        f"[Image Client] cid={args.client_id} train_start round={current_round} "
                        f"device={device} batches={n_batches} train_samples={train_samples}"
                        f"{_sched_note}{_gn_note}",
                        flush=True,
                    )
                _train_t0 = time.monotonic()

                # 支援類別權重：
                # - IMAGE_CLASS_WEIGHTS: 逗號分隔，例如 "1.0,1.0,1.0,1.0"
                # - ROAD_SIGNS_CLASS_WEIGHTS: 若 IMAGE_DATASET=road_signs 時優先使用
                class_weights: Optional[torch.Tensor] = None
                try:
                    dataset = os.environ.get("IMAGE_DATASET", "cifar10").strip().lower()
                    if dataset == "road_signs":
                        w_str = os.environ.get("ROAD_SIGNS_CLASS_WEIGHTS") or os.environ.get("IMAGE_CLASS_WEIGHTS", "")
                    else:
                        w_str = os.environ.get("IMAGE_CLASS_WEIGHTS", "")
                    if w_str:
                        parts = [p.strip() for p in w_str.split(",") if p.strip()]
                        if parts:
                            w_vals = [float(p) for p in parts]
                            class_weights = torch.tensor(w_vals, dtype=torch.float32, device=device)
                except Exception:
                    class_weights = None

                _ls = max(0.0, min(1.0, _env_float("IMAGE_LABEL_SMOOTHING", 0.0)))
                try:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=_ls)
                except TypeError:
                    if _ls > 0.0:
                        print(
                            "[Image Client] ⚠️ 當前 PyTorch 不支援 CrossEntropyLoss(label_smoothing)；"
                            "已退回 label_smoothing=0",
                            flush=True,
                        )
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

                _first_batch_logged = False
                for _ep in range(local_epochs):
                    for xb, yb in train_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        # 增強策略（方案 1）：
                        # - 乾淨樣本：完整增強（可含 crop/flip + 顏色/模糊）
                        # - 投毒樣本：安全增強（只做顏色/模糊/噪聲），避免 trigger 被裁掉/翻面
                        xb = _maybe_augment_batch(xb, mode=("poison_safe" if use_poisoned_train else "clean"))
                        opt.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        if _diag and not _first_batch_logged:
                            print(
                                f"[Image Client] cid={args.client_id} first_batch round={current_round} "
                                f"loss={float(loss.detach().cpu().item()):.4f}",
                                flush=True,
                            )
                            _first_batch_logged = True
                        loss.backward()
                        if _max_gn > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), _max_gn)
                        opt.step()

                _train_wall = time.monotonic() - _train_t0
                if _diag:
                    print(
                        f"[Image Client] cid={args.client_id} train_wall_s={_train_wall:.1f} "
                        f"round={current_round} local_epochs={local_epochs}",
                        flush=True,
                    )

                clean_every = max(1, _env_int("IMAGE_CLEAN_EVAL_EVERY_ROUNDS", 1))
                asr_every = max(1, _env_int("IMAGE_ASR_EVAL_EVERY_ROUNDS", 1))
                do_clean = ((current_round - 1) % clean_every) == 0
                do_asr = ((current_round - 1) % asr_every) == 0

                if do_clean:
                    if _client_eval_enabled:
                        metrics = _eval(model, eval_loader, device)
                        last_metrics = dict(metrics)
                    else:
                        metrics = dict(last_metrics)
                        if _diag:
                            print(
                                f"[Image Client] cid={args.client_id} clean_eval_skipped round={current_round} "
                                f"(IMAGE_CLIENT_EVAL_ENABLED=0)",
                                flush=True,
                            )
                else:
                    metrics = dict(last_metrics)
                    if _diag:
                        print(
                            f"[Image Client] cid={args.client_id} clean_eval_skipped round={current_round} "
                            f"(IMAGE_CLEAN_EVAL_EVERY_ROUNDS={clean_every})",
                            flush=True,
                        )

                if do_asr:
                    asr_metrics = _eval_asr_from_npz(model, npz, device, target_label=target_label)
                    last_asr_metrics = dict(asr_metrics)
                else:
                    asr_metrics = dict(last_asr_metrics)
                    if _diag:
                        print(
                            f"[Image Client] cid={args.client_id} asr_eval_skipped round={current_round} "
                            f"(IMAGE_ASR_EVAL_EVERY_ROUNDS={asr_every})",
                            flush=True,
                        )
                # 指標命名：attack=none -> ftr；attack=backdoor -> asr
                poison_enabled = _env_flag("IMAGE_POISON_ENABLED", "1")
                ftr_val = float(asr_metrics["asr"]) if not poison_enabled else 0.0
                asr_val = float(asr_metrics["asr"]) if poison_enabled else 0.0
                metrics_payload = {
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "ftr": ftr_val,
                    "asr": asr_val,
                    "poisoned_test_n": asr_metrics["poisoned_test_n"],
                    "target_label": int(target_label),
                    "is_poisoned_client": bool(use_poisoned_train),
                }

                # 寫入每個 client 的指標 CSV，方便畫 per-client 曲線
                if args.result_dir:
                    import csv
                    client_dir = Path(args.result_dir) / f"uav{args.client_id}"
                    client_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = client_dir / "client_metrics.csv"
                    file_exists = csv_path.exists()
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        if not file_exists:
                            w.writerow(
                                [
                                    "round",
                                    "clean_acc",
                                    "clean_f1_macro",
                                    "ftr",
                                    "asr",
                                    "trig_n",
                                    "train_samples",
                                    "is_poisoned_client",
                                ]
                            )
                        w.writerow([
                            current_round,
                            metrics["accuracy"],
                            metrics["f1_macro"],
                            ftr_val,
                            asr_val,
                            int(asr_metrics.get("poisoned_test_n", 0.0) or 0.0),
                            train_samples,
                            1 if use_poisoned_train else 0,
                        ])

                # 讓 log 更好讀：每輪至少印一次 local summary
                metric_name = "asr" if poison_enabled else "ftr"
                metric_val = asr_val if poison_enabled else ftr_val
                print(
                    f"[Client {args.client_id}] ✅ local_done round={current_round} "
                    f"poison_enabled={1 if poison_enabled else 0} is_poisoned_client={1 if use_poisoned_train else 0} "
                    f"epochs={local_epochs} bs={batch_size} lr_base={lr} lr_step={lr_now} "
                    f"clean_acc={metrics['accuracy']:.4f} clean_f1={metrics['f1_macro']:.4f} "
                    f"{metric_name}={metric_val:.4f} trig_n={int(asr_metrics.get('poisoned_test_n', 0.0) or 0.0)} "
                    f"poison_train_samples={int(poison_train_samples)} poison_ratio={float(poison_ratio):.4f}",
                    flush=True,
                )

                print(
                    f"[Image Client] cid={args.client_id} upload_start round={current_round} "
                    f"train_samples={train_samples}",
                    flush=True,
                )
                ok, _status_code, _msg = await upload_weights(
                    session=session,
                    aggregator_url=args.aggregator_url,
                    client_id=args.client_id,
                    weights=model.state_dict(),
                    sample_count=train_samples,
                    val_stats={},
                    round_id=current_round,
                    metrics_payload=metrics_payload,
                )
                if ok:
                    last_confirmed_round = current_round
                    print(
                        f"[Image Client] cid={args.client_id} upload_ok round={current_round}",
                        flush=True,
                    )
                else:
                    # 否則 last_attempted_round==current_round 會讓本輪永不再重試
                    last_attempted_round = last_confirmed_round
                    _snippet = (str(_msg) or "").replace("\n", " ")[:240]
                    print(
                        f"[Image Client] cid={args.client_id} upload_fail round={current_round} "
                        f"http={_status_code} msg={_snippet!r} (will_retry_same_round)",
                        flush=True,
                    )

            await asyncio.sleep(2)


async def main_router() -> None:
    if _env_flag("IMAGE_FL", "0"):
        await image_main()
    else:
        await uav_client_fixed.main()


if __name__ == "__main__":
    asyncio.run(main_router())

