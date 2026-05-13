#!/usr/bin/env python3
"""
在產生好 gtsrb_poisoned.npz 之後，依「client 數量」與「attacker 設定」
預先切出每個 client 的訓練集並寫到磁碟，方便檢查或給其他程式用。

分割方式與 image_uav_client 一致（IMAGE_SPLIT=shuffle 或 dirichlet）。

用法：
  cd image_fl_experiments
  python generate_poisoned_gtsrb.py
  IMAGE_NUM_CLIENTS=40 IMAGE_ATTACKER_CLIENTS=0,10,20,30 python split_gtsrb_by_clients.py

輸出目錄：
  gtsrb_poisoned/splits/
    client_0.npz ... client_{N-1}.npz  （每個內含 x_train, y_train）
    manifest.csv
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def _split_indices(n: int, client_id: int, num_clients: int, seed: int = 42) -> list:
    """與 image_uav_client 相同：shuffle 後均分給各 client。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return parts[int(client_id)].tolist()


def _split_indices_dirichlet(
    y: np.ndarray,
    client_id: int,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list:
    """依 Dirichlet(alpha) 做 non-IID 分割，與 image_uav_client 邏輯一致。"""
    y_flat = np.asarray(y).reshape(-1).astype(np.int64)
    num_clients = int(num_clients)
    client_id = int(client_id)
    rng = np.random.default_rng(seed)
    num_classes = int(y_flat.max()) + 1
    client_indices = [[] for _ in range(num_clients)]
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


def main():
    root = Path(__file__).resolve().parent
    npz_path = root / "gtsrb_poisoned" / "gtsrb_poisoned.npz"

    if not npz_path.exists():
        print(f"❌ 找不到 {npz_path}，請先執行 generate_poisoned_gtsrb.py")
        return 1

    num_clients = int(os.environ.get("IMAGE_NUM_CLIENTS", "10"))
    attacker_clients = os.environ.get("IMAGE_ATTACKER_CLIENTS", "0").strip()
    attacker_set = set()
    if attacker_clients:
        try:
            attacker_set = set(int(x) for x in attacker_clients.split(",") if x.strip() != "")
        except Exception:
            attacker_set = {0}
    split_mode = (os.environ.get("IMAGE_SPLIT", "shuffle") or "shuffle").strip().lower()
    dirichlet_alpha = os.environ.get("IMAGE_DIRICHLET_ALPHA", "0.5")

    print(f"📂 載入 {npz_path}")
    data = np.load(npz_path)

    x_train_clean = data["x_train_clean"]
    y_train_clean = data["y_train_clean"]
    x_train_poisoned = data["x_train_poisoned"]
    y_train_poisoned = data["y_train_poisoned"]

    out_dir = root / "gtsrb_poisoned" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    manifest_rows = [["client_id", "n_train", "is_attacker", "split_mode", "dirichlet_alpha"]]

    print(f"✂️ 依 {num_clients} 個 client 分割（attacker: {sorted(attacker_set)}, split={split_mode}）")
    for cid in range(num_clients):
        is_attacker = cid in attacker_set
        if is_attacker:
            x_use, y_use = x_train_poisoned, y_train_poisoned.reshape(-1)
        else:
            x_use, y_use = x_train_clean, y_train_clean.reshape(-1)

        if split_mode == "dirichlet":
            alpha = float(dirichlet_alpha)
            train_idx = _split_indices_dirichlet(y_use, cid, num_clients, alpha=alpha, seed=42)
        else:
            train_idx = _split_indices(len(x_use), cid, num_clients, seed=42)

        x = x_use[train_idx]
        y = y_use[train_idx]

        out_npz = out_dir / f"client_{cid}.npz"
        np.savez_compressed(out_npz, x_train=x, y_train=y)
        manifest_rows.append([str(cid), str(len(train_idx)), "1" if is_attacker else "0", split_mode, dirichlet_alpha])

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(manifest_rows)

    print(f"✅ 已寫入 {num_clients} 個 client 的訓練集到 {out_dir}")
    print("   每個 client_*.npz 內含: x_train, y_train")
    print(f"   總覽: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

