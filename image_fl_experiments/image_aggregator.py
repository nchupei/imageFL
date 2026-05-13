#!/usr/bin/env python3
"""
影像版 Aggregator 啟動入口（保留原本 __main__ 參數與輸出）。

目前先完整沿用 `aggregator_fixed.py` 的 app 與啟動參數：
- --aggregator_id
- --port
"""

import argparse
import os
import sys
from pathlib import Path

# 讓在 image_fl_experiments 下執行時也能 import 到專案模組
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 🔧 影像模式下，強制使用與 image client 相同的 CNN 架構
os.environ.setdefault("IMAGE_FL", "1")

import config_fixed as config  # noqa: E402
from image_uav_client import _get_image_cnn_builder  # noqa: E402
import models.cnn as base_cnn  # noqa: E402


def _build_image_cnn_for_server(input_dim: int, output_dim: int):
    """
    供 aggregator / cloud 使用的影像版 CNN 建構函數。
    與 image_uav_client 一致：依 IMAGE_CNN_SIZE 選 simple 或 medium。
    """
    return _get_image_cnn_builder()(num_classes=output_dim)


# 調整全域模型配置：在影像實驗中使用 CNN 分支
config.MODEL_CONFIG["type"] = "cnn"
dataset = os.environ.get("IMAGE_DATASET", "cifar10").strip().lower()
num_classes = 4 if dataset == "road_signs" else (43 if dataset == "gtsrb" else 10)
config.MODEL_CONFIG["num_classes"] = num_classes
config.MODEL_CONFIG["output_dim"] = num_classes
# 將 models.cnn.build_cnn 替換為影像版 CNN（僅影響本進程）
base_cnn.build_cnn = _build_image_cnn_for_server

import aggregator_fixed  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動聚合器 (影像版本入口，沿用修復版本)")
    parser.add_argument("--aggregator_id", type=int, default=0, help="聚合器ID")
    parser.add_argument("--port", type=int, default=8000, help="端口號")
    args = parser.parse_args()

    # 設置聚合器ID/port（沿用原模組全域變數）
    aggregator_fixed.aggregator_id = args.aggregator_id
    aggregator_fixed.aggregator_port = args.port

    print(f"[Aggregator {aggregator_fixed.aggregator_id}] 🚀 啟動聚合器 (端口: {args.port})")
    # 明確輸出防禦相關設定，避免「env 沒傳下來」時難以診斷
    agg_keys = sorted([k for k in os.environ.keys() if k.startswith("AGG_")])
    if agg_keys:
        print(
            f"[Aggregator {aggregator_fixed.aggregator_id}] 🛡️ AGG_* env: "
            + ", ".join([f"{k}={os.environ.get(k,'')}" for k in agg_keys]),
            flush=True,
        )
    else:
        print(f"[Aggregator {aggregator_fixed.aggregator_id}] 🛡️ AGG_* env: (none)", flush=True)
    print(f"[Aggregator {aggregator_fixed.aggregator_id}] 📊 配置信息:")
    print(f"  - 聚合超時: {aggregator_fixed.AGGREGATION_TIMEOUT}s")
    print(f"  - 最小客戶端數: {aggregator_fixed.config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3)}")
    print(f"  - 訓練流程: {aggregator_fixed.config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')}")
    print(f"  - 平滑因子: {aggregator_fixed.config.FEDERATED_CONFIG.get('aggregation', {}).get('smoothing_factor', 0.8)}")

    import uvicorn

    uvicorn.run(aggregator_fixed.app, host="0.0.0.0", port=args.port)

