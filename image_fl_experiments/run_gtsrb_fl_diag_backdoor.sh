#!/bin/bash
# -*- coding: utf-8 -*-
"""
一鍵啟動影像版 Federated Learning 實驗（GTSRB 後門模式）

此腳本會：
- 啟動 1 個雲端服務器
- 啟動 1 個聚合器
- 啟動 20 個客戶端（客戶端 0, 3, 6 執行資料中毒攻擊）
- 自動推進 5-10 輪聯邦學習
- 輸出結果到 result/ 目錄

使用方式：
  $ cd image_fl_experiments
  $ ./run_gtsrb_fl_diag_backdoor.sh

環境需求：
  - conda 環境 uav_fl 已激活
  - Python 3.10+
  - PyTorch + GPU (可選，無 GPU 也可跑但速度慢)

監控指標：
  - ASR (Attack Success Rate) - 後門觸發成功率，應逐輪增加
  - clean_acc - 乾淨準確度，應穩定或略降
  - image_cloud_per_class_metrics.jsonl - 目標類別的 recall
"""

set -e  # 任何命令失敗即停止

# ============================================================================
# 配置參數
# ============================================================================

# 基準參數（後門訓練，含攻擊）
export IMAGE_FL=1
export IMAGE_POISON_ENABLED=1              # 啟用後門攻擊
export IMAGE_ATTACKER_CLIENTS="0,3,6"      # 攻擊客戶端 (資料中毒端)
export IMAGE_ATTACKER_LOCAL_EPOCHS=1       # 攻擊客戶端本地訓練輪數
export IMAGE_DATASET=gtsrb
export IMAGE_USE_CLIENT_SPLITS=1
export IMAGE_NUM_CLIENTS=20
export IMAGE_CNN_SIZE=resnet18
export IMAGE_GTSRB_NORMALIZE=1
export IMAGE_AUG_ENABLED=0

# 訓練核心參數
export IMAGE_BATCH_SIZE=4
export IMAGE_LR=0.0005
export IMAGE_WEIGHT_DECAY=0.0001
export IMAGE_LOCAL_EPOCHS=1
export IMAGE_LABEL_SMOOTHING=0
export IMAGE_CLIENT_EVAL_ENABLED=0

# 評估參數
export IMAGE_ASR_EVAL_BATCH_SIZE=2
export IMAGE_CLOUD_EVAL_BATCH_SIZE=2
export IMAGE_CLOUD_EVAL_USE_GPU=0

# 聚合/協調參數
export AGG_MIN_PARTIAL_RATIO=0.80
export AGG_MAX_STALENESS=2
export AGG_UPDATE_CLIP_NORM=50             # 防禦：權重裁剪
export AGG_ATTACKER_CLIENTS="0,3,6"        # 聚合層認知的攻擊客戶端
export AGG_ATTACKER_FEDAVG_MULT=1.0        # 攻擊乘數
export AGG_ALPHA_MAX_MULTIPLIER=2.5        # 防禦倍數 (異常檢測)
export AGG_CAS_MAX_RETRIES=3
export AGG_CAS_BACKOFF_S=1.0
export AGG_CAS_UPLOAD_JITTER_S=0.2
export AGG_CAS_STAGGER_PER_ID_S=0.5

# 協調器參數
export COORDINATOR_MIN_ROUND_DWELL_S=120
export COORDINATOR_START_ROUND_TIMEOUT_S=600

# 實驗控制
export IMAGE_FL_FAST_ITERATION=1           # 快速迭代模式
export IMAGE_FL_RESET_AGG_MIN=1            # 重置聚合器最小時間
export IMAGE_CLIENT_STAGGER_S=0.4          # 客戶端錯開時間

# 診斷名稱
export TUNING_NAME=gtsrb_backdoor_baseline

# ============================================================================
# 執行啟動
# ============================================================================

echo "=========================================="
echo "🚀 開始 Image FL GTSRB 後門訓練"
echo "=========================================="
echo ""
echo "📊 實驗配置："
echo "  - 模式: Backdoor (含攻擊)"
echo "  - 客戶端數量: ${IMAGE_NUM_CLIENTS}"
echo "  - 攻擊客戶端: ${IMAGE_ATTACKER_CLIENTS} (資料中毒)"
echo "  - 批次大小: ${IMAGE_BATCH_SIZE}"
echo "  - 學習率: ${IMAGE_LR}"
echo "  - 本地 epoch: ${IMAGE_LOCAL_EPOCHS}"
echo "  - 裁剪範數: ${AGG_UPDATE_CLIP_NORM} (防禦)"
echo "  - 診斷名稱: ${TUNING_NAME}"
echo ""
echo "⚠️  監控重點："
echo "  - ASR (Attack Success Rate) 應逐輪增加"
echo "  - clean_acc 應保持穩定或略降"
echo "  - 檢查目標類別的 recall 變化"
echo ""

# 執行啟動
python start_image_experiment.py \
  --num_aggregators 1 \
  --num_clients "${IMAGE_NUM_CLIENTS}" \
  --seed 0 \
  --max_rounds 10 \
  --clients_per_round 16 \
  --result_root ./result \
  --fast_iteration \
  --attacker_clients 0,3,6

echo ""
echo "=========================================="
echo "✅ 訓練完成！"
echo "=========================================="
echo ""
echo "📁 結果位置："
ls -td result/*/ 2>/dev/null | head -1 | xargs -I {} sh -c 'echo "   {}" && ls -lh {}'
echo ""
echo "📊 檢查 ASR (攻擊成功率)："
echo "   $ grep ASR result/<run_dir>/image_cloud_metrics.csv | tail -3"
echo ""
echo "📊 檢查乾淨準確度："
echo "   $ tail -5 result/<run_dir>/image_cloud_metrics.csv | cut -d',' -f1-4"
echo ""
echo "🔍 驗證完整性："
echo "   $ ./check_run_integrity.sh result/<run_dir>"
