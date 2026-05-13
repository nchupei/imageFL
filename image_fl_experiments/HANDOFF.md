# HANDOFF（最小交接包）

此文件提供接手者可直接執行的最小資訊：目前基準、檢查點、常見風險。

## 1) 目前建議基準（GTSRB）

- 拓樸：
  - `num_aggregators=1`
  - `num_clients=20`
  - `clients_per_round=16`
- 訓練核心：
  - `IMAGE_LOCAL_EPOCHS=1`
  - `IMAGE_LR=0.0005`
  - `IMAGE_BATCH_SIZE=4`
- 聚合/協調：
  - `AGG_MIN_PARTIAL_RATIO=0.80`
  - `AGG_MAX_STALENESS=2`
  - `AGG_UPDATE_CLIP_NORM`：建議做 `50` vs `100` A/B
  - `COORDINATOR_MIN_ROUND_DWELL_S=120`
  - `COORDINATOR_START_ROUND_TIMEOUT_S=600`

## 2) 一鍵啟動範例（可直接改 TUNING_NAME）

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav_fl && cd /home/ubuntuv100/uav/newp1/image_fl_experiments && IMAGE_FL=1 IMAGE_POISON_ENABLED=1 IMAGE_ATTACKER_CLIENTS=0,3,6 IMAGE_ATTACKER_LOCAL_EPOCHS=1 IMAGE_DATASET=gtsrb IMAGE_USE_CLIENT_SPLITS=1 IMAGE_NUM_CLIENTS=20 IMAGE_CNN_SIZE=resnet18 IMAGE_GTSRB_NORMALIZE=1 IMAGE_AUG_ENABLED=0 IMAGE_BATCH_SIZE=4 IMAGE_ASR_EVAL_BATCH_SIZE=2 IMAGE_LR=0.0005 IMAGE_WEIGHT_DECAY=0.0001 IMAGE_LOCAL_EPOCHS=1 IMAGE_LABEL_SMOOTHING=0 IMAGE_CLIENT_EVAL_ENABLED=0 IMAGE_FL_FAST_ITERATION=1 IMAGE_FL_RESET_AGG_MIN=1 COORDINATOR_MIN_ROUND_DWELL_S=120 COORDINATOR_START_ROUND_TIMEOUT_S=600 AGG_ATTACKER_CLIENTS=0,3,6 AGG_ATTACKER_FEDAVG_MULT=1.0 AGG_ALPHA_MAX_MULTIPLIER=2.5 AGG_UPDATE_CLIP_NORM=50 AGG_MIN_PARTIAL_RATIO=0.80 AGG_MAX_STALENESS=2 AGG_CAS_MAX_RETRIES=3 AGG_CAS_BACKOFF_S=1.0 AGG_CAS_UPLOAD_JITTER_S=0.2 AGG_CAS_STAGGER_PER_ID_S=0.5 IMAGE_CLOUD_EVAL_BATCH_SIZE=2 IMAGE_CLOUD_EVAL_USE_GPU=0 IMAGE_CLIENT_STAGGER_S=0.4 TUNING_NAME=gtsrb_handoff_baseline python start_image_experiment.py --num_aggregators 1 --num_clients 20 --seed 0 --max_rounds 10 --clients_per_round 16 --result_root /home/ubuntuv100/uav/newp1/result_image --fast_iteration --attacker_clients 0,3,6
```

## 3) 每次開跑後必查

1. `env_agg.txt`
   - 確認 `AGG_UPDATE_CLIP_NORM / AGG_MIN_PARTIAL_RATIO / AGG_MAX_STALENESS` 是否正確。
2. `run_manifest.jsonl`
   - 確認 `local_epochs / attacker_clients / defense_mode`。
3. `image_cloud_metrics.csv`
   - 觀察前 5~10 輪 `clean_acc` 與 `clean_loss` 斜率。
4. `agg0.out`
   - 是否出現 `pending_aggregation`、`round_advance_blocked`、`rejected_*`。
5. `image_cloud_per_class_metrics.jsonl`
   - 觀察非零 recall 類別數是否增加（避免類別塌縮）。

## 4) 已知風險（重要）

- `IMAGE_ATTACKER_CLIENTS` 與 split 的 `is_attacker` 可能不一致，會影響攻擊實驗解讀。
- clip 過小（例如 5）可能長期全員 clipped，導致「系統穩定但學習停滯」。
- 父 shell 殘留環境變數會覆蓋腳本預設；請優先用顯式 env，並在 run 後檢查 `env_agg.txt`。

## 5) 交接建議流程

1. 先用 `check_run_integrity.sh` 驗證 run 完整性。
2. 再用 `image_cloud_metrics.csv + image_cloud_per_class_metrics.jsonl` 判斷是否「有學習且非塌縮」。
3. 做參數 A/B 時，一次只動一個主參數（先 `AGG_UPDATE_CLIP_NORM`）。
