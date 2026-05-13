# image_fl_experiments 最小交接說明

這個目錄用於影像聯邦學習（Image FL）實驗，主要針對 GTSRB clean/backdoor 設定進行訓練、監控與對照。

## 1) 先決條件

- 作業系統：Linux（需可使用 `bash`）
- Python 環境：建議使用既有 conda 環境 `uav_fl`
- GPU：建議可用 CUDA；若無 GPU 也可跑，但速度較慢

啟用環境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uav_fl
```

## 2) 主要入口檔案

- `run_gtsrb_fl_diag_easy.sh`
  - GTSRB 診斷/收斂流程（可 clean 或 poison）
- `run_gtsrb_fl_diag_backdoor.sh`
  - 後門實驗封裝，預設 `IMAGE_POISON_ENABLED=1`
- `start_image_experiment.py`
  - 實際啟動 cloud/aggregator/client/coordinator
- `gpu_preflight.sh`
  - 開跑前顯示 GPU 佔用（不自動 kill）

## 3) 最常用啟動方式

在本目錄下：

```bash
./run_gtsrb_fl_diag_easy.sh
```

後門版：

```bash
./run_gtsrb_fl_diag_backdoor.sh
```

## 4) 建議的「基準啟動」方式（顯式參數）

若要避免父 shell 殘留環境影響，建議使用顯式環境變數啟動（範例）：

```bash
IMAGE_FL=1 IMAGE_POISON_ENABLED=1 IMAGE_ATTACKER_CLIENTS=0,3,6 IMAGE_LOCAL_EPOCHS=1 \
AGG_MIN_PARTIAL_RATIO=0.80 AGG_MAX_STALENESS=2 AGG_UPDATE_CLIP_NORM=50 \
COORDINATOR_MIN_ROUND_DWELL_S=120 COORDINATOR_START_ROUND_TIMEOUT_S=600 \
python start_image_experiment.py --num_aggregators 1 --num_clients 20 --max_rounds 5 --clients_per_round 16 --fast_iteration
```

## 5) 產物位置

實驗結果預設輸出到：

- `/home/ubuntuv100/uav/newp1/result_image/<run_dir>/`

常用檔案：

- `image_cloud_metrics.csv`
- `agg0.out`
- `cloud_server.out`
- `env_agg.txt`
- `run_manifest.jsonl`
- `aggregator_0_participation.csv`
- `image_cloud_per_class_metrics.jsonl`

## 6) 快速完整性檢查

使用本次新增腳本：

```bash
./check_run_integrity.sh /home/ubuntuv100/uav/newp1/result_image/<run_dir>
```

若只給父目錄，會自動檢查最新的 `gtsrb_backdoor_*` run：

```bash
./check_run_integrity.sh /home/ubuntuv100/uav/newp1/result_image
```

## 7) 已知注意事項

- 參數會有多層覆蓋（環境變數 > 腳本預設 > 程式預設），請優先看 `env_agg.txt` 與 `run_manifest.jsonl` 確認實際生效值。
- `IMAGE_ATTACKER_CLIENTS` 與 split 內 `is_attacker` 可能不一致，會影響攻擊/防禦解讀。
- 若 `AGG_UPDATE_CLIP_NORM` 太小，可能出現長期 clipping 飽和與類別塌縮。
