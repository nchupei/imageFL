# 環境快速檢查與設置

本文件提供環境驗證、依賴檢查和一鍵啟動指南。

## 前置需求

### 1. 檢查 Python 環境

```bash
# 確認 conda 環境已啟用
conda activate uav_fl

# 檢查 Python 版本（需要 3.10+）
python --version
# 預期輸出：Python 3.12.1 (或更新)
```

### 2. 檢查 PyTorch 與 CUDA

```bash
# 驗證 PyTorch 安裝且支援 CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# 預期輸出：PyTorch 2.11.0+cu130, CUDA available: True (若有 GPU)
```

### 3. 驗證核心依賴

```bash
# 檢查所有關鍵模組
python -c "
import fastapi, uvicorn, aiohttp, numpy, PIL
print('✅ 所有核心依賴齊備')
"
```

### 4. 完整依賴安裝（如需要）

若遇到缺失模組，執行：
```bash
pip install -r requirements.txt
```

---

## 一鍵環境驗證腳本

若想一次檢查所有環境，執行以下腳本：

```bash
python3 << 'VERIFY_EOF'
import sys
import subprocess

checks = [
    ("Python version", lambda: f"Python {sys.version.split()[0]}"),
    ("PyTorch", lambda: __import__("torch").__version__),
    ("CUDA available", lambda: str(__import__("torch").cuda.is_available())),
    ("FastAPI", lambda: __import__("fastapi").__version__),
    ("Uvicorn", lambda: __import__("uvicorn").__version__),
    ("Aiohttp", lambda: __import__("aiohttp").__version__),
    ("NumPy", lambda: __import__("numpy").__version__),
    ("Pillow", lambda: __import__("PIL").__version__),
]

print("=" * 60)
print("環境驗證報告")
print("=" * 60)

failed = []
for name, check_fn in checks:
    try:
        result = check_fn()
        print(f"✅ {name}: {result}")
    except Exception as e:
        print(f"❌ {name}: {e}")
        failed.append(name)

print("=" * 60)
if failed:
    print(f"⚠️  缺失依賴: {', '.join(failed)}")
    print("   執行: pip install -r requirements.txt")
else:
    print("✅ 所有環境檢查通過！")
print("=" * 60)
VERIFY_EOF
```

---

## 一鍵啟動

### 選項 1: 清潔訓練 (推薦用於測試)

```bash
cd image_fl_experiments
./run_gtsrb_fl_diag_easy.sh
```

**預期行為:**
- 啟動 1 個雲端服務器
- 啟動 1 個聚合器
- 啟動 20 個客戶端
- 運行 5-10 輪聯邦學習
- 結果輸出到 `/home/ubuntuv100/uav/newp1/result_image/<timestamp>/`

### 選項 2: 後門訓練 (測試攻擊防禦)

```bash
cd image_fl_experiments
./run_gtsrb_fl_diag_backdoor.sh
```

**預期行為:**
- 同上，但包含後門注入（客戶端 0, 3, 6 是攻擊者）
- 監控目標類別準確度下降（ASR）

### 選項 3: 自訂參數啟動

若要細粒度控制，檢視 `image_fl_experiments/HANDOFF.md` 的完整參數列表，然後：

```bash
cd image_fl_experiments

# 範例：修改裁剪範數為 100
IMAGE_FL=1 \
IMAGE_POISON_ENABLED=0 \
AGG_UPDATE_CLIP_NORM=100 \
IMAGE_LOCAL_EPOCHS=2 \
TUNING_NAME=custom_run_v1 \
python start_image_experiment.py \
  --num_aggregators 1 \
  --num_clients 20 \
  --seed 0 \
  --max_rounds 5 \
  --clients_per_round 16 \
  --fast_iteration
```

---

## 運行後檢查清單

每次運行後，請驗證：

### 1. 檢查執行完整性

```bash
# 進入結果目錄
cd /home/ubuntuv100/uav/newp1/result_image/<your_run_dir>

# 運行完整性檢查
../../imageFL/image_fl_experiments/check_run_integrity.sh .
```

**預期輸出:** 所有關鍵文件 (metrics.csv, manifest.jsonl 等) 都存在

### 2. 驗證參數設置

```bash
cat env_agg.txt | grep -E "AGG_UPDATE_CLIP_NORM|AGG_MIN_PARTIAL_RATIO|AGG_MAX_STALENESS"
```

確認顯示的值是您預期的參數。

### 3. 檢查學習曲線

```bash
# 查看訓練損失與準確度前 10 輪
head -15 image_cloud_metrics.csv | cut -d',' -f1-5
```

預期：
- `clean_loss` 應逐輪下降
- `clean_acc` 應逐輪上升

### 4. 避免類別塌縮

```bash
# 檢查多類別指標
cat image_cloud_per_class_metrics.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    if 'round' in data and data['round'] % 2 == 0:  # 每 2 輪檢查一次
        non_zero = sum(1 for v in data.values() if isinstance(v, (int, float)) and v > 0)
        print(f\"Round {data['round']}: {non_zero} 類別有非零 recall\")
        if non_zero < 10:
            print(f\"  ⚠️  警告: 類別數過少，可能發生塌縮\")
"
```

---

## 常見問題快速排查

| 問題 | 症狀 | 解決方案 |
|------|------|---------|
| **CUDA 記憶體不足** | Out of memory 錯誤 | 減少 `IMAGE_BATCH_SIZE`（4→2），或設 `IMAGE_USE_GPU=0` |
| **進程卡住** | 無新日誌超過 2 分鐘 | 檢查 `agg0.out` 日誌，是否有 `pending_aggregation` 或 `timeout` |
| **類別塌縮** | 大多類別 recall = 0 | 增加 `AGG_UPDATE_CLIP_NORM`（50→100）或降低 `IMAGE_LR` |
| **學習停滯** | 準確度不變超過 5 輪 | 檢查環境變數是否被上層 shell 覆蓋，參考 `env_agg.txt` |
| **後門無效** | ASR（目標類別準確度）未下降 | 確認 `IMAGE_ATTACKER_CLIENTS` 與資料分割的 `is_attacker` 一致 |

詳見 `image_fl_experiments/HANDOFF.md` 的「已知風險」章節。

---

## 依賴管理

### 檢查已安裝版本

```bash
pip list | grep -E "torch|numpy|fastapi|uvicorn|aiohttp|Pillow"
```

### 更新所有依賴（謹慎操作）

```bash
pip install --upgrade -r requirements.txt
```

### 鎖定特定版本

若要確保可重現性，建議生成 `requirements-lock.txt`：

```bash
pip freeze > requirements-lock.txt
```

然後與他人共享該文件以確保環境一致性。

---

## 聯絡與支援

- **實驗文檔**: `image_fl_experiments/README.md`
- **交接說明**: `image_fl_experiments/HANDOFF.md`
- **系統架構**: `ARCHITECTURE.md`
- **參數詳解**: `image_fl_experiments/HANDOFF.md` § 1) 目前建議基準

如遇問題，先檢查運行目錄下的：
1. `env_agg.txt` - 實際生效環境變數
2. `run_manifest.jsonl` - 運行配置快照
3. `agg0.out` - 聚合器詳細日誌
