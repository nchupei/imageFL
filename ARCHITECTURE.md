# ImageFL 系統架構

本文件說明 ImageFL 項目的整體架構、模組分工和資料流。

## 1. 高層架構概覽

```
┌─────────────────────────────────────────────────────────┐
│                 雲端服務器 (Cloud Server)               │
│              cloud_server_fixed.py (12.7K 行)           │
│  - 全局模型管理與版本控制                                 │
│  - 聚合結果驗收與評估                                     │
│  - 下發全局權重到聚合器                                   │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/FastAPI 通訊
                   ↓
┌──────────────────────────────┬──────────────────────────────┐
│   聚合器 #0 (Aggregator)      │   聚合器 #N (Aggregator)     │
│  aggregator_fixed.py (3.6K 行)│  aggregator_fixed.py (3.6K 行)│
│ - 接收客戶端梯度上傳           │ - 接收客戶端梯度上傳          │
│ - 執行聚合（FedAvg + 裁剪）   │ - 執行聚合（FedAvg + 裁剪）  │
│ - 維護本地模型版本            │ - 維護本地模型版本           │
└──────────────────┬───────────┴──────────────────┬─────────┘
         HTTP/FastAPI                    HTTP/FastAPI
         通訊 ↙          ↘                     ↙        ↘
┌──────────────────────────────────────────────────────────┐
│              客戶端 (Client) - 分散部署                    │
│         image_uav_client.py (4.1K 行) × 20              │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 流程: 下載權重 → 本地訓練 → 上傳梯度              │   │
│  │ - 從聚合器接收全局權重                           │   │
│  │ - 在本地資料上訓練 IMAGE_LOCAL_EPOCHS 輪         │   │
│  │ - 計算梯度並上傳回聚合器                         │   │
│  │ - 支援資料中毒 (IMAGE_POISON_ENABLED)           │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## 2. 模組結構與職責

```
imageFL/
│
├── 📋 配置與環境管理
│   └── config_fixed.py (1.1K 行)
│       - 統一管理所有超參數與環境變數
│       - 支援 IMAGE_FL / FEDAVG_BASELINE 等多種模式
│       - 日誌、網路、聚合、訓練等 100+ 參數
│
├── 🌐 核心通訊模組
│   ├── cloud_server_fixed.py (12.7K 行)
│   │   - FastAPI Web 伺服器
│   │   - 全局權重管理與版本控制
│   │   - 聚合結果評估與降級
│   │   - 連接多個聚合器與雲端評估
│   │
│   ├── aggregator_fixed.py (3.6K 行)
│   │   - FastAPI Web 伺服器
│   │   - FedAvg 聚合邏輯
│   │   - 權重裁剪 (clipping)
│   │   - 客戶端上傳管理
│   │
│   └── image_uav_client.py (4.1K 行) [在 image_fl_experiments/]
│       - 客戶端訓練邏輯
│       - 本地資料加載與中毒
│       - 模型更新與上傳
│
├── 🧠 模型定義
│   ├── models/
│   │   ├── cnn.py (6.1K 行)
│   │   │   - ResNet18 / 小型 CNN 等
│   │   │   - 支援 GTSRB, CIFAR-10 資料集
│   │   │
│   │   ├── dnn.py (2.2K 行)
│   │   │   - 攻擊模型 (NetworkAttackDNN)
│   │   │   - 用於後門實驗
│   │   │
│   │   └── __init__.py
│   │
│   └── 實驗框架
│       └── image_fl_experiments/
│           ├── start_image_experiment.py
│           │   - 一鍵啟動協調器
│           │   - 管理 Cloud + Aggregators + Clients
│           │
│           ├── image_cloud_server.py
│           │   - Image FL 專用雲端伺服器包裝
│           │
│           ├── image_aggregator.py
│           │   - Image FL 專用聚合器包裝
│           │
│           ├── image_uav_client.py
│           │   - Image FL 客戶端實現
│           │
│           │   
│           │
│           └── 資料與結果
│               ├── GTSRB/ / gtsrb_poisoned/
│               └── result/ (運行結果輸出)
│
└── 📚 文檔
    ├── README.md (根目錄 - 簡述)
    ├── SETUP.md (本文 - 環境與快速開始)
    ├── ARCHITECTURE.md (本文 - 架構細節)
    ├── requirements.txt (依賴清單)
    └── image_fl_experiments/
        ├── README.md (詳細使用說明)
        └── HANDOFF.md (交接清單與參數)
```

## 3. 資料流與通訊協議

### 聯邦學習訓練迴圈（單輪示例）

```
┌─────────────────────────────────────────────────────────────────┐
│ Round T 開始                                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │ 雲端 → 聚合器 → 客戶端    │
        │ 廣播全局權重 wt          │
        └────────────┬─────────────┘
                     │
        ┌────────────▼──────────────────────────────────┐
        │ 客戶端 i 本地訓練                              │
        │ - 下載 wt                                    │
        │ - 在本地資料上訓練 E 個 epoch                │
        │ - 計算梯度 gt_i = (wt - w'i) / lr             │
        │ - 上傳 gt_i 到聚合器                          │
        └────────────┬──────────────────────────────────┘
                     │
        ┌────────────▼──────────────────────────────────┐
        │ 聚合器聚合客戶端更新                           │
        │ - 收集 K 個客戶端的梯度 {gt_i}               │
        │ - 執行 FedAvg: g_avg = (Σ nt * gt_i) / Σ nt │
        │ - 裁剪: clip(g_avg, AGG_UPDATE_CLIP_NORM)   │
        │ - 上傳 g_avg 到雲端                          │
        └────────────┬──────────────────────────────────┘
                     │
        ┌────────────▼──────────────────────────────────┐
        │ 雲端合併與評估                                 │
        │ - 接收所有聚合器的聚合結果                     │
        │ - 執行全局聚合：w(t+1) = wt - β * Σ g_agg   │
        │ - 在評估資料上測評：acc, loss, ASR 等        │
        │ - 將 w(t+1) 寫入檢查點                       │
        └────────────┬──────────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────────┐
        │ Round T+1 開始（迴圈）                        │
        └────────────────────────────────────────────┘
```

### 聚合邏輯細節

```python
# 偽代碼：客戶端訓練與梯度上傳
def client_train_and_upload(wt):
    w_local = wt  # 下載全局權重
    for epoch in range(LOCAL_EPOCHS):
        for batch in local_data:
            loss, grads = compute_loss_and_grads(w_local, batch)
            w_local -= lr * grads
    
    # 計算梯度
    gt_i = (wt - w_local) / lr
    upload(gt_i)  # 上傳回聚合器

# 偽代碼：聚合器聚合
def aggregator_aggregate(grads_list):
    g_avg = sum(nt_i * gt_i for nt_i, gt_i in grads_list) / sum(nt_i)
    g_clipped = clip(g_avg, AGG_UPDATE_CLIP_NORM)
    return g_clipped

# 偽代碼：雲端更新全局權重
def cloud_update_global_weights(agg_results):
    g_global = sum(g_agg for g_agg in agg_results) / num_agg
    wt_new = wt_old - CLOUD_LR * g_global
    
    # 評估
    acc, loss, asr = evaluate(wt_new, eval_data)
    return wt_new, acc, loss, asr
```

## 4. 關鍵參數與超參配置

所有參數在 `config_fixed.py` 中定義，主要分類如下：

### 訓練參數
```python
IMAGE_LR = 0.0005              # 客戶端學習率
IMAGE_BATCH_SIZE = 4           # 客戶端批次大小
IMAGE_LOCAL_EPOCHS = 1         # 客戶端本地訓練輪數
IMAGE_WEIGHT_DECAY = 0.0001    # L2 正則化
CLOUD_LR = 0.00015             # 雲端學習率
```

### 聚合參數
```python
AGG_UPDATE_CLIP_NORM = 50      # 權重裁剪上限（關鍵防禦參數）
AGG_MIN_PARTIAL_RATIO = 0.80   # 聚合所需最少客戶端比例
AGG_MAX_STALENESS = 2          # 容許最大過時輪數
```

### 協調參數
```python
COORDINATOR_MIN_ROUND_DWELL_S = 120    # 輪次間隔（秒）
COORDINATOR_START_ROUND_TIMEOUT_S = 600  # 輪次啟動逾時
```

### 攻擊/防禦參數
```python
IMAGE_POISON_ENABLED = 0       # 是否啟用資料中毒
IMAGE_ATTACKER_CLIENTS = "0,3,6"  # 攻擊客戶端 ID
AGG_ATTACKER_CLIENTS = "0,3,6"    # 聚合層認知的攻擊客戶端
AGG_ALPHA_MAX_MULTIPLIER = 2.5    # 防禦機制：異常倍數
```

詳見 `config_fixed.py` 與 `image_fl_experiments/HANDOFF.md` § 1)。

## 5. 實驗模式

### 模式 1: FedAvg Baseline
```bash
FEDAVG_BASELINE=1 python start_image_experiment.py ...
```
- 禁用雲端訓練與知識蒸餾
- 純 FedAvg 聚合
- 用於對照實驗

### 模式 2: Image FL (Clean)
```bash
IMAGE_FL=1 IMAGE_POISON_ENABLED=0 python start_image_experiment.py ...
```
- 完整 Image FL 框架
- 無攻擊（數據清潔）
- 用於基準測試

### 模式 3: Image FL (Backdoor Attack)
```bash
IMAGE_FL=1 IMAGE_POISON_ENABLED=1 IMAGE_ATTACKER_CLIENTS="0,3,6" python start_image_experiment.py ...
```
- 完整 Image FL 框架
- 特定客戶端進行資料中毒
- 用於攻擊/防禦研究

## 6. 輸出與監控

### 主要輸出文件

運行完成後，在 `result_image/<run_dir>/` 下生成：

```
result/
├── image_cloud_metrics.csv
│   - 主要指標: 輪數, clean_acc, clean_loss, ASR, ...
│   - 用途: 繪製學習曲線，監控收斂
│
├── image_cloud_per_class_metrics.jsonl
│   - 每類別的 precision, recall, f1
│   - 用途: 檢測類別塌縮，分析不均衡
│
├── agg0.out / agg1.out / ...
│   - 聚合器詳細日誌
│   - 用途: 診斷卡頓、超時、異常
│
├── cloud_server.out
│   - 雲端服務器日誌
│   - 用途: 監控全局更新與評估
│
├── env_agg.txt
│   - 實際生效的環境變數快照
│   - 用途: 確認參數是否被覆蓋
│
├── run_manifest.jsonl
│   - 運行配置元資料
│   - 用途: 重現實驗，查詢參數歷史
│
└── aggregator_0_participation.csv
    - 客戶端參與歷史
    - 用途: 分析掉線、延遲等問題
```

### 實時監控命令

```bash
# 監控 Cloud 進度
tail -f cloud_server.out

# 監控聚合器
tail -f agg0.out

# 查看當前準確度
tail -5 image_cloud_metrics.csv | cut -d',' -f1-5

# 檢查是否有長期 clipping（表示防禦過強）
grep -c "all clients clipped" agg0.out
```

## 7. 常見改動點

若您需要修改項目，以下是常見的改動位置：

| 需求 | 文件 | 修改位置 |
|------|------|---------|
| 新增模型 | `models/cnn.py` | `class Model(nn.Module)` |
| 調整損失函數 | `config_fixed.py` | `IMAGE_LOSS_FN` |
| 修改聚合算法 | `aggregator_fixed.py` | `aggregate()` 函數 |
| 新增評估指標 | `cloud_server_fixed.py` | `evaluate()` 函數 |
| 調整超參數 | `config_fixed.py` 或環境變數 | 任何 `IMAGE_*` 或 `AGG_*` |
| 自訂中毒策略 | `image_fl_experiments/image_uav_client.py` | `poison_data()` 函數 |

## 8. 常見問題與調試

### Q: 程式卡在 "Waiting for clients"
**A:** 檢查聚合器日誌 (`agg0.out`)，是否有客戶端連接失敗或超時。

### Q: 準確度持續下降（不收斂）
**A:** 可能是裁剪過強。檢查 `AGG_UPDATE_CLIP_NORM`，嘗試增大（50 → 100）。

### Q: 類別塌縮（某些類別準確度為 0）
**A:** 查看 `image_cloud_per_class_metrics.jsonl`，確認是否大部分類別被忽略。調整 `IMAGE_LABEL_SMOOTHING` 或 `F1_OPTIMIZATION_ENABLED`。

### Q: 後門攻擊無效（ASR 沒有增加）
**A:** 確認 `IMAGE_ATTACKER_CLIENTS` 與資料分割一致，檢查 `run_manifest.jsonl` 裡的 `attacker_clients`。

詳見 `image_fl_experiments/HANDOFF.md` § 4) 已知風險。

---

**更多資訊:**
- 詳細參數列表: `image_fl_experiments/HANDOFF.md` § 1)
- 快速開始: `SETUP.md`
- 完整實驗指南: `image_fl_experiments/README.md`
