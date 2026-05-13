# 🚀 優化後的聯邦學習配置
# 清理重複配置，提高可讀性和維護性

import os
import math
import logging
from typing import Dict, Any, Optional


def _env_flag(name: str, default: str = "0") -> bool:
    """簡易環境變數布林解析"""
    try:
        return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _env_int_or_none(name: str, default: Optional[int] = None) -> Optional[int]:
    """環境變數整數解析，<=0 或 None 代表不限制"""
    try:
        raw = os.environ.get(name, None)
        if raw is None:
            return default
        raw = raw.strip().lower()
        if raw in ("", "none", "null"):
            return None
        val = int(raw)
        return None if val <= 0 else val
    except Exception:
        return default


# =============================================================================
# 🧪 實驗模式旗標（透過環境變數切換 preset）
#   - FEDAVG_BASELINE=1 → 極簡 FedAvg baseline（關掉 KD / 雲端訓練）
# =============================================================================

IS_FEDAVG_BASELINE: bool = os.environ.get("FEDAVG_BASELINE", "0") == "1"
F1_OPTIMIZATION_ENABLED = _env_flag("F1_OPTIMIZATION", "1")  # 🔧 預設啟用：Focal Loss 強化 password/ransomware 等難分類類別

ROUND_CLIENT_LIMIT: Optional[int] = None
SMALL_SCALE_SUMMARY: Dict[str, Any] = {}
SMALL_SCALE_MODE = _env_flag("SMALL_SCALE_MODE", "0")

# =============================================================================
# 📊 基礎配置
# =============================================================================

# 🎯 日誌配置
def set_log_level(level="INFO"):
    """設置日誌級別"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 🎯 核心訓練參數 (平衡穩定性與速度)
# Non-IID 下建議較保守的 client LR；可用環境變數覆寫（LEARNING_RATE / CLIENT_LR）
# 預設 client LR（可用環境變數覆寫）
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-4"))
CLIENT_LR = float(os.environ.get("CLIENT_LR", str(LEARNING_RATE)))  # 向後兼容：客戶端學習率
SERVER_LR = 1.5e-4  # 🚀 再降低 server LR：2e-4 → 1.5e-4（後期更平穩微調）

# 🔧 GPU 記憶體保護：允許透過環境變量調整批次大小，預設為 64（修復客戶端訓練問題）
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))  # 固定使用 exp_id=16 的 batch
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", str(max(TRAIN_BATCH_SIZE, 512))))  # 🚀 加速：增加評估批次大小以加快評估速度（256 → 512）
# 向後兼容舊代碼：BATCH_SIZE 仍沿用訓練批次大小
BATCH_SIZE = TRAIN_BATCH_SIZE
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "1"))   # Non-IID 下減少 client drift、促進全局收斂
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "25"))     # 🚀 調整：控制在約 25 輪內收斂，減少後段震盪
WEIGHT_DECAY = 1e-4  # 🚀 收斂修復：從 5e-5 提高到 1e-4（增加正則化）

# 🎯 系統配置
MODE = "virtual"
NUM_CLIENTS = 60  # 🔧 調整：40 → 60（增加無人機數量以支持擴展性實驗）
TOTAL_CLIENTS = 60  # 🔧 調整：40 → 60
# 🔧 允許通過環境變量覆蓋聚合器數量
NUM_AGGREGATORS = int(os.environ.get("NUM_AGGREGATORS", "6"))  # 調整回 6 個聚合器（60/6=10 台/聚合器）
NUM_CLOUD_SERVERS = 1
TRAIN_RATIO = 0.8

# 🎯 路徑配置
# 🔧 使用新的 Processed_Network_dataset（10 類、Dirichlet non-IID clients）
# 可透過環境變數 DATA_PATH 覆寫（用於 alpha1/alpha3 等實驗）
DATA_PATH = os.environ.get("DATA_PATH", "/home/ubuntuv100/uav/newp1/processed_network_clients_alpha2")
# 🔧 原始數據集路徑（可選）
# 選項1: CIC-DDoS2019 單一合併文件
# RAW_DATA_PATH = "/home/formosa/formosa4T2/uav/cicddos2019_clean/merged.csv"
# 選項2: CICIDS2017 客戶端分割文件（預處理腳本會自動合併）
RAW_DATA_PATH = None  # 設為 None 時，預處理腳本會自動查找 CICIDS2017_generate_dataset 目錄
# 如果使用 CICIDS2017，請將 RAW_DATA_PATH 設為 None，並確保 input_dir 指向正確目錄
MODEL_PATH = "./model"
LOG_DIR = "result"
IP_FILE = "./ip_list.txt"

# 🎯 全域評估配置（雲端評估用）
# 為了讓 cloud baseline 每一輪都「用滿 10 類、全部樣本」，這裡直接使用完整 global_test。
# 若之後想改成子抽樣，可再調整為較小的數字或改回環境變數控制。
GLOBAL_EVAL_MAX_SAMPLES: Optional[int] = 30000

# 向後兼容：模型類型
MODEL_TYPE = "federated"

# 類別權重將在訓練過程中動態計算
# 每個客戶端根據自己的數據分布計算權重
CLASS_WEIGHTS = None  # 將在客戶端訓練時動態計算

# =============================================================================
# 🏗️ 模型配置
# =============================================================================

MODEL_CONFIG = {
    'type': 'dnn',                    # 使用 DNN：目前預設為 NetworkAttackDNN backbone
    'input_dim': 58,                  # 58 維特徵（對應 processed_network_clients_alpha2/feature_cols.json 長度）
    'num_classes': 10,                # 新資料集 10 類（見 ALL_LABELS）
    'output_dim': 10,                 # 輸出維度 10
    'dropout_rate': 0.3,              # Dropout 比例，供 DNN / Transformer 共用
    
    # Transformer 特定配置（新增）
    'd_model': 128,                   # 🚀 Transformer 模型維度（輕量化設計）
    'num_layers': 2,                  # 🚀 Transformer 層數（輕量化：2層，平衡性能和效率）
    'num_heads': 4,                   # 🚀 注意力頭數（4頭，捕捉不同類型的特徵關聯）
    'd_ff': None,                     # 🚀 前饋網絡維度（None = d_model * 2）
    'max_seq_len': 58,                # 最大序列長度（等於 input_dim）
    'use_positional_encoding': True,  # 🚀 是否使用位置編碼（捕捉特徵位置信息）
    
    # DNN 特定配置（NetworkAttackDNN）
    'hidden_dims': [256, 128, 64],    # 與 NetworkAttackDNN 預設結構一致
    'use_batch_norm': True,           # 使用 BatchNorm
    'use_residual': False,            # 關掉殘差，避免目前實作中的維度對不上問題
    'activation': 'relu',             # 與原本 DNN 預設相同
    
    # CNN 特定配置（保留以備後用）
    'channels': [32, 64, 128],        # CNN通道數
    'kernel_sizes': [5, 3, 3],        # CNN卷積核大小
    'pools': [2, 2, 2],               # 池化步長
    'fc_hidden': 128,                 # CNN全連接層
    
    # 🚀 知識蒸餾配置
    'knowledge_distillation': {
        # FEDAVG_BASELINE=1 時，強制關閉 KD（純 FedAvg baseline）
        # 🔧 新增：可透過環境變數 DISABLE_KD=1 強制禁用 KD（用於「僅 Prototype，無 KD」測試）
        'enabled': not IS_FEDAVG_BASELINE and not _env_flag("DISABLE_KD", "0"),  # 是否啟用知識蒸餾
        'temperature': 4.0,          # 🚀 收斂修復：從 2.0 提高到 4.0（更平滑的分布）
        'alpha': float(os.environ.get("KD_ALPHA", "0.1")),  # 可通過環境變數調整 KD alpha（默認 0.1）
        'loss_type': 'kl_div',       # 🚀 核心修復：使用 KL 散度（比 MSE 更穩定，具備機率語義）
        'use_temperature': True,    # 是否使用溫度縮放
        'kd_warmup_rounds': int(os.environ.get("KD_WARMUP_ROUNDS", "10")),  # 🔧 冷啟動保護：前 10 輪完全禁用 KD（可通過環境變數調整，默認 10）
        'diversity_penalty': True,   # 🚀 破局配置 3.0：啟用多樣性懲罰（Entropy Loss），增加預測多樣性，對抗單一類別陷阱
        'diversity_weight': 1.5,     # 🚀 破局配置 3.9：從 0.5 提高到 1.5（加倍權重，強制模型不准只猜一個類別）
    }
}

# 🔢 全域類別數（供 cloud_server 等使用）
# 注意：需與 ALL_LABELS / MODEL_CONFIG['num_classes'] 一致
NUM_CLASSES: int = 10

# =============================================================================
# 🔁 雙向蒸餾：Client 端（GKD）配置
# =============================================================================

CLIENT_KD_CONFIG = {
    # FEDAVG_BASELINE=1 時強制關閉；可用 CLIENT_KD_ENABLED=0 額外禁用
    "enabled": (not IS_FEDAVG_BASELINE) and _env_flag("CLIENT_KD_ENABLED", "1"),
    "alpha": float(os.environ.get("CLIENT_KD_ALPHA", "0.1")),
    "temperature": float(os.environ.get("CLIENT_KD_TEMPERATURE", "4.0")),
    "loss_type": os.environ.get("CLIENT_KD_LOSS_TYPE", "kl_div"),
    "use_temperature": _env_flag("CLIENT_KD_USE_T", "1"),
    "warmup_rounds": int(os.environ.get("CLIENT_KD_WARMUP_ROUNDS", "5")),
}

# 模型頭部層名稱前綴（用於區分 base/head）
MODEL_HEAD_PREFIXES = ["output_layer", "classifier"]

# =============================================================================
# 🎓 學習率調度配置
# =============================================================================

LEARNING_RATE_SCHEDULER = {
    "enabled": True,                 # 🚀 優化：啟用學習率調度器，後期輪次逐漸降低學習率幫助微調
    "mode": "manual",                # 使用 round-aware 調度，避免僅依賴 torch scheduler
    "scheduler_type": "cosine",
    "cycle_rounds": 25,              # 🚀 與新的 MAX_ROUNDS=25 對齊
    "min_lr": 1e-4,                  # 🔧 最小學習率
    "max_lr": 1e-3,                  # 🔧 配合 base_lr 調低，避免過高
    "warmup_rounds": 3,              # 🚀 優化：減少暖啟輪次：5 → 3（更快達到全學習率）
    "warmup_factor": 0.5,            # 🚀 優化：提高暖啟起始：0.3 → 0.5（從50%開始，更快達到全學習率）
    "step_size": 10,                 # 保留舊字段供回退使用
    "gamma": 0.95,
    "f1_based_decay": True,          # 🚀 新增：啟用基於 F1 的學習率衰減
    "f1_threshold": 0.95,            # 🚀 新增：當 F1 > 0.95 時，學習率減半
    "f1_decay_factor": 0.5,          # 🚀 新增：F1 觸發時的學習率衰減因子
    "f1_high_threshold": 0.97,       # 🚀 突破 0.98-0.99：第二階段衰減閾值（當 F1 > 0.97 時）
    "f1_high_decay_factor": 0.5,     # 🚀 新增：高 F1 觸發時的學習率衰減因子
    "f1_ultra_threshold": 0.98,      # 🚀 突破 0.98-0.99：第三階段衰減閾值（當 F1 > 0.98 時，進一步降低學習率）
    "f1_ultra_decay_factor": 0.3,    # 🚀 突破 0.98-0.99：第三階段衰減因子（更激進的衰減，進行精細微調）
}

# =============================================================================
# 🤝 聯邦學習配置
# =============================================================================

FEDERATED_CONFIG = {
    'rounds': MAX_ROUNDS,             # 聯邦學習輪數
    'max_rounds': MAX_ROUNDS,        # 最大輪數
    # 注意：此鍵為舊版遺留，實際以 AGGREGATION_CONFIG['min_clients_for_aggregation'] 為準
    # 'min_clients_for_aggregation': 4,
    'training_flow': 'select_then_train', # 訓練流程
    
    'participation_strategy': {
        # 總共 25 輪時：前 8 輪 100%，其餘 9-25 輪 85%（不再用 60%）
        'initial_rounds': {'threshold': 8, 'ratio': 0.9},   # 前 8 輪：100%
        'early_rounds': {'threshold': 30, 'ratio': 0.8},  # 9-30 輪：85%
        'mid_rounds': {'threshold': 100, 'ratio': 0.9},    # 31-100 輪：90%
        'late_rounds': {'threshold': 200, 'ratio': 0.6},   # 101-200 輪：60%
        'final_rounds': {'ratio': 0.5}                     # 200 輪後：50%
    },
    
    # 同步配置
    'round_sync_interval': 15,        # 輪次同步間隔
    'round_sync_retry_interval': 15,  # 重試間隔
    
    # 客戶端選擇
    'force_participation_threshold': 3, # 強制參與閾值
    'random_seed_base': 1000,         # 隨機種子基數
    
    # 等待配置
    'client_wait_time': 0.5,         # 減少客戶端等待時間，加快響應
    'client_check_interval': 3,      # 檢查間隔
    'warmup_full_participation_rounds': 15,  # 🚀 暖啟輪數：10→15，前 15 輪全參與以穩定早期聚合
    
    # 重試配置
    'upload_max_retries': 30,         # 上傳最大重試
    'upload_retry_delay': 2,          # 重試延遲
    'round_start_max_retries': 3,     # 輪次啟動重試
    
    # 類別平衡
    'class_balance': {
        'enabled': False,             # 關閉類別平衡以避免偏置
        'focus_classes': [],
        'boost_factor': 1.0,
        'min_samples_per_class': 0,
    },
    'aggregation': {
        'smoothing_factor': 0.6       # 🔧 降低 EMA 平滑係數，讓全局模型更快響應
    }
}

# =============================================================================
# 🔄 聚合配置
# =============================================================================

# 🔧 FedProx 近端係數（可用 FEDPROX_MU 覆寫；建議 sweep 0.02–0.1）
FEDPROX_MU = float(os.environ.get("FEDPROX_MU", "0.05"))

# 🔧 修復：適度 FedProx 以平衡穩定性與學習能力
FEDPROX_CONFIG = {
    "enabled": True,                    # 🔧 啟用 FedProx 以穩定訓練
    "mu": FEDPROX_MU,
    "adaptive_mu": False,               # 關閉自適應 mu
    "warmup_rounds": 3,                 # 🔧 縮短暖啟，盡早施加約束
    "proximal_cap": 10.0,               # 🔧 近端項上限，防止 loss 暴漲（可透過環境變數 FEDPROX_PROXIMAL_CAP 覆寫）
    "mu_if_skipped_round": 0.0,         # 🔧 上一輪未參與時 mu 設為 0（或 0.5 表示減半，設 0 完全放開）
    "mu_schedule": {                    # mu 調度策略
        "initial": FEDPROX_MU,
        "final": max(0.0, FEDPROX_MU * 0.5),
        "decay_type": "cosine"
    },
    "high_performance_mu": min(0.02, FEDPROX_MU),  # 高性能區：可更放開約束
    "high_performance_threshold": 0.97,  # 🚀 突破 0.98-0.99：高性能區域閾值
}

# 🚀 新增：區域性聚合配置（Regional Aggregation）
# 參考 FedHSA 和 FedPA 的概念，實現區域性特徵對齊和異常梯度過濾
REGIONAL_AGGREGATION_CONFIG = {
    "enabled": True,                    # 是否啟用區域性聚合
    "confshield": {                     # ConfShield 異常梯度過濾
        "enabled": True,                # 是否啟用異常檢測
        "norm_threshold": 2.0,          # 🚀 放寬：1.2→2.0（norm > mean+2*std 才過濾），減少誤殺 Non-IID 客戶端
        "cosine_threshold": 0.3,        # 🚀 放寬：0.4→0.3（cosine < 0.3 才過濾），允許更多梯度參與
        "drift_threshold": 0.5,         # 🚀 放寬：0.35→0.5（drift > 0.5 才過濾），降低過濾強度
        "enable_filtering": True,         # 是否實際過濾異常客戶端
    },
    "regional_alignment": {            # 區域性特徵對齊
        "enabled": True,                # 是否啟用特徵對齊
        "alignment_method": "mean",     # 對齊方法：'mean', 'median', 'weighted_mean'
        "alignment_strength": 0.2,      # 🚀 優化：從 0.3 降低到 0.2，減少權重擺動，降低震盪
    },
    "fallback_to_fedavg": True,        # 如果區域性聚合失敗，回退到標準 FedAvg
}

# 🔧 混合聚合策略（明顯偏向高性能客戶端，追求更高性能）
AGGREGATION_STRATEGY = {
    "type": "hybrid",                   # 保留結構，關閉過濾門檻/偏重以利穩定
    # 🔧 進一步優化：極度偏向「表現好」的客戶端，讓優秀客戶端主導聚合
    "data_weight": 1.0,                 # 🔧 偏回數據量，避免過度性能偏權
    "performance_weight": 0.0,          # 🔧 關閉性能偏權
    "min_performance": 0.0,             # 🔧 關閉性能門檻過濾
    "warmup_min_performance": 0.0,      # 🔧 關閉暖啟門檻
    "fallback_min_performance": 0.0,    # 🔧 關閉回退門檻
    "performance_metric": "f1_score",    # 使用 F1 分數作為性能指標
    "normalize_weights": True,         # 正規化權重（確保權重和為 1）
    "use_adaptive_weights": True,      # 🔧 啟用自適應權重（早期更重視數據量，後期更重視性能）
    "apply_server_lr": True,           # 🔧 啟用 Server Learning Rate 縮放
    "stability_factor": 0.9,            # 🔧 提高穩定性因子：0.8 → 0.9（進一步降低性能波動對權重的影響，提高訓練穩定性）
    # 🚀 新增：Staleness-aware Weighting（非同步優化）
    "staleness_aware_weighting": {
        "enabled": True,                # 啟用基於過期度的權重調整
        "decay_factor": 0.05,           # 🔧 修復：降低衰減因子：0.1 → 0.05（每過期1輪，權重衰減5%，更溫和）
        "max_staleness": 5,             # 最大過期度（超過此值，權重降至最低）
        "min_staleness_weight": 0.3,     # 🔧 修復：提高最小權重：0.1 → 0.3（即使過期，仍保留30%權重，避免完全丟棄）
    },
    # 🚀 新增：精英領導制（Winner-Take-Most）
    "winner_take_most": {
        "enabled": False,               # 關閉精英領導制
        "performance_gap_threshold": 0.05,  # 🚀 修復單獨訓練問題：0.10 → 0.05（更早觸發精英模式，讓優秀客戶端主導）
        "elite_weight_ratio": 0.75,     # 🚀 修復單獨訓練問題：0.65 → 0.75（極度偏向精英客戶端，最大化利用優秀知識）
        "min_elite_weight": 0.60,       # 🚀 修復單獨訓練問題：0.50 → 0.60（確保精英客戶端有絕對主導權）
    },
    # 🚀 新增：Top-K 客戶端選擇（只聚合表現最好的客戶端）
    "top_k_selection": {
        "enabled": False,               # 關閉 Top-K 選擇
        "top_k_ratio": 0.2,             # 🚀 修復單獨訓練問題：0.5 → 0.2（只選擇前 20% 最好的客戶端，極度激進）
        "min_clients": 3,               # 至少選擇 3 個客戶端（避免過少）
        "apply_after_round": 30,        # 🚀 修復單獨訓練問題：50 → 30（更早應用 Top-K，避免低性能客戶端拖累）
    },
    # 🚀 新增：梯度餘弦相似度檢查（Gradient Consistency Check）
    "gradient_consistency": {
        "enabled": False,               # 🔧 關閉梯度一致性過濾
        "cosine_similarity_threshold": 0.3,  # 餘弦相似度低於此值的 Aggregator 將被降權或排除
        "exclude_opposite": True,       # 是否完全排除方向相反的 Aggregator（相似度 < 0）
        "weight_penalty_factor": 0.1,   # 對低相似度 Aggregator 的權重懲罰因子（0.1 = 降至 10%）
    },
    # 🚀 新增：預判式聚合（Quality Gate）
    "quality_gate": {
        "enabled": False,               # 🔧 關閉品質閥門
        "f1_drop_threshold": 0.15,       # 🔧 進一步降低閾值：0.20 → 0.15（更早觸發保護，避免大幅下降）
        "quick_eval_samples": 5000,     # 🔧 提高評估樣本數：1000 → 5000（提高快速評估的準確性）
        "fallback_to_best_agg": True,   # 如果聚合失敗，是否只接受表現最好的 Aggregator 的更新
        "fusion_ratio": 0.7,            # 🔧 修復：降低融合比例：1.0 → 0.7（70%最佳聚合器 + 30%其他聚合器，避免過度依賴單一聚合器）
    }
}

AGGREGATION_CONFIG = {
    # 🔧 新增：權重範數正則化配置（強化執行，確保嚴格控制在 100-200 之間）
    # 🚀 優化 B：支持動態 hard_limit（基於穩定輪次的範數均值）
    "weight_norm_regularization": {
        "enabled": True,                    # 是否啟用權重範數正則化
        "max_global_l2_norm": 150.0,        # 🔧 全局權重 L2 範數上限：100.0 → 150.0（允許適度增長，但嚴格控制）
        "hard_limit": 200.0,                # 🔧 基礎硬性上限（動態計算的基礎值）
        "use_dynamic_hard_limit": True,      # 🚀 新增：是否使用動態 hard_limit（基於穩定輪次均值）
        "stable_norm_window": 10,            # 🚀 新增：追蹤最近 N 個穩定輪次（默認 10）
        "stable_norm_multiplier": 1.5,       # 🚀 新增：hard_limit = 穩定範數均值 * multiplier（默認 1.5）
        "scaling_factor": 0.90,             # 🔧 降低縮放因子：0.95 → 0.90（更激進的縮放，確保不超過上限）
        "warn_threshold": 120.0,            # 🔧 提高警告閾值：80.0 → 120.0（更早警告）
        "strict_enforcement": True,         # 🔧 新增：嚴格執行模式（即使接近上限也進行正則化）
    },
    # 🚀 新增：精英權重投影配置（防止災難性遺忘）
    # 🔧 緊急修復：增強保護機制，防止 Round 56 類似的災難性下降
    "elite_weight_projection": {
        "enabled": True,                    # 是否啟用精英權重投影
        "max_distance": 0.2,                # 🚀 非同步優化：0.1 → 0.2（更寬鬆，避免鎖死局部最優，允許學習其他節點的Non-IID特徵）
        "method": "cosine",                 # 距離計算方法：'cosine' 或 'euclidean'
        "strength": 0.7,                    # 🚀 非同步優化：0.9 → 0.7（更溫和的投影，避免過度保守）
        "min_best_f1": 0.2,                 # 🔧 進一步降低：0.3 → 0.2（更早啟用，保護更多輪次）
        "enable_f1_based_protection": True,  # 🚀 新增：啟用基於 F1 的保護機制
        "f1_drop_threshold": 0.20,          # 🚀 優化：F1 下降閾值（20%，更寬鬆，避免過於頻繁觸發）
        "observation_period": 3,             # 🚀 優化：觀察期輪數（從 5 縮短到 3，更快觸發回退）
    },
    # 🚀 新增：聚合方法配置（支持中位數聚合和修剪平均）
    # 🔧 緊急修復：切換到中位數聚合，對抗異常值
    "aggregation_method": {
        "type": "mean",                     # 使用穩定的平均聚合
        "trim_ratio": 0.2,                  # 修剪平均的修剪比例（兩端各修剪 20%）
        "use_performance_weighted_mean": False,  # 是否使用性能加權平均
    },
    # 🔧 新增：權重更新條件配置
    "weight_update_condition": {
        "use_numerical_comparison": True,   # 使用數值比較而非 MD5 哈希
        "l2_distance_threshold": 1e-4,     # L2 距離閾值（小於此值認為權重相同）- 🔧 放寬：1e-6 → 1e-4
        "key_layers": ["output_layer.weight", "layers.0.weight", "input_reshape.weight"],  # 用於比較的關鍵層
    },
    'min_clients_for_aggregation': 3,  # 🚀 優化：4 → 3（60台無人機，每個聚合器10台，至少需要3台=30%參與率才能聚合）
    'min_clients_absolute': 3,         # 🚀 優化：4 → 3（與 min_clients_for_aggregation 保持一致）
    'max_wait_time': 480,              # 🔧 放寬等待時間：300 → 480 秒（讓聚合器多等慢速客戶端）
    'min_training_duration': 30,       # 🔧 縮短最短訓練時間，提高聚合頻率（45 → 30秒）
    'min_participation_ratio': 0.4,    # 🔧 放寬：降低參與率要求（0.6 → 0.4），允許更早聚合
    'force_aggregation_after_timeout': True, # 🔧 修復：啟用超時保底聚合，避免訓練停止
    'partial_aggregation_enabled': True,
    'min_partial_ratio': 0.4,          # 🔧 進一步降低部分聚合比例（0.5 → 0.4），確保能觸發聚合
    # 聚合器仲裁（雲端必須等多少個聚合器）
    'aggregator_quorum': 3,            # 提升仲裁門檻：至少 3 個聚合器
    'min_aggregators_for_global': 2,   # 向後兼容鍵

    # 雲端 delta 合併（CAS）：多聚合器同時 POST 時易出現 HTTP 409，靠重試 + 錯開上傳降低衝突率
    # 環境變數可覆寫：AGG_CAS_MAX_RETRIES、AGG_CAS_BACKOFF_S、AGG_CAS_UPLOAD_JITTER_S、AGG_CAS_STAGGER_PER_ID_S
    "cas_max_retries": 12,
    "cas_backoff_seconds": 0.2,
    "cas_upload_jitter_seconds": 0.15,       # 每次上傳前額外 random [0, 此值] 秒
    "cas_upload_stagger_per_id_seconds": 0.04,  # aggregator_id * 此值，讓各聚合器錯開 POST
    
    # 陳舊處理（在不調整 max_train_samples 下減少落後權重參與聚合）
    'stale_policy': 'decay',           # 陳舊策略：只做衰減，不直接丟棄
    'max_staleness': 16,               # 🔧 放寬：4 → 16（ResNet/CPU 等慢 client 不易被協調器超前輪次拒收；可用環境變數 AGG_MAX_STALENESS 覆寫）
    'staleness_decay_lambda': 0.7,     # 🔧 收緊衰減：1.1 → 0.7（落後 1 輪約 0.5 權重，更積極降權）
    
    # 魯棒聚合
    'robust_aggregator': 'clip_then_avg',    # Clip-then-Avg 增強魯棒性
    'clip_norm_max': 5.0,               # 單更新 L2 clip 門檻
    'clip_norm_eps': 1e-6,              # 避免除零
    'trim_ratio': 0.0,                 # 不使用修剪，避免信息丟失
    
    # 🚀 類別平衡聚合
    'avg_by': 'data_size',             # 依各聚合器/客戶端資料量加權
    'class_balance_factor': 1.0,       # 關閉額外類別再加權
    
    # 🔧 修復：BatchNorm 聚合模式
    'bn_aggregation_mode': 'affine_with_running',  # 🔧 聚合 BN 均值/方差，專門處理 num_batches
    'allow_late_upload_buffer': True,   # 🔧 接受遲到上傳並排入下一輪
    'late_upload_max_round_lag': 1,     # 🔧 收緊：3 → 1（僅接受落後 1 輪的遲到上傳，減少舊權重參與）
    'future_round_tolerance': 30,       # 🔧 允許前方輪次的寬容度（避免聚合器超前太多被全部丟棄）
    
    # 🔧 修復：降低 FedProx mu 以允許更多探索（過高的 mu 會限制模型性能提升）
    'fedprox': {
        'enabled': not _env_flag("DISABLE_FEDPROX", "0"),  # 🔧 可通過環境變數 DISABLE_FEDPROX=1 禁用
        'mu': FEDPROX_MU,
        'adaptive_mu': False,          # 關閉自適應 mu
        'target_drift': 0.1,           # 保留此字段以兼容代碼
        'aggregation_method': 'weighted_avg'  # 保留此字段以兼容代碼
    },
    
    # 🚀 FedNova 配置
    'fednova': {
        'enabled': False,             # 備用方案
        'aggregation_method': 'weighted_avg'
    },
    
    # 服務器EMA（啟用以平滑權重更新，減少訓練波動）
    'server_ema': {
        'enabled': not _env_flag("DISABLE_SERVER_EMA", "0"),  # 🔧 可通過環境變數 DISABLE_SERVER_EMA=1 禁用
        'decay': 0.95                 # 🚀 非同步優化：0.99 → 0.95（更積極更新，模型有5%能量吸收新知識，避免過度保守）
    },
    # 雲端非同步合併（delta）通路：預設關閉
    'cloud_async': {
        'enabled': True
    }
    ,
    # 品質門檻（純 FedAvg：關閉）
    'quality_quorum': {
        'enabled': False,             # 關閉品質檢查（純 FedAvg）
        'non_blocking': True,         # 保留此字段以兼容代碼
        'metric': 'delta_macro_f1',
        'min_delta': 0.0,
        'min_pass': 0,
        'weight_strength': 0.0        # 權重強度設為 0（不使用）
    },
    # 分佈對偶加權（Distribution-dual Aggregation）
    'dual_weighting': {
        'enabled': False,             # 預設關閉，開啟後按 JS 散度做溫和降權
        'beta_data': 1.0,             # 數據分佈權重
        'beta_pred': 1.0,             # 預測分佈權重
        'alpha_clip': 0.5             # 限幅比例：每個聚合器權重∈[1-clip, 1+clip]
    },
    # 伺服器端動量聚合（超保守 baseline：關閉，改回純 FedAvg）
    'server_momentum': {
        'enabled': False,             # 🔧 關閉 Server Momentum (FedAvgM)，避免權重更新過猛
        'momentum': 0.9,              # 保留參數以備日後重新啟用
        'nesterov': False            # 不使用 Nesterov 動量
    },
    
    # FedBN（純 FedAvg：關閉）
    'fedbn': {
        'enabled': False,             # 關閉 FedBN（純 FedAvg）
        'layers_to_exclude': [],      # 不排除任何層
    }
}

# 🔧 新增：雲端伺服器訓練開關
#   - 極簡 FedAvg baseline（FEDAVG_BASELINE=1）：自動關閉雲端 KD / Prototype Anchor / Mixup
#   - 其餘模式：維持預設 True，由 cloud_server 決定是否執行 KD 訓練
SERVER_TRAINING_CONFIG = {
    "enabled": not IS_FEDAVG_BASELINE,      # 若設為 False，雲端僅做聚合與評估，不再進行 KD / Prototype Anchor 訓練
}

# 🔧 雲端微調：聚合後用 global 資料做少量微調（已加強：20 epochs、15000 樣本、cosine LR）
#   - 可透過環境變數 CLOUD_FINETUNE=0 關閉
CLOUD_FINETUNE_CONFIG = {
    "enabled": _env_flag("CLOUD_FINETUNE", "1"),
    "epochs": int(os.environ.get("CLOUD_FINETUNE_EPOCHS", "25")),
    "lr": float(os.environ.get("CLOUD_FINETUNE_LR", "7e-5")),
    "lr_min": float(os.environ.get("CLOUD_FINETUNE_LR_MIN", "2e-5")),  # cosine 衰減下限
    "lr_schedule": "cosine",  # cosine | constant
    "batch_size": int(os.environ.get("CLOUD_FINETUNE_BATCH_SIZE", "128")),
    "max_samples": int(os.environ.get("CLOUD_FINETUNE_MAX_SAMPLES", "15000")),  # 15000 以加強微調
}

# =============================================================================
# 🎯 個人化 / 聯合預測配置
# =============================================================================

# 🔧 三方融合係數：本地 / 聚合器 / 全域
#   - 建議總和約為 1.0，但程式會自動正規化
#   - 可視實驗調整：例如偏本地 0.6 / 聚合器 0.2 / 全域 0.2
JOINT_PREDICTION_ALPHA_LOCAL = 0.5
JOINT_PREDICTION_ALPHA_AGGREGATOR = 0.25
JOINT_PREDICTION_ALPHA_GLOBAL = 0.25

# 🔧 個人化聯合預測：本地模型與全域模型的融合係數 α
#   - α 越大越偏向本地（Non-IID 客戶端自己的特性）
#   - (1-α) 越大越偏向全域（所有客戶端共享知識）
#   - 初始設為 0.5：本地 / 全域 參半，如需更強個人化可調高到 0.7~0.8
PERSONALIZATION_ALPHA = 0.5

# =============================================================================
# 🎯 優化器配置
# =============================================================================

OPTIMIZER_CONFIG = {
    "type": "adam",                    # 優化器類型
    "lr": LEARNING_RATE,              # 學習率
    "weight_decay": 2e-4,             # 🚀 優化：進一步增加權重衰減：1e-4 → 2e-4（更強正則化，減少過擬合）
    "momentum": 0.9,                   # 動量
    "beta1": 0.8,                     # 🔧 調整Adam beta1：0.9 → 0.8
    "beta2": 0.99,                    # 🔧 調整Adam beta2：0.999 → 0.99
    "eps": 1e-6                       # 🔧 調整數值穩定性：1e-8 → 1e-6
}

# =============================================================================
# 📉 損失函數配置
# =============================================================================

LOSS_CONFIG = {
    "type": "cross_entropy",          # 基礎損失函數類型（簡化版：以 CE 為主）
    "reduction": "mean",
    "label_smoothing": 0.05,          # 保留輕微 smoothing
    "label_smoothing_majority": 0.1,
    # 👉 簡化：關閉 focal loss，以穩定 CE 為主
    "focal_loss": False,
    "focal_alpha": 0.25,
    "focal_gamma": 2.5,
    # 👉 保留靜態類別權重，但改用較溫和策略
    "class_weights_enabled": True,
    "dynamic_class_weights": True,    # 👉 開啟動態類別權重，結合 Cloud 端動態權重檔
    "class_weight_strategy": "balanced",  # 使用 sklearn 標準 balanced 策略
    "force_all_classes": True,
    "min_class_weight": 1.2,          # 較溫和的權重範圍
    "max_class_weight": 3.0,
    "zero_f1_boost": 1.0,
    # 👉 簡化：關閉 client 多樣性懲罰
    "client_diversity_penalty_enabled": False,
    "client_diversity_weight": 0.0,
    "rare_class_threshold": 0.1,
    "rare_class_boost": 1.5,
    "hard_class_ids": [6, 7],
    "hard_class_min_multiplier": 1.3,
    "late_ce_enabled": False,
    "late_ce_round": 0,
    "late_ce_temperature": 1.0
}

# =============================================================================
# ⚖️ 由 Cloud F1 自動生成並下發的動態 class weight（下一輪生效）
# =============================================================================
# 說明：
# - Cloud Server 每輪評估後，會在 EXPERIMENT_DIR 寫出 dynamic_class_weights.json
# - Client 每輪開始訓練前，會讀取該 JSON 並用於 CrossEntropyLoss(weight=...)
# - 目標：自動補強 F1 較低的類別（例如 DoS_Slowhttptest），減少手動調參
DYNAMIC_CLASS_WEIGHTING = {
    # 👉 啟用 Cloud 端動態 class weight：根據各類別 F1 自動調整權重
    "enabled": True,
    "filename": "dynamic_class_weights.json",
    # 權重公式：w_c ∝ 1 / max(F1_c, eps)
    "eps": 1e-3,
    # 是否將權重強制正規化為「平均=1」
    # 🔧 改為較溫和的做法：關閉強制平均=1，讓弱類別（例如 class3）權重可以偏離 1.0 多一點
    "normalize_mean1": False,
    # 針對權重做 EMA 平滑（避免每輪劇烈抖動）；new = beta*old + (1-beta)*current
    # 🚀 穩定化：提高到 0.85，讓權重變化更平滑（原本 0.7）
    "ema_beta": 0.85,
    # 對權重做裁剪，避免極端值導致訓練不穩或完全偏向某類
    "min_weight": 0.5,
    "max_weight": 12.0,   # 🔧 再提高上限，讓極低 F1 類別有更多補償空間
    # 🚀 進一步自動化：額外強化「當輪 F1 最差的幾個類別」，完全不用手動指定類別 ID
    # top_k: 每輪自動挑出 F1 最低的前幾類（例如 3 類），給額外 boost
    # 🔧 強化 injection/password、ransomware：2 → 3，涵蓋 password(6)、ransomware(7) 等弱類
    "extra_boost_top_k": 3,
    # factor: 對這些最差類別的權重再乘上一個係數（例如 1.35 = 額外 +35%）
    # 🔧 再溫和一點：1.25 → 1.15，減少崩盤輪
    "extra_boost_factor": 1.15,
    "extra_boost_min_f1": 0.88,
    "use_sliding_window": True,
    "sliding_window_size": 5,
    # 每 N 輪才更新一次動態權重，2 → 3 更平滑
    "update_frequency": 3,
}

# =============================================================================
# 🎛️ 客戶端訓練配置
# =============================================================================

CLIENT_TRAINING_CONFIG = {
    "local_epochs": LOCAL_EPOCHS,      # 🔧 修復客戶端訓練：增加到 10（確保足夠的訓練步數）
    "batch_size": BATCH_SIZE,          # 🔧 修復客戶端訓練：降低到 64（確保每個 epoch 有足夠的批次數）
    "learning_rate": LEARNING_RATE,    # 學習率
    "optimizer": "Adam",               # 優化器
    "weight_decay": 1e-4,              # 🔧 修復訓練變慢：1e-3 → 1e-4（降低權重衰減，加快訓練速度）
    "scheduler": "cosine",             # 調度器
    "gradient_clipping": 2.0,         # 🔧 放寬梯度裁剪：1.0 → 2.0
    "early_stopping": False,           # 早停
    "patience": 0,                     # 耐心
    "min_delta": 0.0                   # 最小改善
}

# =============================================================================
# 🌫️ 梯度噪聲配置（提升泛化與抗干擾）
# =============================================================================

GRAD_NOISE_CONFIG = {
    "enabled": False,          # 是否啟用梯度噪聲
    "base_sigma": 0.01,        # 初始噪聲強度
    "decay": 0.0,              # 噪聲衰減（>0 時會隨 epoch 降低）
    "min_sigma": 0.0,          # 最小噪聲強度
    "max_sigma": 0.05,         # 最大噪聲強度（防止過大）
    "mode": "adaptive",        # fixed/adaptive
    "ref_grad_norm": 1.0,      # 目標梯度範數（adaptive 用）
    "log_interval": 50         # 每多少 batch 印一次
}

# =============================================================================
# 🎲 客戶端選擇配置
# =============================================================================

CLIENT_SELECTION_CONFIG = {
    "strategy": "random",               # 選擇策略
    "base_participation_ratio": 0.7,    # 🔧 再提高參與率：0.6 → 0.7
    "min_participation_ratio": 0.6,     # 🔧 提高最小參與率：0.5 → 0.6
    "max_participation_ratio": 0.8,     # 🔧 提高最大參與率：0.7 → 0.8
    "performance_threshold": 0.001,     # 性能閾值
    "improvement_threshold": 0.0001,    # 改善閾值
    "consecutive_failure_limit": 2,    # 連續失敗限制
    "health_check_interval": 2,        # 健康檢查間隔
    "fairness_weight": 0.5,             # 公平性權重
    "performance_weight": 0.5,         # 性能權重
    
    # 健康跳過
    "health_skip": {
        "enabled": True,               # 啟用健康跳過
        "min_accuracy": 0.1,          # 最小準確率
        "max_loss": 3.0,               # 最大損失
        "cooldown_rounds": 3          # 冷卻輪數
    }
}

# =============================================================================
# 📊 數據配置
# =============================================================================

DATA_CONFIG = {
    "train_ratio": 0.8,               # 訓練比例
    "test_ratio": 0.2,                 # 測試比例
    "validation_ratio": 0.1,           # 驗證比例
    "random_state": 42,                # 隨機種子
    "shuffle": True,                  # 是否打亂
    "stratify": True,                  # 是否分層
    "augmentation": True,              # 🔧 啟用數據增強：False → True
    "normalization": "standard",       # 🔧 統一使用standard標準化：robust → standard
    "feature_selection": False,        # 🔧 關閉特徵選擇：True → False
    
    # 🔧 新增：數據使用比例（用於加快讀取和訓練速度）
    "data_usage_ratio": 1.0,           # 🔧 修復：使用全部驗證數據（1.0 = 100%，0.2 = 20%）
    "max_train_samples": None,        # 🚀 不裁切：500→None，使用 client 全部訓練樣本以提升 FL 表現
    "max_test_samples": 1000,         # 🔧 優化：限制驗證集大小為 1000 樣本（加快評估速度，減少 CPU 負擔）
    
    # 🚀 數據平衡配置
    "balancing": {
        "enabled": True,               # 啟用數據平衡
        "method": "borderline_smote",  # 🚀 優化：改用 Borderline-SMOTE（smote → borderline_smote，更適合邊界樣本）
        "k_neighbors": 3,              # 🚀 優化：降低鄰居數：5 → 3（更激進的過採樣，更適合極少數類別）
        "sampling_strategy": "auto",   # 採樣策略
        "random_state": 42             # 隨機種子
    },
    
    # 🚀 預處理配置
    "preprocessing": {
        "scaler_type": "standard",       # 🔧 改為 StandardScaler（避免極端值被放大）
        "handle_missing": "drop",      # 處理缺失值：drop, fill, impute
        "outlier_detection": True,     # 🔧 改進：啟用異常值檢測
        "outlier_method": "isolation_forest",  # 異常值檢測方法：isolation_forest, iqr, zscore
        "outlier_contamination": 0.1,  # 異常值比例（10%）
        "outlier_threshold": 0.1,      # 異常值閾值（移除10%的異常值）
        "feature_scaling": True        # 特徵縮放
    }
}

# =============================================================================
# 🧪 GAN/生成式增強（MVP）
# =============================================================================

GAN_AUGMENTATION_CONFIG = {
    "enabled": False,             # 7 類版本預設關閉 GAN（避免與舊 10 類生成器配置不符）
    "target_label": 6,            # 針對類別（Ransomware，在 7 類設定中為 6）
    "ratio": 0.1,                 # 生成比例（相對於訓練集總量）；可設 "auto"
    "target_class_ratio": 0.1,    # ratio="auto" 時，補到此比例
    "min_class_ratio": 0.05,      # 若目標類別比例 >= 此值則不補
    "max_new_samples": 0,         # 0=不限制，否則上限
    "latent_dim": 42,             # 生成器輸入維度（對應訓練時 noise+label 共 42）
    "generator_path": "weights/cwgan_toniot_43f_10c.pt",  # 生成器權重路徑（可用 GAN_GENERATOR_PATH 覆寫）
    "device": "auto"              # auto/cpu/cuda
}

# =============================================================================
# 🌐 網絡配置
# =============================================================================

NETWORK_CONFIG = {
    'cloud_server': {
        'host': '127.0.0.1',
        'port': 8083,
        'url': 'http://127.0.0.1:8083'
    },
    'aggregators': {
        'base_port': 8000,
        'host': '127.0.0.1',
    'ports': [8000, 8001, 8002, 8003, 8004, 8005]  # 🔧 調整：6 個端口以支持 6 個聚合器
    },
    'clients': {
        'host': '127.0.0.1',
        'base_port': 9000
    },
    'registration_delay': 3,           # 註冊延遲
}

# =============================================================================
# 🏷️ 標籤配置
# =============================================================================

LABEL_COL = "label"
# 🔧 新資料集 10 類標籤（與 preprocess_processed_network.py 的 CLASS_NAMES 一致）
ALL_LABELS = [
    "normal",      # 0
    "backdoor",    # 1
    "dos",         # 2
    "ddos",        # 3
    "injection",   # 4
    "mitm",        # 5
    "password",    # 6
    "ransomware",  # 7
    "scanning",    # 8
    "xss",         # 9
]
ALL_TYPE_NAMES = ALL_LABELS

# 🔧 數值標籤到字串標籤的映射
LABEL_MAPPING = {i: name for i, name in enumerate(ALL_LABELS)}

# 字串標籤到數值標籤的映射
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# =============================================================================
# 📝 日誌配置
# =============================================================================

LOG_CONFIG = {
    "result_log_format": "csv",        # 結果日誌格式
    "unified_naming": True,            # 統一命名
    "result_dir": "result",            # 結果目錄
    "show_client_details": True,        # 顯示客戶端詳情
    "show_evaluation_details": True,   # 顯示評估詳情
    "show_error_details": True,         # 顯示錯誤詳情
    "show_round_summary": True,        # 顯示輪次摘要
    "show_aggregator_status": True     # 顯示聚合器狀態
}

# =============================================================================
# 🚀 F1 ≥ 0.9 優化預設（可透過 F1_OPTIMIZATION=1 啟用）
# =============================================================================
# 啟用時：Focal Loss、較高 LOCAL_EPOCHS、雲端微調等，有助於逼近單機 0.92 上限
# F1_OPTIMIZATION_ENABLED 已於檔案開頭定義

# 雲端微調：聚合後在 global_train 上做少量 epoch 微調，可望再提升 2-5%
CLOUD_FINE_TUNING_CONFIG = {
    "enabled": F1_OPTIMIZATION_ENABLED,  # 與 F1_OPTIMIZATION 聯動
    "epochs": 2,                        # 每輪聚合後微調 epoch 數
    "lr": 1e-4,                        # 微調學習率（較低避免破壞聚合結果）
    "max_samples": 10000,              # 最多使用多少訓練樣本（避免過慢）
}

# =============================================================================
# 🛡️ 攻擊模擬配置（Model Poisoning / Label Flipping）
# =============================================================================

ATTACK_CONFIG = {
    "enabled": _env_flag("ATTACK_ENABLED", "0"),
    "malicious_ratio": float(os.environ.get("MALICIOUS_RATIO", "0.0")),
    "malicious_clients": os.environ.get("MALICIOUS_CLIENTS", "").strip(),  # 例: "0,1,2"
    "seed": int(os.environ.get("MALICIOUS_SEED", "42")),
    "label_flipping": {
        "enabled": _env_flag("LABEL_FLIP_ENABLED", "0"),
        "source_label": os.environ.get("LABEL_FLIP_SOURCE", "DDoS"),
        "target_label": os.environ.get("LABEL_FLIP_TARGET", "BENIGN"),
    },
    "model_poisoning": {
        "enabled": _env_flag("MODEL_POISON_ENABLED", "0"),
        "method": os.environ.get("MODEL_POISON_METHOD", "gaussian"),  # gaussian | random
        "sigma": float(os.environ.get("MODEL_POISON_SIGMA", "0.5")),
        "replace_prob": float(os.environ.get("MODEL_POISON_REPLACE_PROB", "1.0"))
    }
}

# =============================================================================
# 🛡️ ConfShield：權重異常偵測（第一層：DBI / Cluster-based）
#   - 目前先以「監測模式」運行：只計算與列印結果，不實際調整聚合權重
#   - 未來可將 action 改為 "soft" 或 "hard" 讓 Cloud 在聚合前自動降權 / 剔除可疑節點
# =============================================================================

SECURITY_CONFIG = {
    "dbi_weight_anomaly": {
        "enabled": True,              # 是否啟用 DBI 權重異常分析
        "pca_dim": 32,                # PCA 降維維度（過大會變慢，過小會損失資訊）
        "cluster_k": 2,               # k-means 聚類群數（2~4 較常見）
        "min_cluster_ratio": 0.1,     # 小於此比例的 cluster 才視為「小集群」
        "distance_threshold": 1.5,    # 小集群中心與主群中心距離 > 此值才視為可疑
        "action": "monitor",          # monitor | soft | hard
        "soft_factor": 0.3,           # action == "soft" 時，可疑節點權重乘上的係數
        "log_top_k": 5,               # 最多列印多少個 client 的詳細距離 / 群組資訊
    }
}

# =============================================================================
# 🔧 梯度配置
# =============================================================================

GRADIENT_CLIPPING = {
    "enabled": True,                   # 啟用梯度裁剪
    "max_norm": 1.0,                  # 最大範數
    "norm_type": 2,                    # 範數類型
    "adaptive": True,                  # 自適應裁剪
}

# =============================================================================
# 🎯 評估配置
# =============================================================================

EVAL_CONFIG = {
    'benign_logit_bias': 0.0,          # 移除評估偏置
    'eval_every_rounds': 2             # 評估間隔
}

# 向後兼容：評估頻率
EVAL_EVERY_ROUNDS = EVAL_CONFIG['eval_every_rounds']

# =============================================================================
# 🔄 FedProx配置
# =============================================================================

# FEDPROX_CONFIG 已在第 164 行定義（純 FedAvg：關閉）
# 以下為舊定義，已註釋：
# FEDPROX_CONFIG = {
#     "enabled": True,                   # 啟用FedProx
#     "mu": 0.01                         # 🔧 降低約束強度以允許更大更新
# }

# =============================================================================
# 🎯 收斂配置
# =============================================================================

CONVERGENCE_CONFIG = {
    'max_rounds': 25,                  # 與 MAX_ROUNDS 對齊
    'patience': 10,                    # 連續幾輪無提升則早停
    'min_improvement': 0.0005,         # 最小改善門檻
    'min_rounds': 15                   # 至少訓練輪數
}

# =============================================================================
# 🛑 大幅下滑保護（關閉早停時仍能止損）
# =============================================================================

DROP_GUARD_CONFIG = {
    "enabled": True,            # 啟用下滑保護
    "min_rounds": 15,           # 至少訓練輪數
    "drop_threshold": 0.08,     # 與最佳值相比下滑超過 0.08 視為嚴重
    "drop_patience": 2          # 連續幾輪嚴重下滑才停止
}

# =============================================================================
# 🧭 FedProc-lite Prototype Loss 配置
# =============================================================================

PROTOTYPE_LOSS_CONFIG = {
    # 極簡 FedAvg baseline：保持關閉
    # 🔧 新增：可透過環境變數 DISABLE_PROTOTYPE=1 強制禁用 Prototype（用於「僅 KD，無 Prototype」測試）
    # 🔧 新增：可透過環境變數 ENABLE_PROTOTYPE_ONLY=1 強制啟用 Prototype（用於「僅 Prototype，無 KD」測試）
    "enabled": _env_flag("ENABLE_PROTOTYPE_ONLY", "0") and not _env_flag("DISABLE_PROTOTYPE", "0"),
    "lambda_proto": 0.02,              # 基礎權重 λ（用於無排程或作為起點）
    "path": os.path.join("model", "global_prototypes.npy"),  # 全局原型檔案路徑
    "log_every_n_batches": 100,        # 每多少個 batch 輸出一次 prototype loss 日誌
    # 🔄 輪次排程：隨 round 動態調整 lambda_proto
    "schedule": {
        "enabled": True,               # 啟用 prototype loss 排程
        "mode": "cosine",             # 'cosine' 或 'linear'
        "max_rounds": 20,              # 排程只作用前 20 輪，之後固定為 end_lambda
        "start_lambda": 0.05,          # 早期輪次的 λ 起點（加強 early-round 約束）
        "end_lambda": 0.0              # 訓練後期逐漸關閉 prototype loss
    }
}

# =============================================================================
# ⚖️ 全域類別權重 (Class-Balanced Loss) 配置
# =============================================================================

CLASS_WEIGHT_CONFIG = {
    "enabled": False,                                # 🚀 核心修復：禁用全域權重（Non-IID 場景下，本地權重更準確）
    # 理論依據：在 Non-IID 場景下，每個客戶端的類別分佈不同，應該使用本地權重
    # 類別權重是本地優化策略，不參與聚合，因此可以使用不同的權重
    "path": os.path.join("model", "global_class_weights.npy"),  # 權重檔案路徑（備用）
    "normalize": True                                # 是否在載入後再正規化到平均為 1
}

# =============================================================================
# ⚡ 性能配置
# =============================================================================

PERFORMANCE_CONFIG = {
    "use_gpu": False,                  # 使用GPU
    "num_workers": 8,                  # 工作進程數
    "pin_memory": True,                # 固定內存
    "prefetch_factor": 2,             # 預取因子
    "persistent_workers": True,            # 持久工作進程
    "async_loading": False,            # 異步加載
    "compression": False,              # 壓縮
    "quantization": False              # 量化
}

# =============================================================================
# 🔄 Split Learning配置
# =============================================================================

SPLIT_LEARNING_CONFIG = {
    'enabled': False,                   # 全域關閉 Split Learning
    'compression': {
        'enabled': False               # 關閉壓縮
    },
    'disable_cloud_training': True     # 禁用雲端端訓練流程
}

# =============================================================================
# 🚀 實驗配置
# =============================================================================

EXPERIMENT_CONFIG = {
    "name": "optimized_federated_learning",
    "description": "優化的聯邦學習配置",
    "version": "4.0",
    "tags": ["optimized", "cleaned", "efficient"],
    "save_config": True,               # 保存配置
    "track_metrics": True,             # 追蹤指標
    "early_stopping": False,           # 關閉早停（避免震盪期被提早停止）
    "convergence_check": False         # 收斂檢查
}


def _apply_small_scale_overrides():
    """啟用小規模模式時調整系統參數"""
    global NUM_AGGREGATORS, NUM_CLIENTS, TOTAL_CLIENTS, MAX_ROUNDS, ROUND_CLIENT_LIMIT, SMALL_SCALE_SUMMARY
    if not SMALL_SCALE_MODE:
        SMALL_SCALE_SUMMARY = {}
        ROUND_CLIENT_LIMIT = None
        return
    try:
        small_num_aggs = max(1, int(os.environ.get("SMALL_SCALE_NUM_AGGREGATORS", "2")))
        small_num_clients = max(small_num_aggs, int(os.environ.get("SMALL_SCALE_NUM_CLIENTS", "9")))
        small_rounds = max(1, int(os.environ.get("SMALL_SCALE_MAX_ROUNDS", "10")))
        small_clients_per_round = max(1, int(os.environ.get("SMALL_SCALE_CLIENTS_PER_ROUND", "3")))
        small_wait_time = max(60, int(os.environ.get("SMALL_SCALE_MAX_WAIT", "180")))
        
        agg_cfg = NETWORK_CONFIG.get('aggregators', {})
        ports = list(agg_cfg.get('ports', []))
        if ports:
            NUM_AGGREGATORS = max(1, min(NUM_AGGREGATORS, small_num_aggs, len(ports)))
            NETWORK_CONFIG['aggregators']['ports'] = ports[:NUM_AGGREGATORS]
        else:
            NUM_AGGREGATORS = max(1, min(NUM_AGGREGATORS, small_num_aggs))
        
        NUM_CLIENTS = max(NUM_AGGREGATORS, min(NUM_CLIENTS, small_num_clients))
        TOTAL_CLIENTS = NUM_CLIENTS
        MAX_ROUNDS = min(MAX_ROUNDS, small_rounds)
        FEDERATED_CONFIG['rounds'] = min(FEDERATED_CONFIG.get('rounds', MAX_ROUNDS), MAX_ROUNDS)
        FEDERATED_CONFIG['max_rounds'] = FEDERATED_CONFIG['rounds']
        
        clients_per_aggregator = max(1, math.ceil(NUM_CLIENTS / NUM_AGGREGATORS))
        ROUND_CLIENT_LIMIT = min(small_clients_per_round, clients_per_aggregator)
        target_ratio = min(1.0, ROUND_CLIENT_LIMIT / clients_per_aggregator)
        
        # 區域聚合門檻：勿等於「每輪抽樣上限」ROUND_CLIENT_LIMIT（否則變相要求整池幾乎全到齊才聚合）。
        # 改為 6～8 附近或「上限減餘量」，且不大於每區人數減一。
        _cap = int(ROUND_CLIENT_LIMIT)
        _pool = int(clients_per_aggregator)
        min_agg = max(2, min(8, max(6, _cap - 3)))
        min_agg = min(min_agg, max(2, _pool - 1))
        AGGREGATION_CONFIG['min_clients_for_aggregation'] = min_agg
        AGGREGATION_CONFIG['min_clients_absolute'] = min_agg
        AGGREGATION_CONFIG['min_participation_ratio'] = target_ratio
        AGGREGATION_CONFIG['max_wait_time'] = min(AGGREGATION_CONFIG.get('max_wait_time', 420), small_wait_time)
        AGGREGATION_CONFIG['min_training_duration'] = min(AGGREGATION_CONFIG.get('min_training_duration', 45), 30)
        
        CLIENT_SELECTION_CONFIG['base_participation_ratio'] = target_ratio
        CLIENT_SELECTION_CONFIG['min_participation_ratio'] = target_ratio
        CLIENT_SELECTION_CONFIG['max_participation_ratio'] = target_ratio
        
        total_rounds = FEDERATED_CONFIG['rounds']
        early_threshold = min(total_rounds, 3)
        mid_threshold = min(total_rounds, 6)
        late_threshold = min(total_rounds, 9)
        FEDERATED_CONFIG['participation_strategy'] = {
            'early_rounds': {'threshold': early_threshold, 'ratio': target_ratio},
            'mid_rounds': {'threshold': mid_threshold, 'ratio': target_ratio},
            'late_rounds': {'threshold': late_threshold, 'ratio': target_ratio},
            'final_rounds': {'ratio': target_ratio}
        }
        
        SMALL_SCALE_SUMMARY = {
            "enabled": True,
            "num_aggregators": NUM_AGGREGATORS,
            "num_clients": NUM_CLIENTS,
            "max_rounds": FEDERATED_CONFIG['rounds'],
            "clients_per_round_limit": ROUND_CLIENT_LIMIT
        }
    except Exception as exc:
        SMALL_SCALE_SUMMARY = {"error": str(exc)}
        ROUND_CLIENT_LIMIT = None


_apply_small_scale_overrides()

# =============================================================================
# 📊 配置驗證和打印
# =============================================================================

def validate_config():
    """驗證配置一致性"""
    assert LEARNING_RATE > 0, "學習率必須大於0"
    assert BATCH_SIZE > 0, "批次大小必須大於0"
    assert LOCAL_EPOCHS > 0, "本地訓練輪數必須大於0"
    assert MAX_ROUNDS > 0, "最大輪數必須大於0"
    assert NUM_CLIENTS > 0, "客戶端數量必須大於0"
    assert NUM_AGGREGATORS > 0, "聚合器數量必須大於0"
    print("✅ 配置驗證通過")

# 打印配置摘要
print("🎯 優化後的聯邦學習配置已加載")
print(f"📊 核心參數:")
print(f"  - 學習率: {LEARNING_RATE}")
print(f"  - 服務器學習率: {SERVER_LR}")
print(f"  - 批次大小: {BATCH_SIZE}")
print(f"  - 本地訓練輪數: {LOCAL_EPOCHS}")
print(f"  - 最大輪數: {MAX_ROUNDS}")
print(f"  - 客戶端數量: {NUM_CLIENTS}")
print(f"  - 聚合器數量: {NUM_AGGREGATORS}")
print(f"  - 模型類型: {MODEL_CONFIG['type'].upper()}")
print(f"  - 損失函數: {LOSS_CONFIG['type']}")
print(f"  - 參與率: {FEDERATED_CONFIG['participation_strategy']}")
print(f"  - 聚合策略: {AGGREGATION_CONFIG['robust_aggregator']}")
if SMALL_SCALE_MODE:
    print(f"🧪 小規模模式啟用: {SMALL_SCALE_SUMMARY}")

# 驗證配置
validate_config()