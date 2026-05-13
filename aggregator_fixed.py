#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合器 - 修復版本
專注於解決權重分發和輪次管理問題
"""

import asyncio
import json
import pickle
import time
import traceback
import argparse
import sys
import os
from typing import Dict, List, Any, Optional
import datetime
import math
from contextlib import asynccontextmanager

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
import aiohttp

# 動態載入配置模組（支援 CONFIG_MODULE，預設 config_fixed）
import importlib
CONFIG_MODULE = os.environ.get("CONFIG_MODULE", "config_fixed")
try:
    config = importlib.import_module(CONFIG_MODULE)
    print(f"[Aggregator] ✅ 使用配置模組: {CONFIG_MODULE}")
except Exception as e:
    print(f"[Aggregator] ⚠️ 載入配置模組 {CONFIG_MODULE} 失敗，回退到 config_fixed: {e}")
    import config_fixed as config

# 導入模型（改用本地 dnn.py）
from models.dnn import NetworkAttackDNN

# 🚀 新增：導入區域性聚合模組
try:
    from models.regional_aggregation import RegionalAggregator
    REGIONAL_AGGREGATION_AVAILABLE = True
    print("[Aggregator] ✅ 區域性聚合模組已載入")
except ImportError as e:
    REGIONAL_AGGREGATION_AVAILABLE = False
    print(f"[Aggregator] ⚠️ 區域性聚合模組載入失敗: {e}，將使用標準 FedAvg")

# 全局變量

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan 事件"""
    try:
        # 🔧 新增：啟動輪次同步檢查
        asyncio.create_task(periodic_round_sync_check())
        
        # 🔧 修復：延遲註冊到Cloud Server，避免啟動阻塞
        async def delayed_register():
            await asyncio.sleep(3)  # 等待3秒讓服務完全啟動
            await register_with_cloud_server()
        
        asyncio.create_task(delayed_register())
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] ⚠️ 啟動註冊任務失敗: {e}")
    yield

# 創建 FastAPI 應用（使用 lifespan）
app = FastAPI(title="Federated Learning Aggregator", version="2.0", lifespan=lifespan)

# 聚合器狀態
aggregator_id = 0
# federated_status 輪詢極頻繁時降頻 print，避免 I/O 拖慢事件迴圈（見 get_federated_status）
_federated_status_req_seq = 0
round_count = 0
last_completed_round = 0
# 協調器 COORDINATOR_REQUIRE_CLOUD_ACK / 全員同步用：最後一次「雲端已接受 delta（HTTP 200）」或廣播所對應的 FL round_id
last_cloud_ack_round = 0
round_clients = []
client_weights_buffer = {}
client_data_sizes = {}
# 客戶端預測緩衝區（用於跨層教師機制）
client_predictions_buffer = {}
round_start_time = None
client_timeout_status = {}
global_weights = None
server_global_weights = None
model_version = 1
server_model_version = -1
is_training_phase = False
aggregator_port = 8000

# 初始化全局權重
def initialize_global_weights():
    """初始化全局權重"""
    global global_weights
    if global_weights is None:
        print(f"[Aggregator {aggregator_id}] 🔧 初始化全局權重...")
        # 與客戶端完全一致的模型配置
        input_dim = int(config.MODEL_CONFIG.get('input_dim', 22))
        num_classes = int(config.MODEL_CONFIG.get('output_dim', config.MODEL_CONFIG.get('num_classes', 5)))
        # 自動推斷輸入維度
        try:
            if input_dim <= 0:
                data_dir = getattr(config, 'DATA_PATH', '')
                sample_csv = None
                if os.path.isdir(data_dir):
                    for fn in os.listdir(data_dir):
                        if fn.endswith('.csv'):
                            sample_csv = os.path.join(data_dir, fn)
                            break
                if sample_csv:
                    import pandas as pd
                    df_head = pd.read_csv(sample_csv, nrows=1)
                    label_col = getattr(config, 'LABEL_COL', 'label')
                    if label_col in df_head.columns:
                        input_dim = max(1, len(df_head.columns) - 1)
                    else:
                        input_dim = max(1, len(df_head.columns))
        except Exception:
            pass
        
        # 🚀 進階優化：根據配置選擇模型類型（與客戶端和雲端服務器保持一致）
        model_type = config.MODEL_CONFIG.get('type', 'dnn')
        dropout_rate = float(config.MODEL_CONFIG.get('dropout_rate', 0.3))
        
        if model_type == 'transformer':
            # 🚀 使用輕量化 Transformer 模型
            from models.transformer import build_transformer
            model = build_transformer(
                input_dim=input_dim,
                output_dim=num_classes,
                d_model=config.MODEL_CONFIG.get('d_model', 128),
                num_layers=config.MODEL_CONFIG.get('num_layers', 2),
                num_heads=config.MODEL_CONFIG.get('num_heads', 4),
                d_ff=config.MODEL_CONFIG.get('d_ff', None),
                dropout=dropout_rate,
                max_seq_len=config.MODEL_CONFIG.get('max_seq_len', input_dim),
                use_positional_encoding=config.MODEL_CONFIG.get('use_positional_encoding', True)
            )
            print(f"[Aggregator {aggregator_id}] 🚀 使用 Transformer 模型初始化全局權重")
        elif model_type == 'dnn':
            # 使用標準 DNN 模型
            from models.dnn import build_dnn
            model = build_dnn(
                input_dim=input_dim,
                output_dim=num_classes,
                hidden_dims=config.MODEL_CONFIG.get('hidden_dims', [256, 128, 64]),
                dropout_rate=dropout_rate,
                use_batch_norm=config.MODEL_CONFIG.get('use_batch_norm', True),
                use_residual=config.MODEL_CONFIG.get('use_residual', True),
                activation=config.MODEL_CONFIG.get('activation', 'relu')
            )
            print(f"[Aggregator {aggregator_id}] 🔧 使用 DNN 模型初始化全局權重")
        elif model_type == 'cnn':
            # 使用 CNN 模型
            from models.cnn import build_cnn
            model = build_cnn(input_dim=input_dim, output_dim=num_classes)
            print(f"[Aggregator {aggregator_id}] 🔧 使用 CNN 模型初始化全局權重")
        else:
            # 回退到 DNN
            from models.dnn import build_dnn
            model = build_dnn(
                input_dim=input_dim,
                output_dim=num_classes,
                hidden_dims=config.MODEL_CONFIG.get('hidden_dims', [256, 128, 64]),
                dropout_rate=dropout_rate
            )
            print(f"[Aggregator {aggregator_id}] ⚠️ 未知模型類型 {model_type}，回退到 DNN")
        
        global_weights = model.state_dict()
        print(f"[Aggregator {aggregator_id}] ✅ 全局權重初始化完成，權重層數: {len(global_weights)}")
    return global_weights

def reset_global_weights():
    """強制重置全局權重"""
    global global_weights
    print(f"[Aggregator {aggregator_id}] 🔧 強制重置全局權重...")
    global_weights = None
    return initialize_global_weights()

# 配置
federated_config = config.FEDERATED_CONFIG
AGGREGATION_TIMEOUT = int(getattr(config, 'AGGREGATION_CONFIG', {}).get('max_wait_time', 120))
PARTIAL_AGGREGATION_ENABLED = bool(getattr(config, 'AGGREGATION_CONFIG', {}).get('partial_aggregation_enabled', True))
MIN_PARTIAL_RATIO = float(getattr(config, 'AGGREGATION_CONFIG', {}).get('min_partial_ratio', 0.3))
# 允許用環境變數覆寫每輪最小聚合比例（例如 0.5 = 至少 1/2 client）
_mpr_env = os.environ.get("AGG_MIN_PARTIAL_RATIO", "").strip()
if _mpr_env:
    try:
        MIN_PARTIAL_RATIO = max(0.0, min(1.0, float(_mpr_env)))
    except Exception:
        pass

# 進階聚合策略配置（帶預設值，透過 config 覆蓋）
AGG_CFG = getattr(config, 'AGGREGATION_CONFIG', {})
_stale_raw = str(AGG_CFG.get('stale_policy', 'allow'))
# 'decay_then_drop' 與 'decay' 共用衰減邏輯，確保落後權重被降權
STALE_POLICY = 'decay' if _stale_raw in ('decay', 'decay_then_drop') else _stale_raw  # 'allow' | 'strict' | 'decay'
MAX_STALENESS = int(AGG_CFG.get('max_staleness', 1))
# 環境變數覆寫（不需改 config）：例如 ResNet+CPU 實驗 export AGG_MAX_STALENESS=20
_ms_env = os.environ.get("AGG_MAX_STALENESS", "").strip()
if _ms_env:
    try:
        MAX_STALENESS = max(1, int(float(_ms_env)))
    except Exception:
        pass
STALENESS_DECAY_LAMBDA = float(AGG_CFG.get('staleness_decay_lambda', 0.7))
DATASIZE_ALPHA_MAX_MULTIPLIER = float(AGG_CFG.get('alpha_max_multiplier', 3.0))  # cap 為 (multiplier * 1/N)
# 環境變數覆寫（實驗用）：放大某 client 的有效 data_size 後，若仍被 cap 夾死，可一併提高此倍率
_amp_env = os.environ.get("AGG_ALPHA_MAX_MULTIPLIER", "").strip()
if _amp_env:
    try:
        DATASIZE_ALPHA_MAX_MULTIPLIER = float(_amp_env)
    except Exception:
        pass
elif (os.environ.get("IMAGE_FL", "0") or "").strip().lower() in ("1", "true", "yes", "on"):
    # Image FL 預設採較保守 cap，降低 data_size/attacker 放大導致的單輪主導風險。
    DATASIZE_ALPHA_MAX_MULTIPLIER = 2.5

ROBUST_CFG = AGG_CFG.get('robust', {}) or {}
ROBUST_METHOD = str(ROBUST_CFG.get('method', 'none'))  # 'none' | 'median' | 'trimmed_mean'
ROBUST_TRIM_RATIO = float(ROBUST_CFG.get('trim_ratio', 0.2))

CLIP_CFG = AGG_CFG.get('clip', {}) or {}
CLIP_NORM = float(CLIP_CFG.get('norm', 0.0))  # 0 或 <=0 表示關閉

# Norm guard 與信任權重（防 poisoning）
NORM_GUARD_CFG = AGG_CFG.get('norm_guard', {}) or {}
NORM_GUARD_ENABLED = bool(NORM_GUARD_CFG.get('enabled', True))
NORM_GUARD_K = float(NORM_GUARD_CFG.get('k', 3.5))
NORM_GUARD_PENALTY = float(NORM_GUARD_CFG.get('penalty_factor', 0.15))

TRUST_CFG = AGG_CFG.get('trust', {}) or {}
TRUST_ENABLED = bool(TRUST_CFG.get('enabled', True))
TRUST_DECAY = float(TRUST_CFG.get('decay', 0.9))
TRUST_GAIN = float(TRUST_CFG.get('gain', 0.01))

# Cloud Server配置
# 最小改動：允許用環境變數覆寫 Cloud 上傳 timeout（避免 aiohttp TimeoutError() 過於頻繁）。
# - 首選：AGG_CLOUD_TIMEOUT_S
# - 備援：AGG_CLOUD_SERVER_TIMEOUT_S
_cloud_timeout_raw = os.environ.get("AGG_CLOUD_TIMEOUT_S", os.environ.get("AGG_CLOUD_SERVER_TIMEOUT_S", "")).strip()
try:
    if _cloud_timeout_raw:
        _cloud_timeout_s = float(_cloud_timeout_raw)
    else:
        # 影像聯邦 / 大模型：delta pickle 與雲端合併耗時常超過 30s，預設拉長以免聚合器 TimeoutError。
        _img_fl = (os.environ.get("IMAGE_FL", "0") or "").strip().lower() in ("1", "true", "yes", "on")
        _cloud_timeout_s = 600.0 if _img_fl else 30.0
except Exception:
    _cloud_timeout_s = 30.0
_cloud_timeout_s = max(0.1, float(_cloud_timeout_s))

CLOUD_SERVER_CONFIG = {
    "enabled": True,
    "url": config.NETWORK_CONFIG["cloud_server"]["url"],
    "upload_after_aggregation": True,
    "timeout": _cloud_timeout_s,
}

# 簡化客戶端選擇器
simplified_client_selector = None
global_performance = 0.5

# 服務器端模型
server_model = None
optimizer = None
criterion = None

# 聚合統計
total_aggregations_done = 0

# 併發鎖
buffer_lock = asyncio.Lock()
aggregation_lock = asyncio.Lock()

# 簡化聚合策略
simplified_aggregation_strategy = None

# 日誌和監控
should_stop_logging = False

def log_event(event, detail=""):
    """記錄事件（簡化版：僅輸出到控制台）"""
    print(f"[Aggregator {aggregator_id}] 📝 {event}: {detail}")

def save_aggregation_stats():
    """保存聚合統計信息（簡化版：僅輸出到控制台）"""
    stats = {
        "aggregator_id": aggregator_id,
        "round_count": round_count,
        "current_clients": len(round_clients),
        "buffer_size": len(client_weights_buffer),
        "model_version": model_version,
        "is_training_phase": is_training_phase,
    }
    print(f"[Aggregator {aggregator_id}] 📊 統計: {stats}")

# -------------------------
# Cloud Server Integration
# -------------------------
async def register_with_cloud_server():
    """向 Cloud Server 註冊聚合器 - 改進版本"""
    max_retries = 5
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            if not CLOUD_SERVER_CONFIG.get("enabled", False):
                print(f"[Aggregator {aggregator_id}] ℹ️ Cloud Server 連接已禁用")
                return
            
            cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
            if not cloud_url:
                print(f"[Aggregator {aggregator_id}] ⚠️ Cloud Server URL 未配置")
                return
            
            print(f"[Aggregator {aggregator_id}] 🔄 嘗試連接 Cloud Server (嘗試 {attempt + 1}/{max_retries})")
            
            data = aiohttp.FormData()
            data.add_field('aggregator_id', str(aggregator_id))
            data.add_field('status', 'ready')
            data.add_field('port', str(aggregator_port))
            
            timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{cloud_url}/register_aggregator", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"[Aggregator {aggregator_id}] ✅ 已向Cloud Server註冊成功")
                        print(f"  - 響應: {result}")
                        log_event("cloud_register_success", f"status={resp.status}, attempt={attempt + 1}")
                        return  # 成功註冊，退出重試
                    else:
                        text = await resp.text()
                        print(f"[Aggregator {aggregator_id}] ⚠️ Cloud 註冊失敗: HTTP {resp.status}")
                        print(f"  - 錯誤詳情: {text}")
                        log_event("cloud_register_failed", f"status={resp.status}, attempt={attempt + 1}, error={text}")
                        
        except asyncio.TimeoutError:
            print(f"[Aggregator {aggregator_id}] ⏰ Cloud 註冊超時 (嘗試 {attempt + 1}/{max_retries})")
            log_event("cloud_register_timeout", f"attempt={attempt + 1}")
        except aiohttp.ClientConnectorError as e:
            print(f"[Aggregator {aggregator_id}] 🔌 Cloud 連接失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
            log_event("cloud_register_connection_error", f"attempt={attempt + 1}, error={str(e)}")
        except Exception as e:
            print(f"[Aggregator {aggregator_id}] ❌ Cloud 註冊異常 (嘗試 {attempt + 1}/{max_retries}): {e}")
            log_event("cloud_register_exception", f"attempt={attempt + 1}, error={str(e)}")
        
        # 計算延遲時間（指數退避）
        delay = base_delay * (2 ** attempt)
        print(f"[Aggregator {aggregator_id}] ⏳ 等待 {delay:.1f} 秒後重試...")
        await asyncio.sleep(delay)
    
    # 所有重試都失敗
    print(f"[Aggregator {aggregator_id}] ❌ Cloud 註冊失敗，已嘗試 {max_retries} 次")
    log_event("cloud_register_final_failure", f"max_retries={max_retries}")
    
    # 🔧 修復：即使Cloud Server連接失敗，也允許聚合器繼續運行
    print(f"[Aggregator {aggregator_id}] 🔧 允許聚合器在無Cloud Server模式下繼續運行")
    log_event("cloud_server_disabled_fallback", "聚合器將在本地模式下運行")
    
    # 記錄詳細的診斷信息
    try:
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        print(f"[Aggregator {aggregator_id}] 🔍 診斷信息:")
        print(f"  - Cloud URL: {cloud_url}")
        print(f"  - 聚合器端口: {aggregator_port}")
        print(f"  - 超時設置: {CLOUD_SERVER_CONFIG.get('timeout', 30)}s")
        print(f"  - 啟用狀態: {CLOUD_SERVER_CONFIG.get('enabled', False)}")
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] ⚠️ 診斷信息收集失敗: {e}")

async def upload_aggregated_weights_to_cloud(round_id_value: int, weights_state: Dict[str, Any], participating_clients: List[int]):
    """上傳聚合後的權重到 Cloud Server"""
    try:
        if not CLOUD_SERVER_CONFIG.get("enabled", False) or not CLOUD_SERVER_CONFIG.get("upload_after_aggregation", False):
            return
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        if not cloud_url:
            return
        # 準備上傳資料：將權重轉為可聚合的 numpy 陣列
        aggregated_weights_payload = {}
        for key, value in (weights_state or {}).items():
            try:
                if isinstance(value, torch.Tensor):
                    if value.device.type != 'cpu':
                        value_cpu = value.detach().cpu()
                    else:
                        value_cpu = value.detach()
                    aggregated_weights_payload[key] = value_cpu.numpy()
                elif isinstance(value, np.ndarray):
                    aggregated_weights_payload[key] = value
                else:
                    # 轉為 numpy 陣列（若可能）
                    aggregated_weights_payload[key] = np.array(value, dtype=np.float32)
            except Exception:
                # 跳過不可序列化/非法層
                continue

        upload_dict = {
            'aggregated_weights': aggregated_weights_payload,
            'aggregation_stats': {
                'num_layers': len(aggregated_weights_payload),
                'timestamp': time.time(),
                'participating_clients': participating_clients,
            }
        }
        weights_bytes = pickle.dumps(upload_dict)
        data = aiohttp.FormData()
        data.add_field('aggregator_id', str(aggregator_id))
        data.add_field('round_id', str(round_id_value))
        data.add_field('model_version', str(model_version))
        data.add_field('participating_clients', json.dumps(participating_clients))
        data.add_field('weights', weights_bytes, filename='aggregated_weights.pkl')
        timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{cloud_url}/upload_aggregated_weights", data=data) as resp:
                if resp.status == 200:
                    print(f"[Aggregator {aggregator_id}] ☁️ 已上傳聚合權重到Cloud Server (round={round_id_value})")
                    log_event("cloud_upload_success", f"round={round_id_value}")
                else:
                    text = await resp.text()
                    print(f"[Aggregator {aggregator_id}] ⚠️ 上傳聚合權重到Cloud失敗: {resp.status} {text}")
                    log_event("cloud_upload_failed", f"round={round_id_value}, status={resp.status}")
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] ❌ 上傳聚合權重異常: {e}")
        log_event("cloud_upload_exception", f"round={round_id_value}, error={e}")

async def upload_delta_to_cloud(
    base_version: int,
    delta_state: Dict[str, Any],
    round_id_value: int,
    buffered_clients: Optional[int] = None,
) -> bool:
    """上傳 delta 權重到 Cloud，啟用非同步合併。回傳是否成功（200 且流程走完）。"""
    global last_cloud_ack_round
    try:
        # 同輪已被雲端 ACK（或本地已知 ACK）時，不重複上傳，避免重送風暴。
        try:
            if int(last_cloud_ack_round) >= int(round_id_value):
                bc = buffered_clients if buffered_clients is not None else -1
                print(
                    f"[Aggregator {aggregator_id}] ↪️ delta_upload_skip_already_acked "
                    f"round={round_id_value} buffered_clients={bc} last_cloud_ack_round={int(last_cloud_ack_round)}"
                )
                return True
        except Exception:
            pass
        # 同輪已成功上傳過則直接略過（本地去重，避免同 round 重複上傳）。
        try:
            uploaded_rounds = getattr(app.state, "uploaded_rounds", set())
            if int(round_id_value) in uploaded_rounds:
                bc = buffered_clients if buffered_clients is not None else -1
                print(
                    f"[Aggregator {aggregator_id}] ↪️ delta_upload_skip_already_uploaded "
                    f"round={round_id_value} buffered_clients={bc}"
                )
                return True
        except Exception:
            pass

        if not CLOUD_SERVER_CONFIG.get("enabled", False):
            return True
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        if not cloud_url:
            return True
        
        # 輔助函數：計算 delta
        def _compute_delta(local_weights, base_weights_dict):
            """計算 delta = local - base"""
            def _to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().float()
                elif isinstance(x, np.ndarray):
                    return torch.from_numpy(x).float()
                elif isinstance(x, list):
                    try:
                        return torch.tensor(x, dtype=torch.float32)
                    except Exception:
                        return None
                return None
            
            delta_result = {}
            if isinstance(base_weights_dict, dict) and base_weights_dict:
                for k, v in local_weights.items():
                    g = _to_tensor(v)
                    b = _to_tensor(base_weights_dict.get(k))
                    if g is not None and b is not None and g.shape == b.shape:
                        delta_result[k] = (g - b)
                    else:
                        if g is not None:
                            delta_result[k] = g
            else:
                # 無 base -> 上傳全量作為 delta
                for k, v in local_weights.items():
                    t = _to_tensor(v)
                    if t is not None:
                        delta_result[k] = t
            return delta_result
        
        # 輔助函數：序列化 delta
        def _serialize_delta(delta_dict):
            delta_payload = {}
            for k, v in (delta_dict or {}).items():
                if isinstance(v, torch.Tensor):
                    delta_payload[k] = v.detach().cpu().numpy()
                elif isinstance(v, np.ndarray):
                    delta_payload[k] = v
                else:
                    try:
                        delta_payload[k] = np.array(v, dtype=np.float32)
                    except Exception:
                        continue
            return pickle.dumps(delta_payload)
        
        timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
        _agg_cfg = getattr(config, "AGGREGATION_CONFIG", {}) or {}
        max_retries = int(_agg_cfg.get("cas_max_retries", 4))
        base_backoff = float(_agg_cfg.get("cas_backoff_seconds", 0.5))
        jitter_max = float(_agg_cfg.get("cas_upload_jitter_seconds", 0.0))
        stagger_per_id = float(_agg_cfg.get("cas_upload_stagger_per_id_seconds", 0.0))
        import random

        def _env_float(name: str, cur: float) -> float:
            raw = (os.environ.get(name) or "").strip()
            if not raw:
                return cur
            try:
                return max(0.0, float(raw))
            except ValueError:
                return cur

        def _env_int(name: str, cur: int) -> int:
            raw = (os.environ.get(name) or "").strip()
            if not raw:
                return cur
            try:
                return max(1, int(raw))
            except ValueError:
                return cur

        max_retries = _env_int("AGG_CAS_MAX_RETRIES", max_retries)
        base_backoff = _env_float("AGG_CAS_BACKOFF_S", base_backoff)
        jitter_max = _env_float("AGG_CAS_UPLOAD_JITTER_S", jitter_max)
        stagger_per_id = _env_float("AGG_CAS_STAGGER_PER_ID_S", stagger_per_id)
        
        current_base_version = base_version
        current_delta = delta_state
        
        pre_sleep = stagger_per_id * float(aggregator_id)
        if jitter_max > 0.0:
            pre_sleep += random.uniform(0.0, jitter_max)
        if pre_sleep > 0.0:
            await asyncio.sleep(pre_sleep)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            attempt = 0
            while attempt < max_retries:
                # 序列化當前 delta
                delta_bytes = _serialize_delta(current_delta)
                data = aiohttp.FormData()
                data.add_field('aggregator_id', str(aggregator_id))
                data.add_field('base_version', str(current_base_version))
                data.add_field('round_id', str(round_id_value))
                data.add_field('model_version', str(model_version))
                data.add_field('delta', delta_bytes, filename='delta.pkl')
                
                async with session.post(f"{cloud_url}/upload_aggregated_delta", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        bc = buffered_clients if buffered_clients is not None else -1
                        print(
                            f"[Aggregator {aggregator_id}] ☁️ delta_upload_ok "
                            f"round={round_id_value} buffered_clients={bc} "
                            f"base_version={current_base_version} new_version={result.get('new_version')}"
                        )
                        log_event(
                            "cloud_delta_upload_success",
                            f"round={round_id_value}, buffered_clients={bc}, base_version={current_base_version}",
                        )
                        try:
                            rid = int(round_id_value)
                            if rid > int(last_cloud_ack_round):
                                last_cloud_ack_round = rid
                        except Exception:
                            pass
                        try:
                            if not hasattr(app.state, "uploaded_rounds"):
                                app.state.uploaded_rounds = set()
                            app.state.uploaded_rounds.add(int(round_id_value))
                            app.state.last_uploaded_round = int(round_id_value)
                        except Exception:
                            pass
                        return True
                    elif resp.status == 409:
                        # 🔧 修復：版本衝突時，重新獲取最新版本並重新計算 delta
                        bc = buffered_clients if buffered_clients is not None else -1
                        print(
                            f"[Aggregator {aggregator_id}] ⚠️ delta_upload_409 "
                            f"round={round_id_value} buffered_clients={bc} "
                            f"base_version={current_base_version} retry={attempt+1}/{max_retries}"
                        )
                        try:
                            # 重新獲取最新版本
                            async with session.get(f"{cloud_url}/get_global_weights_with_version") as get_resp:
                                if get_resp.status == 200:
                                    payload = pickle.loads(await get_resp.read())
                                    new_base_version = int(payload.get('version', 0))
                                    new_base_weights = payload.get('weights', {})
                                    print(f"[Aggregator {aggregator_id}] 🔄 重新獲取版本: {current_base_version} → {new_base_version}")
                                    # 重新計算 delta
                                    current_base_version = new_base_version
                                    current_delta = _compute_delta(global_weights, new_base_weights)
                                    delay = base_backoff * (2 ** attempt) + random.uniform(-0.1, 0.1)
                                    await asyncio.sleep(max(0.0, delay))
                                    attempt += 1
                                    continue
                                else:
                                    print(f"[Aggregator {aggregator_id}] ⚠️ 重新獲取版本失敗: HTTP {get_resp.status}")
                                    attempt += 1
                                    await asyncio.sleep(base_backoff)
                                    continue
                        except Exception as e:
                            print(f"[Aggregator {aggregator_id}] ⚠️ 重新獲取版本異常: {e}")
                            attempt += 1
                            await asyncio.sleep(base_backoff)
                            continue
                    else:
                        text = await resp.text()
                        bc = buffered_clients if buffered_clients is not None else -1
                        print(
                            f"[Aggregator {aggregator_id}] ❌ delta_upload_fail "
                            f"round={round_id_value} buffered_clients={bc} "
                            f"status={resp.status} msg={text[:200]}"
                        )
                        log_event(
                            "cloud_delta_upload_failed",
                            f"round={round_id_value}, buffered_clients={bc}, status={resp.status}",
                        )
                        return False
            
            # 達到最大重試次數
            print(f"[Aggregator {aggregator_id}] ❌ 上傳 delta 達到最大重試次數 ({max_retries})，放棄")
            log_event("cloud_delta_upload_max_retries", f"max_retries={max_retries}")
            return False
    except Exception as e:
        bc = buffered_clients if buffered_clients is not None else -1
        # 有些例外的 str(e) 可能是空字串，改用 repr(e) 並附上精簡 traceback，避免 log「看起來沒錯」
        err_repr = repr(e)
        tb = traceback.format_exc(limit=6)
        print(
            f"[Aggregator {aggregator_id}] ❌ delta_upload_exception "
            f"round={round_id_value} buffered_clients={bc} err={err_repr}\n"
            f"[Aggregator {aggregator_id}] ❌ delta_upload_exception traceback:\n{tb}"
        )
        log_event("cloud_delta_upload_exception", err_repr)
        return False


async def fetch_global_weights_from_cloud_with_retry(max_retries: int = 15, delay_seconds: float = 2.0) -> bool:
    """從 Cloud Server 拉取全局權重（帶重試），成功則更新本地 global_weights"""
    try:
        if not CLOUD_SERVER_CONFIG.get("enabled", False):
            return False
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        if not cloud_url:
            return False

        timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                        if resp.status == 200:
                            weights_bytes = await resp.read()
                            payload = pickle.loads(weights_bytes)
                            server_version = int(payload.get('version', 0))
                            server_weights = payload.get('weights', {})

                            # 轉換為 torch.Tensor
                            updated_weights = {}
                            for k, v in (server_weights or {}).items():
                                if isinstance(v, torch.Tensor):
                                    updated_weights[k] = v.detach().cpu()
                                elif isinstance(v, np.ndarray):
                                    updated_weights[k] = torch.from_numpy(v).float()
                                elif isinstance(v, list):
                                    try:
                                        updated_weights[k] = torch.tensor(v, dtype=torch.float32)
                                    except Exception:
                                        continue
                                else:
                                    try:
                                        updated_weights[k] = torch.tensor(v, dtype=torch.float32)
                                    except Exception:
                                        continue

                            if updated_weights:
                                global global_weights, model_version
                                global_weights = updated_weights
                                model_version = int(server_version)
                                print(f"[Aggregator {aggregator_id}] ☁️ 已同步Cloud全局權重（cloud_version={server_version}）")
                                log_event("cloud_sync_success", f"cloud_version={server_version}")
                                return True
                        elif resp.status == 404:
                            # Cloud 尚未完成全局聚合
                            print(f"[Aggregator {aggregator_id}] ⏳ Cloud 全局權重尚未可用（404），重試 {attempt+1}/{max_retries}")
                        else:
                            text = await resp.text()
                            print(f"[Aggregator {aggregator_id}] ⚠️ 獲取Cloud全局權重失敗: {resp.status} {text}")
            except Exception as inner_e:
                print(f"[Aggregator {aggregator_id}] ⚠️ 連接Cloud獲取權重失敗（嘗試 {attempt+1}/{max_retries}）: {inner_e}")

            await asyncio.sleep(delay_seconds)

        log_event("cloud_sync_timeout", f"retries={max_retries}")
        return False
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] ❌ 同步Cloud全局權重異常: {e}")
        log_event("cloud_sync_exception", str(e))
        return False

async def periodic_round_sync_check():
    """🔧 新增：定期檢查並同步聚合器輪次"""
    while True:
        try:
            await asyncio.sleep(60)  # 每60秒檢查一次
            await check_and_sync_rounds()
        except Exception as e:
            print(f"[Aggregator {aggregator_id}] ⚠️ 輪次同步檢查異常: {e}")
            await asyncio.sleep(30)  # 異常時縮短間隔

async def check_and_sync_rounds():
    """🔧 新增：檢查並同步所有聚合器輪次"""
    try:
        # 🔧 修復：動態構建所有聚合器的URL列表
        num_aggregators = getattr(config, 'NUM_AGGREGATORS', 4)
        aggregator_urls = [
            f"http://127.0.0.1:{8000 + i}" for i in range(num_aggregators)
        ]
        
        current_rounds = []
        
        for i, url in enumerate(aggregator_urls):
            try:
                async with aiohttp.ClientSession() as session:
                    # 🔧 修復聚合器同步失敗警告：增加超時時間和重試機制
                    timeout = aiohttp.ClientTimeout(total=5)  # 2秒 → 5秒（增加超時時間）
                    async with session.get(f"{url}/current_round", timeout=timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            current_round = data.get('current_round', 0)
                            current_rounds.append(current_round)
                        else:
                            # 🔧 修復：記錄詳細錯誤信息
                            error_text = await response.text()
                            print(f"[Aggregator {aggregator_id}] ⚠️ 檢查聚合器 {i} 輪次失敗: HTTP {response.status} - {error_text[:100]}")
                            current_rounds.append(0)
            except asyncio.TimeoutError:
                # 🔧 修復：區分超時錯誤和其他錯誤
                print(f"[Aggregator {aggregator_id}] ⚠️ 檢查聚合器 {i} 輪次超時（可能未啟動或網絡延遲）")
                current_rounds.append(0)
            except Exception as e:
                # 🔧 修復：記錄詳細錯誤信息，但不中斷同步流程
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"[Aggregator {aggregator_id}] ⚠️ 檢查聚合器 {i} 輪次失敗: {error_msg}")
                current_rounds.append(0)
        
        # 🔧 修復：檢查 aggregator_id 是否在有效範圍內
        if aggregator_id >= len(current_rounds):
            print(f"[Aggregator {aggregator_id}] ⚠️ 聚合器ID {aggregator_id} 超出範圍 (0-{len(current_rounds)-1})，跳過同步檢查")
            return
        
        # 找到最高輪次
        if len(current_rounds) == 0:
            print(f"[Aggregator {aggregator_id}] ⚠️ 無法獲取任何聚合器輪次，跳過同步檢查")
            return
            
        max_round = max(current_rounds)
        current_round = current_rounds[aggregator_id]
        
        # 如果當前聚合器落後太多，自動同步
        if max_round > current_round + 1:  # 落後1輪以上就同步
            print(f"[Aggregator {aggregator_id}] 🔄 檢測到輪次落後: 當前{current_round}輪，最高{max_round}輪，開始同步...")
            
            try:
                async with aiohttp.ClientSession() as session:
                    data = aiohttp.FormData()
                    data.add_field('target_round', str(max_round))
                    
                    async with session.post(f"{aggregator_urls[aggregator_id]}/reset_round", data=data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"[Aggregator {aggregator_id}] ✅ 輪次同步成功: {current_round} -> {max_round}")
                            log_event("round_sync", f"自動同步: {current_round} -> {max_round}")
                        else:
                            print(f"[Aggregator {aggregator_id}] ❌ 輪次同步失敗: HTTP {response.status}")
            except Exception as e:
                print(f"[Aggregator {aggregator_id}] ❌ 輪次同步異常: {e}")
        
        # 記錄輪次狀態
        round_status = ", ".join([f"聚合器{i}={current_rounds[i]}輪" for i in range(len(current_rounds))])
        print(f"[Aggregator {aggregator_id}] 📊 輪次狀態: {round_status}, 最高={max_round}輪")
        
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] ❌ 輪次同步檢查失敗: {e}")
        import traceback
        traceback.print_exc()

def log_training_event_aggregator(event, info: dict):
    """記錄訓練事件（簡化版）"""
    print(f"[Aggregator {aggregator_id}] 📊 {event}: {info}")

def get_global_performance_metric():
    """獲取全局性能指標（簡化版：返回固定值）"""
    return 0.5

def select_clients_for_round(round_id, total_clients=None):
    """🔧 改進：實現分層非同步 FL 的客戶端選擇策略"""
    assigned_clients = get_assigned_clients()
    if not assigned_clients:
        return []
    
    # 🔧 實現隨機抽樣策略（分層非同步 FL 核心）
    import random
    
    # 設置隨機種子確保可重現性
    random.seed(round_id + aggregator_id * 1000)
    
    # 穩定參與率策略：確保足夠的客戶端參與
    # 🚀 使用配置中的參與率策略（前幾輪可設 initial_rounds 接近 1.0）
    participation_cfg = config.FEDERATED_CONFIG.get('participation_strategy', {})
    initial = participation_cfg.get('initial_rounds', {})
    initial_threshold = initial.get('threshold', 0)
    if initial_threshold > 0 and round_id <= initial_threshold:
        participation_ratio = initial.get('ratio', 1.0)
    elif round_id <= participation_cfg.get('early_rounds', {}).get('threshold', 20):
        participation_ratio = participation_cfg.get('early_rounds', {}).get('ratio', 0.8)
    elif round_id <= participation_cfg.get('mid_rounds', {}).get('threshold', 100):
        participation_ratio = participation_cfg.get('mid_rounds', {}).get('ratio', 0.7)
    elif round_id <= participation_cfg.get('late_rounds', {}).get('threshold', 200):
        participation_ratio = participation_cfg.get('late_rounds', {}).get('ratio', 0.6)
    else:
        participation_ratio = participation_cfg.get('final_rounds', {}).get('ratio', 0.5)
    
    # 計算要選擇的客戶端數量
    if total_clients is None:
        total_clients = len(assigned_clients)
    num_to_select = max(2, int(total_clients * participation_ratio))
    
    # 隨機抽樣
    selected_clients = random.sample(assigned_clients, min(num_to_select, len(assigned_clients)))
    
    # 確保公平性：如果某客戶端連續多輪未被選中，強制加入
    if not hasattr(app.state, 'not_selected_streak'):
        app.state.not_selected_streak = {}
    
    # 🔧 修復：確保所有 assigned_clients 都在 not_selected_streak 字典中
    for cid in assigned_clients:
        if cid not in app.state.not_selected_streak:
            app.state.not_selected_streak[cid] = 0
    
    # 更新未被選中的連續輪次
    for cid in assigned_clients:
        if cid in selected_clients:
            app.state.not_selected_streak[cid] = 0
        else:
            app.state.not_selected_streak[cid] = app.state.not_selected_streak.get(cid, 0) + 1
        
    # 強制補入開關（A/B 實驗預設關閉，避免改變抽樣集合）
    enable_forced_clients = os.environ.get("AGG_ENABLE_FORCED_CLIENTS", "0").strip() == "1"
    if enable_forced_clients:
        forced_clients = [cid for cid in assigned_clients if app.state.not_selected_streak.get(cid, 0) >= 3]
        for cid in forced_clients:
            if cid not in selected_clients:
                selected_clients.append(cid)
                print(f"[Aggregator {aggregator_id}] 🔧 強制加入連續3輪未被選中的客戶端: {cid}")
    
    # 去重並排序
    selected_clients = sorted(list(set(selected_clients)))
    
    print(f"[Aggregator {aggregator_id}] 🔧 分層非同步 FL 客戶端選擇: {selected_clients} (參與率: {len(selected_clients)}/{len(assigned_clients)} = {len(selected_clients)/len(assigned_clients):.1%})")
    
    return selected_clients

def calculate_dynamic_participation_ratio(round_id, assigned_clients):
    """計算動態參與率"""
    dynamic_config = config.FEDERATED_CONFIG.get('client_selection', {}).get('dynamic_participation', {})
    base_ratio = float(config.FEDERATED_CONFIG.get('client_selection', {}).get('base_participation_ratio', 0.8))
    min_ratio = float(config.FEDERATED_CONFIG.get('client_selection', {}).get('min_participation_ratio', 0.4))
    max_ratio = float(config.FEDERATED_CONFIG.get('client_selection', {}).get('max_participation_ratio', 1.0))
    
    if not dynamic_config.get('enabled', False):
        return base_ratio
    
    # 初始化性能歷史
    if not hasattr(app.state, 'performance_history'):
        app.state.performance_history = {}
    
    if not hasattr(app.state, 'participation_ratios'):
        app.state.participation_ratios = {base_ratio}
    
    # 計算平均性能改善
    avg_improvement = calculate_average_performance_improvement(assigned_clients, round_id)
    
    # 根據性能改善調整參與率
    improvement_factor = dynamic_config.get('improvement_factor', 0.1)
    penalty_factor = dynamic_config.get('penalty_factor', 0.05)
    min_adjustment = dynamic_config.get('min_adjustment', 0.05)
    max_adjustment = dynamic_config.get('max_adjustment', 0.2)
    smoothing_factor = dynamic_config.get('smoothing_factor', 0.8)
    
    # 計算調整幅度
    if avg_improvement > 0.01:  # 性能改善
        adjustment = min(max_adjustment, improvement_factor * avg_improvement)
        new_ratio = min(max_ratio, base_ratio + adjustment)
    elif avg_improvement < -0.01:  # 性能下降
        adjustment = min(max_adjustment, penalty_factor * abs(avg_improvement))
        new_ratio = max(min_ratio, base_ratio - adjustment)
    else:  # 性能穩定
        new_ratio = base_ratio
    
    # 平滑處理
    if app.state.participation_ratios:
        last_ratio = max(app.state.participation_ratios)
        new_ratio = smoothing_factor * last_ratio + (1 - smoothing_factor) * new_ratio
    
    # 確保在合理範圍內
    new_ratio = max(min_ratio, min(max_ratio, new_ratio))
    
    # 記錄參與率歷史
    app.state.participation_ratios.add(new_ratio)
    
    print(f"[Aggregator {aggregator_id}] 🔧 動態參與率調整: {base_ratio:.3f} -> {new_ratio:.3f} (改善={avg_improvement:.4f})")
    
    return new_ratio

def calculate_client_performance_scores(assigned_clients, round_id):
    """計算客戶端性能分數（簡化版：返回默認分數）"""
    return {cid: 0.5 for cid in assigned_clients}

def calculate_average_performance_improvement(assigned_clients, round_id):
    """計算平均性能改善（簡化版：返回0）"""
    return 0.0

def record_client_performance(client_id, results_data, round_id):
    """記錄客戶端性能（簡化版：僅輸出日誌）"""
    if isinstance(results_data, dict):
        f1 = results_data.get('f1_score', results_data.get('accuracy', 0.0))
        print(f"[Aggregator {aggregator_id}] 📊 客戶端 {client_id} 輪次 {round_id} 性能: {f1:.4f}")


def perform_standard_fedavg(client_weights_list, client_data_sizes_list, client_ids_list=None):
    """執行標準FedAvg聚合"""
    if not client_weights_list:
        return {}
    
    # =============================
    # Defense hooks (image/tabular)
    # =============================
    # 透過環境變數控制（預設 none）：
    # - AGG_DEFENSE_MODE: none | median | trimmed_mean | cosine_filter
    # - AGG_TRIM_RATIO: trimmed_mean 兩端修剪比例（例如 0.2）
    # - AGG_COSINE_THRESHOLD: cosine_filter 固定門檻（例如 0.0~0.5）
    # - AGG_COSINE_PERCENTILE: cosine_filter 動態門檻百分位（0~100），例如 80 = 每輪保留 top20% 以上相似度
    # - AGG_COSINE_DROP_TOP_PCT: cosine_filter 丟掉「最合群(相似度最高)」的前 x%（0~100），用於對抗共謀攻擊
    # - AGG_MIN_CLIENTS_AFTER_FILTER: 過濾後至少保留幾個 client（避免全丟）
    # - AGG_UPDATE_CLIP_NORM: 先對每個 client 的「更新向量」做 L2 clipping（防大幅度惡意更新）
    defense_mode = (os.environ.get("AGG_DEFENSE_MODE", "none") or "none").strip().lower()
    trim_ratio = float(os.environ.get("AGG_TRIM_RATIO", str(ROBUST_TRIM_RATIO)))
    cosine_th = float(os.environ.get("AGG_COSINE_THRESHOLD", "0.0"))
    cosine_pct_raw = (os.environ.get("AGG_COSINE_PERCENTILE", "") or "").strip()
    cosine_drop_top_raw = (os.environ.get("AGG_COSINE_DROP_TOP_PCT", "") or "").strip()
    update_clip_norm = float(os.environ.get("AGG_UPDATE_CLIP_NORM", "0.0") or 0.0)
    min_after = int(os.environ.get("AGG_MIN_CLIENTS_AFTER_FILTER", "2"))
    if update_clip_norm <= 0:
        try:
            if not getattr(perform_standard_fedavg, "_clip_disable_logged", False):
                perform_standard_fedavg._clip_disable_logged = True
                print(
                    "[Aggregator] 🛡️ update_clip: disabled (AGG_UPDATE_CLIP_NORM<=0, no per-client L2 scaling)",
                    flush=True,
                )
        except Exception:
            pass
    if client_ids_list is None or len(client_ids_list) != len(client_weights_list):
        client_ids_list = list(range(len(client_weights_list)))

    def _is_bn_stat(name: str) -> bool:
        return ("running_mean" in name) or ("running_var" in name) or ("num_batches_tracked" in name)

    def _to_tensor_f32(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32)

    def _flatten_update(sd: dict) -> torch.Tensor:
        """以 (client_weights - global_weights) 展平成向量，用於 cosine filter。"""
        vecs = []
        for k, v in sd.items():
            if _is_bn_stat(k):
                continue
            try:
                cv = _to_tensor_f32(v).view(-1)
            except Exception:
                continue
            if global_weights is not None and k in global_weights:
                try:
                    gv = _to_tensor_f32(global_weights[k]).view(-1)
                    cv = cv - gv
                except Exception:
                    pass
            vecs.append(cv.cpu())
        if not vecs:
            return torch.zeros((1,), dtype=torch.float32)
        return torch.cat(vecs, dim=0)

    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float()
        b = b.float()
        if a.numel() != b.numel():
            m = min(a.numel(), b.numel())
            a = a[:m]
            b = b[:m]
        denom = (a.norm() * b.norm()).item()
        if denom <= 1e-12:
            return 0.0
        return float(torch.dot(a, b).item() / denom)

    def _clip_client_update_dict_with_stats(sd: dict):
        """對 client update 做整體 L2 clipping，再還原回權重 dict。"""
        if update_clip_norm <= 0:
            return sd, 0.0, False, 1.0
        try:
            total_sq = 0.0
            prepared = {}
            for k, v in sd.items():
                cv = _to_tensor_f32(v)
                gv = None
                if isinstance(global_weights, dict) and k in global_weights:
                    try:
                        gv = _to_tensor_f32(global_weights[k])
                    except Exception:
                        gv = None
                if gv is not None and cv.shape == gv.shape:
                    upd = (cv - gv).detach().float()
                else:
                    # 若沒有可用 global 對照，退化為直接 clip 權重向量
                    upd = cv.detach().float()
                prepared[k] = (cv, gv, upd)
                total_sq += float(torch.sum(upd * upd).item())
            if total_sq <= 0:
                return sd, 0.0, False, 1.0
            norm = float(total_sq ** 0.5)
            if norm <= update_clip_norm:
                return sd, norm, False, 1.0
            scale = float(update_clip_norm / (norm + 1e-12))
            clipped = {}
            for k, (cv, gv, upd) in prepared.items():
                upd_new = upd * scale
                if gv is not None and cv.shape == gv.shape:
                    clipped[k] = (gv + upd_new).to(dtype=cv.dtype)
                else:
                    clipped[k] = upd_new.to(dtype=cv.dtype)
            return clipped, norm, True, scale
        except Exception:
            return sd, 0.0, False, 1.0

    # 先做 update-level L2 clipping，再進入 cosine/median/trimmed/avg
    if update_clip_norm > 0:
        clipped_list = []
        clip_stats = []
        for cid, sd in zip(client_ids_list, client_weights_list):
            new_sd, pre_norm, did_clip, scale = _clip_client_update_dict_with_stats(sd)
            clipped_list.append(new_sd)
            clip_stats.append(
                {
                    "client_id": cid,
                    "pre_norm": float(pre_norm),
                    "post_norm": float(min(pre_norm, update_clip_norm)) if pre_norm > 0 else 0.0,
                    "clipped": bool(did_clip),
                    "scale": float(scale),
                }
            )
        client_weights_list = clipped_list
        try:
            clipped_n = sum(1 for s in clip_stats if s["clipped"])
            norms = [s["pre_norm"] for s in clip_stats if s["pre_norm"] > 0]
            norms_sorted = sorted(clip_stats, key=lambda s: s["pre_norm"], reverse=True)
            topk = norms_sorted[: min(5, len(norms_sorted))]
            print(
                "[Aggregator] 🛡️ update_clip: "
                f"clip_norm={update_clip_norm:.6g}, clipped={clipped_n}/{len(clip_stats)}"
                + (
                    f", pre_norm(min/mean/max)={min(norms):.6g}/{(sum(norms)/len(norms)):.6g}/{max(norms):.6g}"
                    if norms
                    else ""
                ),
                flush=True,
            )
            print(
                "[Aggregator] 🛡️ update_clip top_pre_norm: "
                + ", ".join(
                    [
                        f"cid={s['client_id']},pre={s['pre_norm']:.6g},post={s['post_norm']:.6g},clipped={int(s['clipped'])},scale={s['scale']:.6g}"
                        for s in topk
                    ]
                ),
                flush=True,
            )
        except Exception:
            pass

    # cosine filter（先過濾再做聚合/robust）
    if defense_mode == "cosine_filter" and len(client_weights_list) >= 3:
        try:
            update_vecs = [_flatten_update(sd) for sd in client_weights_list]
            # 以 element-wise median update 當參考
            ref = torch.median(torch.stack(update_vecs, dim=0), dim=0).values
            sims = [_cosine(v, ref) for v in update_vecs]
            keep_idx: list[int] = []
            dyn_th = cosine_th
            dropped_top_n = 0

            # 方案 A：丟掉「最合群」的前 x%（對抗共謀：攻擊者更新彼此更相似，常落在最高相似度端）
            if cosine_drop_top_raw:
                try:
                    drop_pct = float(cosine_drop_top_raw)
                    if drop_pct <= 1.0:
                        drop_pct = drop_pct * 100.0
                    drop_pct = min(100.0, max(0.0, drop_pct))

                    eligible = [i for i, s in enumerate(sims) if s >= cosine_th]
                    if not eligible:
                        eligible = list(range(len(sims)))

                    eligible_sorted_desc = sorted(eligible, key=lambda i: sims[i], reverse=True)
                    n_eligible = len(eligible_sorted_desc)
                    dropped_top_n = int(np.ceil(n_eligible * (drop_pct / 100.0)))
                    dropped_top_n = min(dropped_top_n, max(0, n_eligible - min_after))

                    keep_idx = eligible_sorted_desc[dropped_top_n:]
                    if len(keep_idx) < min_after:
                        # 兜底：保留「最不合群」的前 min_after 個（避免只剩共謀簇）
                        keep_idx = sorted(eligible_sorted_desc, key=lambda i: sims[i])[:min_after]
                except Exception:
                    keep_idx = []

            # 方案 B：動態 percentile 門檻（原本行為：保留相似度 >= dyn_th 的樣本）
            if not keep_idx:
                if cosine_pct_raw:
                    try:
                        pct = float(cosine_pct_raw)
                        if pct <= 1.0:
                            pct = pct * 100.0
                        pct = min(100.0, max(0.0, pct))
                        dyn_th = float(np.percentile(np.array(sims, dtype=np.float32), pct))
                    except Exception:
                        pass
                keep_idx = [i for i, s in enumerate(sims) if s >= dyn_th]
                if len(keep_idx) < min_after:
                    # 不足則保留最高相似度的前 min_after 個
                    keep_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:min_after]

            keep_idx = sorted(set(keep_idx))
            try:
                cid_sims = [(client_ids_list[i], float(sims[i])) for i in range(len(sims))]
                sims_only = [s for _, s in cid_sims]
                sims_min = float(min(sims_only)) if sims_only else 0.0
                sims_max = float(max(sims_only)) if sims_only else 0.0
                sims_mean = float(sum(sims_only) / max(1, len(sims_only)))
                top_desc = sorted(cid_sims, key=lambda x: x[1], reverse=True)[: min(5, len(cid_sims))]
                bot_asc = sorted(cid_sims, key=lambda x: x[1])[: min(5, len(cid_sims))]
                kept_cids = [client_ids_list[i] for i in keep_idx]
                dropped_idx = [i for i in range(len(client_weights_list)) if i not in set(keep_idx)]
                dropped_cids = [client_ids_list[i] for i in dropped_idx]
                eligible = [i for i, s in enumerate(sims) if s >= cosine_th]
                dropped_top_ids = []
                if cosine_drop_top_raw and dropped_top_n > 0:
                    eligible_sorted_desc = sorted(eligible or list(range(len(sims))), key=lambda i: sims[i], reverse=True)
                    dropped_top_ids = [client_ids_list[i] for i in eligible_sorted_desc[:dropped_top_n]]
                print(
                    "[Aggregator] 🛡️ cosine_filter: "
                    f"th={cosine_th:.3f}, dyn_th={dyn_th:.3f}, pct={cosine_pct_raw or '-'}, "
                    f"drop_top_pct={cosine_drop_top_raw or '-'}, dropped_top={dropped_top_n}, "
                    f"keep={len(keep_idx)}/{len(client_weights_list)}, sims(min/mean/max)={sims_min:.3f}/{sims_mean:.3f}/{sims_max:.3f}",
                    flush=True,
                )
                print(
                    "[Aggregator] 🛡️ cosine_sims top: "
                    + ", ".join([f"cid={cid},sim={s:.3f}" for cid, s in top_desc]),
                    flush=True,
                )
                print(
                    "[Aggregator] 🛡️ cosine_sims bottom: "
                    + ", ".join([f"cid={cid},sim={s:.3f}" for cid, s in bot_asc]),
                    flush=True,
                )
                print(f"[Aggregator] 🛡️ cosine_keep_cids: {kept_cids}", flush=True)
                print(f"[Aggregator] 🛡️ cosine_dropped_cids: {dropped_cids}", flush=True)
                if dropped_top_ids:
                    print(f"[Aggregator] 🛡️ cosine_dropped_top_cids: {dropped_top_ids}", flush=True)
            except Exception:
                print(
                    f"[Aggregator] 🛡️ cosine_filter: th={cosine_th:.3f}, dyn_th={dyn_th:.3f}, "
                    f"pct={cosine_pct_raw or '-'}, drop_top_pct={cosine_drop_top_raw or '-'}, "
                    f"dropped_top={dropped_top_n}, keep={len(keep_idx)}/{len(client_weights_list)} "
                    f"sims={[round(s,3) for s in sims]}",
                    flush=True,
                )
            client_weights_list = [client_weights_list[i] for i in keep_idx]
            client_data_sizes_list = [client_data_sizes_list[i] for i in keep_idx]
            client_ids_list = [client_ids_list[i] for i in keep_idx]
        except Exception as e:  # noqa: BLE001
            print(f"[Aggregator] ⚠️ cosine_filter 失敗，改用原始列表：{e}")
    
    print(f"[Aggregator] 開始標準FedAvg聚合，{len(client_weights_list)}個客戶端")
    
    # ---- FedAvg：可選放大「指定 attacker client」的有效樣本權重 ----
    # - AGG_ATTACKER_FEDAVG_MULT（或別名 AGG_ATTACKER_SAMPLE_MULT）：對 attacker 乘上此係數（預設 1）
    # - AGG_ATTACKER_CLIENTS：逗號分隔 client id；若留空則嘗試讀 IMAGE_ATTACKER_CLIENTS（方便與 client 端同步）
    def _parse_id_set(raw: str) -> set:
        out: set = set()
        for part in (raw or "").split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.add(int(p))
            except Exception:
                continue
        return out

    _att_raw = (os.environ.get("AGG_ATTACKER_CLIENTS", "") or "").strip()
    if not _att_raw:
        _att_raw = (os.environ.get("IMAGE_ATTACKER_CLIENTS", "") or "").strip()
    _attacker_set = _parse_id_set(_att_raw)
    try:
        _att_mult = float(
            os.environ.get("AGG_ATTACKER_FEDAVG_MULT", os.environ.get("AGG_ATTACKER_SAMPLE_MULT", "1.0")) or "1.0"
        )
    except Exception:
        _att_mult = 1.0
    _att_mult = max(0.0, _att_mult)

    _base_sizes = [max(0.0, float(s)) for s in client_data_sizes_list]
    _eff_sizes = list(_base_sizes)
    if _attacker_set and abs(_att_mult - 1.0) > 1e-12:
        _boosted: list[int] = []
        for _i, _cid in enumerate(client_ids_list):
            try:
                _ic = int(_cid)
            except Exception:
                continue
            if _ic in _attacker_set:
                _eff_sizes[_i] = _base_sizes[_i] * _att_mult
                _boosted.append(_ic)
        if _boosted:
            print(
                f"[Aggregator] ⚖️ FedAvg attacker boost: mult={_att_mult:g}, "
                f"attacker_env={_att_raw!r}, boosted_cids={sorted(set(_boosted))}, "
                f"raw_data_sizes={_base_sizes}, effective_sizes={_eff_sizes}",
                flush=True,
            )

    # 計算總數據量（用 effective_sizes）
    total_data_size = max(1.0, sum(_eff_sizes))
    
    # 初步計算每個客戶端的權重
    client_weights = [max(0.0, s) / total_data_size for s in _eff_sizes]

    # 依據 staleness 與策略調整權重（若有提供 round 資訊，於上層已過濾/記錄）
    # 此函式維持純粹依 data_size 權重；staleness 衰減在上層計算後傳入（若需要）。

    # 施加 α 上限並重正規化（防 data_size 操弄）
    try:
        num_clients = max(1, len(client_weights))
        alpha_max = DATASIZE_ALPHA_MAX_MULTIPLIER * (1.0 / num_clients)
        client_weights = [min(w, alpha_max) for w in client_weights]
        s = sum(client_weights)
        if s > 0:
            client_weights = [w / s for w in client_weights]
    except Exception:
        pass
    
    print(f"[Aggregator] 客戶端數據大小(上傳原始): {client_data_sizes_list}")
    if _attacker_set and abs(_att_mult - 1.0) > 1e-12:
        print(f"[Aggregator] 客戶端數據大小(FedAvg有效): {_eff_sizes}", flush=True)
    print(f"[Aggregator] 客戶端權重: {[f'{w:.3f}' for w in client_weights]}")
    print(f"[Aggregator] 總數據量: {total_data_size}")
    
    # 初始化聚合權重
    aggregated_weights = {}
    first_weights = client_weights_list[0]
    
    # 對每個權重層進行聚合（FedBN：跳過 BatchNorm running 統計量）
    for layer_name in first_weights.keys():
        # FedBN: 保留各客戶端 BN running_mean/running_var，不做聚合
        if ('running_mean' in layer_name or 
            'running_var' in layer_name or 
            'num_batches_tracked' in layer_name):
            # 直接沿用當前全局（如無則取第一個客戶）
            if isinstance(global_weights, dict) and layer_name in global_weights and isinstance(global_weights[layer_name], torch.Tensor):
                aggregated_weights[layer_name] = global_weights[layer_name].clone().float()
            else:
                aggregated_weights[layer_name] = (
                    first_weights[layer_name].clone().float()
                    if isinstance(first_weights[layer_name], torch.Tensor)
                    else first_weights[layer_name]
                )
            continue
        first_layer = first_weights[layer_name]
        
        # robust aggregation：median / trimmed mean（座標-wise，不用 data_size 權重）
        if defense_mode in ("median", "trimmed_mean") and isinstance(first_layer, torch.Tensor):
            layers = []
            for client_weight_dict in client_weights_list:
                client_layer = client_weight_dict[layer_name]
                if not isinstance(client_layer, torch.Tensor):
                    client_layer = _to_tensor_f32(client_layer)
                if CLIP_NORM and CLIP_NORM > 0:
                    layer_norm = torch.norm(client_layer.reshape(-1).to(dtype=torch.float64), p=2)
                    if torch.isfinite(layer_norm) and layer_norm > CLIP_NORM:
                        scale = CLIP_NORM / (layer_norm + 1e-12)
                        client_layer = (client_layer.to(dtype=torch.float64) * scale).to(dtype=torch.float32)
                layers.append(client_layer.float())
            stack = torch.stack(layers, dim=0)  # (C, ...)
            if defense_mode == "median":
                agg = torch.median(stack, dim=0).values
            else:
                n_clients = stack.size(0)
                k = int(max(0, min(n_clients // 2 - 1, int(round(trim_ratio * n_clients)))))
                if k == 0:
                    agg = torch.mean(stack, dim=0)
                else:
                    sorted_vals, _ = torch.sort(stack, dim=0)
                    agg = torch.mean(sorted_vals[k : n_clients - k], dim=0)
            aggregated_weights[layer_name] = agg.to(dtype=torch.float32)
            continue

        # 預設：加權 FedAvg
        # 使用 float64 累加，減少數值誤差
        if isinstance(first_layer, torch.Tensor):
            if first_layer.dtype not in (torch.float32, torch.float64):
                first_layer = first_layer.float()
            aggregated_layer = torch.zeros_like(first_layer, dtype=torch.float64)
        else:
            aggregated_layer = 0.0
        for i, client_weight_dict in enumerate(client_weights_list):
            weight = client_weights[i]
            client_layer = client_weight_dict[layer_name]
            if isinstance(client_layer, torch.Tensor):
                if client_layer.dtype not in (torch.float32, torch.float64):
                    client_layer = client_layer.float()
                if CLIP_NORM and CLIP_NORM > 0:
                    layer_norm = torch.norm(client_layer.reshape(-1).to(dtype=torch.float64), p=2)
                    if torch.isfinite(layer_norm) and layer_norm > CLIP_NORM:
                        scale = CLIP_NORM / (layer_norm + 1e-12)
                        client_layer = (client_layer.to(dtype=torch.float64) * scale).to(dtype=client_layer.dtype)
                wt = float(weight)
                aggregated_layer += client_layer.to(dtype=torch.float64) * wt
            elif isinstance(client_layer, list):
                print(f"[Aggregator] ⚠️ 檢測到列表類型權重層 {layer_name}，轉換為張量")
                try:
                    client_layer = torch.tensor(client_layer, dtype=torch.float64)
                    aggregated_layer += client_layer * float(weight)
                except Exception as e:
                    print(f"[Aggregator] ❌ 列表轉張量失敗: {e}")
                    continue
            else:
                try:
                    aggregated_layer += float(weight) * float(client_layer)
                except Exception as e:
                    print(f"[Aggregator] ❌ 權重轉換失敗: {e}")
                    continue
        aggregated_weights[layer_name] = aggregated_layer.to(dtype=torch.float32) if isinstance(aggregated_layer, torch.Tensor) else aggregated_layer
    
    # 應用平滑機制
    aggregated_weights = apply_smoothing_to_weights(aggregated_weights)
    
    print(f"[Aggregator] 標準FedAvg聚合完成")
    print(f"[Aggregator] 聚合權重層數: {len(aggregated_weights)}")
    
    return aggregated_weights

def apply_smoothing_to_weights(new_weights):
    """應用平滑機制到權重，減少震盪"""
    global global_weights
    
    # 前兩輪關閉平滑，避免早期曲線過於平坦
    if (globals().get('round_count', 0) or 0) < 2:
        return new_weights
    
    if global_weights is None or len(global_weights) == 0:
        # 第一次聚合，直接使用新權重
        return new_weights
    
    # 獲取平滑因子
    smoothing_factor = config.FEDERATED_CONFIG.get('aggregation', {}).get('smoothing_factor', 0.5)
    
    print(f"[Aggregator] 🔧 應用平滑機制，平滑因子: {smoothing_factor}")
    
    smoothed_weights = {}
    
    def _to_cpu_f32(x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.device.type != 'cpu':
                x = x.cpu()
            return x.float()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, list):
            try:
                return torch.tensor(x, dtype=torch.float32)
            except Exception:
                return x
        return x
    
    for layer_name in new_weights.keys():
        if layer_name in global_weights:
            # 應用指數移動平均
            old_val = _to_cpu_f32(global_weights[layer_name])
            new_val = _to_cpu_f32(new_weights[layer_name])
            if isinstance(old_val, torch.Tensor) and isinstance(new_val, torch.Tensor):
                smoothed_weights[layer_name] = smoothing_factor * old_val + (1 - smoothing_factor) * new_val
            else:
                try:
                    smoothed_weights[layer_name] = smoothing_factor * float(old_val) + (1 - smoothing_factor) * float(new_val)
                except Exception:
                    smoothed_weights[layer_name] = new_weights[layer_name]
        else:
            # 新層，直接使用
            smoothed_weights[layer_name] = new_weights[layer_name]
    
    print(f"[Aggregator] ✅ 權重平滑完成")
    return smoothed_weights

def check_aggregation_conditions():
    """檢查聚合條件"""
    current_clients = len(client_weights_buffer)
    # 🔧 優化：提高聚合門檻，確保訓練同步
    try:
        expected_clients = len(round_clients)  # 期望所有選中的客戶端
    except Exception:
        expected_clients = 5  # 每個聚合器期望5個客戶端
    configured_min = int(config.AGGREGATION_CONFIG.get("min_clients_for_aggregation", 2))
    # 🔧 緊急修復：降低最小聚合數，避免卡住
    min_clients = max(2, configured_min)  # 至少2個客戶端
    min_clients = min(min_clients, expected_clients)
    max_wait_time = config.AGGREGATION_CONFIG.get("max_wait_time", 60)  # 增加等待時間
    force_aggregation = config.AGGREGATION_CONFIG.get("force_aggregation_after_timeout", True)
    
    print(f"[Aggregator] 🔍 聚合條件檢查:")
    print(f"[Aggregator]   - 當前客戶端: {current_clients}")
    print(f"[Aggregator]   - 期望客戶端: {expected_clients}")
    print(f"[Aggregator]   - 最小要求: {min_clients}")
    print(f"[Aggregator]   - 客戶端列表: {list(client_weights_buffer.keys())}")
    
    # 🔧 緊急修復：降低參與率要求
    participation_ratio = current_clients / expected_clients if expected_clients > 0 else 0
    min_participation_ratio = 0.0  # 臨時解卡：不限制參與比例
    
    # 達到最小聚合數且參與率達標
    if current_clients >= min_clients and participation_ratio >= min_participation_ratio:
        print(f"[Aggregator] ✅ 達到聚合條件: {current_clients} 個客戶端 (參與率: {participation_ratio:.1%})")
        return True
    
    # 🔧 優化：延長快速聚合時間
    if round_start_time is not None:
        elapsed_time = time.time() - round_start_time
        quick_time = max_wait_time * 0.6  # 提早到 60% 等待時間可提前聚合
        
        if elapsed_time > quick_time and current_clients >= min_clients:
            print(f"[Aggregator] ⚠️ 快速聚合（等待時間60%）: {current_clients} 個客戶端 (等待 {elapsed_time:.1f}s)")
            return True
        
    # 超時強制聚合：超過 max_wait_time，只要 buffer_size >= min_clients_for_aggregation 就強制聚合
    if round_start_time is not None:
        elapsed_time = time.time() - round_start_time
        if elapsed_time > max_wait_time:
            threshold = min_clients  # 直接使用前面算好的 min_clients_for_aggregation
            if current_clients >= threshold:
                print(f"[Aggregator] ⚠️ 超時強制聚合（使用 min_clients_for_aggregation）: {current_clients} 個客戶端 (等待 {elapsed_time:.1f}s, 門檻={threshold})")
                return True
            else:
                print(f"[Aggregator] ❌ 超時但客戶端數量不足，維持等待狀態: {current_clients} < {threshold}, elapsed={elapsed_time:.1f}s")
    
    print(f"[Aggregator] ⏳ 等待更多客戶端: {current_clients}/{min_clients} (參與率: {participation_ratio:.1%})")
    return False

def get_assigned_clients():
    """獲取分配給此聚合器的客戶端列表"""
    assigned_clients = []
    total_clients = int(getattr(config, 'NUM_CLIENTS', 10))
    for client_id in range(total_clients):
        if client_id % config.NUM_AGGREGATORS == aggregator_id:
            assigned_clients.append(client_id)
    
    print(f"[Aggregator {aggregator_id}] 📊 分配客戶端: {assigned_clients}")
    return assigned_clients

def convert_numpy_values(obj):
    """轉換numpy值為原生Python類型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_values(item) for item in obj]
    else:
        return obj

@app.get("/health")
async def health_check():
    """健康檢查"""
    try:
        health_status = {
            "status": "healthy",
            "aggregator_id": aggregator_id,
            "round_count": round_count,
            "buffer_size": len(client_weights_buffer),
            "global_weights_available": global_weights is not None,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": model_version,
            "training_phase": is_training_phase
        }
        
        return convert_numpy_values(health_status)
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@app.post("/start_federated_round")
async def start_federated_round(round_id: int = Form(...), force: bool = Form(False)):
    """開始新一輪聯邦學習"""
    global round_count, round_clients, client_weights_buffer, round_start_time, client_timeout_status, is_training_phase
    
    print(f"[Aggregator] 🔍 收到開始第{round_id}輪聯邦學習請求 (強制模式: {force})")
    print(f"[Aggregator] 🔍 當前狀態: 輪次={round_count}, 選中客戶端={round_clients}")
    
    # 記錄事件
    log_event("start_federated_round", f"收到輪次 {round_id} 啟動請求 (強制: {force})")
    
    # 檢查輪次狀態，避免未聚合就提前推進
    min_clients_required = config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3)
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')

    # 客戶端發來舊輪次請求
    if round_id < round_count and not force:
        print(f"[Aggregator] ⚠️ 收到過期輪次啟動請求: {round_id} < {round_count}")
        log_event("round_start_stale_request", f"請求輪次 {round_id} 小於當前 {round_count}")
        return {
            "round_id": round_count,
            "selected_clients": round_clients,
            "min_clients_required": min_clients_required,
            "aggregation_timeout": AGGREGATION_TIMEOUT,
            "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
            "selection_strategy": "simplified",
            "training_flow": training_flow,
            "status": "stale_round_request"
        }

    # 🔧 修復：改進輪次已啟動檢查邏輯
    if round_id == round_count and round_start_time is not None and not force:
        # 檢查是否真的卡住了（超過5分鐘沒有進展）
        current_time = time.time()
        if current_time - round_start_time > 600:  # 10分鐘
            print(f"[Aggregator] 🔧 檢測到輪次卡住超過10分鐘，允許重啟第{round_id}輪")
            log_event("round_stuck_detected", f"輪次 {round_id} 卡住超過10分鐘，強制重啟")
            # 重置輪次狀態，允許重新啟動
            round_start_time = None
            is_training_phase = False
            # 清空緩衝區，重新開始
            client_weights_buffer = {}
            client_timeout_status = {}
            app.state.round_upload_window_closed = False
        else:
            print(f"[Aggregator] ⚠️ 第{round_id}輪已經啟動，跳過重複啟動")
            log_event("round_already_started", f"輪次 {round_id} 已啟動")
            return {
                "round_id": round_id,
                "selected_clients": round_clients,
                "min_clients_required": min_clients_required,
                "aggregation_timeout": AGGREGATION_TIMEOUT,
                "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                "selection_strategy": "simplified",
                "training_flow": training_flow,
                "status": "already_started"
            }

    # 🔧 修復：改進輪次同步邏輯
    if force:
        # 強制模式：直接設置輪次
        print(f"[Aggregator] 🔧 強制模式：設置輪次 {round_id}")
        round_count = round_id
        round_start_time = time.time()
        is_training_phase = True
        # 修復：緩衝區應為字典（client_id -> weights），不能重設為列表
        client_weights_buffer = {}
        client_timeout_status = {}
        app.state.round_upload_window_closed = False
        
        # 重新選擇客戶端
        round_clients = select_clients_for_round(round_id)
        print(f"[Aggregator] 🔧 強制模式：選中客戶端 {round_clients}")
        
        log_event("round_force_started", f"強制啟動輪次 {round_id}, 選中客戶端 {round_clients}")
        return {
            "round_id": round_id,
            "selected_clients": round_clients,
            "min_clients_required": min_clients_required,
            "aggregation_timeout": AGGREGATION_TIMEOUT,
            "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
            "selection_strategy": "simplified",
            "training_flow": training_flow,
            "status": "force_started"
        }
    
    # 🔧 修復：改進輪次推進邏輯，確保連續性
    if round_id > round_count:
        if round_id > round_count + 1:  # 🔧 修復：只允許跳躍1輪
            print(f"[Aggregator] ⚠️ 輪次跳躍過大: 請求={round_id}, 當前={round_count}")
            log_event("round_advance_blocked", f"large_skip: current={round_count}, request={round_id}")
            return {
                "round_id": round_count,
                "selected_clients": round_clients,
                "min_clients_required": min_clients_required,
                "aggregation_timeout": AGGREGATION_TIMEOUT,
                "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                "selection_strategy": "simplified",
                "training_flow": training_flow,
                "status": "large_skip_rejected"
            }
        
        # ✅ 關鍵修復：若上一輪仍有未聚合的更新，禁止推進輪次
        # 避免「尚未聚合就被 start_federated_round 清空 buffer」導致 Cloud 永遠收不到 delta。
        if not force and len(client_weights_buffer) > 0:
            # 🔧 修復：若已等待一段時間仍有 pending buffer，先「沖刷聚合」再推進，避免整體輪次分裂卡住
            min_flush_wait = float(getattr(config, "AGGREGATION_CONFIG", {}).get("min_training_duration", 30))
            _env_mtd = os.environ.get("AGG_MIN_TRAINING_DURATION_S", "").strip()
            if _env_mtd:
                try:
                    min_flush_wait = float(_env_mtd)
                except ValueError:
                    pass
            elapsed = 0.0
            try:
                if round_start_time is not None:
                    elapsed = time.time() - round_start_time
                else:
                    # round_stuck 等路徑會把 round_start_time 清為 None；若 client 已再次寫入 buffer，
                    # 仍用 elapsed=0 則永遠 < min_flush_wait，導致無限 pending（log 常見 elapsed=0.0s）。
                    elapsed = float(min_flush_wait)
                    print(
                        f"[Aggregator] 🔧 pending buffer 但 round_start_time=None，"
                        f"以 elapsed>={min_flush_wait:.1f}s 視為可 flush（避免卡死）",
                        flush=True,
                    )
            except Exception:
                elapsed = 0.0
            can_flush = elapsed >= min_flush_wait
            if aggregation_lock.locked():
                can_flush = False
            # Image FL：禁止「半包」flush（會先佔用雲端該輪唯一 merge 名額，導致後續滿員 delta 被 skip / 缺輪）
            _image_fl_no_partial = (os.environ.get("IMAGE_FL", "") or "").strip() == "1" and str(
                os.environ.get("IMAGE_FL_ALLOW_PARTIAL_FLUSH", "0") or "0"
            ).strip().lower() not in ("1", "true", "yes", "on")
            if _image_fl_no_partial and round_clients and len(client_weights_buffer) < len(round_clients):
                can_flush = False
                print(
                    f"[Aggregator] ⏳ Image FL：拒絕 flush 半包（buffer={len(client_weights_buffer)}/{len(round_clients)}），"
                    f"等待本輪所選客戶端到齊再上傳後才允許協調器推進到 round={round_id}",
                    flush=True,
                )
            if can_flush:
                print(
                    f"[Aggregator] 🔧 pending buffer={len(client_weights_buffer)} 且 elapsed={elapsed:.1f}s >= {min_flush_wait:.1f}s，"
                    f"先嘗試聚合並上傳 delta，再推進到 round={round_id}"
                )
                try:
                    async with aggregation_lock:
                        # 併發保護：快照 buffer（避免聚合期間被修改）
                        async with buffer_lock:
                            client_ids_in_buffer = list(client_weights_buffer.keys())
                            client_weights_list = [client_weights_buffer[cid] for cid in client_ids_in_buffer]
                            client_data_sizes_list = [client_data_sizes.get(cid, 1) for cid in client_ids_in_buffer]

                        if client_weights_list:
                            log_event("aggregation_started", f"開始聚合 {len(client_weights_list)} 個客戶端權重 (flush_before_advance)")
                            aggregated_weights = perform_standard_fedavg(
                                client_weights_list, client_data_sizes_list, client_ids_list=client_ids_in_buffer
                            )
                        else:
                            aggregated_weights = None

                        if aggregated_weights:
                            # 更新本地全局權重與版本
                            global global_weights, model_version
                            global_weights = aggregated_weights
                            model_version += 1

                            # 上傳 delta 到 cloud（使用既有流程：取 base -> 算 delta -> upload）
                            upload_ok = True
                            try:
                                cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
                                base_version = 0
                                base_weights = None
                                if cloud_url:
                                    timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                                    async with aiohttp.ClientSession(timeout=timeout) as session:
                                        async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                            if resp.status == 200:
                                                payload = pickle.loads(await resp.read())
                                                base_version = int(payload.get('version', 0))
                                                base_weights = payload.get('weights', {})

                                def _to_tensor(x):
                                    if isinstance(x, torch.Tensor):
                                        return x.detach().cpu().float()
                                    if isinstance(x, np.ndarray):
                                        return torch.from_numpy(x).float()
                                    if isinstance(x, list):
                                        try:
                                            return torch.tensor(x, dtype=torch.float32)
                                        except Exception:
                                            return None
                                    return None

                                delta = {}
                                if isinstance(base_weights, dict) and base_weights:
                                    for k, v in global_weights.items():
                                        g = _to_tensor(v)
                                        b = _to_tensor(base_weights.get(k))
                                        if g is not None and b is not None and g.shape == b.shape:
                                            delta[k] = (g - b)
                                        else:
                                            if g is not None:
                                                delta[k] = g
                                else:
                                    for k, v in global_weights.items():
                                        t = _to_tensor(v)
                                        if t is not None:
                                            delta[k] = t

                                upload_ok = await upload_delta_to_cloud(
                                    base_version, delta, round_count, buffered_clients=len(client_weights_list)
                                )
                            except Exception as e:
                                print(f"[Aggregator {aggregator_id}] ⚠️ flush 聚合後上傳 delta 失敗: {e}")
                                log_event("cloud_sync_failed", f"flush_before_advance: {e}")
                                upload_ok = False
                            _imf = (os.environ.get("IMAGE_FL", "") or "").strip() == "1"
                            _cloud_cfg_url = (CLOUD_SERVER_CONFIG.get("url") or "").strip()
                            if _imf and _cloud_cfg_url and not upload_ok:
                                print(
                                    f"[Aggregator {aggregator_id}] ⚠️ IMAGE_FL：flush 上傳 delta 未成功，保留 buffer、不推進",
                                    flush=True,
                                )
                                log_event("round_advance_blocked", "flush_before_advance_upload_failed")
                                return {
                                    "round_id": round_count,
                                    "selected_clients": round_clients,
                                    "min_clients_required": min_clients_required,
                                    "aggregation_timeout": AGGREGATION_TIMEOUT,
                                    "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                                    "selection_strategy": "simplified",
                                    "training_flow": training_flow,
                                    "status": "pending_aggregation",
                                }

                            # 清空本輪 buffer，允許推進
                            async with buffer_lock:
                                client_weights_buffer.clear()
                                client_data_sizes.clear()
                            app.state.round_upload_window_closed = True

                            log_event("aggregation_completed", f"flush_before_advance 參與客戶端: {client_ids_in_buffer}")
                        else:
                            print(f"[Aggregator] ⚠️ flush 聚合無輸出，保留 buffer，避免丟失更新")
                            log_event("round_advance_blocked", f"flush_failed_no_output buffer={len(client_weights_buffer)} elapsed={elapsed:.1f}s")
                            return {
                                "round_id": round_count,
                                "selected_clients": round_clients,
                                "min_clients_required": min_clients_required,
                                "aggregation_timeout": AGGREGATION_TIMEOUT,
                                "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                                "selection_strategy": "simplified",
                                "training_flow": training_flow,
                                "status": "pending_aggregation"
                            }
                except Exception as e:
                    print(f"[Aggregator] ⚠️ flush_before_advance 失敗，維持 pending: {e}")
                    log_event("round_advance_blocked", f"flush_exception buffer={len(client_weights_buffer)} elapsed={elapsed:.1f}s")
                    return {
                        "round_id": round_count,
                        "selected_clients": round_clients,
                        "min_clients_required": min_clients_required,
                        "aggregation_timeout": AGGREGATION_TIMEOUT,
                        "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                        "selection_strategy": "simplified",
                        "training_flow": training_flow,
                        "status": "pending_aggregation"
                    }
            else:
                print(
                    f"[Aggregator] ⏳ 尚有未聚合更新（buffer_size={len(client_weights_buffer)}），"
                    f"拒絕推進到 round={round_id}，請先完成本輪聚合（elapsed={elapsed:.1f}s < {min_flush_wait:.1f}s）"
                )
                log_event("round_advance_blocked", f"pending_aggregation buffer={len(client_weights_buffer)} elapsed={elapsed:.1f}s")
                return {
                    "round_id": round_count,
                    "selected_clients": round_clients,
                    "min_clients_required": min_clients_required,
                    "aggregation_timeout": AGGREGATION_TIMEOUT,
                    "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                    "selection_strategy": "simplified",
                    "training_flow": training_flow,
                    "status": "pending_aggregation",
                }

        print(f"[Aggregator] 🔧 輪次更新: {round_count} -> {round_id}")
        round_count = round_id
    
    # 檢查訓練流程配置
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
    
    if training_flow == 'train_then_select':
        # 新流程：先讓所有客戶端訓練，然後基於結果選擇
        print(f"[Aggregator] 🚀 使用新流程：先訓練後選擇")
        assigned_clients = []
        for client_id in range(10):
            if client_id % config.NUM_AGGREGATORS == aggregator_id:
                assigned_clients.append(client_id)
        round_clients = assigned_clients
        is_training_phase = True
        print(f"[Aggregator] ✅ 新流程：分配給此聚合器的客戶端可以參與訓練: {round_clients}")
    else:
        # 原有流程：先選擇客戶端再訓練
        print(f"[Aggregator] 🚀 使用原有流程：先選擇後訓練")
        round_clients = select_clients_for_round(round_id)
        is_training_phase = False
    
    # 清空緩衝區，開始新輪次
    client_weights_buffer.clear()
    client_data_sizes.clear()
    
    # 重置輪次計時器和超時狀態
    round_start_time = time.time()
    client_timeout_status.clear()
    # 受控混合：重置本輪聚合標記
    if not hasattr(app.state, 'aggregation_done'):
        app.state.aggregation_done = False
        app.state.aggregation_done_round = -1
    else:
        app.state.aggregation_done = False
        app.state.aggregation_done_round = round_count
    app.state.round_upload_window_closed = False
    app.state.inflight_round = -1
    app.state.inflight_started_at = 0.0
    app.state.round_claimed_for_aggregation = -1
    app.state.uploaded_rounds = set()
    app.state.last_uploaded_round = -1
    
    print(f"[Aggregator] ✅ 開始第{round_id}輪聯邦學習")
    print(f"[Aggregator] ✅ 客戶端選擇策略: {training_flow}")
    print(f"[Aggregator] ✅ 聚合超時設置: {AGGREGATION_TIMEOUT}s")
    print(f"[Aggregator] 🚀 選中客戶端列表: {round_clients} (共 {len(round_clients)} 台)")
    
    log_training_event_aggregator(
        'round_started',
        {
            'round_id': round_id,
            'selected_len': len(round_clients),
            'buffer_size': len(client_weights_buffer),
            'model_version': model_version,
            'detail': str(round_clients)
        }
    )
    
    return {
        "round_id": round_id,
        "selected_clients": round_clients,
        "min_clients_required": min_clients_required,
        "aggregation_timeout": AGGREGATION_TIMEOUT,
        "partial_aggregation_enabled": PARTIAL_AGGREGATION_ENABLED,
        "selection_strategy": "simplified",
        "training_flow": training_flow
    }

@app.post("/upload_federated_weights")
async def upload_federated_weights(
    client_id: int = Form(...),
    data_size: int = Form(...),
    round_id: int = Form(...),
    commit_id: str = Form(""),
    # 客戶端預測概率分布（用於跨層教師機制）
    client_predictions: Optional[UploadFile] = File(None),
    weights: Optional[UploadFile] = File(None)
):
    """接收客戶端上傳的聯邦學習權重"""
    global global_weights  # 🚀 修復：在函數開始處聲明全局變量
    global client_weights_buffer, client_data_sizes, client_predictions_buffer, round_count, round_clients, global_performance, is_training_phase, global_weights, model_version
    
    print(f"[Aggregator] 🔍 收到客戶端 {client_id} 權重上傳請求")
    print(f"[Aggregator] 🔍 當前狀態: 輪次={round_count}, 客戶端輪次={round_id}")
    print(f"[Aggregator] 🔍 當前緩衝區: {len(client_weights_buffer)} 個客戶端")
    print(f"[Aggregator] 🔍 選中客戶端: {round_clients}")
    
    # Stale 政策處理
    if STALE_POLICY == 'strict':
        if round_id != round_count:
            print(f"[Aggregator] ❌ 嚴格同步：拒收非當前輪次的權重 (client_round={round_id}, agg_round={round_count})")
            return {
                "status": "rejected",
                "message": "嚴格同步策略：僅接受當前輪次",
                "expected_round": round_count
            }
    elif STALE_POLICY == 'decay':
        if round_id < round_count - MAX_STALENESS:
            print(f"[Aggregator] ❌ 衰減策略：超過允許的落後輪次 MAX_STALENESS={MAX_STALENESS}")
            return {
                "status": "rejected",
                "message": "超過允許落後輪次",
                "expected_round": round_count
            }
        elif round_id < round_count:
            print(f"[Aggregator] ℹ️ 衰減策略：接受落後 {round_count - round_id} 輪，將在權重中施加衰減")
            # 實際衰減在後續加權階段應用；此處僅標記 staleness
            staleness = round_count - round_id
        else:
            staleness = 0
    else:
        # 🔧 修復：更寬容的輪次處理，允許落後50輪以內的客戶端參與
        if round_id < round_count - 50:
            print(f"[Aggregator] ⚠️ 客戶端輪次 {round_id} 嚴重落後於聚合器輪次 {round_count}，拒絕處理")
            log_event("weight_upload_rejected", f"客戶端輪次嚴重落後: {round_id} < {round_count - 10}")
            return {
                "status": "rejected",
                "message": "輪次嚴重不匹配",
                "expected_round": round_count
            }
        elif round_id < round_count:
            print(f"[Aggregator] 🔧 客戶端輪次 {round_id} 落後{round_count - round_id}輪，但允許參與當前輪次 {round_count}")
            round_id = round_count
    
    # 輪次管理
    if round_id != round_count:
        print(f"[Aggregator] 輪次不匹配: 客戶端輪次={round_id}, 聚合器輪次={round_count}")
        
        # 如果聚合器輪次為0（初始狀態），則調整到客戶端輪次
        if round_count == 0:
            print(f"[Aggregator] 聚合器初始狀態，調整到客戶端輪次 {round_id}")
            round_count = round_id
            training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
            if training_flow == 'train_then_select':
                round_clients = get_assigned_clients()
                is_training_phase = True
            else:
                round_clients = select_clients_for_round(round_id)
                is_training_phase = False
            print(f"[Aggregator] 開始第{round_id}輪")
        else:
            # 如果客戶端輪次超前，檢查是否需要清空緩衝區
            if round_id > round_count:
                print(f"[Aggregator] 客戶端輪次超前，調整到客戶端輪次 {round_id}")
                
                if len(client_weights_buffer) == 0:
                    print(f"[Aggregator] 緩衝區為空，安全調整輪次")
                    round_count = round_id
                    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
                    if training_flow == 'train_then_select':
                        round_clients = get_assigned_clients()
                        is_training_phase = True
                    else:
                        round_clients = select_clients_for_round(round_id)
                        is_training_phase = False
                    print(f"[Aggregator] 開始第{round_id}輪")
                else:
                    print(f"[Aggregator] ⚠️ 緩衝區有 {len(client_weights_buffer)} 個客戶端權重，保持當前輪次等待聚合")
                    round_id = round_count
            elif round_id < round_count:
                print(f"[Aggregator] 客戶端輪次落後，保持當前輪次 {round_count}")
            else:
                print(f"[Aggregator] 輪次匹配，保持當前輪次 {round_count}")
    
    # 本輪已「提交」聚合並清空 buffer 後，拒收同輪遲到上傳，避免 phantom buffer 卡住 coordinator 推進
    if bool(getattr(app.state, "aggregation_done", False)) and int(getattr(app.state, "aggregation_done_round", -1)) == int(round_count):
        print(
            f"[Aggregator] 🚫 本輪 round={round_count} 已完成聚合，"
            f"拒收同輪遲到/重複上傳 (client={client_id})",
            flush=True,
        )
        log_event("weight_upload_rejected", f"round_already_aggregated_late_upload client={client_id} round={round_id}")
        return {
            "status": "rejected",
            "reason": "round_already_aggregated",
            "message": "本輪已完成聚合，同輪晚到更新已忽略",
            "expected_round": round_count,
            "current_round": round_count,
        }

    if getattr(app.state, "round_upload_window_closed", False) and int(round_id) == int(round_count):
        print(
            f"[Aggregator] 🚫 本輪 round={round_count} 收包窗口已關閉（聚合已完成），"
            f"拒收遲到權重 (client={client_id})",
            flush=True,
        )
        log_event("weight_upload_rejected", f"round_closed_late_upload client={client_id} round={round_id}")
        return {
            "status": "rejected",
            "reason": "round_closed_late_upload",
            "message": "本輪已完成聚合，遲到權重已忽略",
            "expected_round": round_count,
            "current_round": round_count,
        }
    
    # 檢查客戶端是否在選中列表中
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
    
    if training_flow == 'train_then_select':
        if client_id not in round_clients:
            print(f"[Aggregator] ⚠️ 客戶端 {client_id} 不在分配列表中，但仍在訓練階段接受")
    else:
        if client_id not in round_clients:
            print(f"[Aggregator] ⚠️ 客戶端 {client_id} 不在選中列表中")
            app.state.rejected_not_selected_count = int(getattr(app.state, "rejected_not_selected_count", 0)) + 1
            print(
                f"[Aggregator] 🚫 拒收非選中客戶端 {client_id} 權重 "
                f"(rejected_not_selected_count={app.state.rejected_not_selected_count})"
            )
            return {
                "status": "rejected",
                "message": f"client {client_id} is not selected for round {round_count}",
                "reason": "not_selected",
                "current_round": round_count
            }
    
    try:
        # 解析客戶端預測概率分布（用於跨層教師機制）
        client_prediction = None
        if client_predictions is not None:
            try:
                predictions_bytes = await client_predictions.read()
                client_prediction = pickle.loads(predictions_bytes)
                print(f"[Aggregator] 🚀 收到客戶端 {client_id} 預測概率分布: {client_prediction}")
            except Exception as pred_error:
                print(f"[Aggregator] ⚠️ 解析客戶端預測概率分布失敗: {pred_error}")
                client_prediction = None
        
        # 解析權重數據
        if weights is None:
            return {
                "status": "rejected",
                "message": "weights field is required for standard federated learning",
                "current_round": round_count
            }
        
        weights_data = pickle.loads(await weights.read())
        # 統一權重到 CPU/float32，提升後續聚合穩定性
        def _to_cpu_f32_layer(x):
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.device.type != 'cpu':
                    x = x.cpu()
                return x.float()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            elif isinstance(x, list):
                try:
                    return torch.tensor(x, dtype=torch.float32)
                except Exception:
                    return x
            return x
        if isinstance(weights_data, dict):
            normalized_weights = {}
            for k, v in weights_data.items():
                # 🔧 修復：過濾掉 GKD teacher 模型權重
                if k.startswith('_teacher_model'):
                    continue
                normalized_weights[k] = _to_cpu_f32_layer(v)
            weights_data = normalized_weights
        
        # 併發與冪等保護
        async with buffer_lock:
            # 冪等鍵： (round, client_id, commit_id)
            if commit_id:
                if not hasattr(app.state, 'idempotent_commits'):
                    app.state.idempotent_commits = set()
                idem_key = (int(round_id), int(client_id), str(commit_id))
                if idem_key in app.state.idempotent_commits:
                    print(f"[Aggregator] ℹ️ 冪等：忽略重複提交 (round={round_id}, client={client_id}, commit={commit_id})")
                    return {
                        "status": "received",
                        "accepted": True,
                        "server_round": int(round_count),
                        "reason": "duplicate_commit_ignored",
                        "message": "duplicate_commit_ignored",
                        "buffer_size": len(client_weights_buffer),
                        "current_round": round_count
                    }
                app.state.idempotent_commits.add(idem_key)

            # 基礎驗證：鍵集合/形狀/NaN/Inf
            if global_weights is None:
                # 🚀 改進：若尚未有全局權重模板，先嘗試初始化（根據配置），否則使用客戶端權重
                try:
                    initialize_global_weights()
                    if global_weights is not None:
                        print(f"[Aggregator] 🔧 全局權重為空，已根據配置初始化（模型類型: {config.MODEL_CONFIG.get('type', 'dnn')}）")
                    else:
                        # 如果初始化失敗，使用客戶端權重作為模板
                        print(f"[Aggregator] 🔧 全局權重為空，使用客戶端 {client_id} 的權重作為初始模板")
                        filtered_weights = {
                            k: (v.detach().cpu().float().clone() if isinstance(v, torch.Tensor) else v)
                            for k, v in (weights_data.items() if isinstance(weights_data, dict) else [])
                            if not k.startswith('_teacher_model')
                        }
                        global_weights = filtered_weights
                except Exception as e:
                    # 如果初始化失敗，使用客戶端權重作為模板
                    print(f"[Aggregator] ⚠️ 全局權重初始化失敗: {e}，使用客戶端 {client_id} 的權重作為初始模板")
                    filtered_weights = {
                        k: (v.detach().cpu().float().clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in (weights_data.items() if isinstance(weights_data, dict) else [])
                        if not k.startswith('_teacher_model')
                    }
                    global_weights = filtered_weights
            if isinstance(weights_data, dict) and isinstance(global_weights, dict):
                gw_keys = set(global_weights.keys())
                wk_keys = set(weights_data.keys())
                if gw_keys != wk_keys:
                    missing = list(gw_keys - wk_keys)
                    extra = list(wk_keys - gw_keys)
                    
                    # 🔧 修復：允許 BatchNorm 參數差異和 GKD teacher 模型權重
                    # 過濾掉 BatchNorm 相關的差異和 teacher 模型權重
                    filtered_missing = [k for k in missing if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                    filtered_extra = [k for k in extra if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                    
                    # 🚀 改進：檢測架構變更（例如從 DNN 切換到 Transformer）
                    # 檢測全局權重是舊架構（DNN），客戶端權重是新架構（Transformer）
                    global_is_old_arch = (
                        any('layers.' in k or 'residual_layers.' in k or 'batch_norms.' in k for k in filtered_missing) and
                        any('transformer_blocks.' in k or 'input_projection.' in k or 'classifier.' in k for k in filtered_extra)
                    )
                    # 檢測全局權重是新架構（Transformer），客戶端權重是舊架構（DNN）
                    client_is_old_arch = (
                        any('transformer_blocks.' in k or 'input_projection.' in k or 'classifier.' in k for k in filtered_missing) and
                        any('layers.' in k or 'residual_layers.' in k or 'batch_norms.' in k for k in filtered_extra)
                    )
                    
                    if global_is_old_arch:
                        # 全局權重是舊架構，客戶端權重是新架構，更新全局權重
                        model_type = config.MODEL_CONFIG.get('type', 'dnn')
                        print(f"[Aggregator] 🔄 檢測到架構變更（全局權重是舊架構，客戶端權重是新架構），更新全局權重架構")
                        print(f"  - 缺失的鍵（舊架構）: {len(filtered_missing)} 個")
                        print(f"  - 新增的鍵（新架構）: {len(filtered_extra)} 個")
                        # 🚀 修復：使用 reset_global_weights() 強制重置並重新初始化
                        reset_global_weights()
                        # 重新獲取全局權重鍵（確保使用新架構）
                        gw_keys = set(global_weights.keys())
                        # 重新檢查客戶端權重與新全局權重的差異
                        missing = list(gw_keys - wk_keys)
                        extra = list(wk_keys - gw_keys)
                        filtered_missing = [k for k in missing if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                        filtered_extra = [k for k in extra if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                        print(f"[Aggregator] ✅ 全局權重已重新初始化為 {model_type} 架構，權重層數: {len(global_weights)}")
                        print(f"[Aggregator] 🔍 重新檢查後 - missing: {len(filtered_missing)}, extra: {len(filtered_extra)}")
                    elif client_is_old_arch:
                        # 全局權重是新架構，客戶端權重是舊架構，拒絕客戶端權重
                        print(f"[Aggregator] ⚠️ 檢測到客戶端 {client_id} 使用舊架構（DNN），全局權重已更新為新架構（Transformer）")
                        print(f"  - 全局權重缺少的鍵（新架構）: {len(filtered_missing)} 個")
                        print(f"  - 客戶端權重多餘的鍵（舊架構）: {len(filtered_extra)} 個")
                        print(f"[Aggregator] ❌ 拒絕客戶端 {client_id} 的權重，請客戶端更新為 Transformer 架構後重試")
                        raise ValueError(f"客戶端 {client_id} 使用舊架構（DNN），全局權重已更新為新架構（Transformer）。請客戶端更新後重試。")
                    
                    if filtered_missing or filtered_extra:
                        missing_display = filtered_missing[:5]
                        extra_display = filtered_extra[:5]
                        raise ValueError(f"權重鍵不一致 missing={missing_display} extra={extra_display}")
                    else:
                        print(f"[Aggregator] ⚠️ 檢測到 BatchNorm 參數或 teacher 模型權重差異，自動忽略: missing={len(missing)} extra={len(extra)}")
                        # 只保留匹配的鍵，並過濾掉 teacher 模型權重
                        common_keys = gw_keys & wk_keys
                        weights_data = {k: weights_data[k] for k in common_keys if k in weights_data and not k.startswith('_teacher_model')}
                        global_weights = {k: global_weights[k] for k in common_keys if k in global_weights and not k.startswith('_teacher_model')}
                for k, v in weights_data.items():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            raise ValueError(f"權重含 NaN/Inf: {k}")
                        if k in global_weights and isinstance(global_weights[k], torch.Tensor):
                            if tuple(v.shape) != tuple(global_weights[k].shape):
                                # 🔧 形狀不符時，不再整體報錯；改為：
                                #    1) 記錄一次警告（最多列出部分層）
                                #    2) 直接用客戶端的權重覆蓋全局該層，將全局模板遷移到新架構
                                print(
                                    f"[Aggregator] ⚠️ 權重形狀不符，將以客戶端權重覆蓋全局模板: "
                                    f"{k} {tuple(global_weights[k].shape)} -> {tuple(v.shape)}"
                                )
                                global_weights[k] = v.detach().cpu().float().clone()

            # 檢查是否已存在該客戶端的權重（覆蓋為最新）
        if client_id in client_weights_buffer:
            print(f"[Aggregator] ⚠️ 客戶端 {client_id} 權重已存在，覆蓋舊權重")
        
        # 寫入緩衝
        client_weights_buffer[client_id] = weights_data
        client_data_sizes[client_id] = max(0, int(data_size))
        if client_prediction is not None:
            client_predictions_buffer[client_id] = client_prediction
        # 記錄 staleness（供後續加權使用）
        if STALE_POLICY == 'decay':
            if not hasattr(app.state, 'client_staleness'):
                app.state.client_staleness = {}
            app.state.client_staleness[client_id] = int(locals().get('staleness', 0))
        
        print(f"[Aggregator] ✅ 客戶端 {client_id} 權重已接收")
        print(f"[Aggregator] 📊 緩衝區狀態: {len(client_weights_buffer)}/{len(round_clients)} 客戶端")

        # 記錄接收事件
        log_training_event_aggregator(
            'weights_received',
            {
                'round_id': round_count,
                'selected_len': len(round_clients),
                'buffer_size': len(client_weights_buffer),
                'received_client': client_id,
                'model_version': model_version
            }
        )
        
        # 🔧 修復：使用「部分聚合」門檻（與 config 同步），避免 buffer 一直達不到而永不聚合
        do_aggregate = False
        try:
            # 部分聚合門檻：預設使用 config 的 MIN_PARTIAL_RATIO（例如 0.4）
            # 若關閉 partial aggregation，則回退到較保守的 0.7
            required_ratio = float(MIN_PARTIAL_RATIO) if PARTIAL_AGGREGATION_ENABLED else 0.7
            required_min = int(config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3))
            current_clients = len(client_weights_buffer)
            expected_clients = max(1, len(round_clients))
            participation = current_clients / expected_clients
            
            # 達門檻即可聚合
            required_clients = max(required_min, int(math.ceil(expected_clients * required_ratio)))
            if current_clients >= required_clients:
                do_aggregate = True
                print(f"[Aggregator] ✅ 達到聚合門檻: {current_clients}/{expected_clients} ({participation:.1%}), required={required_clients} (ratio={required_ratio:.2f})")
            # 超時強制聚合（但門檻更高）
            elif round_start_time is not None and (time.time() - round_start_time) > AGGREGATION_TIMEOUT:
                min_clients = max(required_min, int(config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', required_min)))
                if current_clients >= min_clients:
                    do_aggregate = True
                    print(f"[Aggregator] ⚠️ 超時強制聚合: {current_clients} 個客戶端")
                else:
                    print(f"[Aggregator] ⏳ 超時但客戶端不足: {current_clients} < {min_clients}")
            else:
                print(f"[Aggregator] ⏳ 等待更多客戶端: {current_clients}/{expected_clients} ({participation:.1%}), required_ratio={required_ratio:.2f}")
        except Exception:
            do_aggregate = check_aggregation_conditions()

        # 同輪上傳中防重入（含超時自動釋放，避免旗標卡死）
        try:
            inflight_round = int(getattr(app.state, "inflight_round", -1))
            inflight_started_at = float(getattr(app.state, "inflight_started_at", 0.0) or 0.0)
            inflight_ttl = float(os.environ.get("AGG_INFLIGHT_TTL_S", "900") or "900")
            if inflight_round == int(round_count):
                if inflight_started_at > 0 and (time.time() - inflight_started_at) > max(30.0, inflight_ttl):
                    app.state.inflight_round = -1
                    app.state.inflight_started_at = 0.0
                    log_event("inflight_expired", f"round={round_count}")
                else:
                    return {
                        "status": "received",
                        "accepted": True,
                        "server_round": int(round_count),
                        "reason": "aggregation_inflight_same_round",
                        "message": "aggregation_inflight_same_round",
                        "buffer_size": len(client_weights_buffer),
                        "current_round": round_count,
                    }
        except Exception:
            pass

        if do_aggregate:
            # 防止重入：若已有聚合在進行，僅回覆已接收
            if aggregation_lock.locked():
                print(f"[Aggregator] ⏳ 聚合進行中，暫不重入")
                return {
                    "status": "received",
                    "accepted": True,
                    "server_round": int(round_count),
                    "reason": "aggregation_in_progress",
                    "message": "aggregation_in_progress",
                    "buffer_size": len(client_weights_buffer),
                    "current_round": round_count
                }
            async with aggregation_lock:
                current_round = int(round_count)
                # 同輪只允許一次有效聚合/上傳，最終一致性判斷放在鎖內
                try:
                    if bool(getattr(app.state, "aggregation_done", False)) and int(getattr(app.state, "aggregation_done_round", -1)) == current_round:
                        return {
                            "status": "received",
                            "accepted": True,
                            "server_round": current_round,
                            "reason": "round_already_aggregated",
                            "message": "round_already_aggregated",
                            "buffer_size": len(client_weights_buffer),
                            "current_round": current_round,
                        }
                except Exception:
                    pass
                try:
                    uploaded_rounds = getattr(app.state, "uploaded_rounds", set())
                    # 首輪 cold-start 再硬擋一次，避免 round=1 仍被重複提交。
                    if current_round == 1 and current_round in uploaded_rounds:
                        return {
                            "status": "received",
                            "accepted": True,
                            "server_round": current_round,
                            "reason": "round_already_uploaded",
                            "message": "round_already_uploaded",
                            "buffer_size": len(client_weights_buffer),
                            "current_round": current_round,
                        }
                    if current_round in uploaded_rounds:
                        return {
                            "status": "received",
                            "accepted": True,
                            "server_round": current_round,
                            "reason": "round_already_uploaded",
                            "message": "round_already_uploaded",
                            "buffer_size": len(client_weights_buffer),
                            "current_round": current_round,
                        }
                except Exception:
                    pass
                try:
                    claimed_round = int(getattr(app.state, "round_claimed_for_aggregation", -1))
                    if claimed_round == current_round:
                        return {
                            "status": "received",
                            "accepted": True,
                            "server_round": current_round,
                            "reason": "round_claimed_for_aggregation",
                            "message": "round_claimed_for_aggregation",
                            "buffer_size": len(client_weights_buffer),
                            "current_round": current_round,
                        }
                except Exception:
                    pass

                # 鎖內再檢查一次聚合門檻，避免 do_aggregate 與實際 buffer 發生競態
                do_aggregate_locked = False
                try:
                    required_ratio = float(MIN_PARTIAL_RATIO) if PARTIAL_AGGREGATION_ENABLED else 0.7
                    required_min = int(config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3))
                    current_clients = len(client_weights_buffer)
                    expected_clients = max(1, len(round_clients))
                    required_clients = max(required_min, int(math.ceil(expected_clients * required_ratio)))
                    if current_clients >= required_clients:
                        do_aggregate_locked = True
                    elif round_start_time is not None and (time.time() - round_start_time) > AGGREGATION_TIMEOUT:
                        min_clients = max(required_min, int(config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', required_min)))
                        do_aggregate_locked = current_clients >= min_clients
                except Exception:
                    do_aggregate_locked = do_aggregate
                if not do_aggregate_locked:
                    return {
                        "status": "received",
                        "accepted": True,
                        "server_round": int(round_count),
                        "reason": "buffered_waiting_for_aggregation",
                        "message": f"權重已接收，等待聚合",
                        "buffer_size": len(client_weights_buffer),
                        "expected_clients": len(round_clients),
                        "current_round": round_count
                    }

                # 若與上次聚合的客戶集合相同，避免重複聚合
                # 但當前 round 已達門檻（尤其已收齊 expected_clients）時，必須立即放行聚合，
                # 否則在協調器重送 start_federated_round 的情境下，可能反覆重置等待窗口而死鎖。
                try:
                    last_ids = getattr(app.state, 'last_aggregated_client_ids', set())
                    last_round = getattr(app.state, 'last_aggregated_round', -1)
                    current_ids = set(client_weights_buffer.keys())
                    if last_round == round_count and current_ids == last_ids:
                        # 強制即時 flush：達到聚合門檻（特別是收齊本輪 selected clients）就直接聚合
                        # 可用 AGG_IMMEDIATE_FLUSH_ON_THRESHOLD=0 關閉，回退舊行為。
                        immediate_flush = str(os.environ.get("AGG_IMMEDIATE_FLUSH_ON_THRESHOLD", "1") or "1").strip().lower() in (
                            "1",
                            "true",
                            "yes",
                            "on",
                        )
                        expected_clients = max(1, len(round_clients))
                        current_clients = len(client_weights_buffer)
                        reached_full = current_clients >= expected_clients
                        if immediate_flush and reached_full:
                            print(
                                f"[Aggregator] ⚡ 同集合但本輪已收齊 {current_clients}/{expected_clients}，"
                                f"立即聚合上傳（跳過等待窗口）",
                                flush=True,
                            )
                        else:
                            # 加入超時豁免：相同集合連續等待超過60秒則仍允許聚合
                            allow_after = 60
                            elapsed = 0
                            if round_start_time is not None:
                                try:
                                    elapsed = time.time() - round_start_time
                                except Exception:
                                    elapsed = 0
                                if elapsed < allow_after:
                                    print(f"[Aggregator] ⏳ 聚合條件達成但與上次聚合客戶相同，等待更多客戶... ({elapsed:.0f}s/<{allow_after}s)")
                                    return {
                                        "status": "received",
                                        "accepted": True,
                                        "server_round": int(round_count),
                                        "reason": "same_clients_already_aggregated",
                                        "message": "已聚合相同客戶集合，等待新客戶加入",
                                        "buffer_size": len(client_weights_buffer),
                                        "current_round": round_count
                                    }
                            else:
                                print(f"[Aggregator] ⏳ 相同客戶集合等待超過 {allow_after}s，放行聚合以推進輪次")
                except Exception:
                    pass
                # 先搶佔本輪聚合提交權，避免同輪第二條請求進入鎖外重算/重上傳。
                app.state.round_claimed_for_aggregation = current_round
                app.state.inflight_round = current_round
                app.state.inflight_started_at = time.time()

            print(f"[Aggregator] 🔄 滿足聚合條件，開始聚合...")
            
            # 準備聚合數據
            client_ids_in_buffer = list(client_weights_buffer.keys())
            client_weights_list = [client_weights_buffer[cid] for cid in client_ids_in_buffer]
            client_data_sizes_list = [client_data_sizes[cid] for cid in client_ids_in_buffer]

            # 可選：基於 Δ 的 norm guard 與 trust 權重調整（粗略近似：用層拼接 L2 估算）
            if NORM_GUARD_ENABLED or TRUST_ENABLED:
                # 計算每個客戶端向量的 L2
                l2_list = []
                for w in client_weights_list:
                    total = 0.0
                    for v in w.values():
                        if isinstance(v, torch.Tensor):
                            total += float(torch.norm(v.reshape(-1).to(dtype=torch.float64), p=2))
                        elif isinstance(v, np.ndarray):
                            total += float(np.linalg.norm(v.reshape(-1)))
                        elif isinstance(v, list):
                            try:
                                total += float(np.linalg.norm(np.array(v, dtype=np.float32).reshape(-1)))
                            except Exception:
                                continue
                    l2_list.append(total)
                if l2_list:
                    median_l2 = float(np.median(l2_list))
                else:
                    median_l2 = 0.0

                # 增強自適應權重聚合
                if NORM_GUARD_ENABLED:
                    # 初始化自適應表
                    if not hasattr(app.state, 'client_adaptive_weights'):
                        app.state.client_adaptive_weights = {}
                    if not hasattr(app.state, 'client_performance_history'):
                        app.state.client_performance_history = {}
                    
                    new_sizes = []
                    for idx, cid in enumerate(client_ids_in_buffer):
                        size = float(client_data_sizes_list[idx])
                        adj = 1.0
                        
                        # 1. 異常值檢測和懲罰
                        if median_l2 > 0 and l2_list[idx] > NORM_GUARD_K * median_l2:
                            adj *= NORM_GUARD_PENALTY
                            # 降低該客戶端的自適應權重
                            app.state.client_adaptive_weights[cid] = app.state.client_adaptive_weights.get(cid, 1.0) * 0.95
                        
                        # 2. 自適應權重調整
                        adaptive_weight = app.state.client_adaptive_weights.get(cid, 1.0)
                        
                        # 3. 基於L2 norm和性能的動態調整
                        if median_l2 > 0:
                            norm_ratio = l2_list[idx] / median_l2
                            
                            # 基於梯度大小（L2 norm）的調整
                            # 目的：讓 adaptive_weights 在早期更容易出現「分化」，便於觀察是否真的在做偏權。
                            # - 放寬「低 norm」區間（更容易加權）
                            # - 收緊「高 norm」區間（更容易降權）
                            if norm_ratio < 0.85:  # 梯度偏小：可能較穩定/已收斂
                                adaptive_weight *= 1.30
                            elif norm_ratio < 1.00:  # 梯度略小
                                adaptive_weight *= 1.10
                            elif norm_ratio > 1.60:  # 梯度偏大：可能噪音/異常
                                adaptive_weight *= 0.75
                            elif norm_ratio > 1.20:  # 梯度略大
                                adaptive_weight *= 0.90
                            
                            # 基於歷史性能的調整
                            if hasattr(app.state, 'client_performance_history'):
                                if cid in app.state.client_performance_history:
                                    perf_history = app.state.client_performance_history[cid]
                                    if len(perf_history) >= 3:
                                        recent_avg = sum(perf_history[-3:]) / 3
                                        # 讓歷史表現也更容易影響偏權（仍維持溫和）
                                        if recent_avg > 0.26:  # 性能良好
                                            adaptive_weight *= 1.08
                                        elif recent_avg < 0.19:  # 性能較差
                                            adaptive_weight *= 0.92
                        
                        # 4. 限制權重範圍
                        adaptive_weight = max(0.3, min(3.0, adaptive_weight))
                        app.state.client_adaptive_weights[cid] = adaptive_weight
                        
                        # 5. 記錄性能歷史
                        if not hasattr(app.state, 'client_performance_history'):
                            app.state.client_performance_history = {}
                        if cid not in app.state.client_performance_history:
                            app.state.client_performance_history[cid] = []
                        
                        # 基於當前L2 norm估算性能（簡化版本）
                        if median_l2 > 0:
                            estimated_performance = max(0.1, min(0.5, 1.0 - norm_ratio * 0.3))
                            app.state.client_performance_history[cid].append(estimated_performance)
                            # 只保留最近10個記錄
                            if len(app.state.client_performance_history[cid]) > 10:
                                app.state.client_performance_history[cid] = app.state.client_performance_history[cid][-10:]
                        
                        # 5. 應用自適應權重
                        adj *= adaptive_weight
                        
                        new_sizes.append(size * adj)
                    
                    client_data_sizes_list = new_sizes
                    
                    # 記錄自適應權重統計
                    adaptive_weights = [app.state.client_adaptive_weights.get(cid, 1.0) for cid in client_ids_in_buffer]
                    avg_weight = sum(adaptive_weights) / len(adaptive_weights)
                    print(f"[Aggregator] 🔧 自適應權重統計: 平均={avg_weight:.3f}, 範圍=[{min(adaptive_weights):.3f}, {max(adaptive_weights):.3f}]")
                
                    # 6. 過擬合檢測和早停
                    if not hasattr(app.state, 'performance_history'):
                        app.state.performance_history = []
                    
                    # 基於平均L2 norm估算整體性能
                    if median_l2 > 0:
                        avg_norm_ratio = sum(l2_list) / len(l2_list) / median_l2
                        estimated_global_performance = max(0.1, min(0.5, 1.0 - avg_norm_ratio * 0.3))
                        app.state.performance_history.append(estimated_global_performance)
                        
                        # 只保留最近20個記錄
                        if len(app.state.performance_history) > 20:
                            app.state.performance_history = app.state.performance_history[-20:]
                        
                        # 檢測性能下降趨勢
                        if len(app.state.performance_history) >= 5:
                            recent_5 = app.state.performance_history[-5:]
                            if all(recent_5[i] <= recent_5[i-1] for i in range(1, len(recent_5))):
                                print(f"[Aggregator] ⚠️ 檢測到性能下降趨勢，可能過擬合")
                                # 降低所有客戶端權重
                                for cid in client_ids_in_buffer:
                                    current_weight = app.state.client_adaptive_weights.get(cid, 1.0)
                                    app.state.client_adaptive_weights[cid] = max(0.3, current_weight * 0.9)
            # 若採用衰減策略，對 α 施加 staleness 衰減（透過 data_size 權重修正為等效）
            if STALE_POLICY == 'decay':
                staleness_list = [int(getattr(app.state, 'client_staleness', {}).get(cid, 0)) for cid in client_ids_in_buffer]
                decay_weights = [math.exp(-STALENESS_DECAY_LAMBDA * s) for s in staleness_list]
                # 將 data_size 以衰減權重縮放，等效於在 α 上做衰減
                client_data_sizes_list = [max(0.0, ds) * float(dw) for ds, dw in zip(client_data_sizes_list, decay_weights)]
            
            # 🚀 新增：在聚合前保存客戶端權重統計信息（用於計算真正的權重差異度）
            client_weight_stats = {}  # {client_id: {'norm': float, 'mean': float, 'std': float}}
            try:
                # 🔧 修復：torch 已在文件頂部導入，不需要局部導入
                for idx, client_id in enumerate(client_ids_in_buffer):
                    if idx < len(client_weights_list):
                        weights = client_weights_list[idx]
                        # 計算權重統計
                        all_weights_flat = []
                        for key, value in weights.items():
                            if isinstance(value, torch.Tensor):
                                all_weights_flat.extend(value.cpu().numpy().flatten().tolist())
                            elif isinstance(value, (list, np.ndarray)):
                                if isinstance(value, np.ndarray):
                                    all_weights_flat.extend(value.flatten().tolist())
                                else:
                                    all_weights_flat.extend(value)
                        
                        if all_weights_flat:
                            client_weight_stats[client_id] = {
                                'norm': float(np.linalg.norm(all_weights_flat)),
                                'mean': float(np.mean(all_weights_flat)),
                                'std': float(np.std(all_weights_flat)),
                                'min': float(np.min(all_weights_flat)),
                                'max': float(np.max(all_weights_flat))
                            }
            except Exception as e:
                print(f"[Aggregator] ⚠️ 計算客戶端權重統計失敗: {e}")
            
            # 執行聚合
            try:
                log_event("aggregation_started", f"開始聚合 {len(client_weights_list)} 個客戶端權重")
                
                # 🚀 新增：區域性聚合（如果啟用）
                regional_config = getattr(config, 'REGIONAL_AGGREGATION_CONFIG', {}) or {}
                defense_mode_env = (os.environ.get("AGG_DEFENSE_MODE", "none") or "none").strip().lower()
                # 對照組硬保證：AGG_DEFENSE_MODE=none 時，禁止進入區域性 ConfShield 過濾路徑。
                use_regional = (
                    defense_mode_env != "none"
                    and regional_config.get('enabled', False)
                    and REGIONAL_AGGREGATION_AVAILABLE
                )
                if defense_mode_env == "none":
                    print("[Aggregator] ℹ️ AGG_DEFENSE_MODE=none，跳過區域性聚合/ConfShield，使用純標準 FedAvg")
                
                # =========================================================
                # 強制可驗證：drop-top cosine_filter（先篩選，再進入 regional / fedavg）
                # 目的：確保「一定會執行」並輸出 dropped client id（便於驗證）
                # =========================================================
                try:
                    import os as _os
                    defense_mode = (_os.environ.get("AGG_DEFENSE_MODE", "none") or "none").strip().lower()
                    cosine_th = float(_os.environ.get("AGG_COSINE_THRESHOLD", "0.0") or 0.0)
                    cosine_pct_raw = (_os.environ.get("AGG_COSINE_PERCENTILE", "") or "").strip()
                    cosine_drop_top_raw = (_os.environ.get("AGG_COSINE_DROP_TOP_PCT", "") or "").strip()
                    min_after = int(_os.environ.get("AGG_MIN_CLIENTS_AFTER_FILTER", "2") or 2)

                    if (
                        defense_mode != "none"
                        and defense_mode == "cosine_filter"
                        and cosine_drop_top_raw
                        and len(client_weights_list) >= 3
                    ):
                        drop_pct = float(cosine_drop_top_raw)
                        if drop_pct <= 1.0:
                            drop_pct = drop_pct * 100.0
                        drop_pct = min(100.0, max(0.0, drop_pct))

                        def _is_bn_stat(name: str) -> bool:
                            return ("running_mean" in name) or ("running_var" in name) or ("num_batches_tracked" in name)

                        def _to_tensor_f32(x):
                            if isinstance(x, torch.Tensor):
                                return x.detach().float()
                            if isinstance(x, np.ndarray):
                                return torch.from_numpy(x).float()
                            if isinstance(x, list):
                                return torch.tensor(x, dtype=torch.float32)
                            return torch.tensor(x, dtype=torch.float32)

                        def _flatten_update(sd: dict) -> torch.Tensor:
                            vecs = []
                            for k, v in sd.items():
                                if _is_bn_stat(k):
                                    continue
                                try:
                                    cv = _to_tensor_f32(v).view(-1)
                                except Exception:
                                    continue
                                if global_weights is not None and k in global_weights:
                                    try:
                                        gv = _to_tensor_f32(global_weights[k]).view(-1)
                                        cv = cv - gv
                                    except Exception:
                                        pass
                                vecs.append(cv.cpu())
                            if not vecs:
                                return torch.zeros((1,), dtype=torch.float32)
                            return torch.cat(vecs, dim=0)

                        def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
                            a = a.float()
                            b = b.float()
                            if a.numel() != b.numel():
                                m = min(a.numel(), b.numel())
                                a = a[:m]
                                b = b[:m]
                            denom = (a.norm() * b.norm()).item()
                            if denom <= 1e-12:
                                return 0.0
                            return float(torch.dot(a, b).item() / denom)

                        update_vecs = [_flatten_update(sd) for sd in client_weights_list]
                        ref = torch.median(torch.stack(update_vecs, dim=0), dim=0).values
                        sims = [_cosine(v, ref) for v in update_vecs]

                        eligible = [i for i, s in enumerate(sims) if s >= cosine_th]
                        if not eligible:
                            eligible = list(range(len(sims)))
                        eligible_sorted_desc = sorted(eligible, key=lambda i: sims[i], reverse=True)
                        n_eligible = len(eligible_sorted_desc)

                        dropped_top_n = int(np.ceil(n_eligible * (drop_pct / 100.0)))
                        dropped_top_n = min(dropped_top_n, max(0, n_eligible - min_after))
                        dropped_idx = eligible_sorted_desc[:dropped_top_n]
                        keep_idx = eligible_sorted_desc[dropped_top_n:]

                        # 兜底：若 keep 太少，保留「最不合群」的 min_after 個
                        if len(keep_idx) < min_after:
                            keep_idx = sorted(eligible_sorted_desc, key=lambda i: sims[i])[:min_after]
                            dropped_idx = [i for i in range(len(client_weights_list)) if i not in set(keep_idx)]

                        # 另外保留原本 percentile 門檻，作為 log 參考（不影響 drop-top 行為）
                        dyn_th = cosine_th
                        if cosine_pct_raw:
                            try:
                                pct = float(cosine_pct_raw)
                                if pct <= 1.0:
                                    pct = pct * 100.0
                                pct = min(100.0, max(0.0, pct))
                                dyn_th = float(np.percentile(np.array(sims, dtype=np.float32), pct))
                            except Exception:
                                pass

                        keep_idx = sorted(set(keep_idx))
                        dropped_idx = sorted(set(dropped_idx))
                        kept_cids = [client_ids_in_buffer[i] for i in keep_idx]
                        dropped_cids = [client_ids_in_buffer[i] for i in dropped_idx]
                        dropped_top_cids = [client_ids_in_buffer[i] for i in dropped_idx if i in set(dropped_idx)]

                        print(
                            "[Aggregator] 🧪 FORCE cosine_drop_top (pre-regional): "
                            f"th={cosine_th:.3f}, dyn_th={dyn_th:.3f}, pct={cosine_pct_raw or '-'}, "
                            f"drop_top_pct={cosine_drop_top_raw}, dropped_top={len(dropped_idx)}, "
                            f"keep={len(keep_idx)}/{len(client_weights_list)}",
                            flush=True,
                        )
                        print(f"[Aggregator] 🧪 FORCE cosine_keep_cids: {kept_cids}", flush=True)
                        print(f"[Aggregator] 🧪 FORCE cosine_dropped_cids: {dropped_cids}", flush=True)

                        client_weights_list = [client_weights_list[i] for i in keep_idx]
                        client_data_sizes_list = [client_data_sizes_list[i] for i in keep_idx]
                        client_ids_in_buffer = [client_ids_in_buffer[i] for i in keep_idx]
                except Exception as e:
                    print(f"[Aggregator] ⚠️ FORCE cosine_drop_top 失敗（忽略，照常聚合）: {e}", flush=True)
                
                if use_regional:
                    print(f"[Aggregator] 🚀 使用區域性聚合策略（ConfShield + Regional Alignment）")
                    try:
                        # 初始化區域性聚合器
                        confshield_cfg = regional_config.get('confshield', {})
                        alignment_cfg = regional_config.get('regional_alignment', {})
                        
                        regional_aggregator = RegionalAggregator(
                            confshield_config=confshield_cfg,
                            alignment_config=alignment_cfg
                        )
                        
                        # 執行區域性聚合
                        aggregated_weights, agg_info = regional_aggregator.aggregate(
                            client_weights_list,
                            client_data_sizes_list,
                            global_weights if global_weights else {},
                            client_ids=client_ids_in_buffer,
                        )
                        
                        # 記錄聚合信息
                        print(f"[Aggregator] 📊 區域性聚合信息:")
                        print(f"  - 原始客戶端數: {agg_info['original_count']}")
                        print(f"  - 過濾後客戶端數: {agg_info['filtered_count']}")
                        print(f"  - 特徵對齊: {'已應用' if agg_info['alignment_applied'] else '未應用'}")
                        
                        if not aggregated_weights and regional_config.get('fallback_to_fedavg', True):
                            print(f"[Aggregator] ⚠️ 區域性聚合失敗，回退到標準 FedAvg")
                            aggregated_weights = perform_standard_fedavg(
                                client_weights_list, client_data_sizes_list, client_ids_list=client_ids_in_buffer
                            )
                    except Exception as reg_e:
                        print(f"[Aggregator] ⚠️ 區域性聚合異常: {reg_e}，回退到標準 FedAvg")
                        if regional_config.get('fallback_to_fedavg', True):
                            aggregated_weights = perform_standard_fedavg(
                                client_weights_list, client_data_sizes_list, client_ids_list=client_ids_in_buffer
                            )
                        else:
                            raise
                else:
                    # 使用標準 FedAvg
                    aggregated_weights = perform_standard_fedavg(
                        client_weights_list, client_data_sizes_list, client_ids_list=client_ids_in_buffer
                    )
                
                print(f"[Aggregator] ✅ 聚合完成")
                
            except Exception as e:
                print(f"[Aggregator] ❌ 聚合失敗: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "message": f"聚合失敗: {str(e)}"
                }
            
            if aggregated_weights:
                # 🚀 計算聚合權重的統計信息（用於權重差異度計算）
                weight_stats = {}
                try:
                    # 🔧 修復：torch 已在文件頂部導入，不需要局部導入
                    # 計算聚合權重的統計信息
                    all_weights_flat = []
                    for key, value in aggregated_weights.items():
                        if isinstance(value, torch.Tensor):
                            all_weights_flat.extend(value.cpu().numpy().flatten().tolist())
                        elif isinstance(value, (list, np.ndarray)):
                            if isinstance(value, np.ndarray):
                                all_weights_flat.extend(value.flatten().tolist())
                            else:
                                all_weights_flat.extend(value)
                    
                    if all_weights_flat:
                        weight_stats = {
                            'mean': float(np.mean(all_weights_flat)),
                            'std': float(np.std(all_weights_flat)),
                            'min': float(np.min(all_weights_flat)),
                            'max': float(np.max(all_weights_flat)),
                            'norm': float(np.linalg.norm(all_weights_flat))
                        }
                except Exception as e:
                    print(f"[Aggregator] ⚠️ 計算聚合權重統計失敗: {e}")
                    weight_stats = {}
                # 追加更詳細的客戶端參與統計到CSV
                try:
                    import csv, datetime
                    result_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
                    os.makedirs(result_dir, exist_ok=True)
                    csv_path = os.path.join(result_dir, f"aggregator_{aggregator_id}_participation.csv")
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                        fieldnames = [
                            'timestamp','round_id','model_version','num_selected','num_buffered',
                            'participation_ratio','client_ids','data_sizes','adaptive_weights',
                            'weight_mean','weight_std','weight_norm',  # 🚀 新增：聚合權重統計
                            'client_weight_norms','client_weight_means','client_weight_stds'  # 🚀 新增：客戶端權重統計（用於真正的權重差異度）
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        adaptive_weights_row = []
                        try:
                            adaptive_weights_row = [app.state.client_adaptive_weights.get(cid, 1.0) for cid in client_ids_in_buffer]
                        except Exception:
                            adaptive_weights_row = []
                        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        participation_ratio = 0.0
                        try:
                            denom_clients = max(1, len(round_clients))
                            participation_ratio = len(client_weights_buffer) / denom_clients
                        except Exception:
                            participation_ratio = 0.0
                        
                        
                        # 🚀 新增：收集客戶端權重統計（用於真正的權重差異度計算）
                        client_weight_norms_str = ''
                        client_weight_means_str = ''
                        client_weight_stds_str = ''
                        try:
                            # 按照 client_ids_in_buffer 的順序收集統計信息
                            norms_list = [f"{client_weight_stats.get(cid, {}).get('norm', 0.0):.6f}" for cid in client_ids_in_buffer]
                            means_list = [f"{client_weight_stats.get(cid, {}).get('mean', 0.0):.6f}" for cid in client_ids_in_buffer]
                            stds_list = [f"{client_weight_stats.get(cid, {}).get('std', 0.0):.6f}" for cid in client_ids_in_buffer]
                            client_weight_norms_str = ','.join(norms_list)
                            client_weight_means_str = ','.join(means_list)
                            client_weight_stds_str = ','.join(stds_list)
                        except Exception as e:
                            print(f"[Aggregator] ⚠️ 收集客戶端權重統計失敗: {e}")
                        
                        writer.writerow({
                            'timestamp': ts,
                            'round_id': round_count,
                            'model_version': model_version,
                            'num_selected': len(round_clients),
                            'num_buffered': len(client_weights_buffer),
                            'participation_ratio': participation_ratio,
                            'client_ids': ','.join(map(str, client_ids_in_buffer)),
                            'data_sizes': ','.join(map(str, [client_data_sizes.get(cid, 0) for cid in client_ids_in_buffer])),
                            'adaptive_weights': ','.join([f"{w:.3f}" for w in adaptive_weights_row]) if adaptive_weights_row else '',
                            'weight_mean': weight_stats.get('mean', 0.0),  # 🚀 新增：聚合權重統計
                            'weight_std': weight_stats.get('std', 0.0),  # 🚀 新增：聚合權重統計
                            'weight_norm': weight_stats.get('norm', 0.0),  # 🚀 新增：聚合權重統計
                            'client_weight_norms': client_weight_norms_str,  # 🚀 新增：客戶端權重範數
                            'client_weight_means': client_weight_means_str,  # 🚀 新增：客戶端權重均值
                            'client_weight_stds': client_weight_stds_str  # 🚀 新增：客戶端權重標準差
                        })
                except Exception as _e:
                    print(f"[Aggregator] ⚠️ 寫入參與統計失敗: {_e}")
                # 更新全局性能指標
                if global_performance is None:
                    global_performance = 0.5
                else:
                    denom_clients = max(1, len(round_clients))
                    participation_ratio = len(client_weights_buffer) / denom_clients
                    global_performance = 0.9 * global_performance + 0.1 * participation_ratio
                
                print(f"[Aggregator] 📈 更新全局性能指標: {global_performance:.3f}")
                
                # 穩定輪次推進：確保足夠的客戶端參與（先算 will_advance，再決定是否嚴格「先上傳雲端再提交本地」）
                denom_clients2 = max(1, len(round_clients))
                participation_ratio = len(client_weights_list) / denom_clients2
                will_advance = participation_ratio >= 0.4  # 40% 參與率即可推進
                cloud_url_pre = (CLOUD_SERVER_CONFIG.get("url") or "").strip().rstrip("/")
                _image_fl_agg = (os.environ.get("IMAGE_FL", "") or "").strip() == "1"
                _strict_cloud = bool(_image_fl_agg and cloud_url_pre and will_advance)
                old_gw = None
                old_mv = None
                if _strict_cloud:
                    old_gw = global_weights
                    old_mv = model_version
                
                # 保存聚合結果到緩存，不立即推進/同步
                global_weights = aggregated_weights
                
                # 🔧 修復：更新 model_version
                model_version += 1
                print(f"[Aggregator] 🔄 更新 model_version: {model_version}")
                
                if will_advance:
                    # round_count += 1  # 🔥 緊急修復：禁用自動輪次推進
                    print(f"[Aggregator] 🔄 聚合完成，保持在第 {round_count} 輪")
                    print(f"[Aggregator] 📊 本輪聚合統計: {len(client_weights_list)} 個客戶端參與 (參與率: {participation_ratio:.1%})")
                else:
                    print(f"[Aggregator] ⚠️ 參與率不足，不推進輪次: {len(client_weights_list)}/{denom_clients2} ({participation_ratio:.1%})")
                print(f"[Aggregator] 📊 本輪聚合統計: {len(client_weights_list)} 個客戶端參與")

                # IMAGE_FL + 雲端：先確保 delta 上傳成功，再標記完成／清空 buffer（避免協調器已推進但雲端未收到該輪）
                if _strict_cloud:
                    upload_ok_early = False
                    try:
                        cloud_url = cloud_url_pre
                        base_version = 0
                        base_weights = None
                        try:
                            timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                            async with aiohttp.ClientSession(timeout=timeout) as session:
                                async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                    if resp.status == 200:
                                        payload = pickle.loads(await resp.read())
                                        base_version = int(payload.get('version', 0))
                                        base_weights = payload.get('weights', {})
                        except Exception as e:
                            print(f"[Aggregator {aggregator_id}] ⚠️ 取得 cloud base 版本失敗: {e}")

                        def _to_tensor_sc(x):
                            if isinstance(x, torch.Tensor):
                                return x.detach().cpu().float()
                            if isinstance(x, np.ndarray):
                                return torch.from_numpy(x).float()
                            if isinstance(x, list):
                                try:
                                    return torch.tensor(x, dtype=torch.float32)
                                except Exception:
                                    return None
                            return None

                        delta_sc = {}
                        if isinstance(base_weights, dict) and base_weights:
                            for k, v in global_weights.items():
                                g = _to_tensor_sc(v)
                                b = _to_tensor_sc(base_weights.get(k))
                                if g is not None and b is not None and g.shape == b.shape:
                                    delta_sc[k] = (g - b)
                                else:
                                    if g is not None:
                                        delta_sc[k] = g
                        else:
                            for k, v in global_weights.items():
                                t = _to_tensor_sc(v)
                                if t is not None:
                                    delta_sc[k] = t

                        upload_ok_early = await upload_delta_to_cloud(
                            base_version, delta_sc, current_round, buffered_clients=len(client_weights_list)
                        )
                        if not upload_ok_early:
                            try:
                                timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                                async with aiohttp.ClientSession(timeout=timeout) as session:
                                    async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                        if resp.status == 200:
                                            payload = pickle.loads(await resp.read())
                                            base_version = int(payload.get('version', 0))
                                            base_weights = payload.get('weights', {})
                                            delta_sc = {}
                                            for k, v in global_weights.items():
                                                g = _to_tensor_sc(v)
                                                b = _to_tensor_sc(base_weights.get(k))
                                                if g is not None and b is not None and g.shape == b.shape:
                                                    delta_sc[k] = (g - b)
                                                else:
                                                    if g is not None:
                                                        delta_sc[k] = g
                                            upload_ok_early = await upload_delta_to_cloud(
                                                base_version, delta_sc, current_round, buffered_clients=len(client_weights_list)
                                            )
                            except Exception as e2:
                                print(f"[Aggregator {aggregator_id}] ❌ IMAGE_FL 嚴格上傳重試仍失敗: {e2}")
                                upload_ok_early = False
                    except Exception as e:
                        print(f"[Aggregator {aggregator_id}] ⚠️ IMAGE_FL 嚴格雲端同步失敗: {e}")
                        log_event("cloud_sync_failed", f"strict_before_buffer_clear: {e}")
                        upload_ok_early = False

                    if not upload_ok_early:
                        app.state.inflight_round = -1
                        app.state.inflight_started_at = 0.0
                        app.state.round_claimed_for_aggregation = -1
                        global_weights = old_gw
                        model_version = old_mv if old_mv is not None else max(0, model_version - 1)
                        print(
                            f"[Aggregator {aggregator_id}] ⚠️ IMAGE_FL：本輪 delta 未成功上傳雲端，回滾 model、保留 buffer",
                            flush=True,
                        )
                        log_event("aggregation_blocked", "strict_cloud_upload_failed_keep_buffer")
                        return {
                            "status": "pending_cloud_upload",
                            "accepted": False,
                            "server_round": int(current_round),
                            "reason": "cloud_delta_upload_failed",
                            "message": "聚合結果已計算但雲端上傳失敗，已回滾全局權重並保留緩衝區以便重試",
                            "global_performance": global_performance,
                            "participation_ratio": participation_ratio,
                            "next_round": current_round,
                        }
                
                # 標記本輪已聚合，記錄本次聚合使用的客戶集合
                # 👉 聚合的實際輪次就是當前 round_count，不再使用 round_count-1 的偏移，
                #    避免對 Cloud 回報錯誤輪次（導致雲端一直把更新當成上一輪的 delta）。
                app.state.aggregated_in_round = current_round
                app.state.aggregation_done = True
                app.state.aggregation_done_round = current_round
                app.state.last_aggregated_client_ids = set(client_weights_buffer.keys())
                app.state.last_aggregated_round = current_round
                # 不在同輪聚合成功後立即清 inflight，避免同輪晚到請求再次觸發重複聚合/上傳。
                # inflight 於 start_federated_round/reset_round 重置。
                
                # 在清空前記錄參與客戶端
                participating_clients = list(client_weights_buffer.keys())
                # 僅在推進輪次時清空緩衝區；否則保留，讓更多客戶加入再嘗試
                if will_advance:
                    client_weights_buffer.clear()
                    client_data_sizes.clear()
                    app.state.round_upload_window_closed = True
                
                print(f"[Aggregator] ✅ 聚合完成，清空緩衝區")
                log_event("aggregation_completed", f"聚合完成，參與客戶端: {participating_clients}")
                # 標記上一輪完成（round_count 已 +1，所以上一輪是 round_count-1）
                try:
                    global last_completed_round
                    last_completed_round = round_count
                except Exception:
                    pass
                # 聚合成功次數 +1
                try:
                    globals()['total_aggregations_done'] = int(globals().get('total_aggregations_done', 0)) + 1
                except Exception:
                    globals()['total_aggregations_done'] = 1
                save_aggregation_stats()

                # 只在輪次推進時做 Cloud 同步與拉取（IMAGE_FL 嚴格路徑已於清 buffer 前上傳）
                if not _strict_cloud:
                    try:
                        # 先獲取 cloud 當前版本與權重作為 base
                        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
                        base_version = 0
                        base_weights = None
                        if cloud_url:
                            try:
                                timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                                async with aiohttp.ClientSession(timeout=timeout) as session:
                                    async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                        if resp.status == 200:
                                            payload = pickle.loads(await resp.read())
                                            base_version = int(payload.get('version', 0))
                                            base_weights = payload.get('weights', {})
                            except Exception as e:
                                print(f"[Aggregator {aggregator_id}] ⚠️ 取得 cloud base 版本失敗: {e}")
                        # 計算 delta = local - base
                        def _to_tensor(x):
                            if isinstance(x, torch.Tensor):
                                return x.detach().cpu().float()
                            elif isinstance(x, np.ndarray):
                                return torch.from_numpy(x).float()
                            elif isinstance(x, list):
                                try:
                                    return torch.tensor(x, dtype=torch.float32)
                                except Exception:
                                    return None
                            return None
                        delta = {}
                        if isinstance(base_weights, dict) and base_weights:
                            for k, v in global_weights.items():
                                g = _to_tensor(v)
                                b = _to_tensor(base_weights.get(k))
                                if g is not None and b is not None and g.shape == b.shape:
                                    delta[k] = (g - b)
                                else:
                                    # 若 base 不存在該鍵或形狀不匹配，視為全量更新
                                    if g is not None:
                                        delta[k] = g
                        else:
                            # 無 base -> 上傳全量作為 delta
                            for k, v in global_weights.items():
                                t = _to_tensor(v)
                                if t is not None:
                                    delta[k] = t

                        # 嘗試上傳 delta；若 CAS 衝突（409），重取版本後重算 delta 再傳一次
                        try:
                            # round_id 必須使用當前 round_count（避免雲端永遠只收到上一輪）
                            await upload_delta_to_cloud(base_version, delta, current_round, buffered_clients=len(client_weights_list))
                        except Exception as e:
                            print(f"[Aggregator {aggregator_id}] ⚠️ 上傳 delta 初次失敗: {e}")
                            # 重取
                            try:
                                timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                                async with aiohttp.ClientSession(timeout=timeout) as session:
                                    async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                        if resp.status == 200:
                                            payload = pickle.loads(await resp.read())
                                            base_version = int(payload.get('version', 0))
                                            base_weights = payload.get('weights', {})
                                            # 重新計算 delta
                                            delta = {}
                                            for k, v in global_weights.items():
                                                g = _to_tensor(v)
                                                b = _to_tensor(base_weights.get(k))
                                                if g is not None and b is not None and g.shape == b.shape:
                                                    delta[k] = (g - b)
                                                else:
                                                    if g is not None:
                                                        delta[k] = g
                                            await upload_delta_to_cloud(base_version, delta, current_round, buffered_clients=len(client_weights_list))
                            except Exception as e2:
                                print(f"[Aggregator {aggregator_id}] ❌ 上傳 delta 重試仍失敗: {e2}")
                    except Exception as e:
                        print(f"[Aggregator {aggregator_id}] ⚠️ Cloud同步失敗: {e}")
                        log_event("cloud_sync_failed", str(e))

                log_training_event_aggregator(
                    'aggregated',
                    {
                        'round_id': current_round,
                        'aggregated_count': len(client_weights_list),
                        'participation_ratio': len(client_weights_list) / max(1, len(round_clients)),
                        'model_version': model_version
                    }
                )
                
                return {
                    "status": "aggregated",
                    "accepted": True,
                    "server_round": int(current_round),
                    "reason": "aggregated",
                    "message": f"聚合完成，{len(client_weights_list)}個客戶端參與",
                    "global_performance": global_performance,
                    "participation_ratio": len(client_weights_list) / max(1, len(round_clients)),
                    "next_round": current_round
                }
        
        return {
            "status": "received",
            "accepted": True,
            "server_round": int(round_count),
            "reason": "buffered_waiting_for_aggregation",
            "message": f"權重已接收，等待聚合",
            "buffer_size": len(client_weights_buffer),
            "expected_clients": len(round_clients),
            "current_round": round_count
        }
        
    except Exception as e:
        try:
            app.state.inflight_round = -1
            app.state.inflight_started_at = 0.0
            app.state.round_claimed_for_aggregation = -1
        except Exception:
            pass
        print(f"[Aggregator] ❌ 處理客戶端 {client_id} 權重時發生錯誤: {str(e)}")
        import traceback as tb_module
        print(f"[Aggregator] ❌ 錯誤詳情: {tb_module.format_exc()}")
        raise HTTPException(status_code=500, detail=f"權重處理失敗: {str(e)}")

@app.get("/get_global_weights")
async def get_global_weights():
    """獲取全局權重"""
    # 如果全局權重為空，先初始化
    if global_weights is None:
        initialize_global_weights()
    
    if global_weights is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "全局權重初始化失敗", "available": False}
        )
    
    # 將PyTorch張量轉換為JSON可序列化的格式
    json_weights = {}
    if global_weights:
        for key, value in global_weights.items():
            if isinstance(value, torch.Tensor):
                # 確保張量在CPU上並轉換為numpy數組
                if value.device.type != 'cpu':
                    value = value.cpu()
                json_weights[key] = value.numpy().tolist()
            elif isinstance(value, (int, float, bool)):
                # 直接使用基本類型
                json_weights[key] = value
            elif isinstance(value, (list, tuple)):
                # 處理列表或元組
                json_weights[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
            else:
                # 其他類型，嘗試轉換為float
                try:
                    json_weights[key] = float(value)
                except (ValueError, TypeError):
                    print(f"[Aggregator] ⚠️ 無法序列化權重 {key}: {type(value)}")
                    json_weights[key] = 0.0
    
    # 伺服器模型權重（可選）
    json_server_weights = {}
    if server_global_weights:
        for key, value in server_global_weights.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cpu':
                    value = value.cpu()
                json_server_weights[key] = value.numpy().tolist()
            elif isinstance(value, (int, float, bool)):
                json_server_weights[key] = value
            elif isinstance(value, (list, tuple)):
                json_server_weights[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
            else:
                try:
                    json_server_weights[key] = float(value)
                except (ValueError, TypeError):
                    print(f"[Aggregator] ⚠️ 無法序列化伺服器權重 {key}: {type(value)}")
                    json_server_weights[key] = 0.0
    
    # 返回JSON格式的權重信息
    weights_info = {
        "status": "success",
        "available": True,
        "global_weights": json_weights,
        "weights_count": len(global_weights) if global_weights else 0,
        "model_version": model_version,
        "server_weights": json_server_weights if json_server_weights else None,
        "server_model_version": server_model_version,
        "round_count": round_count,
        "aggregator_id": aggregator_id,
        "timestamp": time.time()
    }
    
    return JSONResponse(content=weights_info)

@app.get("/aggregation_status")
async def get_aggregation_status():
    """獲取聚合狀態"""
    try:
        status = {
            'federated_round': {
                'current_round': round_count,
                'selected_clients': round_clients,
                'buffer_size': len(client_weights_buffer),
                'min_clients_required': 2,  # 🔧 修復：設置為2，確保多客戶端參與
                'round_start_time': round_start_time,
                'elapsed_time': time.time() - round_start_time if round_start_time else 0,
                'timeout_clients': [cid for cid, timeout in client_timeout_status.items() if timeout]
            },
            'global_weights': {
                'available': global_weights is not None,
                'model_version': model_version
            },
            'timeout_config': {
                'aggregation_timeout': AGGREGATION_TIMEOUT,
                'partial_aggregation_enabled': PARTIAL_AGGREGATION_ENABLED,
                'min_partial_ratio': MIN_PARTIAL_RATIO
            }
        }
        
        return convert_numpy_values(status)
        
    except Exception as e:
        error_msg = f"獲取聚合狀態時發生錯誤: {str(e)}"
        print(f"[Aggregator {aggregator_id}] ❌ {error_msg}")
        log_event("aggregation_status_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/enhanced_status")
async def get_enhanced_status():
    """獲取增強狀態信息"""
    status = {
        "aggregator_id": aggregator_id,
        "round_count": round_count,
        "buffer_size": len(client_weights_buffer),
        "global_weights_available": global_weights is not None,
        "model_info": {
            "version": model_version,
            "available": global_weights is not None
        },
        "training_phase": is_training_phase,
        "selected_clients": round_clients,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return convert_numpy_values(status)

def _federated_status_log_every_n() -> int:
    """每 N 次請求記錄一次；N<=0 或 IMAGE_FL 未開時改為每次記錄。覆寫：AGG_FEDERATED_STATUS_LOG_EVERY_N"""
    raw = (os.environ.get("AGG_FEDERATED_STATUS_LOG_EVERY_N", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if (os.environ.get("IMAGE_FL", "") or "").strip() == "1":
        return 200
    return 1


@app.get("/federated_status")
async def get_federated_status():
    """獲取聯邦學習狀態 - 客戶端需要的端點"""
    try:
        global _federated_status_req_seq, last_cloud_ack_round
        _federated_status_req_seq += 1
        _n = _federated_status_req_seq
        _every = _federated_status_log_every_n()
        _do_log = _every <= 1 or _n == 1 or (_n % _every == 0)
        if _do_log:
            print(
                f"[Aggregator {aggregator_id}] 🔍 federated_status 查詢: round_count={round_count}, "
                f"selected_clients={round_clients}, buffer_size={len(client_weights_buffer)}"
            )
        
        status = {
            "aggregator_id": aggregator_id,
            "round_count": round_count,
            "current_round": round_count,
            "last_cloud_ack_round": int(last_cloud_ack_round),
            "buffer_size": len(client_weights_buffer),
            "global_weights_available": global_weights is not None,
            "model_info": {
                "version": model_version,
                "available": global_weights is not None
            },
            "training_phase": is_training_phase,
            "selected_clients": round_clients,
            "min_clients_required": 2,  # 🔧 修復：設置為2，確保多客戶端參與
            "aggregation_timeout": AGGREGATION_TIMEOUT,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        result = convert_numpy_values(status)
        if _do_log:
            print(
                f"[Aggregator {aggregator_id}] ✅ federated_status 返回: round={round_count}, "
                f"selected_count={len(round_clients)}, selected={result.get('selected_clients', [])}"
            )
        
        return result
    except Exception as e:
        error_msg = f"獲取聯邦學習狀態時發生錯誤: {str(e)}"
        print(f"[Aggregator {aggregator_id}] ❌ {error_msg}")
        log_event("federated_status_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/register_client")
async def register_client(client_id: int = Form(...)):
    """註冊客戶端"""
    try:
        print(f"[Aggregator {aggregator_id}] 📝 客戶端 {client_id} 註冊")
        
        # 檢查客戶端是否在分配列表中
        assigned_clients = get_assigned_clients()
        if client_id in assigned_clients:
            print(f"[Aggregator {aggregator_id}] ✅ 客戶端 {client_id} 註冊成功")
            return {
                "status": "success",
                "message": f"客戶端 {client_id} 註冊成功",
                "aggregator_id": aggregator_id,
                "assigned": True
            }
        else:
            print(f"[Aggregator {aggregator_id}] ⚠️ 客戶端 {client_id} 不在分配列表中")
            return {
                "status": "success",
                "message": f"客戶端 {client_id} 註冊成功（但不在分配列表中）",
                "aggregator_id": aggregator_id,
                "assigned": False
            }
        
    except Exception as e:
        error_msg = f"客戶端註冊失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] ❌ {error_msg}")
        log_event("client_registration_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/current_round")
async def get_current_round():
    """獲取當前輪次"""
    return {
        "current_round": round_count,
        "aggregator_id": aggregator_id
    }

@app.post("/sync_state")
async def sync_state(client_id: int = Form(...), last_confirmed_round: int = Form(...)):
    """同步客戶端狀態（回傳當前輪次與選中客戶端）"""
    return {
        "status": "success",
        "aggregator_id": aggregator_id,
        "current_round": round_count,
        "last_confirmed_round": last_confirmed_round,
        "selected_clients": list(round_clients) if round_clients else []
    }

@app.post("/reset_round")
async def reset_round(target_round: int = Form(...)):
    """重置輪次"""
    global round_count, round_clients, client_weights_buffer, round_start_time, client_timeout_status, last_completed_round, is_training_phase
    
    print(f"[Aggregator {aggregator_id}] 🔄 重置輪次: {round_count} -> {target_round}")
    
    # 🔧 修復：完全重置所有輪次相關狀態
    round_count = target_round
    last_completed_round = target_round - 1  # 🔧 修復：重置完成輪次
    round_clients = []
    client_weights_buffer.clear()
    client_data_sizes.clear()
    round_start_time = None
    client_timeout_status.clear()
    is_training_phase = False
    app.state.round_upload_window_closed = False
    app.state.inflight_round = -1
    app.state.inflight_started_at = 0.0
    app.state.round_claimed_for_aggregation = -1
    app.state.uploaded_rounds = set()
    app.state.last_uploaded_round = -1
    
    # 🔧 修復：重置選擇器狀態
    if hasattr(app.state, 'not_selected_streak'):
        app.state.not_selected_streak = {}
    if hasattr(app.state, 'last_selected_clients'):
        app.state.last_selected_clients = []
    if hasattr(app.state, 'training_results_buffer'):
        app.state.training_results_buffer.clear()
    
    # 重新選擇客戶端
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
    if training_flow == 'train_then_select':
        round_clients = get_assigned_clients()
    else:
        round_clients = select_clients_for_round(target_round, total_clients=len(get_assigned_clients()))
    
    print(f"[Aggregator {aggregator_id}] ✅ 輪次重置完成，選中客戶端: {round_clients}")
    
    # 記錄事件
    log_event("round_reset", f"重置到輪次 {target_round}")
    
    return {
            "status": "success",
        "message": f"輪次重置為 {target_round}",
                "current_round": round_count,
        "selected_clients": round_clients
    }

@app.post("/report_availability")
async def report_availability(
    client_id: int = Form(...),
    cpu_usage: float = Form(...),
    memory_usage: float = Form(...),
    battery_level: float = Form(...)
):
    """報告客戶端可用性"""
    try:
        print(f"[Aggregator {aggregator_id}] 📊 客戶端 {client_id} 可用性報告:")
        print(f"  - CPU使用率: {cpu_usage:.2f}%")
        print(f"  - 內存使用率: {memory_usage:.2f}%")
        print(f"  - 電池電量: {battery_level:.2f}%")
        
        return {
            "status": "success",
            "message": "可用性報告已接收",
            "client_id": client_id
        }
        
    except Exception as e:
        error_msg = f"處理可用性報告失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] ❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/receive_global_weights")
async def receive_global_weights(
    weights: UploadFile = File(...),
    global_version: int = Form(...),
    broadcast_type: str = Form("periodic"),
    round_id: Optional[int] = Form(None),
):
    """🔧 新增：接收 Cloud Server 廣播的全局權重（分層非同步 FL 核心）"""
    global server_global_weights, server_model_version, last_cloud_ack_round
    
    try:
        print(f"[Aggregator {aggregator_id}] 📥 收到 Cloud Server 廣播的全局權重 (版本: {global_version}, 類型: {broadcast_type})")
        
        # 解析權重數據
        weights_data = pickle.loads(weights.file.read())
        
        if isinstance(weights_data, dict) and ('server_weights' in weights_data or 'global_weights' in weights_data):
            # 新格式：包含元數據的字典
            new_server_weights = weights_data.get('server_weights', weights_data.get('global_weights'))
            broadcast_version = weights_data.get('global_version', global_version)
            broadcast_timestamp = weights_data.get('timestamp', time.time())
        else:
            # 舊格式：直接是權重字典
            new_server_weights = weights_data
            broadcast_version = global_version
            broadcast_timestamp = time.time()
        
        # 檢查版本是否更新
        if broadcast_version > server_model_version:
            # 更新伺服器端權重與版本（不覆蓋聚合器全模型權重）
            server_global_weights = new_server_weights
            server_model_version = broadcast_version
            if round_id is not None:
                try:
                    rid = int(round_id)
                    if rid > int(last_cloud_ack_round):
                        last_cloud_ack_round = rid
                except Exception:
                    pass
            
            print(f"[Aggregator {aggregator_id}] ✅ 成功更新伺服器模型權重: 版本 {server_model_version}")
            log_event("server_weights_updated", f"version={server_model_version},broadcast_type={broadcast_type}")
            
            return {
                "status": "success",
                "message": f"伺服器模型權重已更新到版本 {server_model_version}",
                "aggregator_id": aggregator_id,
                "new_version": server_model_version
            }
        else:
            print(f"[Aggregator {aggregator_id}] ℹ️ 廣播版本 {broadcast_version} 不新於當前版本 {server_model_version}，跳過更新")
            return {
                "status": "skipped",
                "message": f"版本 {broadcast_version} 不新於當前版本 {server_model_version}",
                "aggregator_id": aggregator_id,
                "current_version": server_model_version
            }
        
    except Exception as e:
        error_msg = f"處理廣播全局權重失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] ❌ {error_msg}")
        log_event("broadcast_receive_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/select_clients_after_training")
async def select_clients_after_training(
    client_id: int = Form(...),
    training_results: str = Form(...),
    round_id: int = Form(...)
):
    """訓練後選擇客戶端 - 用於train_then_select流程"""
    try:
        print(f"[Aggregator {aggregator_id}] 📊 收到客戶端 {client_id} 訓練後選擇請求")
        
        # 解析訓練結果
        try:
            import json
            results_data = json.loads(training_results)
        except json.JSONDecodeError:
            # 如果JSON解析失敗，嘗試pickle格式
            try:
                results_data = pickle.loads(training_results.encode())
            except Exception as e:
                print(f"[Aggregator {aggregator_id}] ❌ 訓練結果解析失敗: {e}")
                results_data = {}
        
        # 🔧 新增：記錄客戶端性能
        record_client_performance(client_id, results_data, round_id)
        
        # 存儲訓練結果
        if not hasattr(app.state, 'training_results_buffer'):
            app.state.training_results_buffer = {}
        
        app.state.training_results_buffer[client_id] = {
            'results': results_data,
            'round_id': round_id,
            'timestamp': time.time()
        }
        
        print(f"[Aggregator {aggregator_id}] ✅ 客戶端 {client_id} 訓練結果已接收")
        print(f"[Aggregator {aggregator_id}] 📊 當前訓練結果緩衝區: {len(app.state.training_results_buffer)} 個客戶端")
        
        # 檢查是否需要進行客戶端選擇（允許部分客戶端，避免長時間等待）
        current_count = len(app.state.training_results_buffer)
        total_expected = max(1, len(round_clients))
        min_partial_ratio = MIN_PARTIAL_RATIO if 'MIN_PARTIAL_RATIO' in globals() else 0.3
        min_required = max(1, min(total_expected, int(math.ceil(min_partial_ratio * total_expected))))

        if current_count >= min_required:
            print(f"[Aggregator {aggregator_id}] 🔄 收到 {current_count}/{total_expected} 份訓練結果（門檻 {min_required}），開始選擇最佳客戶端")
            
            # 基於訓練結果選擇客戶端
            selected_clients = select_best_clients_after_training()
            
            return {
                "status": "selection_complete",
                "message": "客戶端選擇完成",
                "selected_clients": selected_clients,
                "total_clients": current_count
            }
        else:
            return {
                "status": "waiting",
                "message": f"等待更多客戶端完成訓練 ({current_count}/{total_expected})",
                "selected_clients": [],
                "total_clients": current_count,
                "min_required": min_required
            }
        
    except Exception as e:
        error_msg = f"處理訓練後選擇請求失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] ❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

def select_best_clients_after_training():
    """基於訓練結果選擇最佳客戶端（簡化版）"""
    if not hasattr(app.state, 'training_results_buffer') or not app.state.training_results_buffer:
        return []
    
    # 簡化：返回所有有訓練結果的客戶端
    selected_clients = list(app.state.training_results_buffer.keys())
    app.state.training_results_buffer.clear()
    
    print(f"[Aggregator {aggregator_id}] ✅ 選擇客戶端: {selected_clients}")
    return selected_clients

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動聚合器 (修復版本)")
    parser.add_argument("--aggregator_id", type=int, default=0, help="聚合器ID")
    parser.add_argument("--port", type=int, default=8000, help="端口號")
    
    args = parser.parse_args()

    # 設置聚合器ID
    aggregator_id = args.aggregator_id
    aggregator_port = args.port
    
    print(f"[Aggregator {aggregator_id}] 🚀 啟動聚合器 (端口: {args.port})")
    print(f"[Aggregator {aggregator_id}] 📊 配置信息:")
    print(f"  - 聚合超時: {AGGREGATION_TIMEOUT}s")
    print(f"  - 最小客戶端數: {config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3)}")
    print(f"  - 訓練流程: {config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')}")
    print(f"  - 平滑因子: {config.FEDERATED_CONFIG.get('aggregation', {}).get('smoothing_factor', 0.8)}")
    print(f"  - MAX_STALENESS: {MAX_STALENESS}（環境變數 AGG_MAX_STALENESS 可覆寫 config）")
    
    # 啟動服務
    uvicorn.run(app, host="0.0.0.0", port=args.port)
