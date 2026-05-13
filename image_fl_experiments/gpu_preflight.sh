#!/usr/bin/env bash
# 跑影像 FL 實驗前檢查 GPU：印出 nvidia-smi 與佔用 GPU 的行程。
# 不自動 kill 任何行程；請依輸出自行判斷殘留的 python/訓練/Notebook 並手動結束（例如 kill <PID>）。
#
# 用法：
#   ./gpu_preflight.sh
#   SKIP_GPU_PREFLIGHT=1 ./run_gtsrb_fl_diag_easy.sh   # 略過預檢，由 run 腳本處理
#
set -u

echo "========== GPU 預檢 $(date -Iseconds 2>/dev/null || date) =========="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu_preflight] 未找到 nvidia-smi（可能無 NVIDIA 驅動）。若 client 用 CPU 仍能跑，可略過。"
  exit 0
fi

nvidia-smi
echo ""
echo "========== Compute apps（若驅動支援；否則看上方 Processes）=========="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true

echo ""
echo "[gpu_preflight] 請確認沒有不需的訓練／Jupyter／瀏覽器佔用 GPU。"
echo "[gpu_preflight] 若要手動釋放顯存：先記下 PID，再執行  kill <PID>  或  kill -9 <PID>（謹慎使用）。"
exit 0
