#!/usr/bin/env python3
"""
image_cloud_metrics.csv 讀取輔助：並發寫入時同一 round 可能出現多列，分析時應去重。

規則：同一 round 保留「檔案中較晚出現」的那一列，再依 round 遞增排序。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_raw_metric_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        return [dict(r) for r in reader]


def dedupe_metric_rows_by_round(rows: List[Dict[str, str]], *, round_key: str = "round") -> List[Dict[str, str]]:
    """同一 round 多列時保留最後一列；輸出依 round 遞增排序。"""
    by_round: Dict[int, Dict[str, str]] = {}
    for r in rows:
        try:
            rid = int(float(str(r.get(round_key, "0") or "0").strip() or "0"))
        except ValueError:
            continue
        by_round[rid] = r
    return [by_round[k] for k in sorted(by_round.keys())]


def read_image_cloud_metrics_rows_deduped(csv_path: Path) -> List[Dict[str, str]]:
    return dedupe_metric_rows_by_round(read_raw_metric_rows(csv_path))


def _parse_typed_metric_row(r: Dict[str, str]) -> Optional[Dict[str, Any]]:
    try:
        ftr_val = float(r.get("ftr", "0") or 0)
        asr_val = float(r.get("asr", "0") or 0)
        return {
            "round": int(float(r.get("round", 0) or 0)),
            "clean_acc": float(r.get("clean_acc", 0) or 0),
            "clean_f1_macro": float(r.get("clean_f1_macro", 0) or 0),
            "clean_loss": float(r.get("clean_loss", 0) or 0),
            "ftr": ftr_val,
            "asr": asr_val,
            "trig_n": int(float(r.get("trig_n", r.get("asr_n", 0)) or 0)),
            "target_label": int(float(r.get("target_label", 0) or 0)),
            "poison_enabled": int(float(r.get("poison_enabled", 0) or 0)),
            "attacker_clients": str(r.get("attacker_clients", "") or ""),
            "defense_mode": str(r.get("defense_mode", "") or ""),
        }
    except Exception:
        return None


def metrics_last_and_best_f1(csv_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """去重後：last = round 最大者；best = clean_f1_macro 最大者。"""
    rows: List[Dict[str, Any]] = []
    for r in read_image_cloud_metrics_rows_deduped(csv_path):
        p = _parse_typed_metric_row(r)
        if p is not None:
            rows.append(p)
    if not rows:
        return None, None
    last = max(rows, key=lambda x: int(x.get("round", -1)))
    best = max(rows, key=lambda x: float(x.get("clean_f1_macro", 0.0)))
    return last, best
