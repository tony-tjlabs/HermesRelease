"""
MAC Stitching — Release stub.
Only evaluation/display functions are kept for the Stitching tab UI.
Core stitching logic (Level 1 / Level 2) is in the development codebase only.
"""
from typing import Any, Dict

import pandas as pd

from src.config.constants import (
    DEVICE_TYPE_APPLE,
    DEVICE_TYPE_ANDROID,
    DWELL_SHORT_MAX,
    DWELL_MEDIUM_MAX,
)


def stitching_evaluation(
    raw_sessions: pd.DataFrame,
    l1_sessions: pd.DataFrame,
    l2_sessions: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Raw -> Level 1 -> Level 2 3-stage comparison report.
    Returns dict with stage-by-stage metrics.
    """
    def _stage_metrics(sdf: pd.DataFrame, label: str) -> Dict[str, Any]:
        if sdf.empty:
            return {f"{label}_sessions": 0}
        dwell = sdf["dwell_seconds"] if "dwell_seconds" in sdf.columns else pd.Series(dtype=float)
        n = len(sdf)
        apple_n = int((sdf["device_type"] == DEVICE_TYPE_APPLE).sum()) if "device_type" in sdf.columns else 0
        android_n = int((sdf["device_type"] == DEVICE_TYPE_ANDROID).sum()) if "device_type" in sdf.columns else 0

        short = int((dwell < DWELL_SHORT_MAX).sum()) if not dwell.empty else 0
        medium = int(((dwell >= DWELL_SHORT_MAX) & (dwell < DWELL_MEDIUM_MAX)).sum()) if not dwell.empty else 0
        long_ = int((dwell >= DWELL_MEDIUM_MAX).sum()) if not dwell.empty else 0

        return {
            f"{label}_sessions": n,
            f"{label}_dwell_mean": round(float(dwell.mean()), 1) if not dwell.empty else 0.0,
            f"{label}_dwell_median": round(float(dwell.median()), 1) if not dwell.empty else 0.0,
            f"{label}_dwell_p90": round(float(dwell.quantile(0.9)), 1) if not dwell.empty else 0.0,
            f"{label}_apple": apple_n,
            f"{label}_android": android_n,
            f"{label}_short": short,
            f"{label}_medium": medium,
            f"{label}_long": long_,
        }

    raw_m = _stage_metrics(raw_sessions, "raw")
    l1_m = _stage_metrics(l1_sessions, "l1")
    l2_m = _stage_metrics(l2_sessions, "l2")

    raw_n = raw_m.get("raw_sessions", 0)
    l1_n = l1_m.get("l1_sessions", 0)
    l2_n = l2_m.get("l2_sessions", 0)

    result = {**raw_m, **l1_m, **l2_m}
    result["l1_merge_pct"] = round((raw_n - l1_n) / raw_n * 100, 1) if raw_n else 0.0
    result["l2_merge_pct"] = round((l1_n - l2_n) / l1_n * 100, 1) if l1_n else 0.0
    result["total_merge_pct"] = round((raw_n - l2_n) / raw_n * 100, 1) if raw_n else 0.0

    raw_dwell = raw_m.get("raw_dwell_mean", 0)
    l2_dwell = l2_m.get("l2_dwell_mean", 0)
    result["dwell_improvement_pct"] = (
        round((l2_dwell - raw_dwell) / raw_dwell * 100, 1) if raw_dwell else 0.0
    )

    return result


def stitching_daily_summary(
    raw_sessions: pd.DataFrame,
    l1_sessions: pd.DataFrame,
    l2_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-date Raw -> L1 -> L2 comparison table.
    """
    def _daily_agg(sdf: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if sdf.empty or "date" not in sdf.columns:
            return pd.DataFrame()
        grp = sdf.groupby("date").agg(
            sessions=("mac_address", "count"),
            dwell_mean=("dwell_seconds", "mean"),
            dwell_median=("dwell_seconds", "median"),
        ).reset_index()
        grp.columns = ["date", f"{prefix}_sessions", f"{prefix}_dwell_mean", f"{prefix}_dwell_median"]
        grp[f"{prefix}_dwell_mean"] = grp[f"{prefix}_dwell_mean"].round(1)
        grp[f"{prefix}_dwell_median"] = grp[f"{prefix}_dwell_median"].round(1)
        return grp

    raw_d = _daily_agg(raw_sessions, "raw")
    l1_d = _daily_agg(l1_sessions, "l1")
    l2_d = _daily_agg(l2_sessions, "l2")

    if raw_d.empty:
        return pd.DataFrame()

    result = raw_d.copy()
    if not l1_d.empty:
        result = result.merge(l1_d, on="date", how="left")
    if not l2_d.empty:
        result = result.merge(l2_d, on="date", how="left")

    return result
