"""
체류 시간대별 분포 (Short / Medium / Long Stay).
Dwell Intelligence.
"""
from typing import Dict, List

import pandas as pd

from src.config.constants import DWELL_SHORT_MAX, DWELL_MEDIUM_MAX


def classify_dwell(seconds: float) -> str:
    if seconds < DWELL_SHORT_MAX:
        return "Short (<3min)"
    if seconds < DWELL_MEDIUM_MAX:
        return "Medium (3–10min)"
    return "Long (10min+)"


def dwell_distribution(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    sessions_df: columns [dwell_seconds, ...]
    Returns: segment, count, ratio
    """
    if sessions_df.empty or "dwell_seconds" not in sessions_df.columns:
        return pd.DataFrame(columns=["segment", "count", "ratio"])
    s = sessions_df["dwell_seconds"].map(classify_dwell)
    counts = s.value_counts().reindex(["Short (<3min)", "Medium (3–10min)", "Long (10min+)"])
    counts = counts.fillna(0).astype(int)
    total = counts.sum()
    ratio = (counts / total * 100).round(1) if total else counts * 0
    return pd.DataFrame({
        "segment": counts.index,
        "count": counts.values,
        "ratio": ratio.values,
    })
