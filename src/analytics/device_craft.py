"""
Apple vs Android 비중 (Device Craftsmanship).
세션/방문객 기준 디바이스 비율.
"""
from typing import Dict, Optional

import pandas as pd

from src.config.constants import DEVICE_TYPE_APPLE, DEVICE_TYPE_ANDROID


def device_mix_summary(device_mix: pd.DataFrame) -> Dict[str, float]:
    """
    device_mix: columns [date, device_type, count]
    Returns: apple_ratio, android_ratio, apple_count, android_count
    """
    if device_mix.empty or "device_type" not in device_mix.columns:
        return {"apple_ratio": 0.0, "android_ratio": 0.0, "apple_count": 0, "android_count": 0}
    total = device_mix["count"].sum()
    apple = device_mix[device_mix["device_type"] == DEVICE_TYPE_APPLE]["count"].sum()
    android = device_mix[device_mix["device_type"] == DEVICE_TYPE_ANDROID]["count"].sum()
    return {
        "apple_ratio": apple / total if total else 0.0,
        "android_ratio": android / total if total else 0.0,
        "apple_count": int(apple),
        "android_count": int(android),
        "total": int(total),
    }


def device_type_name(dt: int) -> str:
    if dt == DEVICE_TYPE_APPLE:
        return "Apple"
    if dt == DEVICE_TYPE_ANDROID:
        return "Android"
    return "Other"


def device_mix_by_date(device_mix: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date iOS / Android share (percent).
    Columns: date, ios_pct, android_pct, ios_count, android_count
    """
    if device_mix.empty or "date" not in device_mix.columns or "device_type" not in device_mix.columns:
        return pd.DataFrame(columns=["date", "ios_pct", "android_pct", "ios_count", "android_count"])
    rows = []
    for date, grp in device_mix.groupby("date"):
        apple = grp[grp["device_type"] == DEVICE_TYPE_APPLE]["count"].sum()
        android = grp[grp["device_type"] == DEVICE_TYPE_ANDROID]["count"].sum()
        total = apple + android
        if total:
            rows.append({
                "date": date,
                "ios_pct": round(100 * apple / total, 1),
                "android_pct": round(100 * android / total, 1),
                "ios_count": int(apple),
                "android_count": int(android),
            })
        else:
            rows.append({
                "date": date,
                "ios_pct": 0.0,
                "android_pct": 0.0,
                "ios_count": 0,
                "android_count": 0,
            })
    return pd.DataFrame(rows)
