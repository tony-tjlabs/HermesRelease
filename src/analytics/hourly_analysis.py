"""
Hourly stats: 1h or 30min bins, single or multi-date.
Used for "select date(s) → show intraday charts".
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.constants import TIME_UNIT_SECONDS, DEVICE_TYPE_APPLE, DEVICE_TYPE_ANDROID, SECONDS_PER_HOUR


def hourly_stats_for_date(
    daily_hourly: pd.DataFrame,
    sessions_all: pd.DataFrame,
    date_str: str,
) -> pd.DataFrame:
    """
    Build one row per hour (0–23) for the given date.
    Columns: hour, floating_count, visitor_count, cvr_pct, dwell_sec_mean, ios_pct, android_pct
    """
    hours = list(range(24))
    # All hourly metrics read directly from the single-source cache (daily_hourly.parquet).
    # CVR is also taken from cache — avoids recomputing from floating_count/visitor_count
    # and ensures Phase 3 is identical to the source used in Phase 1.
    sub = daily_hourly[daily_hourly["date"] == date_str]
    if sub.empty:
        fp_map = {h: 0 for h in hours}
        v_map = {h: 0 for h in hours}
        cvr_map = {h: 0.0 for h in hours}
    else:
        sub_idx = sub.set_index("hour")
        fp_map = sub_idx["floating_count"].to_dict()
        v_map = sub_idx["visitor_count"].to_dict()
        # Use pre-computed CVR from cache (same source as daily_stats CVR)
        cvr_map = sub_idx["conversion_rate"].to_dict() if "conversion_rate" in sub_idx.columns else {}
        for h in hours:
            fp_map.setdefault(h, 0)
            v_map.setdefault(h, 0)
            cvr_map.setdefault(h, 0.0)

    # Optional: floating by device (floating_apple, floating_android) for iOS/Android % of floating pop
    has_floating_device = not sub.empty and "floating_apple" in sub.columns and "floating_android" in sub.columns
    if has_floating_device:
        fa_map = sub.set_index("hour")["floating_apple"].to_dict()
        fb_map = sub.set_index("hour")["floating_android"].to_dict()
    else:
        fa_map = fb_map = {}

    rows = []
    for h in hours:
        fp = fp_map.get(h, 0)
        v = v_map.get(h, 0)
        cvr = cvr_map.get(h, 0.0)
        row = {
            "hour": h,
            "floating_count": fp,
            "visitor_count": v,
            "cvr_pct": round(cvr, 1),
        }
        if has_floating_device:
            fa = fa_map.get(h, 0)
            fb = fb_map.get(h, 0)
            total_f = fa + fb
            row["floating_ios_pct"] = round(100 * fa / total_f, 1) if total_f else 0.0
            row["floating_android_pct"] = round(100 * fb / total_f, 1) if total_f else 0.0
        rows.append(row)

    hourly_df = pd.DataFrame(rows)

    # Dwell and device mix from sessions for this date
    sessions_date = sessions_all[sessions_all["date"] == date_str] if "date" in sessions_all.columns and not sessions_all.empty else pd.DataFrame()
    if not sessions_date.empty and "entry_time_index" in sessions_date.columns:
        sessions_date = sessions_date.copy()
        # Use entry_time_index: consistent with ble_engine and runner — "when did visitor arrive?"
        sessions_date["hour"] = (sessions_date["entry_time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int).clip(0, 23)
    elif not sessions_date.empty and "hour" not in sessions_date.columns:
        sessions_date["hour"] = 0

    if not sessions_date.empty:
        dwell_by_h = sessions_date.groupby("hour")["dwell_seconds"].mean().reindex(hours).fillna(0)
        hourly_df["dwell_sec_mean"] = dwell_by_h.values
        # iOS / Android per hour
        device_by_h = sessions_date.groupby(["hour", "device_type"]).size().unstack(fill_value=0)
        device_by_h = device_by_h.reindex(hours).fillna(0).astype(int)
        device_by_h = device_by_h.reindex(columns=[DEVICE_TYPE_APPLE, DEVICE_TYPE_ANDROID], fill_value=0)
        ios_pct_list = []
        android_pct_list = []
        for h in hours:
            row = device_by_h.loc[h]
            apple = int(row.get(DEVICE_TYPE_APPLE, 0))
            android = int(row.get(DEVICE_TYPE_ANDROID, 0))
            total = apple + android
            if total:
                ios_pct_list.append(round(100 * apple / total, 1))
                android_pct_list.append(round(100 * android / total, 1))
            else:
                ios_pct_list.append(0.0)
                android_pct_list.append(0.0)
        hourly_df["ios_pct"] = ios_pct_list
        hourly_df["android_pct"] = android_pct_list
    else:
        hourly_df["dwell_sec_mean"] = 0.0
        hourly_df["ios_pct"] = 0.0
        hourly_df["android_pct"] = 0.0

    if not has_floating_device:
        hourly_df["floating_ios_pct"] = 0.0
        hourly_df["floating_android_pct"] = 0.0

    return hourly_df


# ── 30-min bin helpers ────────────────────────────────────────────────────────

def _compute_30min_stats(
    sessions: pd.DataFrame,
    daily_timeseries: pd.DataFrame,
    date_str: str,
) -> pd.DataFrame:
    """
    Compute 30-minute-binned stats for a single date from sessions + timeseries.

    Returns DataFrame with 48 rows (bins 00:00–23:30).
    Columns: bin_label, floating_count, visitor_count, cvr_pct,
             dwell_sec_mean, ios_pct, android_pct
    """
    bins = list(range(48))
    bin_labels = [f"{b // 2:02d}:{(b % 2) * 30:02d}" for b in bins]

    # ── Floating Population from timeseries (minute-level) ─────────────────
    fp_by_bin = {b: 0 for b in bins}
    ts_day = daily_timeseries[daily_timeseries["date"] == date_str] if not daily_timeseries.empty and "date" in daily_timeseries.columns else pd.DataFrame()
    if not ts_day.empty and "minute" in ts_day.columns and "floating_count" in ts_day.columns:
        ts_day = ts_day.copy()
        ts_day["bin30"] = ts_day["minute"] // 30
        fp_agg = ts_day.groupby("bin30")["floating_count"].max()
        for b_idx, val in fp_agg.items():
            if 0 <= b_idx < 48:
                fp_by_bin[int(b_idx)] = int(val)

    # ── Visitors, dwell, device from sessions ──────────────────────────────
    v_by_bin = {b: 0 for b in bins}
    dwell_by_bin = {b: 0.0 for b in bins}
    ios_by_bin = {b: 0 for b in bins}
    android_by_bin = {b: 0 for b in bins}

    sess_day = sessions[sessions["date"] == date_str] if not sessions.empty and "date" in sessions.columns else pd.DataFrame()
    if not sess_day.empty and "entry_time_index" in sess_day.columns:
        sess_day = sess_day.copy()
        # entry_time_index * 10sec → minutes → 30min bin
        sess_day["bin30"] = (sess_day["entry_time_index"] * TIME_UNIT_SECONDS // 60 // 30).astype(int).clip(0, 47)

        v_counts = sess_day.groupby("bin30").size()
        for b_idx, cnt in v_counts.items():
            if 0 <= b_idx < 48:
                v_by_bin[int(b_idx)] = int(cnt)

        if "dwell_seconds" in sess_day.columns:
            dwell_means = sess_day.groupby("bin30")["dwell_seconds"].mean()
            for b_idx, val in dwell_means.items():
                if 0 <= b_idx < 48:
                    dwell_by_bin[int(b_idx)] = float(val)

        if "device_type" in sess_day.columns:
            dev_counts = sess_day.groupby(["bin30", "device_type"]).size().unstack(fill_value=0)
            for b_idx in dev_counts.index:
                if 0 <= b_idx < 48:
                    apple = int(dev_counts.at[b_idx, DEVICE_TYPE_APPLE]) if DEVICE_TYPE_APPLE in dev_counts.columns else 0
                    android = int(dev_counts.at[b_idx, DEVICE_TYPE_ANDROID]) if DEVICE_TYPE_ANDROID in dev_counts.columns else 0
                    ios_by_bin[int(b_idx)] = apple
                    android_by_bin[int(b_idx)] = android

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    rows = []
    for b in bins:
        fp = fp_by_bin[b]
        v = v_by_bin[b]
        cvr = round(v / fp * 100, 1) if fp > 0 else 0.0
        total_dev = ios_by_bin[b] + android_by_bin[b]
        rows.append({
            "bin_label": bin_labels[b],
            "bin_idx": b,
            "floating_count": fp,
            "visitor_count": v,
            "cvr_pct": cvr,
            "dwell_sec_mean": round(dwell_by_bin[b], 1),
            "ios_pct": round(100 * ios_by_bin[b] / total_dev, 1) if total_dev else 0.0,
            "android_pct": round(100 * android_by_bin[b] / total_dev, 1) if total_dev else 0.0,
        })
    return pd.DataFrame(rows)


def hourly_stats_flexible(
    daily_hourly: pd.DataFrame,
    sessions: pd.DataFrame,
    daily_timeseries: pd.DataFrame,
    dates: List[str],
    bin_minutes: int = 60,
) -> pd.DataFrame:
    """
    Compute time-binned stats across one or more dates, then average.

    Parameters
    ----------
    daily_hourly : Cached hourly data (used for 60-min bins)
    sessions : Session-level data (used for all bin sizes)
    daily_timeseries : Minute-level timeseries (used for 30-min FP)
    dates : List of date strings ('YYYY-MM-DD')
    bin_minutes : 30 or 60

    Returns
    -------
    DataFrame[bin_label, floating_count, visitor_count, cvr_pct,
              dwell_sec_mean, ios_pct, android_pct]
    """
    if not dates:
        return pd.DataFrame()

    frames = []
    for date_str in dates:
        if bin_minutes == 30:
            df = _compute_30min_stats(sessions, daily_timeseries, date_str)
        else:
            # 60min: reuse existing function, add bin_label
            df = hourly_stats_for_date(daily_hourly, sessions, date_str)
            df["bin_label"] = df["hour"].apply(lambda h: f"{h:02d}:00")
            df["bin_idx"] = df["hour"]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0]

    # Average across dates
    combined = pd.concat(frames, ignore_index=True)
    num_cols = ["floating_count", "visitor_count", "cvr_pct", "dwell_sec_mean", "ios_pct", "android_pct"]
    existing_cols = [c for c in num_cols if c in combined.columns]
    avg_df = combined.groupby("bin_idx", as_index=False)[existing_cols].mean()

    # Round
    for c in existing_cols:
        avg_df[c] = avg_df[c].round(1)
    avg_df["floating_count"] = avg_df["floating_count"].round(0).astype(int)
    avg_df["visitor_count"] = avg_df["visitor_count"].round(0).astype(int)

    # Recompute CVR from averaged FP/visitors
    safe_fp = avg_df["floating_count"].replace(0, 1)
    avg_df["cvr_pct"] = (avg_df["visitor_count"] / safe_fp * 100).round(1)
    avg_df.loc[avg_df["floating_count"] == 0, "cvr_pct"] = 0.0

    # Add bin_label
    if bin_minutes == 30:
        avg_df["bin_label"] = avg_df["bin_idx"].apply(lambda b: f"{b // 2:02d}:{(b % 2) * 30:02d}")
    else:
        avg_df["bin_label"] = avg_df["bin_idx"].apply(lambda h: f"{int(h):02d}:00")

    return avg_df.sort_values("bin_idx").reset_index(drop=True)


def identify_peak_hours(
    hourly_df: pd.DataFrame,
    metric: str = "visitor_count",
    top_n: int = 3,
) -> List[Dict]:
    """
    Return top N peak time bins sorted by metric descending.

    Returns
    -------
    List of dicts: [{bin_label, value, rank}, ...]
    """
    if hourly_df.empty or metric not in hourly_df.columns:
        return []

    top = hourly_df.nlargest(top_n, metric)
    results = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        results.append({
            "bin_label": row.get("bin_label", ""),
            "value": row[metric],
            "rank": rank,
            "fp": row.get("floating_count", 0),
            "visitors": row.get("visitor_count", 0),
            "cvr": row.get("cvr_pct", 0.0),
            "dwell": row.get("dwell_sec_mean", 0.0),
        })
    return results
