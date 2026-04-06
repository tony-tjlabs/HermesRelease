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

def _compute_subhour_stats(
    sessions: pd.DataFrame,
    daily_timeseries: pd.DataFrame,
    daily_hourly: pd.DataFrame,
    date_str: str,
    bin_minutes: int = 30,
    daily_fp: int = 0,
) -> pd.DataFrame:
    """
    Compute sub-hourly binned stats.

    FP (floating_count):
      Daily FP (from daily_stats, including Android correction) is distributed
      proportionally across sub-bins using the hourly FP shape.
      This ensures: sum of all bin FPs ≈ daily FP.

    Visitors (visitor_count):
      Unique visitor MACs from sessions, binned by entry_time_index.

    CVR = Visitors / FP → consistent with daily CVR.
    """
    n_bins = 24 * 60 // bin_minutes
    bins_per_hour = 60 // bin_minutes
    bins = list(range(n_bins))
    bin_labels = [
        f"{(b * bin_minutes) // 60:02d}:{(b * bin_minutes) % 60:02d}"
        for b in bins
    ]

    # ── FP: distribute daily_fp proportionally using hourly shape ──────────
    hourly_fp = {}
    h_day = daily_hourly[daily_hourly["date"] == date_str] if not daily_hourly.empty and "date" in daily_hourly.columns else pd.DataFrame()
    if not h_day.empty and "hour" in h_day.columns and "floating_count" in h_day.columns:
        hourly_fp = dict(zip(h_day["hour"].astype(int), h_day["floating_count"].astype(int)))

    # Step 2: Get timeseries shape for proportional distribution
    ts_day = daily_timeseries[daily_timeseries["date"] == date_str] if not daily_timeseries.empty and "date" in daily_timeseries.columns else pd.DataFrame()
    ts_by_bin = {}
    if not ts_day.empty and "minute" in ts_day.columns and "floating_count" in ts_day.columns:
        ts_day = ts_day.copy()
        ts_day["bin_n"] = ts_day["minute"] // bin_minutes
        ts_by_bin = ts_day.groupby("bin_n")["floating_count"].sum().to_dict()

    # Step 3: Distribute daily_fp across all bins proportionally
    # Use hourly shape (further split by timeseries within each hour)
    # This ensures: sum(all bins) ≈ daily_fp
    fp_by_bin = {b: 0 for b in bins}

    # Build weights for all bins using timeseries shape
    all_weights = []
    for b in bins:
        all_weights.append(max(ts_by_bin.get(b, 0), 0))
    total_weight = sum(all_weights)

    use_daily_fp = daily_fp if daily_fp > 0 else sum(hourly_fp.values())

    if total_weight > 0 and use_daily_fp > 0:
        for b in bins:
            fp_by_bin[b] = int(round(use_daily_fp * all_weights[b] / total_weight))
    elif use_daily_fp > 0:
        # Even distribution if no timeseries
        per_bin = use_daily_fp // n_bins
        for b in bins:
            fp_by_bin[b] = per_bin

    # ── Visitors: unique MACs from sessions per bin ────────────────────────
    v_by_bin = {b: 0 for b in bins}
    dwell_by_bin = {b: 0.0 for b in bins}
    ios_by_bin = {b: 0 for b in bins}
    android_by_bin = {b: 0 for b in bins}

    sess_day = sessions[sessions["date"] == date_str] if not sessions.empty and "date" in sessions.columns else pd.DataFrame()
    if not sess_day.empty and "entry_time_index" in sess_day.columns:
        sess_day = sess_day.copy()
        sess_day["bin_n"] = (
            sess_day["entry_time_index"] * TIME_UNIT_SECONDS // 60 // bin_minutes
        ).astype(int).clip(0, n_bins - 1)

        # Unique visitor MACs per bin
        v_counts = sess_day.groupby("bin_n")["mac_address"].nunique()
        for b_idx, cnt in v_counts.items():
            if 0 <= b_idx < n_bins:
                v_by_bin[int(b_idx)] = int(cnt)

        if "dwell_seconds" in sess_day.columns:
            dwell_means = sess_day.groupby("bin_n")["dwell_seconds"].mean()
            for b_idx, val in dwell_means.items():
                if 0 <= b_idx < n_bins:
                    dwell_by_bin[int(b_idx)] = float(val)

        if "device_type" in sess_day.columns:
            dev_counts = sess_day.groupby(["bin_n", "device_type"]).size().unstack(fill_value=0)
            for b_idx in dev_counts.index:
                if 0 <= b_idx < n_bins:
                    apple = int(dev_counts.at[b_idx, DEVICE_TYPE_APPLE]) if DEVICE_TYPE_APPLE in dev_counts.columns else 0
                    android = int(dev_counts.at[b_idx, DEVICE_TYPE_ANDROID]) if DEVICE_TYPE_ANDROID in dev_counts.columns else 0
                    ios_by_bin[int(b_idx)] = apple
                    android_by_bin[int(b_idx)] = android

    # ── Assemble — guarantee FP >= Visitors ────────────────────────────────
    rows = []
    for b in bins:
        v = v_by_bin[b]
        fp = max(fp_by_bin[b], v)  # safety: FP can never be less than visitors
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
    daily_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute time-binned stats across one or more dates, then average.

    Parameters
    ----------
    daily_hourly : Cached hourly data (used for 60-min bins)
    sessions : Session-level data (used for all bin sizes)
    daily_timeseries : Minute-level timeseries (used for sub-hour bins)
    dates : List of date strings ('YYYY-MM-DD')
    bin_minutes : 5, 10, 15, 30, or 60
    daily_stats : daily_stats DataFrame (for daily_fp lookup in sub-hour bins)

    Returns
    -------
    DataFrame[bin_label, floating_count, visitor_count, cvr_pct,
              dwell_sec_mean, ios_pct, android_pct]
    """
    if not dates:
        return pd.DataFrame()

    frames = []
    for date_str in dates:
        if bin_minutes == 60:
            # 60min: reuse existing hourly function (already validated)
            df = hourly_stats_for_date(daily_hourly, sessions, date_str)
            df["bin_label"] = df["hour"].apply(lambda h: f"{h:02d}:00")
            df["bin_idx"] = df["hour"]
        else:
            # Sub-hourly: distribute daily FP proportionally across bins
            d_fp = 0
            if daily_stats is not None and not daily_stats.empty:
                ds_row = daily_stats[daily_stats["date"].astype(str) == date_str]
                if not ds_row.empty:
                    d_fp = int(ds_row.iloc[0].get("floating_unique", 0))
            df = _compute_subhour_stats(
                sessions, daily_timeseries, daily_hourly,
                date_str, bin_minutes=bin_minutes, daily_fp=d_fp,
            )
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
    if bin_minutes == 60:
        avg_df["bin_label"] = avg_df["bin_idx"].apply(lambda h: f"{int(h):02d}:00")
    elif bin_minutes == 30:
        avg_df["bin_label"] = avg_df["bin_idx"].apply(lambda b: f"{b // 2:02d}:{(b % 2) * 30:02d}")
    else:
        avg_df["bin_label"] = avg_df["bin_idx"].apply(
            lambda b: f"{(b * bin_minutes) // 60:02d}:{(b * bin_minutes) % 60:02d}"
        )

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
