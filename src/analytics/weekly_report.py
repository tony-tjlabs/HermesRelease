"""
Weekly report data aggregation and next-week prediction.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd


def get_last_two_weeks(daily_stats: pd.DataFrame, days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split last 2 weeks into this_week / prev_week.
    Returns (this_week_df, prev_week_df).
    """
    df = daily_stats.sort_values("date").tail(days * 2)
    if len(df) < days:
        return pd.DataFrame(), pd.DataFrame()
    return df.tail(days), df.head(days)


def predict_next_week(
    daily_stats: pd.DataFrame,
    report_end_date: str,
    forecast_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Predict next week from report_end_date + 1 through report_end_date + 7.
    report_end_date: last day of report period (e.g. "2026-02-18").
    forecast_df: from fetch_weather_forecast (may be filtered to 7 days).
    Returns list of dicts with date, weather, day_type, floating_mean, floating_std, quality_cvr_mean, etc.
    """
    if daily_stats.empty:
        return []
    from src.analytics.day_type import add_day_type_to_daily_stats, get_day_type

    try:
        end_d = datetime.strptime(report_end_date, "%Y-%m-%d").date()
    except ValueError:
        return []
    next_start = end_d + timedelta(days=1)
    next_week_dates = [next_start + timedelta(days=i) for i in range(7)]
    next_week_strs = [d.strftime("%Y-%m-%d") for d in next_week_dates]

    # Build forecast rows for exactly these 7 dates (fallback if API missed a day)
    fallback_row = {"weather": "Unknown", "temp_max": None, "temp_min": None}
    forecast_by_date: Dict[str, Dict] = {}
    if not forecast_df.empty and "date" in forecast_df.columns:
        next_week_set = set(next_week_strs)
        for rec in forecast_df.to_dict("records"):
            d = str(rec.get("date", ""))[:10]
            if d in next_week_set:
                w = rec.get("weather")
                if w is None or (isinstance(w, str) and len(w) > 20):
                    w = "Unknown"
                else:
                    w = str(w).strip() or "Unknown"
                forecast_by_date[d] = {
                    "weather": w,
                    "temp_max": rec.get("temp_max"),
                    "temp_min": rec.get("temp_min"),
                }
    for d in next_week_strs:
        if d not in forecast_by_date:
            forecast_by_date[d] = fallback_row.copy()

    df = add_day_type_to_daily_stats(daily_stats.copy())
    if "weather" not in df.columns:
        df["weather"] = "Unknown"
    qc = "quality_cvr" if "quality_cvr" in df.columns else "conversion_rate"
    qv = "quality_visitor_count" if "quality_visitor_count" in df.columns else "visitor_count"

    grp = df.groupby(["weather", "day_type"])
    lookup = grp.agg(
        floating_mean=("floating_unique", "mean"),
        floating_std=("floating_unique", "std"),
        qc_mean=(qc, "mean"),
        qv_mean=(qv, "mean"),
    ).reset_index()
    lookup["floating_std"] = lookup["floating_std"].fillna(0)
    overall_fp = df["floating_unique"].mean()
    overall_qc = df[qc].mean()
    overall_qv = df[qv].mean()

    result = []
    for i, date_str in enumerate(next_week_strs):
        info = forecast_by_date.get(date_str, fallback_row)
        weather = str(info.get("weather", "Unknown")).strip() or "Unknown"
        if not weather or len(weather) > 50:
            weather = "Unknown"
        day_type = get_day_type(date_str)
        match = lookup[(lookup["weather"] == weather) & (lookup["day_type"] == day_type)]
        if match.empty:
            match = lookup[lookup["weather"] == weather]
        if match.empty:
            match = lookup[lookup["day_type"] == day_type]
        if match.empty:
            fp_mean, fp_std = overall_fp, 0.0
            qc_mean, qv_mean = overall_qc, overall_qv
        else:
            m = match.iloc[0]
            fp_mean = m["floating_mean"]
            fp_std = float(m["floating_std"]) if pd.notna(m.get("floating_std")) else 0.0
            qc_mean = m.get("qc_mean", overall_qc)
            qv_mean = m.get("qv_mean", overall_qv)
        d_obj = next_week_dates[i]
        result.append({
            "date": date_str,
            "date_obj": d_obj,
            "weather": weather,
            "day_type": day_type,
            "floating_mean": round(float(fp_mean), 1),
            "floating_std": round(float(fp_std), 1),
            "quality_cvr_mean": round(float(qc_mean), 2),
            "quality_visitor_mean": round(float(qv_mean), 1),
            "temp_max": info.get("temp_max"),
            "temp_min": info.get("temp_min"),
            "data_sufficient": True,
        })
    return result
