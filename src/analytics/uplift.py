"""
평일 대비 휴일/특정일 uplift 계산.
Cause–Effect: 기준점(평일 평균) 대비 특정 시점 증감률.
"""
from typing import Dict, List, Optional

import pandas as pd


def compute_baseline_weekday(daily_stats: pd.DataFrame) -> Dict[str, float]:
    """
    평일(weekday)만으로 일별 평균 지표 계산.
    daily_stats에 day_type 필요 (add_day_type_to_daily_stats 적용 후).
    """
    wd = daily_stats[daily_stats["day_type"] == "weekday"]
    if wd.empty:
        return {}
    return {
        "floating_unique_avg": float(wd["floating_unique"].mean()),
        "visitor_count_avg": float(wd["visitor_count"].mean()),
        "conversion_rate_avg": float(wd["conversion_rate"].mean()),
        "dwell_seconds_mean_avg": float(wd["dwell_seconds_mean"].mean()),
    }


def compute_week_over_week(
    daily_stats: pd.DataFrame,
    days_per_week: int = 7,
    fp_col: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    최근 2주 데이터를 이번 주 / 전주로 나누어 평균 및 delta 반환.

    Parameters
    ----------
    fp_col : str, optional
        Floating population column to use. Defaults to "floating_unique".

    Returns
    -------
    {
        "this_week": { "floating_unique", "quality_visitor_count", "quality_cvr", "dwell_median_seconds", ... },
        "prev_week": { ... },
        "delta": { "floating_pct", "quality_visitor_pct", "quality_cvr_pp", "dwell_median" }
    }
    """
    df = daily_stats.sort_values("date").tail(days_per_week * 2)
    if len(df) < days_per_week:
        return {"this_week": {}, "prev_week": {}, "delta": {}}

    this_df = df.tail(days_per_week)
    prev_df = df.head(days_per_week)

    # fp_col 선택: 요청한 컬럼이 없으면 floating_unique로 fallback
    _fp_col = fp_col if (fp_col and fp_col in df.columns) else "floating_unique"
    qv_col = "quality_visitor_count" if "quality_visitor_count" in df.columns else "visitor_count"
    qc_col = "quality_cvr" if "quality_cvr" in df.columns else "conversion_rate"
    dm_col = "dwell_median_seconds" if "dwell_median_seconds" in df.columns else "dwell_seconds_mean"

    def _avg(d: pd.DataFrame, col: str) -> float:
        return float(d[col].mean()) if col in d.columns else 0.0

    tw = {
        "floating_unique": _avg(this_df, _fp_col),
        "quality_visitor_count": _avg(this_df, qv_col),
        "quality_cvr": _avg(this_df, qc_col),
        "dwell_median_seconds": _avg(this_df, dm_col),
    }
    pw = {
        "floating_unique": _avg(prev_df, _fp_col),
        "quality_visitor_count": _avg(prev_df, qv_col),
        "quality_cvr": _avg(prev_df, qc_col),
        "dwell_median_seconds": _avg(prev_df, dm_col),
    }

    def _pct(a: float, b: float) -> float:
        return (a - b) / b * 100.0 if b else 0.0

    delta = {
        "floating_pct": _pct(tw["floating_unique"], pw["floating_unique"]),
        "quality_visitor_pct": _pct(tw["quality_visitor_count"], pw["quality_visitor_count"]),
        "quality_cvr_pp": tw["quality_cvr"] - pw["quality_cvr"],  # percentage point
        "dwell_median": tw["dwell_median_seconds"] - pw["dwell_median_seconds"],
    }
    return {"this_week": tw, "prev_week": pw, "delta": delta}


def compute_uplift(
    daily_stats: pd.DataFrame,
    baseline: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    일별 uplift (대비 기준 대비 증감률 %).
    baseline 없으면 평일 평균으로 계산.
    """
    df = daily_stats.copy()
    if "day_type" not in df.columns:
        from src.analytics.day_type import add_day_type_to_daily_stats
        df = add_day_type_to_daily_stats(df)
    base = baseline or compute_baseline_weekday(df)
    if not base:
        df["uplift_floating"] = 0.0
        df["uplift_visitor"] = 0.0
        df["uplift_cvr"] = 0.0
        df["uplift_dwell"] = 0.0
        return df

    def pct_change(val, avg):
        if avg is None or avg == 0:
            return 0.0
        return (val - avg) / avg * 100.0

    df["uplift_floating"] = df["floating_unique"].map(
        lambda x: pct_change(x, base.get("floating_unique_avg"))
    )
    df["uplift_visitor"] = df["visitor_count"].map(
        lambda x: pct_change(x, base.get("visitor_count_avg"))
    )
    df["uplift_cvr"] = df["conversion_rate"].map(
        lambda x: pct_change(x, base.get("conversion_rate_avg"))
    )
    df["uplift_dwell"] = df["dwell_seconds_mean"].map(
        lambda x: pct_change(x, base.get("dwell_seconds_mean_avg"))
    )
    return df
