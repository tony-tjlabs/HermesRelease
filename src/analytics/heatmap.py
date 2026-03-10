"""
요일×시간별 혼잡/트래픽 히트맵 데이터 생성.
Hermes Flow Heatmap.
"""
from typing import Optional

import pandas as pd


def build_weekday_hour_heatmap(
    daily_hourly: pd.DataFrame,
    daily_stats: pd.DataFrame,
    value_column: str = "visitor_count",
) -> pd.DataFrame:
    """
    요일(0~6) × hour(0~23) 별 value_column 평균.
    daily_hourly에 date, hour 필요. daily_stats에 date, weekday 필요.
    """
    if daily_hourly.empty or daily_stats.empty:
        return pd.DataFrame()
    merge = daily_hourly.merge(
        daily_stats[["date", "weekday"]].drop_duplicates(),
        on="date",
        how="left",
    )
    if value_column not in merge.columns:
        return pd.DataFrame()
    agg = merge.groupby(["weekday", "hour"], as_index=False)[value_column].mean()
    agg.columns = ["weekday", "hour", "value"]
    return agg


def pivot_heatmap(agg: pd.DataFrame) -> pd.DataFrame:
    """weekday × hour 피벗 (행=weekday, 열=hour)."""
    if agg.empty:
        return pd.DataFrame()
    return agg.pivot(index="weekday", columns="hour", values="value").fillna(0)
