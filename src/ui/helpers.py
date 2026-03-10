"""
Hermes UI helpers — shared functions for all page modules.
"""
import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.day_type import add_day_type_to_daily_stats
from src.ui.chart_theme import apply_theme
from src.config.constants import TIME_UNIT_SECONDS

# Palette
DEEP_NAVY = "#0f172a"
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"
TEAL = "#0d9488"
ROSE = "#e11d48"

_WEATHER_ICON = {
    "Sunny": "☀️ Sunny",
    "Rain": "🌧 Rain",
    "Snow": "❄️ Snow",
    "Unknown": "— Unknown",
}


def ensure_day_type(loader: CacheLoader) -> pd.DataFrame:
    daily_stats = loader.get_daily_stats()
    if daily_stats.empty:
        return daily_stats
    return add_day_type_to_daily_stats(daily_stats)


def info(text: str, label: str = "📖 About this") -> None:
    """Render a collapsed expander with explanatory text."""
    with st.expander(label, expanded=False):
        st.markdown(text)


def weather_color(weather: str) -> str:
    return {
        "Sunny": GOLD,
        "Rain": SLATE_GRAY,
        "Snow": "#93c5fd",
        "Unknown": "#d1d5db",
    }.get(weather, SLATE_GRAY)


def has_api_key() -> bool:
    try:
        st_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        st_key = ""
    return bool(st_key) or bool(os.environ.get("ANTHROPIC_API_KEY", ""))


def build_occupancy_timeseries(
    sessions: pd.DataFrame,
    date_str: str,
    bin_minutes: int = 1,
) -> pd.DataFrame:
    """
    세션 DataFrame에서 특정 날짜의 분 단위 동시 재실 인원(occupancy)을 계산한다.
    """
    if sessions.empty:
        return pd.DataFrame(columns=["time_label", "occupancy"])

    df = sessions[sessions["date"] == date_str].copy() if "date" in sessions.columns else sessions.copy()
    if df.empty:
        return pd.DataFrame(columns=["time_label", "occupancy"])

    slots_per_bin = bin_minutes * 60 // TIME_UNIT_SECONDS
    total_slots = 24 * 3600 // TIME_UNIT_SECONDS
    bins = range(0, total_slots, slots_per_bin)

    entry = df["entry_time_index"].to_numpy()
    exit_ = df["exit_time_index"].to_numpy()

    records = []
    for b in bins:
        b_end = b + slots_per_bin - 1
        occ = int(((entry <= b_end) & (exit_ >= b)).sum())
        hour = (b * TIME_UNIT_SECONDS) // 3600
        minute = ((b * TIME_UNIT_SECONDS) % 3600) // 60
        records.append({"time_label": f"{hour:02d}:{minute:02d}", "occupancy": occ})

    return pd.DataFrame(records)


def metric_card(label: str, value: str, delta: str = "", delta_color: str = "normal") -> None:
    """Wrapper for st.metric with consistent styling."""
    st.metric(label, value, delta, delta_color=delta_color)
