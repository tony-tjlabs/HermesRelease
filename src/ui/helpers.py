"""
Hermes UI helpers — dark theme utilities, shared rendering functions.

Inspired by TheHyundaiSeoul: metric-card, section-header, ai-comment,
make_plotly_layout for consistent dark chart styling.
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.day_type import add_day_type_to_daily_stats
from src.config.constants import TIME_UNIT_SECONDS

# -- Dark Theme Palette ----------------------------------------------------

BG_COLOR = "#0E1117"
CARD_BG = "#1E2130"
CARD_BORDER = "#2d3456"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#ccd6f6"
TEXT_MUTED = "#a8b2d1"
TEXT_DIMMED = "#8892b0"
GRID_COLOR = "#1a2035"
ACCENT_TEAL = "#64ffda"

# Legacy palette names (used by some modules)
# Brightened from #0f172a for visibility on dark (#0E1117) background
DEEP_NAVY = "#4A90D9"
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"
TEAL = "#0d9488"
ROSE = "#e11d48"


# -- Rendering Utilities ---------------------------------------------------

def render_metric_card(label: str, value: str, sub: str = "") -> None:
    """Render a TheHyundaiSeoul-style metric card using custom HTML."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(text: str) -> None:
    """Render a section header with bottom border."""
    st.markdown(
        f'<div class="section-header">{text}</div>',
        unsafe_allow_html=True,
    )


def render_ai_comment(title: str, text: str) -> None:
    """Render an AI insight comment box with teal accent."""
    if text:
        st.markdown(
            f'<div class="ai-comment">'
            f'<span style="color:{ACCENT_TEAL}; font-weight:600;">{title}</span><br>'
            f'{text}</div>',
            unsafe_allow_html=True,
        )


def make_plotly_layout(title: str = "", height: int = 400) -> dict:
    """
    Common Plotly layout for dark theme.

    Matches TheHyundaiSeoul's make_plotly_layout exactly:
    dark background, muted grid, horizontal legend.
    """
    return dict(
        title=dict(text=title, font=dict(size=14, color=TEXT_SECONDARY)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_SECONDARY, size=11),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )


# -- Data Utilities --------------------------------------------------------

def ensure_day_type(loader: CacheLoader) -> pd.DataFrame:
    daily_stats = loader.get_daily_stats()
    if daily_stats.empty:
        return daily_stats
    return add_day_type_to_daily_stats(daily_stats)


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
    """Compute per-minute occupancy from session data."""
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
    """Wrapper for st.metric (backward compat)."""
    st.metric(label, value, delta, delta_color=delta_color)


def info(text: str, label: str = "About this") -> None:
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


# -- Date Filter Helpers ---------------------------------------------------

def filter_by_date_range(
    df: pd.DataFrame,
    date_range: tuple | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if df.empty or date_range is None:
        return df
    if date_col not in df.columns:
        return df
    if date_range[0] == "last":
        n_days = date_range[1]
        dates = pd.to_datetime(df[date_col])
        max_date = dates.max()
        cutoff = max_date - timedelta(days=n_days - 1)
        return df[dates >= cutoff].copy()
    else:
        start_str, end_str = date_range
        dates = pd.to_datetime(df[date_col])
        mask = (dates >= pd.to_datetime(start_str)) & (dates <= pd.to_datetime(end_str))
        return df[mask].copy()


def get_date_range_description(
    df: pd.DataFrame,
    date_range: tuple | None,
    date_col: str = "date",
) -> str:
    """Get human-readable description of the date range."""
    if df.empty:
        return "No data"
    if date_col not in df.columns:
        return "N/A"
    dates = pd.to_datetime(df[date_col])
    min_date = dates.min().strftime("%Y-%m-%d")
    max_date = dates.max().strftime("%Y-%m-%d")
    n_days = len(df[date_col].unique())
    if date_range is not None and date_range[0] == "last":
        return f"Last {date_range[1]} days ({min_date} ~ {max_date})"
    elif date_range is not None:
        return f"{min_date} ~ {max_date}"
    else:
        return f"All data ({n_days} days: {min_date} ~ {max_date})"


def compute_week_over_week(
    current_value: float,
    previous_value: float,
) -> tuple[float, str]:
    """Compute week-over-week change."""
    if previous_value == 0:
        return 0.0, "N/A"
    delta_pct = (current_value - previous_value) / previous_value * 100
    delta_str = f"{delta_pct:+.1f}%"
    return delta_pct, delta_str
