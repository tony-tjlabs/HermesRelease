"""
Hermes Dashboard pages — Retail Manager-Focused Dashboard.

Tab 1: Overview 매장현황 — Store status with core metrics
Tab 2: Hourly 시간대분석 — Time-binned analysis (30min / 1hr) + multi-date
Tab 3: Patterns 패턴분석 — Day-type, weather, heatmaps, dwell, anomaly
Tab 4: Report 리포트 — Weekly traffic report + PDF download
"""
from typing import Optional
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.day_type import add_day_type_to_daily_stats, get_day_context, weekday_names_en
from src.analytics.uplift import compute_baseline_weekday, compute_uplift, compute_week_over_week
from src.analytics.heatmap import build_weekday_hour_heatmap, pivot_heatmap
from src.analytics.device_craft import device_mix_summary, device_mix_by_date
from src.analytics.dwell_intelligence import dwell_distribution
from src.analytics.hourly_analysis import hourly_stats_for_date, hourly_stats_flexible, identify_peak_hours
from src.ui.chart_theme import apply_theme
from src.ai import (
    call_claude,
    generate_weekly_report_insight,
    generate_prediction_comment,
    generate_kpi_summary,
    generate_context_comment,
)
from src.analytics.weekly_report import get_last_two_weeks, predict_next_week
from src.data.external_api import fetch_weather_forecast
from src.config.constants import TIME_UNIT_SECONDS
# NOTE: src.report (fpdf2) is imported lazily inside render_report_tab()
# to avoid ImportError on environments where fpdf2 may not be installed.

# Palette
DEEP_NAVY = "#0f172a"
GOLD      = "#c49a3a"
AMBER     = "#d97706"
SLATE_GRAY = "#64748b"

_WEATHER_ICON = {
    "Sunny": "☀️ Sunny", "Rain": "🌧 Rain",
    "Snow": "❄️ Snow",  "Unknown": "— Unknown",
}


# ── Cached analytics wrappers — prevent recomputation on every widget rerun ───

@st.cache_data(show_spinner=False)
def _cached_day_type(daily_stats: pd.DataFrame) -> pd.DataFrame:
    """add_day_type_to_daily_stats is called 4× across tabs — cache it."""
    return add_day_type_to_daily_stats(daily_stats)


@st.cache_data(show_spinner=False)
def _cached_heatmap(
    daily_hourly: pd.DataFrame,
    daily_stats: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """Weekday×hour heatmap — O(dates × hours) join + groupby, expensive on repeat."""
    return build_weekday_hour_heatmap(daily_hourly, daily_stats, col)


@st.cache_data(show_spinner=False)
def _cached_uplift(daily_stats: pd.DataFrame) -> pd.DataFrame:
    return compute_uplift(daily_stats)


@st.cache_data(show_spinner=False)
def _cached_wow(daily_stats: pd.DataFrame, days_per_week: int = 7) -> dict:
    return compute_week_over_week(daily_stats, days_per_week=days_per_week)


@st.cache_data(show_spinner=False)
def _cached_dwell_dist(sessions: pd.DataFrame) -> pd.DataFrame:
    return dwell_distribution(sessions)


@st.cache_data(show_spinner=False)
def _cached_device_mix_summary(device_mix: pd.DataFrame) -> dict:
    return device_mix_summary(device_mix)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_day_type(loader: CacheLoader) -> pd.DataFrame:
    daily_stats = loader.get_daily_stats()
    if daily_stats.empty:
        return daily_stats
    return _cached_day_type(daily_stats)  # use cached computation


def _info(text: str, label: str = "📖 About this") -> None:
    """Render a collapsed expander with explanatory text."""
    with st.expander(label, expanded=False):
        st.markdown(text)


def _build_occupancy_timeseries(
    sessions: pd.DataFrame,
    date_str: str,
    bin_minutes: int = 1,
) -> pd.DataFrame:
    """
    Build minute-level occupancy (concurrent in-store visitors) for a date.
    """
    if sessions.empty:
        return pd.DataFrame(columns=["time_label", "occupancy"])

    df = sessions[sessions["date"] == date_str].copy() if "date" in sessions.columns else sessions.copy()
    if df.empty:
        return pd.DataFrame(columns=["time_label", "occupancy"])

    slots_per_bin = bin_minutes * 60 // TIME_UNIT_SECONDS   # e.g. 1min=6, 5min=30
    total_slots = 24 * 3600 // TIME_UNIT_SECONDS            # 8640 (one day)
    bins = range(0, total_slots, slots_per_bin)

    entry = df["entry_time_index"].to_numpy()
    exit_ = df["exit_time_index"].to_numpy()

    records = []
    for b in bins:
        b_end = b + slots_per_bin - 1
        occ = int(((entry <= b_end) & (exit_ >= b)).sum())
        hour   = (b * TIME_UNIT_SECONDS) // 3600
        minute = ((b * TIME_UNIT_SECONDS) % 3600) // 60
        records.append({"time_label": f"{hour:02d}:{minute:02d}", "occupancy": occ})

    return pd.DataFrame(records)


def _weather_color(weather: str) -> str:
    return {
        "Sunny":   GOLD,
        "Rain":    SLATE_GRAY,
        "Snow":    "#93c5fd",
        "Unknown": "#d1d5db",
    }.get(weather, SLATE_GRAY)


def _has_api_key() -> bool:
    try:
        st_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        st_key = ""
    return bool(st_key) or bool(os.environ.get("ANTHROPIC_API_KEY", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview 매장현황
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(space_name: str, loader: CacheLoader) -> None:
    st.subheader("Store Overview")
    st.markdown(
        "View store status with **precisely defined standard metrics**. "
        "See each section below for measurement and calculation methods."
    )

    daily_results    = loader.get_daily_results()
    daily_stats      = loader.get_daily_stats()
    device_mix       = loader.get_device_mix()
    daily_timeseries = loader.get_daily_timeseries()
    daily_hourly     = loader.get_daily_hourly()

    if not daily_results or daily_stats.empty:
        st.info("No daily results in cache. Run precompute first.")
        return

    dr = loader.get_date_range()
    if dr:
        st.caption(f"Period: {dr[0]} ~ {dr[-1]} ({len(dr)} days)")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    _info("""
**Core Metrics (핵심 지표)**

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Floating Pop (유동인구)** | Foot traffic near store | Unique MACs at entrance S-Ward with RSSI ≥ −80 dBm. Daily dedup. |
| **Visitors (방문자)** | In-store visitors | Sessions passing **Strict Entry** (≥3 hits in 1 min + all RSSI ≥ −80 dBm) at inside S-Ward. |
| **CVR (방문율)** | FP → visitor conversion | Visitors ÷ Floating Pop × 100 (%). |
| **Dwell Time (체류시간)** | Time spent in store | Entry to exit. Exit back-dated to last signal. Android 120s / Apple 180s buffer. |

> **Strict Entry**: Only counts as a visit when **≥3 hits in 1 min** and all RSSI strong enough. Filters passers-by.

> **Delta**: Week-over-week change. "Period avg" when comparison not available.
""")

    # Week-over-week comparison (when ≥7 days)
    wow = _cached_wow(daily_stats) if len(daily_stats) >= 7 else {}
    tw = wow.get("this_week", {})
    pw = wow.get("prev_week", {})
    d = wow.get("delta", {})

    # ── Period label: show exactly which dates are "this week" vs "prev week" ──
    if len(daily_stats) >= 14:
        sorted_ds = daily_stats.sort_values("date")
        last14 = sorted_ds.tail(14)
        this_7 = last14.tail(7)
        prev_7 = last14.head(7)
        this_start = str(this_7["date"].iloc[0])[:10]
        this_end   = str(this_7["date"].iloc[-1])[:10]
        prev_start = str(prev_7["date"].iloc[0])[:10]
        prev_end   = str(prev_7["date"].iloc[-1])[:10]
        st.markdown(
            f"<div style='background:#f1f5f9;border-left:3px solid #c49a3a;"
            f"padding:8px 14px;border-radius:4px;margin-bottom:12px;font-size:0.85rem;'>"
            f"📅 <b>This period</b> (current): {this_start} ~ {this_end} &nbsp;｜&nbsp; "
            f"<b>Reference (prev period)</b>: {prev_start} ~ {prev_end}"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif len(daily_stats) >= 7:
        sorted_ds = daily_stats.sort_values("date")
        this_7 = sorted_ds.tail(7)
        this_start = str(this_7["date"].iloc[0])[:10]
        this_end   = str(this_7["date"].iloc[-1])[:10]
        st.markdown(
            f"<div style='background:#f1f5f9;border-left:3px solid #94a3b8;"
            f"padding:8px 14px;border-radius:4px;margin-bottom:12px;font-size:0.85rem;'>"
            f"📅 <b>This period</b>: {this_start} ~ {this_end} &nbsp;｜&nbsp; "
            f"Reference: insufficient data (need ≥14 days for prev week comparison)"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Period avg only — insufficient data for week-over-week comparison (need ≥7 days).")

    has_quality = "quality_cvr" in daily_stats.columns and "quality_visitor_count" in daily_stats.columns
    has_median = "dwell_median_seconds" in daily_stats.columns

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = tw.get("floating_unique", daily_stats["floating_unique"].iloc[-1] if len(daily_stats) else 0)
        delta_str = f"{d.get('floating_pct', 0):+.1f}% vs prev week" if d else "Period avg"
        st.metric("Floating Pop (유동인구)", f"{val:,.0f}", delta_str, delta_color="normal")
    with col2:
        val = tw.get("quality_visitor_count", daily_stats["visitor_count"].iloc[-1] if len(daily_stats) else 0)
        lbl = "Quality Visitors" if has_quality else "Visitors (방문자)"
        delta_str = f"{d.get('quality_visitor_pct', 0):+.1f}% vs prev week" if d else "Period avg"
        st.metric(lbl, f"{val:,.0f}", delta_str, delta_color="normal")
    with col3:
        val = tw.get("quality_cvr", daily_stats["conversion_rate"].iloc[-1] if len(daily_stats) else 0.0)
        lbl = "Quality CVR" if has_quality else "CVR (방문율)"
        delta_str = f"{d.get('quality_cvr_pp', 0):+.1f}%p vs prev week" if d else "Period avg"
        st.metric(lbl, f"{val:.1f}%", delta_str, delta_color="normal")
    with col4:
        dm = tw.get("dwell_median_seconds", 0) if has_median else (daily_stats["dwell_seconds_mean"].iloc[-1] if len(daily_stats) else 0)
        mm, ss = int(dm) // 60, int(dm) % 60
        val_str = f"{mm}m {ss}s" if has_median else f"{dm/60:.1f} min"
        delta_val = d.get("dwell_median", 0)
        delta_str = f"{delta_val:+.0f}s vs prev week" if d and delta_val != 0 else ("Period avg" if not d else "Same")
        st.metric("Median Dwell (체류시간)" if has_median else "Avg Dwell (체류시간)", val_str, delta_str, delta_color="normal")

    st.markdown("---")

    # ── Daily trend chart ─────────────────────────────────────────────────────
    st.markdown("#### Daily FP vs Visitors")
    _info("""
- **Floating Pop** (navy): Daily unique devices at entrance. Patterns vary by weekend/holiday/weather.
- **Visitors** (gold): Sessions passing Strict Entry. Tighter gap vs FP = higher CVR.
- Both up = strong day; high FP + low visitors = window shopping; high visitor share = strong pull.
""")

    daily_stats_sorted = daily_stats.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stats_sorted["date"], y=daily_stats_sorted["floating_unique"],
        name="Floating Pop", line=dict(color=DEEP_NAVY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=daily_stats_sorted["date"], y=daily_stats_sorted["visitor_count"],
        name="Visitors", line=dict(color=GOLD, width=2),
    ))
    fig.update_layout(
        title="Daily FP vs Visitors",
        xaxis_title="Date", yaxis_title="Count",
        template="plotly_white", height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative Visitors ────────────────────────────────────────────────────
    st.markdown("#### Cumulative Visitors & FP")
    _info("""
**Cumulative Visitors**: Running total of daily visitors over the period.
Shows the overall growth trajectory. Steeper slope = busier period.
Dotted line: cumulative floating population for reference.
""", label="📖 Cumulative")

    ds_cum = daily_stats_sorted.copy()
    ds_cum["cum_visitors"] = ds_cum["visitor_count"].cumsum()
    ds_cum["cum_floating"] = ds_cum["floating_unique"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=ds_cum["date"], y=ds_cum["cum_floating"],
        name="Cumulative FP",
        line=dict(color=DEEP_NAVY, width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.05)",
    ))
    fig_cum.add_trace(go.Scatter(
        x=ds_cum["date"], y=ds_cum["cum_visitors"],
        name="Cumulative Visitors",
        line=dict(color=GOLD, width=2.5),
        fill="tozeroy", fillcolor="rgba(196,154,58,0.08)",
    ))
    fig_cum.update_layout(
        title="Cumulative Visitors & FP",
        xaxis_title="Date", yaxis_title="Cumulative Count",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_theme(fig_cum)
    st.plotly_chart(fig_cum, use_container_width=True)

    total_fp_cum = int(ds_cum["cum_floating"].iloc[-1]) if len(ds_cum) else 0
    total_v_cum = int(ds_cum["cum_visitors"].iloc[-1]) if len(ds_cum) else 0
    avg_daily_v = total_v_cum / len(ds_cum) if len(ds_cum) else 0
    period_cvr = total_v_cum / total_fp_cum * 100 if total_fp_cum else 0
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Total FP", f"{total_fp_cum:,}")
    cc2.metric("Total Visitors", f"{total_v_cum:,}")
    cc3.metric("Avg Daily Visitors", f"{avg_daily_v:.0f}")
    cc4.metric("Period CVR", f"{period_cvr:.1f}%")

    st.markdown("---")

    # ── iOS vs Android — Daily Trend ───────────────────────────────────────────
    if not device_mix.empty:
        st.markdown("#### iOS vs Android — Daily Trend")
        _info("""
Daily iOS and Android visitor share (%). Track device composition changes over time.
Note: iOS MAC randomization may affect counts; focus on trends.
""", label="📖 Device trend")
        mix_daily = device_mix_by_date(device_mix)
        if not mix_daily.empty:
            fig_dev_daily = go.Figure()
            fig_dev_daily.add_trace(go.Scatter(
                x=mix_daily["date"], y=mix_daily["ios_pct"],
                name="iOS (%)", line=dict(color=DEEP_NAVY, width=2),
                fill="tozeroy", fillcolor="rgba(15,23,42,0.06)",
            ))
            fig_dev_daily.add_trace(go.Scatter(
                x=mix_daily["date"], y=mix_daily["android_pct"],
                name="Android (%)", line=dict(color=GOLD, width=2),
            ))
            fig_dev_daily.update_layout(
                title="Daily iOS vs Android Share",
                xaxis_title="Date", yaxis_title="Share (%)",
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            apply_theme(fig_dev_daily)
            st.plotly_chart(fig_dev_daily, use_container_width=True)

            avg_ios = mix_daily["ios_pct"].mean()
            avg_android = mix_daily["android_pct"].mean()
            dc1, dc2 = st.columns(2)
            dc1.metric("Avg iOS Share", f"{avg_ios:.1f}%")
            dc2.metric("Avg Android Share", f"{avg_android:.1f}%")
            st.markdown("---")

    # ── Dwell Funnel Stack Chart ───────────────────────────────────────────────
    if all(c in daily_stats.columns for c in ["short_dwell_count", "medium_dwell_count", "long_dwell_count", "quality_cvr"]):
        st.markdown("---")
        st.markdown("#### Dwell Funnel — Short / Medium / Long")
        _info("""
| Segment | Dwell (체류시간) | Meaning |
|---------|-------|---------|
| **Short** | <3 min | Browse, pass-through |
| **Medium** | 3–10 min | Interested, exploring |
| **Long** | 10+ min | Purchase intent |

**Quality CVR (품질방문율)** = (medium + long) / FP × 100.
""", label="📖 Dwell funnel")
        ds = daily_stats.sort_values("date")
        fig_funnel = go.Figure()
        fig_funnel.add_trace(go.Bar(
            x=ds["date"], y=ds["short_dwell_count"], name="Short (<3min)", marker_color=SLATE_GRAY,
        ))
        fig_funnel.add_trace(go.Bar(
            x=ds["date"], y=ds["medium_dwell_count"], name="Medium (3–10min)", marker_color=DEEP_NAVY,
        ))
        fig_funnel.add_trace(go.Bar(
            x=ds["date"], y=ds["long_dwell_count"], name="Long (10min+)", marker_color=GOLD,
        ))
        fig_funnel.update_layout(barmode="stack", height=320, xaxis_title="Date", yaxis_title="Count")
        fig_funnel.add_trace(go.Scatter(
            x=ds["date"], y=ds["quality_cvr"],
            name="Quality CVR (%)", yaxis="y2", line=dict(color=AMBER, width=2, dash="dot"),
        ))
        fig_funnel.update_layout(
            yaxis2=dict(overlaying="y", side="right", range=[0, max(ds["quality_cvr"].max() * 1.2, 10)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_funnel)
        st.plotly_chart(fig_funnel, use_container_width=True)
        # Summary metrics
        total_v = daily_stats["visitor_count"].sum()
        qv = daily_stats["quality_visitor_count"].sum() if "quality_visitor_count" in daily_stats.columns else total_v
        long_ratio = (daily_stats["long_dwell_count"].sum() / total_v * 100) if total_v else 0
        qcvr = (qv / daily_stats["floating_unique"].sum() * 100) if daily_stats["floating_unique"].sum() else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality visitor ratio", f"{qv/total_v*100:.1f}%" if total_v else "—", "Medium+Long / total")
        c2.metric("Long dwell ratio", f"{long_ratio:.1f}%", "10min+")
        c3.metric("Quality CVR (period)", f"{qcvr:.1f}%", "Quality visitors / FP")

    st.markdown("---")

    # ── Intraday timeseries ───────────────────────────────────────────────────
    st.markdown("#### Intraday Traffic — Minute-level flow")
    _info("""
| Chart | Meaning |
|-------|---------|
| **Floating Pop (유동인구)** (navy) | Unique MACs per minute at entrance. |
| **Active Visitors (방문자)** (gold) | In-store sessions per minute (occupancy). |

**Resolution**: 1 min = fine peaks; 5 min = overall flow. High FP + low Active = window shopping; high Active = busy in-store.
""")

    if not daily_timeseries.empty and dr:
        ts_date = st.selectbox(
            "Date (Intraday)", options=dr, key="overview_ts_date",
        )
        resolution = st.radio(
            "Resolution", options=["1 min", "5 min"], index=1,
            horizontal=True, key="overview_ts_resolution",
        )
        res_min = 1 if resolution == "1 min" else 5

        ts_sub = daily_timeseries[daily_timeseries["date"] == ts_date].copy()
        if ts_sub.empty:
            st.caption("No timeseries data for this date. Re-run precompute.")
        else:
            ts_sub["bucket"] = (ts_sub["minute"] // res_min) * res_min
            ts_agg = ts_sub.groupby("bucket", as_index=False).agg(
                floating_count=("floating_count", "max"),
                active_visitors=("active_visitors", "max"),
            )
            ts_agg["time_label"] = ts_agg["bucket"].apply(
                lambda m: f"{m // 60:02d}:{m % 60:02d}"
            )

            fig_ts_fp = go.Figure()
            fig_ts_fp.add_trace(go.Scatter(
                x=ts_agg["time_label"], y=ts_agg["floating_count"],
                name="Floating Pop",
                line=dict(color=DEEP_NAVY, width=1.5),
                fill="tozeroy", fillcolor="rgba(15,23,42,0.08)",
            ))
            fig_ts_fp.update_layout(
                title=f"Floating Population — {ts_date} ({resolution})",
                xaxis_title="Time", yaxis_title="Unique MACs",
                height=300, xaxis=dict(tickangle=-45, nticks=25),
            )
            apply_theme(fig_ts_fp)
            st.plotly_chart(fig_ts_fp, use_container_width=True)

            fig_ts_v = go.Figure()
            fig_ts_v.add_trace(go.Scatter(
                x=ts_agg["time_label"], y=ts_agg["active_visitors"],
                name="Active Visitors",
                line=dict(color=GOLD, width=1.5),
                fill="tozeroy", fillcolor="rgba(196,154,58,0.12)",
            ))
            fig_ts_v.update_layout(
                title=f"Active Visitors (Occupancy) — {ts_date} ({resolution})",
                xaxis_title="Time", yaxis_title="Sessions in-store",
                height=300, xaxis=dict(tickangle=-45, nticks=25),
            )
            apply_theme(fig_ts_v)
            st.plotly_chart(fig_ts_v, use_container_width=True)
    else:
        st.caption("No timeseries cache — re-run precompute to generate minute-by-minute data.")

    st.markdown("---")

    # ── Daily detail table ────────────────────────────────────────────────────
    st.markdown("#### Daily Detail")
    _info("""
| Column | Description |
|--------|-------------|
| **Floating Pop (유동인구)** | Daily unique devices at entrance |
| **Visitors (방문자)** | Sessions passing Strict Entry |
| **CVR (방문율) (%)** | Visitors ÷ FP × 100 |
| **Avg Dwell (min)** | Mean dwell (entry to last signal) |
| **Day Type** | weekday / weekend / holiday |
| **Weather** | Open-Meteo tag |
| **Visitor / FP iOS (%)** | Visitor sessions vs FP by device. Difference suggests device skew in conversion. |
""")

    daily_with_type = add_day_type_to_daily_stats(daily_stats)
    detail = daily_with_type[[
        "date", "floating_unique", "visitor_count",
        "conversion_rate", "dwell_seconds_mean", "day_type", "weekday"
    ]].copy()
    detail = detail.rename(columns={
        "floating_unique":   "Floating Pop",
        "visitor_count":     "Visitors",
        "conversion_rate":   "CVR (%)",
        "dwell_seconds_mean":"Avg Dwell (sec)",
        "day_type":          "Day Type",
    })
    detail["Avg Dwell (min)"] = (detail["Avg Dwell (sec)"] / 60).round(1)
    detail["CVR (%)"] = detail["CVR (%)"].round(1)

    if not device_mix.empty:
        mix_by_date = device_mix_by_date(device_mix)
        detail = detail.merge(mix_by_date[["date", "ios_pct", "android_pct"]], on="date", how="left")
        detail["Visitor iOS (%)"]     = detail["ios_pct"].fillna(0).round(1)
        detail["Visitor Android (%)"] = detail["android_pct"].fillna(0).round(1)
    else:
        detail["Visitor iOS (%)"] = 0.0
        detail["Visitor Android (%)"] = 0.0

    has_fp_device = False
    if not daily_hourly.empty and "floating_apple" in daily_hourly.columns:
        fp_device = (
            daily_hourly.groupby("date")[["floating_apple", "floating_android"]]
            .sum().reset_index()
        )
        fp_device["fp_total"] = fp_device["floating_apple"] + fp_device["floating_android"]
        safe_total = fp_device["fp_total"].replace(0, 1)
        fp_device["FP iOS (%)"] = (100 * fp_device["floating_apple"] / safe_total).round(1)
        fp_device["FP Android (%)"] = (100 * fp_device["floating_android"] / safe_total).round(1)
        fp_device.loc[fp_device["fp_total"] == 0, ["FP iOS (%)", "FP Android (%)"]] = 0.0
        detail = detail.merge(fp_device[["date", "FP iOS (%)", "FP Android (%)"]], on="date", how="left")
        detail["FP iOS (%)"]     = detail["FP iOS (%)"].fillna(0.0)
        detail["FP Android (%)"] = detail["FP Android (%)"].fillna(0.0)
        has_fp_device = True

    has_weather = "weather" in daily_with_type.columns
    if has_weather:
        detail["Weather"]   = daily_with_type["weather"].map(lambda w: _WEATHER_ICON.get(w, w)).values
        detail["Rain (mm)"] = daily_with_type["precipitation"].fillna(0).round(1).values

    display_cols = ["date", "Floating Pop", "Visitors", "CVR (%)", "Avg Dwell (min)", "Day Type"]
    if has_weather:
        display_cols += ["Weather", "Rain (mm)"]
    else:
        st.caption("Weather data not in cache — run precompute or update_weather.py.")
    display_cols += ["Visitor iOS (%)", "Visitor Android (%)"]
    if has_fp_device:
        display_cols += ["FP iOS (%)", "FP Android (%)"]

    st.dataframe(detail[display_cols], use_container_width=True, hide_index=True)

    # ── AI Daily Summary ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 AI Daily Summary")
    st.caption(
        "Claude AI summarizes notable patterns and action items from the metrics."
    )

    if not _has_api_key():
        st.info(
            "Set ANTHROPIC_API_KEY in `.streamlit/secrets.toml` or environment to use AI features."
        )
    else:
        if st.button("🤖 Generate AI insights", key="ai_overview_btn"):
            try:
                daily_with_type_ai = _ensure_day_type(loader)

                avg_fp    = float(daily_stats["floating_unique"].mean())  if "floating_unique"    in daily_stats.columns else 0.0
                avg_v     = float(daily_stats["visitor_count"].mean())    if "visitor_count"      in daily_stats.columns else 0.0
                avg_cvr   = float(daily_stats["conversion_rate"].mean())  if "conversion_rate"    in daily_stats.columns else 0.0
                avg_dwell = float(daily_stats["dwell_seconds_mean"].mean()) / 60 if "dwell_seconds_mean" in daily_stats.columns else 0.0

                last = daily_results[-1] if daily_results else {}
                last_date    = last.get("date", "N/A")
                last_fp      = last.get("floating_unique", 0)
                last_v       = last.get("visitor_count", 0)
                last_cvr     = last.get("conversion_rate", 0.0)

                day_type_cvr_lines = []
                if "day_type" in daily_with_type_ai.columns and "conversion_rate" in daily_with_type_ai.columns:
                    dt_cvr = (
                        daily_with_type_ai.groupby("day_type")["conversion_rate"]
                        .mean()
                        .reset_index()
                        .rename(columns={"day_type": "Type", "conversion_rate": "Avg CVR (%)"})
                    )
                    day_type_cvr_lines = [
                        f"  - {r['Type']}: {r['Avg CVR (%)']:.1f}%"
                        for r in dt_cvr.to_dict("records")
                    ]
                day_type_cvr_table = "\n".join(day_type_cvr_lines) if day_type_cvr_lines else "  (no data)"

                weather_section = ""
                if "weather" in daily_stats.columns and "conversion_rate" in daily_stats.columns:
                    w_cvr = (
                        daily_stats.groupby("weather")["conversion_rate"]
                        .mean()
                        .reset_index()
                    )
                    lines = [f"  - {r['weather']}: {r['conversion_rate']:.1f}%" for r in w_cvr.to_dict("records")]
                    weather_section = "\n[Avg CVR by Weather]\n" + "\n".join(lines)

                dr_list = loader.get_date_range() or []
                period_str = f"{dr_list[0]} ~ {dr_list[-1]}" if len(dr_list) >= 2 else (dr_list[0] if dr_list else "N/A")
                n_days = len(dr_list)

                system_prompt = (
                    "You are a Hermes retail insight analyst. Apply Cause-Effect and provide concise, data-driven insights in English. Include numbers and two actionable items for store operations."
                )
                user_prompt = f"""[Store]
Space: {space_name}
Period: {period_str} ({n_days} days)

[Period averages]
- FP: {avg_fp:.0f}/day
- Visitors: {avg_v:.0f}/day
- CVR: {avg_cvr:.1f}%
- Avg dwell: {avg_dwell:.1f} min

[Last day ({last_date})]
- FP: {last_fp}, Visitors: {last_v}, CVR: {last_cvr:.1f}%

[Avg CVR by day type]
{day_type_cvr_table}
{weather_section}

Provide:
1. 2–3 notable patterns (with numbers)
2. Two actionable items for store operations.
Keep it concise."""

                with st.spinner("🤖 Analyzing data..."):
                    result = call_claude(
                        user_prompt,
                        system=system_prompt,
                        space_notes=st.session_state.get("current_space_notes", ""),
                    )

                if "⚠️" in result:
                    st.warning(result)
                else:
                    st.markdown(result)

            except Exception as _exc:
                st.warning(f"🤖 AI analysis error: {_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Hourly 시간대분석
# ─────────────────────────────────────────────────────────────────────────────

def render_hourly(space_name: str, loader: CacheLoader) -> None:
    """Hourly Analysis — 30min/1hr bins, multi-date averaging, peak timing."""
    st.subheader("Hourly Analysis")
    st.markdown(
        "Analyze traffic by **time of day**. Select date(s) and time bin resolution. "
        "Multi-date selection averages metrics across selected dates."
    )

    daily_hourly     = loader.get_daily_hourly()
    daily_timeseries = loader.get_daily_timeseries()
    daily_stats      = loader.get_daily_stats()
    sessions_all     = loader.get_sessions_all()
    date_range       = loader.get_date_range()

    sessions_stitched = loader.get_sessions_stitched()
    sessions_display = sessions_stitched if not sessions_stitched.empty else sessions_all

    if not date_range:
        st.info("No cached data. Run precompute first.")
        return

    # ── Controls: date multiselect + bin toggle ────────────────────────────────
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        selected_dates = st.multiselect(
            "Select date(s) — multi-select averages",
            options=date_range,
            default=[date_range[-1]] if date_range else [],
            key="hourly_dates_multi",
        )
    with ctrl2:
        bin_choice = st.radio(
            "Time bin",
            options=["1 hour", "30 min"],
            index=0,
            horizontal=True,
            key="hourly_bin_radio",
        )

    bin_minutes = 30 if bin_choice == "30 min" else 60

    if not selected_dates:
        st.caption("Select at least one date to view hourly analysis.")
        return

    # ── Compute hourly stats ──────────────────────────────────────────────────
    h_df = hourly_stats_flexible(
        daily_hourly, sessions_display, daily_timeseries,
        selected_dates, bin_minutes=bin_minutes,
    )

    if h_df.empty:
        st.caption("No hourly data for selected dates.")
        return

    date_label = selected_dates[0] if len(selected_dates) == 1 else f"{len(selected_dates)} dates avg"
    bin_label = "30-min bins" if bin_minutes == 30 else "1-hour bins"
    st.caption(f"**{date_label}** — {bin_label}")

    # ── Day KPI Summary for selected dates ────────────────────────────────────
    if not daily_stats.empty:
        sel_mask = daily_stats["date"].astype(str).isin(selected_dates)
        sel_stats = daily_stats[sel_mask]
        if not sel_stats.empty:
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Total FP (유동인구)", f"{int(sel_stats['floating_unique'].sum()):,}")
            kc2.metric("Total Visitors (방문자)", f"{int(sel_stats['visitor_count'].sum()):,}")
            avg_cvr = sel_stats["conversion_rate"].mean() if "conversion_rate" in sel_stats.columns else 0
            kc3.metric("Avg CVR (방문율)", f"{avg_cvr:.1f}%")
            avg_dwell = sel_stats["dwell_seconds_mean"].mean() / 60 if "dwell_seconds_mean" in sel_stats.columns else 0
            kc4.metric("Avg Dwell (체류시간)", f"{avg_dwell:.1f} min")
            st.markdown("---")

    # ── Peak Timing ───────────────────────────────────────────────────────────
    peaks = identify_peak_hours(h_df, metric="visitor_count", top_n=3)
    if peaks:
        st.markdown("##### 🔥 Peak Hours — Top 3 by Visitors")
        _info("""
**Peak Hours**: The 3 time bins with the highest visitor count. Use for staffing, promotions, and energy management.
""", label="📖 Peak timing")
        peak_cols = st.columns(len(peaks))
        for i, p in enumerate(peaks):
            with peak_cols[i]:
                st.metric(
                    f"#{p['rank']} {p['bin_label']}",
                    f"{int(p['visitors'])} visitors",
                    f"FP: {int(p['fp'])} | CVR: {p['cvr']:.1f}%",
                )
        st.markdown("---")

    # ── Floating Population by time bin ────────────────────────────────────────
    fig_fp = go.Figure()
    fig_fp.add_trace(go.Bar(
        x=h_df["bin_label"], y=h_df["floating_count"],
        name="Floating Pop", marker_color=DEEP_NAVY,
    ))
    fig_fp.update_layout(
        title=f"Floating Population (유동인구) by {bin_label}",
        xaxis_title="Time", yaxis_title="Count", height=320,
        xaxis=dict(tickangle=-45),
    )
    apply_theme(fig_fp)
    st.plotly_chart(fig_fp, use_container_width=True)

    # ── Visitors by time bin ───────────────────────────────────────────────────
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(
        x=h_df["bin_label"], y=h_df["visitor_count"],
        name="Visitors", marker_color=GOLD,
    ))
    fig_v.update_layout(
        title=f"Visitors (방문자) by {bin_label}",
        xaxis_title="Time", yaxis_title="Count", height=320,
        xaxis=dict(tickangle=-45),
    )
    apply_theme(fig_v)
    st.plotly_chart(fig_v, use_container_width=True)

    # ── CVR by time bin ────────────────────────────────────────────────────────
    fig_cvr = go.Figure()
    fig_cvr.add_trace(go.Scatter(
        x=h_df["bin_label"], y=h_df["cvr_pct"],
        name="CVR (%)", line=dict(color=AMBER, width=2),
        mode="lines+markers",
    ))
    fig_cvr.update_layout(
        title=f"CVR (방문율) by {bin_label}",
        xaxis_title="Time", yaxis_title="CVR (%)", height=300,
        xaxis=dict(tickangle=-45),
    )
    apply_theme(fig_cvr)
    st.plotly_chart(fig_cvr, use_container_width=True)

    # ── Avg Dwell by time bin ──────────────────────────────────────────────────
    if "dwell_sec_mean" in h_df.columns:
        h_df["dwell_min"] = (h_df["dwell_sec_mean"] / 60).round(1)
        fig_dwell = go.Figure()
        fig_dwell.add_trace(go.Scatter(
            x=h_df["bin_label"], y=h_df["dwell_min"],
            name="Avg Dwell (min)", line=dict(color=SLATE_GRAY, width=2),
            mode="lines+markers",
        ))
        fig_dwell.update_layout(
            title=f"Avg Dwell Time (체류시간) by {bin_label}",
            xaxis_title="Time", yaxis_title="Avg Dwell (min)", height=300,
            xaxis=dict(tickangle=-45),
        )
        apply_theme(fig_dwell)
        st.plotly_chart(fig_dwell, use_container_width=True)

    # ── iOS vs Android by time bin ─────────────────────────────────────────────
    if "ios_pct" in h_df.columns and h_df["ios_pct"].sum() > 0:
        fig_dev = go.Figure()
        fig_dev.add_trace(go.Scatter(
            x=h_df["bin_label"], y=h_df["ios_pct"],
            name="iOS (%)", line=dict(color=DEEP_NAVY, width=2),
            mode="lines+markers",
        ))
        fig_dev.add_trace(go.Scatter(
            x=h_df["bin_label"], y=h_df["android_pct"],
            name="Android (%)", line=dict(color=GOLD, width=2),
            mode="lines+markers",
        ))
        fig_dev.update_layout(
            title=f"iOS vs Android by {bin_label} — Visitors",
            xaxis_title="Time", yaxis_title="Share (%)", height=300,
            xaxis=dict(tickangle=-45),
        )
        apply_theme(fig_dev)
        st.plotly_chart(fig_dev, use_container_width=True)

    st.markdown("---")

    # ── Intraday Occupancy ─────────────────────────────────────────────────────
    st.markdown("#### Intraday Occupancy — minute-level in-store count")
    _info("""
In-store visitor count by minute (stitched data). Shows concurrent visitor occupancy at each time slot.
Peak occupancy = max simultaneous visitors.
""", label="📖 Intraday Occupancy")

    occ_col1, occ_col2 = st.columns([2, 1])
    with occ_col1:
        occ_date = st.selectbox(
            "Date (Occupancy)",
            options=date_range if date_range else [],
            key="occ_date_select",
        )
    with occ_col2:
        occ_bin = st.radio(
            "Bin",
            options=[1, 5],
            format_func=lambda x: f"{x} min",
            horizontal=True,
            key="occ_bin_radio",
        )

    if occ_date:
        ts_occ = _build_occupancy_timeseries(sessions_display, occ_date, bin_minutes=occ_bin)

        fig_occ = go.Figure()
        if not ts_occ.empty:
            fig_occ.add_trace(go.Scatter(
                x=ts_occ["time_label"],
                y=ts_occ["occupancy"],
                mode="lines",
                name="In-store visitors",
                line=dict(color=GOLD, width=2),
                fill="tozeroy",
                fillcolor="rgba(196,154,58,0.12)",
            ))

        step_label = f"{occ_bin} min"
        fig_occ.update_layout(
            height=340,
            title=f"In-store visitors ({occ_date}, {step_label})",
            xaxis_title="Time",
            yaxis_title="In-store count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                tickmode="array",
                tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)],
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            ),
        )
        apply_theme(fig_occ)
        st.plotly_chart(fig_occ, use_container_width=True)

        if not ts_occ.empty:
            peak = int(ts_occ["occupancy"].max())
            avg_occ = ts_occ["occupancy"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Peak Occupancy", f"{peak}")
            c2.metric("Average Occupancy", f"{avg_occ:.1f}")
    else:
        st.caption("Select a date to see minute-level occupancy.")

    st.markdown("---")

    # ── Hourly data table ──────────────────────────────────────────────────────
    st.markdown("#### Hourly Data Table")
    display_df = h_df.copy()
    rename_map = {
        "bin_label": "Time",
        "floating_count": "Floating Pop",
        "visitor_count": "Visitors",
        "cvr_pct": "CVR (%)",
        "dwell_sec_mean": "Avg Dwell (sec)",
        "ios_pct": "iOS (%)",
        "android_pct": "Android (%)",
    }
    display_cols = [c for c in rename_map if c in display_df.columns]
    st.dataframe(
        display_df[display_cols].rename(columns=rename_map),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Patterns 패턴분석
# ─────────────────────────────────────────────────────────────────────────────

def render_patterns(space_name: str, loader: CacheLoader) -> None:
    """Pattern Analysis — day-type, weather, heatmaps, dwell, device, anomaly."""
    st.subheader("Pattern Analysis")
    st.markdown(
        "Discover **patterns and causes** behind traffic changes. "
        "Analyze by day type, weather, weekday × hour, device mix, and dwell behavior."
    )

    daily_stats = _ensure_day_type(loader)
    if daily_stats.empty:
        st.info("No cache data.")
        return

    daily_hourly  = loader.get_daily_hourly()
    device_mix    = loader.get_device_mix()
    sessions_all  = loader.get_sessions_all()
    date_range    = loader.get_date_range()
    wd_en         = weekday_names_en()

    sessions_stitched = loader.get_sessions_stitched()
    sessions_display = sessions_stitched if not sessions_stitched.empty else sessions_all

    # ── 1. Day-of-Week Analysis (요일별 분석) — PRIMARY SECTION ─────────────────
    st.markdown("#### Day-of-Week Analysis (요일별 분석)")
    _info("""
**Most critical dimension for retail management.**

- Which day brings the most foot traffic?
- Which day has the best conversion (CVR)?
- Which day do customers stay the longest?
- How predictable is each day? (Consistency — lower CV = more predictable)

Gold bars = Weekend (Sat/Sun). Navy bars = Weekday. Error bars = ±1 std dev.
""")

    _WD_SHORT = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    _WD_KO    = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    _WEEKEND  = {5, 6}

    if "weekday" in daily_stats.columns:
        wd_stats = daily_stats.groupby("weekday", as_index=False).agg(
            days=("date", "count"),
            avg_fp=("floating_unique", "mean"),
            std_fp=("floating_unique", "std"),
            avg_visitors=("visitor_count", "mean"),
            std_visitors=("visitor_count", "std"),
            avg_cvr=("conversion_rate", "mean"),
            std_cvr=("conversion_rate", "std"),
            avg_dwell=("dwell_seconds_mean", "mean"),
        ).sort_values("weekday")

        wd_stats["std_fp"]       = wd_stats["std_fp"].fillna(0)
        wd_stats["std_visitors"] = wd_stats["std_visitors"].fillna(0)
        wd_stats["std_cvr"]      = wd_stats["std_cvr"].fillna(0)
        # Coefficient of Variation (%) — lower = more consistent/predictable
        safe_avg_v = wd_stats["avg_visitors"].replace(0, 1)
        wd_stats["cv_visitors"]  = (wd_stats["std_visitors"] / safe_avg_v * 100).round(1)
        # Display labels: "Mon (월)", "Sat (토)" etc.
        wd_stats["wd_label"]     = wd_stats["weekday"].apply(
            lambda w: f"{_WD_SHORT.get(w, '?')} ({_WD_KO.get(w, '?')})"
        )
        wd_stats["is_weekend"]   = wd_stats["weekday"].isin(_WEEKEND)

        # ── Best/Worst highlights ─────────────────────────────────────────────
        best_v_row   = wd_stats.loc[wd_stats["avg_visitors"].idxmax()]
        best_cvr_row = wd_stats.loc[wd_stats["avg_cvr"].idxmax()]
        best_d_row   = wd_stats.loc[wd_stats["avg_dwell"].idxmax()]
        worst_v_row  = wd_stats.loc[wd_stats["avg_visitors"].idxmin()]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "🏆 Busiest Day (방문자)",
            best_v_row["wd_label"],
            f"avg {best_v_row['avg_visitors']:.0f} visitors",
        )
        k2.metric(
            "🎯 Best CVR Day (방문율)",
            best_cvr_row["wd_label"],
            f"avg {best_cvr_row['avg_cvr']:.1f}%",
        )
        k3.metric(
            "⏱ Longest Dwell (체류)",
            best_d_row["wd_label"],
            f"{int(best_d_row['avg_dwell'])//60}m {int(best_d_row['avg_dwell'])%60}s",
        )
        k4.metric(
            "📉 Slowest Day (최저 방문자)",
            worst_v_row["wd_label"],
            f"avg {worst_v_row['avg_visitors']:.0f} visitors",
            delta_color="inverse",
        )
        st.markdown("---")

        # ── Chart A: Avg FP + Visitors by weekday (grouped, with error bars) ──
        bar_colors_fp = [GOLD if r else DEEP_NAVY for r in wd_stats["is_weekend"]]
        bar_colors_v  = ["#d4a21a" if r else "#334155" for r in wd_stats["is_weekend"]]

        fig_wd_traf = go.Figure()
        fig_wd_traf.add_trace(go.Bar(
            x=wd_stats["wd_label"],
            y=wd_stats["avg_fp"].round(0),
            name="Avg FP (유동인구)",
            marker_color=bar_colors_fp,
            opacity=0.65,
            error_y=dict(type="data", array=wd_stats["std_fp"].round(0), visible=True, color="#94a3b8"),
            hovertemplate="%{x}<br>Avg FP: %{y:.0f}<extra></extra>",
        ))
        fig_wd_traf.add_trace(go.Bar(
            x=wd_stats["wd_label"],
            y=wd_stats["avg_visitors"].round(1),
            name="Avg Visitors (방문자)",
            marker_color=bar_colors_v,
            error_y=dict(type="data", array=wd_stats["std_visitors"].round(0), visible=True, color="#94a3b8"),
            hovertemplate="%{x}<br>Avg Visitors: %{y:.1f}<extra></extra>",
        ))
        fig_wd_traf.update_layout(
            title="Avg FP & Visitors by Weekday — Gold = Weekend",
            xaxis_title="Day of Week", yaxis_title="Count",
            barmode="group", height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_wd_traf)
        st.plotly_chart(fig_wd_traf, use_container_width=True)

        # ── Chart B: CVR by weekday (highlight best day) ───────────────────────
        cvr_colors = []
        max_cvr = wd_stats["avg_cvr"].max()
        for _, row in wd_stats.iterrows():
            if row["avg_cvr"] == max_cvr:
                cvr_colors.append(AMBER)         # best day = amber
            elif row["is_weekend"]:
                cvr_colors.append(GOLD)          # weekend = gold
            else:
                cvr_colors.append(DEEP_NAVY)     # weekday = navy

        fig_wd_cvr = go.Figure()
        fig_wd_cvr.add_trace(go.Bar(
            x=wd_stats["wd_label"],
            y=wd_stats["avg_cvr"].round(2),
            name="Avg CVR (%)",
            marker_color=cvr_colors,
            error_y=dict(type="data", array=wd_stats["std_cvr"].round(2), visible=True, color="#94a3b8"),
            text=wd_stats["avg_cvr"].round(1).astype(str) + "%",
            textposition="outside",
            hovertemplate="%{x}<br>Avg CVR: %{y:.1f}%<extra></extra>",
        ))
        fig_wd_cvr.update_layout(
            title="Avg CVR (방문율) by Weekday — Amber = Best day",
            xaxis_title="Day of Week", yaxis_title="CVR (%)",
            height=340,
        )
        apply_theme(fig_wd_cvr)
        st.plotly_chart(fig_wd_cvr, use_container_width=True)

        # ── Chart C: Dwell by weekday ──────────────────────────────────────────
        wd_stats["avg_dwell_min"] = (wd_stats["avg_dwell"] / 60).round(1)
        dwell_colors = [GOLD if r else DEEP_NAVY for r in wd_stats["is_weekend"]]
        fig_wd_dwell = go.Figure()
        fig_wd_dwell.add_trace(go.Bar(
            x=wd_stats["wd_label"],
            y=wd_stats["avg_dwell_min"],
            name="Avg Dwell (min)",
            marker_color=dwell_colors,
            text=wd_stats["avg_dwell_min"].astype(str) + " min",
            textposition="outside",
            hovertemplate="%{x}<br>Avg Dwell: %{y:.1f} min<extra></extra>",
        ))
        fig_wd_dwell.update_layout(
            title="Avg Dwell Time (체류시간) by Weekday",
            xaxis_title="Day of Week", yaxis_title="Avg Dwell (min)",
            height=320,
        )
        apply_theme(fig_wd_dwell)
        st.plotly_chart(fig_wd_dwell, use_container_width=True)

        # ── Chart D: Consistency (CV%) by weekday ─────────────────────────────
        st.markdown("##### Day Consistency (예측 가능성)")
        _info("""
**Coefficient of Variation (CV %)** = std / avg × 100.
Lower CV = more consistent and predictable.
Higher CV = highly variable — hard to plan staffing.
""", label="📖 Consistency")

        cv_colors = []
        for cv in wd_stats["cv_visitors"]:
            if cv < 15:
                cv_colors.append("#22c55e")   # green = very consistent
            elif cv < 30:
                cv_colors.append(AMBER)       # amber = moderate
            else:
                cv_colors.append("#ef4444")   # red = high variability

        fig_wd_cv = go.Figure()
        fig_wd_cv.add_trace(go.Bar(
            x=wd_stats["wd_label"],
            y=wd_stats["cv_visitors"],
            name="CV (%) — Visitor variability",
            marker_color=cv_colors,
            text=wd_stats["cv_visitors"].astype(str) + "%",
            textposition="outside",
            hovertemplate="%{x}<br>CV: %{y:.1f}%<extra></extra>",
        ))
        fig_wd_cv.add_hline(y=15, line_dash="dot", line_color="#22c55e",
                             annotation_text="15% (consistent)", annotation_position="top right")
        fig_wd_cv.add_hline(y=30, line_dash="dot", line_color="#ef4444",
                             annotation_text="30% (high variability)", annotation_position="top right")
        fig_wd_cv.update_layout(
            title="Visitor Variability by Weekday (green < 15% = predictable)",
            xaxis_title="Day of Week", yaxis_title="CV (%)",
            height=300,
        )
        apply_theme(fig_wd_cv)
        st.plotly_chart(fig_wd_cv, use_container_width=True)

        # ── Detail Table ───────────────────────────────────────────────────────
        st.markdown("##### Full Weekday Summary Table")
        tbl = wd_stats[[
            "wd_label", "days", "avg_fp", "avg_visitors", "avg_cvr", "avg_dwell_min", "cv_visitors"
        ]].copy()
        tbl["avg_fp"]       = tbl["avg_fp"].round(0).astype(int)
        tbl["avg_visitors"] = tbl["avg_visitors"].round(1)
        tbl["avg_cvr"]      = tbl["avg_cvr"].round(2)
        tbl["avg_dwell_min"]= tbl["avg_dwell_min"].round(1)
        tbl = tbl.rename(columns={
            "wd_label": "Day",
            "days": "# Days",
            "avg_fp": "Avg FP",
            "avg_visitors": "Avg Visitors",
            "avg_cvr": "Avg CVR (%)",
            "avg_dwell_min": "Avg Dwell (min)",
            "cv_visitors": "Variability CV (%)",
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ── Compact Day Type Summary ───────────────────────────────────────────
        with st.expander("📊 Weekday / Weekend / Holiday Summary", expanded=False):
            summary = daily_stats.groupby("day_type", as_index=False).agg({
                "floating_unique": "mean", "visitor_count": "mean",
                "conversion_rate": "mean", "dwell_seconds_mean": "mean", "date": "count",
            }).rename(columns={
                "date": "# Days", "floating_unique": "Avg FP",
                "visitor_count": "Avg Visitors", "conversion_rate": "Avg CVR (%)",
                "dwell_seconds_mean": "Avg Dwell (sec)",
            })
            summary["Avg Dwell (min)"] = (summary["Avg Dwell (sec)"] / 60).round(1)
            summary["Avg CVR (%)"]     = summary["Avg CVR (%)"].round(1)
            st.dataframe(
                summary[["day_type", "# Days", "Avg FP", "Avg Visitors", "Avg CVR (%)", "Avg Dwell (min)"]],
                use_container_width=True, hide_index=True,
            )

        # ── Auto-insight (rule-based, always shown) ────────────────────────────
        st.markdown("##### 📊 Weekday Insights")

        wd_only = wd_stats[~wd_stats["is_weekend"]]
        we_only = wd_stats[wd_stats["is_weekend"]]
        top_v_row    = wd_stats.loc[wd_stats["avg_visitors"].idxmax()]
        bot_v_row    = wd_stats.loc[wd_stats["avg_visitors"].idxmin()]
        top_cvr_row  = wd_stats.loc[wd_stats["avg_cvr"].idxmax()]
        bot_cvr_row  = wd_stats.loc[wd_stats["avg_cvr"].idxmin()]
        best_cv_row  = wd_stats.loc[wd_stats["cv_visitors"].idxmin()]
        worst_cv_row = wd_stats.loc[wd_stats["cv_visitors"].idxmax()]

        auto_insights = []
        # Weekend vs weekday
        if not wd_only.empty and not we_only.empty:
            wd_avg_v   = wd_only["avg_visitors"].mean()
            we_avg_v   = we_only["avg_visitors"].mean()
            wd_avg_cvr = wd_only["avg_cvr"].mean()
            we_avg_cvr = we_only["avg_cvr"].mean()
            we_ratio   = we_avg_v / wd_avg_v if wd_avg_v > 0 else 1
            cvr_gap    = we_avg_cvr - wd_avg_cvr
            if abs(we_ratio - 1) > 0.1:
                direction = "higher" if we_ratio > 1 else "lower"
                auto_insights.append(
                    f"**Weekend traffic is {we_ratio:.1f}x {'higher' if we_ratio > 1 else 'lower'} than weekday** "
                    f"(avg {we_avg_v:.0f} vs {wd_avg_v:.0f} visitors). "
                    f"CVR is {cvr_gap:+.1f}pp {'higher' if cvr_gap > 0 else 'lower'} on weekends."
                )
            else:
                auto_insights.append(
                    f"Weekend and weekday traffic are similar ({we_ratio:.2f}x ratio). "
                    f"CVR gap: {cvr_gap:+.1f}pp."
                )

        # Best vs worst day
        v_ratio = top_v_row["avg_visitors"] / bot_v_row["avg_visitors"] if bot_v_row["avg_visitors"] > 0 else 1
        auto_insights.append(
            f"**{top_v_row['wd_label']} is the busiest day** ({top_v_row['avg_visitors']:.0f} avg visitors) "
            f"— {v_ratio:.1f}× more than {bot_v_row['wd_label']} ({bot_v_row['avg_visitors']:.0f})."
        )

        # CVR insight
        cvr_gap_bw = top_cvr_row["avg_cvr"] - bot_cvr_row["avg_cvr"]
        auto_insights.append(
            f"**Highest CVR on {top_cvr_row['wd_label']}** ({top_cvr_row['avg_cvr']:.1f}%) vs lowest on "
            f"{bot_cvr_row['wd_label']} ({bot_cvr_row['avg_cvr']:.1f}%) — {cvr_gap_bw:.1f}pp gap. "
            f"→ Run promotions on {top_cvr_row['wd_label']} for maximum conversion efficiency."
        )

        # Consistency insight
        auto_insights.append(
            f"**{best_cv_row['wd_label']} is the most predictable** (CV {best_cv_row['cv_visitors']:.0f}%). "
            f"**{worst_cv_row['wd_label']} is the most variable** (CV {worst_cv_row['cv_visitors']:.0f}%) "
            f"— harder to plan staffing for this day."
        )

        for ins in auto_insights:
            st.markdown(f"- {ins}")

        # ── AI Deep Analysis button ────────────────────────────────────────────
        if _has_api_key():
            st.markdown("")
            if st.button("🤖 AI Weekday Pattern Analysis", key="ai_weekday_btn"):
                try:
                    wd_summary_lines = []
                    for _, row in wd_stats.iterrows():
                        wd_summary_lines.append(
                            f"  {row['wd_label']}: avg_visitors={row['avg_visitors']:.1f}, "
                            f"avg_fp={row['avg_fp']:.0f}, avg_cvr={row['avg_cvr']:.2f}%, "
                            f"avg_dwell={row['avg_dwell_min']:.1f}min, cv={row['cv_visitors']:.1f}%, "
                            f"days={int(row['days'])}"
                        )
                    wd_text = "\n".join(wd_summary_lines)
                    ai_prompt = f"""[Store: {space_name}]
Weekday performance data (averages across all weeks in period):
{wd_text}

Provide a concise analysis (4–5 bullet points) covering:
1. Most notable weekday pattern (e.g., weekend surge, midweek dip)
2. Best day for promotion timing (high CVR) and why
3. Staffing recommendation based on traffic + variability
4. One surprising or counter-intuitive finding from the data
5. One watch-out or risk (e.g., high variability day, low CVR despite high traffic)

Be specific with numbers. English only."""
                    ai_system = "You are a retail analytics expert. Give specific, data-driven insights for store operations. Concise bullet points only."
                    with st.spinner("🤖 Analyzing weekday patterns..."):
                        ai_result = call_claude(
                            ai_prompt,
                            system=ai_system,
                            space_notes=st.session_state.get("current_space_notes", ""),
                        )
                    if "⚠️" in ai_result:
                        st.warning(ai_result)
                    else:
                        st.markdown(ai_result)
                except Exception as _e:
                    st.warning(f"AI analysis error: {_e}")

    else:
        # Fallback if weekday column not available
        st.caption("Weekday data not available — run precompute to generate day-of-week analysis.")

    st.markdown("---")

    # ── 2. Heatmap — Weekday × Hour (시간대 × 요일) ────────────────────────────
    st.markdown("#### Weekday × Hour Heatmap")
    _info("""
Rows = day of week, Columns = hour (0–23). Darker cell = higher average.

Use together with the charts above:
- FP heatmap → when does foot traffic arrive by day?
- CVR heatmap → which day + hour combo has the best conversion?
""")

    if not daily_hourly.empty and not daily_stats.empty:
        # Compute heatmaps before tabs so they can be used for insight below
        heatmap_fp  = _cached_heatmap(daily_hourly, daily_stats, "floating_count")
        heatmap_cvr = _cached_heatmap(daily_hourly, daily_stats, "conversion_rate")

        tab_fp, tab_cvr = st.tabs(["FP Heatmap", "CVR Heatmap"])
        with tab_fp:
            if not heatmap_fp.empty:
                pivot_fp = pivot_heatmap(heatmap_fp)
                pivot_fp.index = [wd_en.get(i, str(i)) for i in pivot_fp.index]
                fig_hm_fp = px.imshow(pivot_fp, labels=dict(x="Hour", y="Weekday", color="Floating Pop"),
                                      aspect="auto",
                                      color_continuous_scale=["#f8fafc", AMBER, DEEP_NAVY])
                fig_hm_fp.update_layout(height=400, title="Floating Pop — Weekday × Hour (avg)")
                apply_theme(fig_hm_fp)
                st.plotly_chart(fig_hm_fp, use_container_width=True)
            else:
                st.caption("No floating data.")
        with tab_cvr:
            if not heatmap_cvr.empty:
                pivot_cvr = pivot_heatmap(heatmap_cvr)
                pivot_cvr.index = [wd_en.get(i, str(i)) for i in pivot_cvr.index]
                fig_hm_cvr = px.imshow(pivot_cvr, labels=dict(x="Hour", y="Weekday", color="CVR (%)"),
                                       aspect="auto",
                                       color_continuous_scale=["#f8fafc", GOLD, DEEP_NAVY])
                fig_hm_cvr.update_layout(height=400, title="CVR — Weekday × Hour (avg)")
                apply_theme(fig_hm_cvr)
                st.plotly_chart(fig_hm_cvr, use_container_width=True)
            else:
                st.caption("No CVR data.")

        # ── Auto-insight: best time slots ──────────────────────────────────────
        _wd_hm = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        hm_insight_parts = []
        if not heatmap_fp.empty:
            best_fp_row = heatmap_fp.loc[heatmap_fp["value"].idxmax()]
            hm_insight_parts.append(
                f"📈 <b>Peak FP slot</b>: {_wd_hm.get(int(best_fp_row['weekday']), '?')} "
                f"{int(best_fp_row['hour']):02d}:00 — avg {best_fp_row['value']:.0f} devices in front of store"
            )
        if not heatmap_cvr.empty:
            best_cvr_row = heatmap_cvr.loc[heatmap_cvr["value"].idxmax()]
            hm_insight_parts.append(
                f"🎯 <b>Best CVR slot</b>: {_wd_hm.get(int(best_cvr_row['weekday']), '?')} "
                f"{int(best_cvr_row['hour']):02d}:00 — avg CVR {best_cvr_row['value']:.1f}% "
                f"→ highest conversion moment of the week"
            )
            # Check if peak FP and peak CVR align
            if not heatmap_fp.empty:
                fp_best_wd  = int(heatmap_fp.loc[heatmap_fp["value"].idxmax(), "weekday"])
                fp_best_hr  = int(heatmap_fp.loc[heatmap_fp["value"].idxmax(), "hour"])
                cvr_best_wd = int(heatmap_cvr.loc[heatmap_cvr["value"].idxmax(), "weekday"])
                cvr_best_hr = int(heatmap_cvr.loc[heatmap_cvr["value"].idxmax(), "hour"])
                if fp_best_wd == cvr_best_wd and fp_best_hr == cvr_best_hr:
                    hm_insight_parts.append(
                        "✅ <b>Peak FP and CVR align</b> — prioritize staffing during this slot for maximum impact."
                    )
                else:
                    hm_insight_parts.append(
                        f"⚡ <b>FP and CVR peaks differ</b> — "
                        f"most traffic arrives {_wd_hm.get(fp_best_wd, '?')} {fp_best_hr:02d}:00, "
                        f"but best conversion is {_wd_hm.get(cvr_best_wd, '?')} {cvr_best_hr:02d}:00. "
                        f"Consider targeted promotions at the high-traffic slot to lift CVR there too."
                    )
        if hm_insight_parts:
            st.markdown(
                '<div style="background:#f1f5f9;border-left:3px solid #c49a3a;padding:10px 14px;'
                'border-radius:4px;margin:8px 0;font-size:0.9rem;">'
                + "<br>".join(hm_insight_parts)
                + "</div>",
                unsafe_allow_html=True,
            )

        # ── AI Heatmap Analysis button ─────────────────────────────────────────
        if _has_api_key():
            if st.button("🤖 AI Heatmap Insight", key="ai_heatmap_btn"):
                try:
                    hm_lines = []
                    if not heatmap_fp.empty:
                        for _, hr in heatmap_fp.nlargest(12, "value").iterrows():
                            hm_lines.append(
                                f"FP  | {_wd_hm.get(int(hr['weekday']), '?')} {int(hr['hour']):02d}:00 → {hr['value']:.0f}"
                            )
                    if not heatmap_cvr.empty:
                        for _, hr in heatmap_cvr.nlargest(12, "value").iterrows():
                            hm_lines.append(
                                f"CVR | {_wd_hm.get(int(hr['weekday']), '?')} {int(hr['hour']):02d}:00 → {hr['value']:.1f}%"
                            )
                    hm_prompt = (
                        f"[Weekday × Hour heatmap — top slots — {space_name}]\n"
                        + "\n".join(hm_lines)
                        + "\n\nAnalyze:\n"
                        "1. What time windows consistently drive the most foot traffic (FP)?\n"
                        "2. When does CVR peak — does it align with FP or is there a mismatch?\n"
                        "3. Recommend 2 concrete operational actions (staffing, promotions) based on these patterns.\n"
                        "Keep response to 3 paragraphs max."
                    )
                    with st.spinner("🤖 Analyzing heatmap patterns..."):
                        hm_result = call_claude(
                            hm_prompt,
                            system="You are a retail analytics assistant. Provide actionable insights from weekday×hour traffic and CVR heatmap data. Be specific and concise.",
                            space_notes=st.session_state.get("current_space_notes", ""),
                        )
                    if "⚠️" in hm_result:
                        st.warning(hm_result)
                    else:
                        st.markdown(hm_result)
                except Exception as _e:
                    st.warning(f"Heatmap AI error: {_e}")

    st.markdown("---")

    # ── 3. Weather Effect ──────────────────────────────────────────────────────
    st.markdown("#### Weather Effect — Cause analysis")
    _info("""
Weather as **cause**; visitors and CVR as **effect**.
Chart 1: visitors by weather. Chart 2: visitors vs precipitation (dual axis).
Table: summary by weather; CVR vs Sunny = pp diff vs sunny baseline.
""")

    has_weather = "weather" in daily_stats.columns and daily_stats["weather"].notna().any()
    if has_weather:
        ds        = daily_stats.copy()
        ds_sorted = ds.sort_values("date")

        # ── Chart A: Daily bars, each colored by that day's weather (no overlap) ─
        st.markdown("##### Daily Visitors — colored by weather")
        _info("""
Each bar represents one day. Color = that day's weather condition.
No overlap — each day has exactly one weather type.
""", label="📖 Chart guide")

        bar_colors  = [_weather_color(w) for w in ds_sorted["weather"]]
        bar_hovers  = [_WEATHER_ICON.get(w, w) for w in ds_sorted["weather"]]

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=ds_sorted["date"],
            y=ds_sorted["visitor_count"],
            marker_color=bar_colors,
            customdata=bar_hovers,
            hovertemplate="%{x}<br><b>Visitors</b>: %{y}<br>%{customdata}<extra></extra>",
            name="Visitors",
        ))
        # Legend: dummy scatter traces per weather type (so legend shows colors)
        present_weathers = ds_sorted["weather"].dropna().unique()
        for wv in ["Sunny", "Rain", "Snow", "Unknown"]:
            if wv in present_weathers:
                fig_daily.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=_WEATHER_ICON.get(wv, wv),
                    marker=dict(size=12, color=_weather_color(wv), symbol="square"),
                    showlegend=True,
                ))
        fig_daily.update_layout(
            title="Daily Visitors by Weather",
            xaxis_title="Date", yaxis_title="Visitors",
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_daily)
        st.plotly_chart(fig_daily, use_container_width=True)

        # ── Chart B: Weather Impact Summary — avg visitors + CVR per type ─────────
        st.markdown("##### Weather Impact — Average per Condition")
        _info("""
**Cause (날씨) → Effect (방문자, CVR)**

Each bar = average visitors on days with that weather.
Gold line = average CVR (%). This directly shows how weather affects store performance.
""", label="📖 How to read")

        weather_summary = ds.groupby("weather", as_index=False).agg(
            Days=("date", "count"),
            Avg_FP=("floating_unique", "mean"),
            Avg_Visitors=("visitor_count", "mean"),
            Avg_CVR=("conversion_rate", "mean"),
            Avg_Dwell_sec=("dwell_seconds_mean", "mean"),
            Avg_Precip=("precipitation", "mean"),
        ).sort_values("Avg_Visitors", ascending=False)
        weather_summary["Avg_FP"]        = weather_summary["Avg_FP"].round(0).astype(int)
        weather_summary["Avg_Visitors"]  = weather_summary["Avg_Visitors"].round(1)
        weather_summary["Avg_CVR"]       = weather_summary["Avg_CVR"].round(2)
        weather_summary["Avg_Dwell_min"] = (weather_summary["Avg_Dwell_sec"] / 60).round(1)
        weather_summary["Avg_Precip"]    = weather_summary["Avg_Precip"].fillna(0).round(1)
        weather_summary["Weather"]       = weather_summary["weather"].map(lambda w: _WEATHER_ICON.get(w, w))
        # x-axis labels: icon + day count
        x_labels = [
            f"{_WEATHER_ICON.get(w, w)}\n({int(d)} days)"
            for w, d in zip(weather_summary["weather"], weather_summary["Days"])
        ]

        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(
            x=x_labels,
            y=weather_summary["Avg_Visitors"],
            name="Avg Visitors",
            marker_color=[_weather_color(w) for w in weather_summary["weather"]],
            text=weather_summary["Avg_Visitors"].round(0).astype(int),
            textposition="outside",
        ))
        fig_impact.add_trace(go.Scatter(
            x=x_labels,
            y=weather_summary["Avg_CVR"],
            name="Avg CVR (%)",
            yaxis="y2",
            line=dict(color=AMBER, width=2.5),
            mode="lines+markers",
            marker=dict(size=10),
        ))
        cvr_max = float(weather_summary["Avg_CVR"].max()) if not weather_summary.empty else 10
        fig_impact.update_layout(
            title="Weather Impact on Visitors & CVR",
            xaxis_title="Weather Condition",
            yaxis=dict(title="Avg Visitors / day"),
            yaxis2=dict(
                title="Avg CVR (%)", overlaying="y", side="right",
                showgrid=False,
                range=[0, max(cvr_max * 1.4, 5)],
            ),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_impact)
        st.plotly_chart(fig_impact, use_container_width=True)

        # ── Chart C: Visitors vs Precipitation (time-series) ──────────────────────
        if "precipitation" in ds_sorted.columns:
            st.markdown("##### Precipitation vs Visitors — Time Series")
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=ds_sorted["date"], y=ds_sorted["precipitation"].fillna(0),
                name="Precipitation (mm)", marker_color="#93c5fd", opacity=0.7, yaxis="y2",
            ))
            fig_rain.add_trace(go.Scatter(
                x=ds_sorted["date"], y=ds_sorted["visitor_count"],
                name="Visitors", line=dict(color=DEEP_NAVY, width=2),
            ))
            fig_rain.update_layout(
                title="Visitors vs Precipitation",
                xaxis_title="Date",
                yaxis=dict(title="Visitors"),
                yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right", showgrid=False),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            apply_theme(fig_rain)
            st.plotly_chart(fig_rain, use_container_width=True)

        # ── Table D: Weather Summary ───────────────────────────────────────────────
        st.markdown("##### Summary by Weather")
        sunny_cvr    = weather_summary.loc[weather_summary["weather"] == "Sunny", "Avg_CVR"]
        baseline_cvr = float(sunny_cvr.iloc[0]) if not sunny_cvr.empty else None
        weather_summary["CVR vs Sunny"] = (
            weather_summary["Avg_CVR"].apply(lambda v: f"{v - baseline_cvr:+.1f}pp")
            if baseline_cvr and baseline_cvr > 0 else "—"
        )
        st.dataframe(
            weather_summary[["Weather","Days","Avg_FP","Avg_Visitors","Avg_CVR","CVR vs Sunny","Avg_Dwell_min","Avg_Precip"]].rename(columns={
                "Avg_FP":"Avg FP", "Avg_Visitors":"Avg Visitors",
                "Avg_CVR":"Avg CVR (%)", "Avg_Dwell_min":"Avg Dwell (min)", "Avg_Precip":"Avg Rain (mm)",
            }),
            use_container_width=True, hide_index=True,
        )

        # ── Weather auto-insight ───────────────────────────────────────────────
        if len(weather_summary) >= 2:
            best_w  = weather_summary.iloc[0]   # sorted by Avg_Visitors desc
            worst_w = weather_summary.iloc[-1]
            best_icon  = _WEATHER_ICON.get(best_w["weather"],  best_w["weather"])
            worst_icon = _WEATHER_ICON.get(worst_w["weather"], worst_w["weather"])
            visitor_diff = best_w["Avg_Visitors"] - worst_w["Avg_Visitors"]
            visitor_pct  = visitor_diff / max(worst_w["Avg_Visitors"], 1) * 100
            w_insight_parts = [
                f"🌤 <b>Best weather</b>: {best_icon} — avg {best_w['Avg_Visitors']:.0f} visitors/day "
                f"(CVR {best_w['Avg_CVR']:.1f}%)",
                f"☁️ <b>Worst weather</b>: {worst_icon} — avg {worst_w['Avg_Visitors']:.0f} visitors/day "
                f"(CVR {worst_w['Avg_CVR']:.1f}%)",
                f"📊 <b>Weather impact</b>: {visitor_pct:+.0f}% visitor difference between best and worst condition",
            ]
            if baseline_cvr and baseline_cvr > 0:
                cvr_delta = worst_w["Avg_CVR"] - baseline_cvr
                if cvr_delta < -2:
                    w_insight_parts.append(
                        f"💡 On {worst_icon} days, CVR drops {abs(cvr_delta):.1f}pp vs Sunny "
                        "— consider targeted in-store promotions on bad-weather days to compensate."
                    )
                elif cvr_delta > 0:
                    w_insight_parts.append(
                        f"💡 On {worst_icon} days, CVR is actually {cvr_delta:+.1f}pp vs Sunny "
                        "— visitors in bad weather are more purposeful buyers. "
                        "Staff accordingly for high-intent customers."
                    )
            st.markdown(
                '<div style="background:#f1f5f9;border-left:3px solid #c49a3a;padding:10px 14px;'
                'border-radius:4px;margin:8px 0;font-size:0.9rem;">'
                + "<br>".join(w_insight_parts)
                + "</div>",
                unsafe_allow_html=True,
            )

        # ── AI Weather Analysis button ─────────────────────────────────────────
        if _has_api_key():
            if st.button("🤖 AI Weather Impact Analysis", key="ai_weather_btn"):
                try:
                    ws_text = weather_summary[
                        ["weather", "Days", "Avg_FP", "Avg_Visitors", "Avg_CVR", "Avg_Dwell_min", "Avg_Precip"]
                    ].to_string(index=False)
                    w_prompt = (
                        f"[Weather impact summary — {space_name}]\n{ws_text}\n\n"
                        "Analyze:\n"
                        "1. How significantly does weather affect visitor traffic and CVR?\n"
                        "2. What promotional or operational strategies should be applied on poor-weather days?\n"
                        "3. Are there any unexpected patterns (e.g., rain days with higher CVR)?\n"
                        "Keep response to 3 paragraphs."
                    )
                    with st.spinner("🤖 Analyzing weather patterns..."):
                        w_result = call_claude(
                            w_prompt,
                            system="You are a retail analytics assistant. Analyze weather impact on store performance with actionable recommendations. Be concise.",
                            space_notes=st.session_state.get("current_space_notes", ""),
                        )
                    if "⚠️" in w_result:
                        st.warning(w_result)
                    else:
                        st.markdown(w_result)
                except Exception as _e:
                    st.warning(f"Weather AI error: {_e}")
    else:
        st.caption("Weather data not in cache. Re-run precompute to enable weather analysis.")

    st.markdown("---")

    # ── 4. Traffic Uplift ──────────────────────────────────────────────────────
    st.markdown("#### Traffic Uplift vs Weekday Average")
    _info("""
**Traffic Uplift** = % change vs weekday average.
0% dashed line = weekday baseline; above = better, below = worse.
Visitor Uplift (amber): % vs weekday visitors. CVR Uplift (gold): % vs weekday CVR.
""")

    uplift_df = _cached_uplift(daily_stats)
    st.caption("Baseline: weekday average. Uplift = % change vs baseline.")

    fig_uplift = go.Figure()
    fig_uplift.add_trace(go.Scatter(
        x=uplift_df["date"], y=uplift_df["uplift_visitor"],
        name="Visitor Uplift %", line=dict(color=AMBER, width=2),
    ))
    fig_uplift.add_trace(go.Scatter(
        x=uplift_df["date"], y=uplift_df["uplift_cvr"],
        name="CVR Uplift %", line=dict(color=GOLD, width=2),
    ))
    fig_uplift.add_hline(y=0, line_dash="dot", line_color=SLATE_GRAY)
    fig_uplift.update_layout(
        title="Traffic Uplift vs Weekday Average",
        xaxis_title="Date", yaxis_title="Uplift (%)",
        template="plotly_white", height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_theme(fig_uplift)
    st.plotly_chart(fig_uplift, use_container_width=True)

    # ── 5. Traffic Context Scatter ─────────────────────────────────────────────
    qc_col = "quality_cvr" if "quality_cvr" in daily_stats.columns else "conversion_rate"
    if has_weather and "day_type" in daily_stats.columns:
        st.markdown("#### Traffic Context — FP × CVR")
        _info("""
- X: Floating Pop | Y: Quality CVR. Color: weather. Markers: day type.
- Cross = group mean. Use to spot clusters and outliers.
""", label="📖 Scatter")
        ds = daily_stats.copy()
        ds["qcvr"] = ds[qc_col]
        fig_scatter = go.Figure()
        _sym = {"weekday": "circle", "weekend": "square", "holiday": "diamond"}
        for weather in ["Sunny", "Rain", "Snow", "Unknown"]:
            sub = ds[ds["weather"] == weather]
            if sub.empty:
                continue
            syms = [_sym.get(str(dt).lower(), "circle") for dt in sub["day_type"]]
            fig_scatter.add_trace(go.Scatter(
                x=sub["floating_unique"], y=sub["qcvr"],
                mode="markers", name=weather,
                marker=dict(size=12, color=_weather_color(weather), symbol=syms),
                text=sub.apply(lambda r: f"{r['date']} | {r['weather']} | {r['day_type']}<br>FP:{r['floating_unique']} CVR:{r['qcvr']:.1f}%", axis=1),
                hovertemplate="%{text}<extra></extra>",
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[sub["floating_unique"].mean()], y=[sub["qcvr"].mean()],
                mode="markers", name=f"{weather} avg",
                marker=dict(symbol="x", size=14, color=_weather_color(weather), line=dict(width=2)),
                showlegend=False,
            ))
        fig_scatter.update_layout(
            xaxis_title="Floating Pop (FP)", yaxis_title="Quality CVR (%)",
            height=400, template="plotly_white",
        )
        apply_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Insight caption
        by_weather = daily_stats.groupby("weather", as_index=False)[qc_col].mean()
        parts = [
            f"{_WEATHER_ICON.get(rec['weather'], rec['weather'])}: avg CVR {rec[qc_col]:.1f}%"
            for rec in by_weather.to_dict("records")
        ]
        best_dt = daily_stats.groupby("day_type", as_index=False)[qc_col].mean().sort_values(qc_col, ascending=False)
        if not best_dt.empty:
            parts.append(f"| Highest CVR day type: {best_dt.iloc[0]['day_type']}")
        st.caption(" ".join(parts))

    st.markdown("---")

    # ── 6. Dwell Intelligence ──────────────────────────────────────────────────
    st.markdown("#### Dwell Intelligence — dwell segments")
    _info("""
Dwell time (체류시간) in 3 segments: **Short** (<3 min), **Medium** (3–10 min), **Long** (10+ min).
High short = quick visits or low engagement; high long = high-engagement customers.
Medium-dominant = typical browsing; promotions tend to work.
""")

    if not sessions_display.empty and "dwell_seconds" in sessions_display.columns:
        dist = _cached_dwell_dist(sessions_display)
        if not dist.empty:
            fig_dwell = px.bar(dist, x="segment", y="count", color="ratio",
                               color_continuous_scale=["#f8fafc", GOLD, DEEP_NAVY])
            fig_dwell.update_layout(height=320, xaxis_title="Segment",
                                    yaxis_title="Count", title="Dwell time distribution")
            apply_theme(fig_dwell)
            st.plotly_chart(fig_dwell, use_container_width=True)
            st.dataframe(
                dist.rename(columns={"segment":"Segment","count":"Count","ratio":"Ratio (%)"}),
                use_container_width=True, hide_index=True,
            )

            # ── Dwell auto-insight ─────────────────────────────────────────────
            dominant     = dist.loc[dist["count"].idxmax()]
            dom_seg      = dominant["segment"]
            dom_ratio    = float(dominant["ratio"])
            short_ratio  = float(dist.loc[dist["segment"] == "Short",  "ratio"].iloc[0]) if "Short"  in dist["segment"].values else 0.0
            medium_ratio = float(dist.loc[dist["segment"] == "Medium", "ratio"].iloc[0]) if "Medium" in dist["segment"].values else 0.0
            long_ratio   = float(dist.loc[dist["segment"] == "Long",   "ratio"].iloc[0]) if "Long"   in dist["segment"].values else 0.0
            _dwell_seg_msg = {
                "Short":  "Most visitors leave quickly — consider entrance-side promotions and impulse-buy displays.",
                "Medium": "Visitors are actively browsing — promotions and in-store displays are most effective.",
                "Long":   "High-engagement customers dominate — loyalty programs and staff assistance can maximise conversion.",
            }
            dw_insight_parts = [
                f"📊 <b>Dominant segment</b>: {dom_seg} ({dom_ratio:.0f}% of sessions) — "
                + _dwell_seg_msg.get(dom_seg, ""),
            ]
            if short_ratio > 50:
                dw_insight_parts.append(
                    "⚠️ Over <b>50% Short-dwell sessions</b> (<3 min) — high bounce rate. "
                    "Investigate entrance friction, product visibility, or pricing signals."
                )
            if long_ratio > 30:
                dw_insight_parts.append(
                    "💡 Significant <b>Long-dwell segment</b> (10+ min) — "
                    "these high-engagement visitors are prime candidates for loyalty programs and upselling."
                )
            if medium_ratio > 50:
                dw_insight_parts.append(
                    "✅ <b>Medium-dwell dominant</b> (3–10 min) — healthy browsing behaviour. "
                    "In-store promotions and product displays are well-positioned to drive conversion."
                )
            st.markdown(
                '<div style="background:#f1f5f9;border-left:3px solid #c49a3a;padding:10px 14px;'
                'border-radius:4px;margin:8px 0;font-size:0.9rem;">'
                + "<br>".join(dw_insight_parts)
                + "</div>",
                unsafe_allow_html=True,
            )

            # ── AI Dwell Analysis button ───────────────────────────────────────
            if _has_api_key():
                if st.button("🤖 AI Dwell Analysis", key="ai_dwell_btn"):
                    try:
                        dwell_text = dist.to_string(index=False)
                        d_prompt = (
                            f"[Dwell time distribution — {space_name}]\n{dwell_text}\n\n"
                            "Analyze:\n"
                            "1. What does this dwell distribution reveal about customer engagement level?\n"
                            "2. What are 2 specific actions to improve dwell time and conversion?\n"
                            "3. Is there anything surprising or concerning?\n"
                            "Keep response to 3 paragraphs."
                        )
                        with st.spinner("🤖 Analyzing dwell patterns..."):
                            d_result = call_claude(
                                d_prompt,
                                system="You are a retail analytics assistant specialising in customer behaviour and dwell time analysis. Be concise and actionable.",
                                space_notes=st.session_state.get("current_space_notes", ""),
                            )
                        if "⚠️" in d_result:
                            st.warning(d_result)
                        else:
                            st.markdown(d_result)
                    except Exception as _e:
                        st.warning(f"Dwell AI error: {_e}")
    else:
        st.caption("No session data for dwell analysis.")

    st.markdown("---")

    # ── 7. Device Mix ──────────────────────────────────────────────────────────
    st.markdown("#### Device Mix — iOS vs Android")
    _info("""
Visitor sessions by iOS (Apple) vs Android. Track ratio changes over time.
Note: iOS MAC randomization may affect counts; focus on trends rather than absolute share.
""")

    if not device_mix.empty:
        mix = _cached_device_mix_summary(device_mix)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("iOS Share", f"{mix['apple_ratio']*100:.1f}%", f"{mix['apple_count']} sessions")
        with c2:
            st.metric("Android Share", f"{mix['android_ratio']*100:.1f}%", f"{mix['android_count']} sessions")

        fig_pie = go.Figure(data=[go.Pie(
            labels=["iOS", "Android"],
            values=[mix["apple_count"], mix["android_count"]],
            hole=0.5, marker_colors=[DEEP_NAVY, GOLD],
        )])
        fig_pie.update_layout(height=280, showlegend=True, margin=dict(t=20))
        apply_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

        # ── Device mix auto-insight ────────────────────────────────────────────
        ios_pct = mix["apple_ratio"] * 100
        adr_pct = mix["android_ratio"] * 100
        dm_insight_parts = [
            f"📱 <b>iOS</b>: {ios_pct:.1f}% of sessions &nbsp;|&nbsp; 🤖 <b>Android</b>: {adr_pct:.1f}%",
        ]
        dominant_device = "iOS" if ios_pct >= adr_pct else "Android"
        if abs(ios_pct - adr_pct) > 20:
            dm_insight_parts.append(
                f"💡 <b>Strong {dominant_device} dominance</b> ({max(ios_pct, adr_pct):.0f}%). "
                + (
                    "iOS MAC randomization is more aggressive — session counts may be underestimated. "
                    "Focus on AST (accumulated stay time) as a more stable engagement metric."
                    if dominant_device == "iOS" else
                    "Android provides more stable session tracking. Visitor counts here are reliable."
                )
            )
        elif abs(ios_pct - adr_pct) <= 10:
            dm_insight_parts.append(
                "✅ <b>Balanced device mix</b> — near-equal iOS/Android split. "
                "This is typical for general retail; no significant demographic skew detected."
            )
        st.markdown(
            '<div style="background:#f1f5f9;border-left:3px solid #c49a3a;padding:10px 14px;'
            'border-radius:4px;margin:8px 0;font-size:0.9rem;">'
            + "<br>".join(dm_insight_parts)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No session/device data.")

    st.markdown("---")

    # ── 8. AI Anomaly Detection ────────────────────────────────────────────────
    st.markdown("#### 🤖 AI Anomaly Detection")
    st.caption(
        "Find dates outside mean ± 2σ and interpret causes (weather, holiday, day type)."
    )

    if not _has_api_key():
        st.info("Set ANTHROPIC_API_KEY to use AI anomaly detection.")
    elif daily_stats.empty:
        st.caption("No daily data. Run precompute first.")
    else:
        metric_choice = st.selectbox(
            "Metric for anomaly (mean ± 2σ)",
            options=["conversion_rate", "visitor_count", "floating_unique", "dwell_seconds_mean"],
            format_func=lambda x: {
                "conversion_rate": "CVR (%)",
                "visitor_count": "Visitors",
                "floating_unique": "Floating Pop (FP)",
                "dwell_seconds_mean": "Avg dwell (sec)",
            }.get(x, x),
            key="anomaly_metric_select",
        )

        if st.button("🤖 Run anomaly analysis", key="ai_anomaly_btn"):
            try:
                anom_ds = _ensure_day_type(loader)
                col = metric_choice
                if col not in anom_ds.columns or anom_ds[col].dropna().empty:
                    st.warning("No data for selected metric.")
                else:
                    mu  = anom_ds[col].mean()
                    sig = anom_ds[col].std()
                    anom_rows = anom_ds[((anom_ds[col] - mu).abs() > 2 * sig)].copy() if sig > 0 else anom_ds.head(0)

                    if anom_rows.empty:
                        st.info("No dates outside 2σ. Data is stable.")
                        anom_summary = "(no anomalies)"
                    else:
                        disp_cols = [c for c in ["date", "day_type", "weather", col,
                                                  "visitor_count", "floating_unique", "conversion_rate"] if c in anom_rows.columns]
                        st.dataframe(anom_rows[disp_cols].sort_values("date"), use_container_width=True, hide_index=True)
                        anom_summary = anom_rows[disp_cols].to_json(orient="records", force_ascii=False, date_format="iso")

                    stats_cols = [c for c in ["date", "day_type", "weather", "visitor_count",
                                               "floating_unique", "conversion_rate", "dwell_seconds_mean"] if c in anom_ds.columns]
                    stats_json = anom_ds[stats_cols].to_json(orient="records", force_ascii=False, date_format="iso")
                    metric_label = {
                        "conversion_rate": "CVR (%)",
                        "visitor_count": "Visitors",
                        "floating_unique": "Floating Pop (FP)",
                        "dwell_seconds_mean": "Avg dwell (sec)",
                    }.get(col, col)

                    anom_prompt = f"""[Daily data — {space_name}]
{stats_json}

[Anomaly dates — {metric_label}, outside mean ± 2σ]
Mean(μ): {mu:.2f}, Std(σ): {sig:.2f}
{anom_summary}

Analyze:
1. For each anomaly date, explain possible causes (weather, holiday, day type) with evidence.
2. Two operational insights from the anomaly pattern.
If no anomalies, assess stability and notable features."""

                    anom_system = (
                        "You are a Hermes anomaly analyst. Interpret statistical anomalies with Cause-Effect and give actionable insights in English."
                    )

                    with st.spinner("🤖 Analyzing anomalies..."):
                        anom_result = call_claude(
                            anom_prompt,
                            system=anom_system,
                            space_notes=st.session_state.get("current_space_notes", ""),
                        )

                    if "⚠️" in anom_result:
                        st.warning(anom_result)
                    else:
                        st.markdown(anom_result)

            except Exception as _anom_exc:
                st.warning(f"🤖 Anomaly analysis error: {_anom_exc}")

    st.markdown("---")

    # ── 9. AI Cause Analysis Chat ──────────────────────────────────────────────
    st.markdown("#### 🤖 Cause Analysis Chat")
    st.caption("Ask questions based on the data. e.g. 'Why is CVR lower on rainy days?'")

    if not _has_api_key():
        st.info("Set ANTHROPIC_API_KEY in secrets or environment to use AI.")
    else:
        user_question = st.text_input(
            "🤖 Ask a question",
            key="ai_chat_input",
            placeholder="e.g. Why are weekend visitors lower than weekday?",
        )

        if user_question:
            try:
                ce_daily = loader.get_daily_stats()
                ce_daily_with_type = add_day_type_to_daily_stats(ce_daily) if not ce_daily.empty else ce_daily

                ctx_cols = [c for c in [
                    "date", "floating_unique", "visitor_count",
                    "conversion_rate", "dwell_seconds_mean", "day_type", "weather",
                ] if c in ce_daily_with_type.columns]

                ctx_df   = ce_daily_with_type[ctx_cols].tail(30)
                ctx_json = ctx_df.to_json(orient="records", force_ascii=False, date_format="iso")

                system_prompt_ce = (
                    "You are a Hermes cause-effect analyst. Answer in English based only on the given data; mark speculation as 'Beyond data:'. Focus on weather, day type, holiday vs traffic metrics."
                )
                user_prompt_ce = f"""[Store data — {space_name}]
{ctx_json}

[Question]
{user_question}

Answer concisely based on the data. Mark any speculation as "Beyond data:"
"""

                with st.spinner("🤖 Analyzing..."):
                    ce_result = call_claude(
                        user_prompt_ce,
                        system=system_prompt_ce,
                        space_notes=st.session_state.get("current_space_notes", ""),
                    )

                st.session_state["ai_last_response"] = ce_result

                if "⚠️" in ce_result:
                    st.warning(ce_result)
                else:
                    st.markdown(ce_result)

            except Exception as _ce_exc:
                st.warning(f"🤖 AI error: {_ce_exc}")

        elif st.session_state.get("ai_last_response"):
            st.markdown("**Last response:**")
            st.markdown(st.session_state["ai_last_response"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Report 리포트
# ─────────────────────────────────────────────────────────────────────────────

def _render_report_config_panel(daily_stats: pd.DataFrame):
    """Render report period and options. Returns (period_tuple, include_options_dict)."""
    st.markdown("##### 📋 Weekly Traffic Report")
    use_auto = st.radio(
        "Report period",
        options=["Auto (last week)", "Custom dates"],
        index=0,
        horizontal=True,
        key="report_period_mode",
    )
    period = None
    if use_auto == "Auto (last week)":
        ds = daily_stats.sort_values("date").tail(14)
        if len(ds) >= 7:
            this_week = ds.tail(7)
            period = (str(this_week["date"].iloc[0]), str(this_week["date"].iloc[-1]))
        else:
            st.caption("At least 7 days of data required.")
    else:
        dr = sorted(daily_stats["date"].dropna().unique())
        if len(dr):
            def _to_date(x):
                if hasattr(x, "date"): return x.date()
                return pd.Timestamp(str(x)).date()
            start = st.date_input("Start date", value=_to_date(dr[-7]) if len(dr) >= 7 else _to_date(dr[0]), key="report_start")
            end = st.date_input("End date", value=_to_date(dr[-1]), key="report_end")
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            if start <= end and (end - start).days <= 14:
                period = (start_str, end_str)
            else:
                st.caption("Period must be at most 14 days.")
    st.markdown("**Include**")
    include = {
        "kpi": st.checkbox("Key KPIs & vs prev week", value=True, key="inc_kpi"),
        "funnel": st.checkbox("Dwell funnel", value=True, key="inc_funnel"),
        "context": st.checkbox("Weather & day context", value=True, key="inc_context"),
        "ai": st.checkbox("AI insights (Claude)", value=True, key="inc_ai"),
        "prediction": st.checkbox("Next week prediction", value=True, key="inc_pred"),
    }
    return period, include


def _prepare_report_data(daily_stats: pd.DataFrame, period: tuple, space_notes: str):
    """Build report_data dict for the given period (start_date, end_date)."""
    if not period or len(period) != 2:
        return {}
    start, end = period[0], period[1]
    df = daily_stats[daily_stats["date"].astype(str) >= start]
    df = df[df["date"].astype(str) <= end].sort_values("date")
    if df.empty or len(df) < 1:
        return {}
    prev = daily_stats[daily_stats["date"].astype(str) < start].sort_values("date").tail(7)
    this_for_wow = df.tail(7)
    combined = pd.concat([prev, this_for_wow], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    wow = _cached_wow(combined, days_per_week=7) if len(combined) >= 7 else {}
    qc = "quality_cvr" if "quality_cvr" in df.columns else "conversion_rate"
    qv = "quality_visitor_count" if "quality_visitor_count" in df.columns else "visitor_count"
    fp_mean, fp_std = df["floating_unique"].mean(), df["floating_unique"].std() or 0
    cvr_mean, cvr_std = df[qc].mean(), df[qc].std() or 0
    fp_anomaly = (df["floating_unique"] - fp_mean).abs() > 1.5 * fp_std if fp_std else pd.Series(False, index=df.index)
    cvr_anomaly = (df[qc] - cvr_mean).abs() > 1.5 * cvr_std if cvr_std else pd.Series(False, index=df.index)
    anomaly_dates = set(df.loc[fp_anomaly | cvr_anomaly, "date"].astype(str))
    total_v = df["visitor_count"].sum() or 1
    short = df["short_dwell_count"].sum() if "short_dwell_count" in df.columns else 0
    medium = df["medium_dwell_count"].sum() if "medium_dwell_count" in df.columns else 0
    long = df["long_dwell_count"].sum() if "long_dwell_count" in df.columns else 0
    quality_v = (medium + long) if (medium + long) else df[qv].sum() if qv in df.columns else 0
    fp_total = df["floating_unique"].sum() or 1
    funnel = {
        "short_pct": short / total_v * 100 if total_v else 0,
        "medium_pct": medium / total_v * 100 if total_v else 0,
        "long_pct": long / total_v * 100 if total_v else 0,
        "quality_visitor_ratio": quality_v / total_v * 100 if total_v else 0,
        "long_ratio": long / total_v * 100 if total_v else 0,
        "quality_cvr": quality_v / fp_total * 100 if fp_total else 0,
    }
    first_row = df.iloc[0]
    daily_weather = [
        {"date": str(r["date"]), "weather": r.get("weather", "Unknown"), "temp_max": r.get("temp_max"), "temp_min": r.get("temp_min")}
        for r in df.to_dict("records")
    ]
    ctx = {
        "daily_weather": daily_weather,
        "season": first_row.get("season", "-"),
        "month_label": first_row.get("month_label", "-"),
        "holiday_period": first_row.get("holiday_period", "-"),
        "space_notes": space_notes or "",
    }
    forecast = fetch_weather_forecast(days=16)
    preds = predict_next_week(daily_stats, report_end_date=end, forecast_df=forecast)
    this_week = {
        "floating": float(df["floating_unique"].sum()),
        "quality_cvr": float(df[qc].mean()) if qc in df.columns else 0.0,
    }
    weather_col = df["weather"].astype(str) if "weather" in df.columns else pd.Series(["Unknown"] * len(df))
    dominant_weather = weather_col.mode().iloc[0] if len(weather_col) else "Sunny"
    week_stats = {
        "date_range": f"{start} - {end}",
        "dominant_weather": dominant_weather,
    }
    hol_df = df[df["is_holiday"] == True] if "is_holiday" in df.columns else pd.DataFrame()
    holiday_info = {
        "period": first_row.get("holiday_period", "None") or "None",
        "days": int(hol_df.shape[0]) if not hol_df.empty else 0,
    }
    report_data = {
        "kpi": wow,
        "daily": df,
        "anomaly_dates": list(anomaly_dates),
        "funnel": funnel,
        "context": ctx,
        "predictions": preds,
        "prediction_comment": "",
        "this_week": this_week,
        "week_stats": week_stats,
        "holiday_info": holiday_info,
        "kpi_summary": "",
        "context_comment": "",
    }
    return report_data


def _build_dwell_funnel_chart(report_data: dict):
    """Donut chart: Short / Medium / Long dwell ratio for the week."""
    funnel = report_data.get("funnel", {})
    short = funnel.get("short_pct", 0) or 0
    medium = funnel.get("medium_pct", 0) or 0
    long_ = funnel.get("long_pct", 0) or 0
    if short == 0 and medium == 0 and long_ == 0:
        return None
    fig = go.Figure(go.Pie(
        labels=["Short (<3min)", "Medium (3-10min)", "Long (10min+)"],
        values=[short, medium, long_],
        hole=0.55,
        marker_colors=[SLATE_GRAY, DEEP_NAVY, GOLD],
        textinfo="label+percent",
        textfont_size=10,
    ))
    fig.add_annotation(
        text=f"Quality<br>{medium + long_:.1f}%",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color=DEEP_NAVY, family="Helvetica"),
    )
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        paper_bgcolor="#ffffff",
    )
    apply_theme(fig)
    return fig


def _build_prediction_chart(predictions: list, this_week_daily_avg: float):
    """Bar chart: next week daily FP with this week avg as dashed line."""
    if not predictions:
        return None
    dates = [p.get("date_obj") or p.get("date") for p in predictions]
    date_labels = []
    for d in dates:
        if hasattr(d, "strftime"):
            wd = d.weekday()
            date_labels.append(f"{d.strftime('%b %d')}\n({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][wd]})")
        else:
            date_labels.append(str(d)[:10])
    fp_means = [p.get("floating_mean", 0) for p in predictions]
    fp_stds = [p.get("floating_std", 0) for p in predictions]
    colors = [GOLD if (hasattr(d, "weekday") and d.weekday() >= 5) else DEEP_NAVY for d in dates]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=date_labels, y=fp_means,
        error_y=dict(type="data", array=fp_stds, visible=True),
        marker_color=colors,
        name="Predicted FP",
    ))
    fig.add_hline(
        y=this_week_daily_avg,
        line_dash="dash",
        line_color=AMBER,
        annotation_text=f"This week avg: {this_week_daily_avg:,.0f}",
        annotation_position="top right",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=30, t=30, b=50),
        plot_bgcolor="#f8f9fc",
        paper_bgcolor="#ffffff",
        showlegend=False,
        yaxis_title="Floating Pop",
        font=dict(size=10),
        xaxis=dict(tickangle=0),
    )
    apply_theme(fig)
    return fig


def _build_report_charts(report_data: dict):
    """Build Plotly figures for traffic, funnel (stacked bar + donut), and prediction."""
    figs = {}
    df = report_data.get("daily", pd.DataFrame())
    qc = "quality_cvr" if "quality_cvr" in df.columns else "conversion_rate"
    anomaly = set(report_data.get("anomaly_dates", []))
    if not df.empty:
        fig_t = go.Figure()
        colors = [AMBER if str(d) in anomaly else DEEP_NAVY for d in df["date"]]
        fig_t.add_trace(go.Bar(x=df["date"], y=df["floating_unique"], name="Floating Pop", marker_color=colors))
        fig_t.add_trace(go.Scatter(
            x=df["date"], y=df[qc], name=f"{qc} (%)", yaxis="y2",
            line=dict(color=GOLD, width=2), mode="lines+markers",
        ))
        fig_t.update_layout(
            barmode="overlay", height=380,
            yaxis=dict(title="Floating Pop"),
            yaxis2=dict(overlaying="y", side="right", title=f"{qc} (%)", range=[0, max(float(df[qc].max()) * 1.2, 5)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_t)
        figs["traffic"] = fig_t
    if not df.empty and all(c in df.columns for c in ["short_dwell_count", "medium_dwell_count", "long_dwell_count", qc]):
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(x=df["date"], y=df["short_dwell_count"], name="Short (<3min)", marker_color=SLATE_GRAY))
        fig_f.add_trace(go.Bar(x=df["date"], y=df["medium_dwell_count"], name="Medium (3-10min)", marker_color=DEEP_NAVY))
        fig_f.add_trace(go.Bar(x=df["date"], y=df["long_dwell_count"], name="Long (10min+)", marker_color=GOLD))
        fig_f.update_layout(barmode="stack", height=320, xaxis_title="Date", yaxis_title="Count")
        fig_f.add_trace(go.Scatter(
            x=df["date"], y=df[qc], name="Quality CVR (%)", yaxis="y2",
            line=dict(color=AMBER, width=2, dash="dot"),
        ))
        fig_f.update_layout(
            yaxis2=dict(overlaying="y", side="right", range=[0, max(float(df[qc].max()) * 1.2, 10)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_f)
        figs["funnel"] = fig_f
    fig_donut = _build_dwell_funnel_chart(report_data)
    if fig_donut is not None:
        figs["dwell_funnel"] = fig_donut
    preds = report_data.get("predictions", [])
    this_week = report_data.get("this_week", {})
    this_avg = (this_week.get("floating") or 0) / 7 if preds else 0
    fig_pred = _build_prediction_chart(preds, this_avg)
    if fig_pred is not None:
        figs["prediction"] = fig_pred
    return figs


def _render_report_preview(report_data: dict, ai_insight: str, chart_figs: dict):
    """Render report preview with styled sections matching PDF layout."""
    st.markdown("---")
    st.markdown("#### Report Preview")

    kpi = report_data.get("kpi", {})
    tw = kpi.get("this_week", {})
    d = kpi.get("delta", {})

    # ── Section 1: Key Metrics ─────────────────────────────────────────────────
    st.markdown(
        '<div class="report-section-title">1. Key Metrics — This Week</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Floating Pop (유동인구)", f"{tw.get('floating_unique', 0):,.0f}",
              f"{d.get('floating_pct', 0):+.1f}% vs prev week" if d else "")
    c2.metric("Quality Visitors (방문자)", f"{tw.get('quality_visitor_count', 0):,.0f}",
              f"{d.get('quality_visitor_pct', 0):+.1f}% vs prev week" if d else "")
    c3.metric("Quality CVR (방문율)", f"{tw.get('quality_cvr', 0):.1f}%",
              f"{d.get('quality_cvr_pp', 0):+.1f}%p vs prev week" if d else "")
    dm = tw.get("dwell_median_seconds", 0)
    c4.metric("Median Dwell (체류시간)", f"{int(dm)//60}m {int(dm)%60}s",
              f"{d.get('dwell_median', 0):+.0f}s vs prev week" if d else "")

    kpi_summary = report_data.get("kpi_summary", "")
    if kpi_summary:
        st.markdown(
            f'<div class="report-ai-box">{kpi_summary}</div>',
            unsafe_allow_html=True,
        )

    # ── Section 2: Weekly Traffic ──────────────────────────────────────────────
    st.markdown(
        '<div class="report-section-title">2. Weekly Traffic Flow</div>',
        unsafe_allow_html=True,
    )
    if "traffic" in chart_figs:
        st.plotly_chart(chart_figs["traffic"], use_container_width=True)

    # ── Section 3: Dwell Funnel ────────────────────────────────────────────────
    st.markdown(
        '<div class="report-section-title">3. Dwell Funnel</div>',
        unsafe_allow_html=True,
    )
    funnel = report_data.get("funnel", {})
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Short (<3min)", f"{funnel.get('short_pct', 0):.1f}%")
    fc2.metric("Medium (3-10min)", f"{funnel.get('medium_pct', 0):.1f}%")
    fc3.metric("Long (10min+)", f"{funnel.get('long_pct', 0):.1f}%")
    fc4.metric("Quality CVR", f"{funnel.get('quality_cvr', 0):.1f}%")
    if "funnel" in chart_figs:
        st.plotly_chart(chart_figs["funnel"], use_container_width=True)
    if "dwell_funnel" in chart_figs:
        st.plotly_chart(chart_figs["dwell_funnel"], use_container_width=True)

    # ── Section 4: This Week Context ───────────────────────────────────────────
    st.markdown(
        '<div class="report-section-title">4. This Week Context</div>',
        unsafe_allow_html=True,
    )
    ctx = report_data.get("context", {})
    weather_rows = ctx.get("daily_weather", [])[:7]
    if weather_rows:
        ctx_df = pd.DataFrame(weather_rows)
        rename_cols = {}
        if "date" in ctx_df.columns:
            rename_cols["date"] = "Date"
        if "weather" in ctx_df.columns:
            ctx_df["weather"] = ctx_df["weather"].map(lambda w: _WEATHER_ICON.get(w, w))
            rename_cols["weather"] = "Weather"
        if "temp_max" in ctx_df.columns:
            rename_cols["temp_max"] = "High (°C)"
        if "temp_min" in ctx_df.columns:
            rename_cols["temp_min"] = "Low (°C)"
        st.dataframe(ctx_df.rename(columns=rename_cols), use_container_width=True, hide_index=True)

    st.caption(f"Season: {ctx.get('season', '-')} | Month: {ctx.get('month_label', '-')} | Holiday: {ctx.get('holiday_period', '-')}")

    context_comment = report_data.get("context_comment", "")
    if context_comment:
        st.markdown(
            f'<div class="report-ai-box">{context_comment}</div>',
            unsafe_allow_html=True,
        )

    if ctx.get("space_notes"):
        with st.expander("📝 Space Notes", expanded=False):
            st.text(ctx["space_notes"])

    # ── Section 5: AI Insights ─────────────────────────────────────────────────
    st.markdown(
        '<div class="report-section-title">5. AI Insights</div>',
        unsafe_allow_html=True,
    )
    if ai_insight:
        st.markdown(
            f'<div class="report-ai-box">{ai_insight}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("AI insights not generated for this report.")

    # ── Section 6: Next Week Outlook ───────────────────────────────────────────
    st.markdown(
        '<div class="report-section-title">6. Next Week Outlook (Reference)</div>',
        unsafe_allow_html=True,
    )
    preds = report_data.get("predictions", [])[:7]
    if preds:
        pred_rows = []
        for p in preds:
            pred_rows.append({
                "Date": str(p.get("date", ""))[:10],
                "Weather": _WEATHER_ICON.get(p.get("weather", "Unknown"), p.get("weather", "Unknown")),
                "Day Type": p.get("day_type", "-"),
                "FP (predicted)": f"{p.get('floating_mean', 0):.0f} ± {p.get('floating_std', 0):.0f}",
                "CVR (predicted)": f"{p.get('quality_cvr_mean', 0):.1f}%",
            })
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    if "prediction" in chart_figs:
        st.plotly_chart(chart_figs["prediction"], use_container_width=True)

    pred_comment = report_data.get("prediction_comment", "")
    if pred_comment:
        st.markdown(
            f'<div class="report-ai-box">{pred_comment}</div>',
            unsafe_allow_html=True,
        )


def render_report_tab(space_name: str, loader: CacheLoader) -> None:
    """Render the Report tab: config panel, generate button, preview, PDF download."""
    st.subheader("📋 Report")
    st.markdown("Generate a weekly traffic report and download as PDF.")
    daily_stats = _ensure_day_type(loader)
    if daily_stats.empty:
        st.warning("No daily data in cache. Run precompute first.")
        return
    period, include = _render_report_config_panel(daily_stats)
    if st.button("📄 Generate report", type="primary", key="report_generate_btn"):
        if not period:
            st.warning("Select report period.")
        else:
            with st.spinner("Generating report..."):
                report_data = _prepare_report_data(daily_stats, period, st.session_state.get("current_space_notes", ""))
                if not report_data:
                    st.warning("Cannot build data for selected period.")
                else:
                    space_notes = st.session_state.get("current_space_notes", "")
                    kpi_data = report_data.get("kpi", {})
                    tw = kpi_data.get("this_week", {})
                    delta = kpi_data.get("delta", {})
                    ctx = report_data.get("context", {})
                    this_week_for_ai = {
                        "floating": tw.get("floating_unique", 0),
                        "fp_delta": delta.get("floating_pct") or 0,
                        "quality_visitor": tw.get("quality_visitor_count", 0),
                        "qv_delta": delta.get("quality_visitor_pct") or 0,
                        "quality_cvr": tw.get("quality_cvr", 0),
                        "cvr_delta": delta.get("quality_cvr_pp") or 0,
                        "dwell_median_str": f"{int(tw.get('dwell_median_seconds', 0))//60}m {int(tw.get('dwell_median_seconds', 0))%60}s",
                        "dwell_delta_str": f"{delta.get('dwell_median') or 0:+.0f}s",
                    }
                    prev_week_for_ai = {}
                    if include.get("ai"):
                        try:
                            report_data["kpi_summary"] = generate_kpi_summary(
                                this_week_for_ai, prev_week_for_ai, ctx, space_notes=space_notes
                            )
                        except Exception:
                            report_data["kpi_summary"] = ""
                        try:
                            report_data["context_comment"] = generate_context_comment(
                                report_data.get("week_stats", {}),
                                report_data.get("holiday_info", {}),
                                ctx.get("season", ""),
                                space_notes=space_notes,
                            )
                        except Exception:
                            report_data["context_comment"] = ""
                    keep_cols = [c for c in ["date", "floating_unique", "quality_visitor_count", "quality_cvr", "dwell_median_seconds", "weather", "day_type"] if c in report_data["daily"].columns]
                    weekly_json = report_data["daily"][keep_cols].to_dict(orient="records")
                    prev_df = daily_stats[daily_stats["date"].astype(str) < period[0]].sort_values("date").tail(7)
                    prev_cols = [c for c in keep_cols if c in prev_df.columns]
                    prev_json = prev_df[prev_cols].to_dict(orient="records") if not prev_df.empty and prev_cols else []
                    weather_json = report_data["daily"][["date", "weather"]].to_dict(orient="records") if "weather" in report_data["daily"].columns else []
                    ctx_list = [get_day_context(str(d)) for d in report_data["daily"]["date"]]
                    ai_insight = ""
                    if include.get("ai"):
                        ai_insight = generate_weekly_report_insight(
                            weekly_json, prev_json, weather_json, ctx_list,
                            space_notes=space_notes,
                        )
                    if report_data.get("predictions") and include.get("ai"):
                        try:
                            report_data["prediction_comment"] = generate_prediction_comment(
                                report_data["predictions"], space_notes=space_notes
                            )
                        except Exception:
                            pass
                    chart_figs = _build_report_charts(report_data)
                    try:
                        from src.report import generate_weekly_report_pdf  # lazy — fpdf2 optional
                        pdf_bytes = generate_weekly_report_pdf(
                            report_data=report_data,
                            chart_figures=chart_figs,
                            space_name=space_name,
                            date_range=period,
                            ai_insight=ai_insight,
                        )
                    except Exception as e:
                        pdf_bytes = b""
                        st.warning(f"PDF generation failed: {e}")
                    st.session_state["report_ready"] = True
                    st.session_state["report_pdf"] = pdf_bytes
                    st.session_state["report_data"] = report_data
                    st.session_state["report_ai"] = ai_insight
                    st.session_state["report_charts"] = chart_figs
                    st.session_state["report_period"] = period
                    st.success("Report ready. Check preview below and download PDF.")
    if st.session_state.get("report_ready"):
        _render_report_preview(
            st.session_state["report_data"],
            st.session_state["report_ai"],
            st.session_state["report_charts"],
        )
        period_dl = st.session_state.get("report_period", ("", ""))
        fname = f"Hermes_Report_{space_name}_{period_dl[0]}_{period_dl[1]}.pdf"
        pdf_data = st.session_state.get("report_pdf", b"")
        if isinstance(pdf_data, bytearray):
            pdf_data = bytes(pdf_data)
        st.download_button(
            label="⬇ Download PDF",
            data=pdf_data,
            file_name=fname,
            mime="application/pdf",
            type="primary",
            key="report_pdf_dl",
        )
