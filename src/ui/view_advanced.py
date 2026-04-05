"""
Hermes Advanced Analytics View — Expert analysis for Admin users.

Contains 6 sub-tabs extracted from render_patterns:
1. Weekday Patterns
2. Heatmaps
3. Weather Effect
4. Dwell Detail
5. Device Mix
6. AI Tools (Anomaly Detection + Cause Analysis Chat)
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.day_type import add_day_type_to_daily_stats, weekday_names_en
from src.analytics.uplift import compute_uplift
from src.analytics.heatmap import build_weekday_hour_heatmap, pivot_heatmap
from src.analytics.device_craft import device_mix_summary
from src.analytics.dwell_intelligence import dwell_distribution
from src.ui.chart_theme import apply_theme
from src.ui.helpers import has_api_key
from src.ai import call_claude

# Palette (dark theme — DEEP_NAVY brightened for visibility on #0E1117 background)
DEEP_NAVY = "#4A90D9"
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"

_WEATHER_ICON = {
    "Sunny": "sun Sunny",
    "Rain": "rain Rain",
    "Snow": "snow Snow",
    "Unknown": "-- Unknown",
}


# ── Cached wrappers ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_day_type(daily_stats: pd.DataFrame) -> pd.DataFrame:
    return add_day_type_to_daily_stats(daily_stats)


@st.cache_data(show_spinner=False)
def _cached_heatmap(daily_hourly: pd.DataFrame, daily_stats: pd.DataFrame, col: str) -> pd.DataFrame:
    return build_weekday_hour_heatmap(daily_hourly, daily_stats, col)


@st.cache_data(show_spinner=False)
def _cached_uplift(daily_stats: pd.DataFrame) -> pd.DataFrame:
    return compute_uplift(daily_stats)


@st.cache_data(show_spinner=False)
def _cached_dwell_dist(sessions: pd.DataFrame) -> pd.DataFrame:
    return dwell_distribution(sessions)


@st.cache_data(show_spinner=False)
def _cached_device_mix_summary(device_mix: pd.DataFrame) -> dict:
    return device_mix_summary(device_mix)


def _ensure_day_type(loader: CacheLoader) -> pd.DataFrame:
    daily_stats = loader.get_daily_stats()
    if daily_stats.empty:
        return daily_stats
    return _cached_day_type(daily_stats)


def _info(text: str, label: str = "About this") -> None:
    """Render a collapsed expander with explanatory text."""
    with st.expander(label, expanded=False):
        st.markdown(text)


def _weather_color(weather: str) -> str:
    return {
        "Sunny": GOLD,
        "Rain": SLATE_GRAY,
        "Snow": "#93c5fd",
        "Unknown": "#d1d5db",
    }.get(weather, SLATE_GRAY)


# ── Main render function ─────────────────────────────────────────────────────

def render_advanced(space_name: str, loader: CacheLoader) -> None:
    """Render Advanced Analytics view with 6 sub-tabs. Admin only."""
    from src.auth import is_admin
    if not is_admin():
        st.error("Administrator access required.")
        return
    st.subheader("Advanced Analytics")
    st.markdown("Expert-level pattern analysis for store operations optimization.")

    daily_stats = _ensure_day_type(loader)
    if daily_stats.empty:
        st.info("No cache data available.")
        return

    daily_hourly = loader.get_daily_hourly()
    device_mix = loader.get_device_mix()
    sessions_all = loader.get_sessions_all()
    sessions_stitched = loader.get_sessions_stitched()
    sessions_display = sessions_stitched if not sessions_stitched.empty else sessions_all
    wd_en = weekday_names_en()

    tabs = st.tabs([
        "Weekday Patterns",
        "Heatmaps",
        "Weather",
        "Dwell Detail",
        "Device Mix",
        "AI Tools",
    ])

    with tabs[0]:
        _render_weekday_analysis(space_name, daily_stats, wd_en)

    with tabs[1]:
        _render_heatmap_analysis(space_name, daily_hourly, daily_stats, wd_en)

    with tabs[2]:
        _render_weather_analysis(space_name, daily_stats)

    with tabs[3]:
        _render_dwell_detail(space_name, sessions_display, daily_stats)

    with tabs[4]:
        _render_device_analysis(space_name, device_mix)

    with tabs[5]:
        _render_ai_tools(space_name, loader, daily_stats)


# ── Tab 1: Weekday Patterns ──────────────────────────────────────────────────

def _render_weekday_analysis(
    space_name: str,
    daily_stats: pd.DataFrame,
    wd_en: dict,
) -> None:
    """Day-of-week analysis with charts and AI."""
    st.markdown("#### Day-of-Week Analysis")
    _info("""
**Most critical dimension for retail management.**

- Which day brings the most foot traffic?
- Which day has the best conversion (CVR)?
- Which day do customers stay the longest?
- How predictable is each day? (Consistency - lower CV = more predictable)

Gold bars = Weekend (Sat/Sun). Navy bars = Weekday. Error bars = +/- 1 std dev.
""")

    _WD_SHORT = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    _WD_KO = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    _WEEKEND = {5, 6}

    if "weekday" not in daily_stats.columns:
        st.caption("Weekday data not available - run precompute to generate day-of-week analysis.")
        return

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

    wd_stats["std_fp"] = wd_stats["std_fp"].fillna(0)
    wd_stats["std_visitors"] = wd_stats["std_visitors"].fillna(0)
    wd_stats["std_cvr"] = wd_stats["std_cvr"].fillna(0)
    safe_avg_v = wd_stats["avg_visitors"].replace(0, 1)
    wd_stats["cv_visitors"] = (wd_stats["std_visitors"] / safe_avg_v * 100).round(1)
    wd_stats["wd_label"] = wd_stats["weekday"].apply(
        lambda w: f"{_WD_SHORT.get(w, '?')} ({_WD_KO.get(w, '?')})"
    )
    wd_stats["is_weekend"] = wd_stats["weekday"].isin(_WEEKEND)

    # Best/Worst highlights
    best_v_row = wd_stats.loc[wd_stats["avg_visitors"].idxmax()]
    best_cvr_row = wd_stats.loc[wd_stats["avg_cvr"].idxmax()]
    best_d_row = wd_stats.loc[wd_stats["avg_dwell"].idxmax()]
    worst_v_row = wd_stats.loc[wd_stats["avg_visitors"].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Busiest Day", best_v_row["wd_label"], f"avg {best_v_row['avg_visitors']:.0f} visitors")
    k2.metric("Best CVR Day", best_cvr_row["wd_label"], f"avg {best_cvr_row['avg_cvr']:.1f}%")
    k3.metric("Longest Dwell", best_d_row["wd_label"], f"{int(best_d_row['avg_dwell'])//60}m {int(best_d_row['avg_dwell'])%60}s")
    k4.metric("Slowest Day", worst_v_row["wd_label"], f"avg {worst_v_row['avg_visitors']:.0f} visitors", delta_color="inverse")

    st.markdown("---")

    # Chart A: Avg FP + Visitors by weekday
    bar_colors_fp = [GOLD if r else DEEP_NAVY for r in wd_stats["is_weekend"]]
    bar_colors_v = ["#d4a21a" if r else "#64748b" for r in wd_stats["is_weekend"]]

    fig_wd_traf = go.Figure()
    fig_wd_traf.add_trace(go.Bar(
        x=wd_stats["wd_label"], y=wd_stats["avg_fp"].round(0),
        name="Avg FP", marker_color=bar_colors_fp, opacity=0.65,
        error_y=dict(type="data", array=wd_stats["std_fp"].round(0), visible=True, color="#94a3b8"),
    ))
    fig_wd_traf.add_trace(go.Bar(
        x=wd_stats["wd_label"], y=wd_stats["avg_visitors"].round(1),
        name="Avg Visitors", marker_color=bar_colors_v,
        error_y=dict(type="data", array=wd_stats["std_visitors"].round(0), visible=True, color="#94a3b8"),
    ))
    fig_wd_traf.update_layout(
        title="Avg FP & Visitors by Weekday - Gold = Weekend",
        xaxis_title="Day of Week", yaxis_title="Count",
        barmode="group", height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_theme(fig_wd_traf)
    st.plotly_chart(fig_wd_traf, use_container_width=True)

    # Chart B: CVR by weekday
    cvr_colors = []
    max_cvr = wd_stats["avg_cvr"].max()
    for _, row in wd_stats.iterrows():
        if row["avg_cvr"] == max_cvr:
            cvr_colors.append(AMBER)
        elif row["is_weekend"]:
            cvr_colors.append(GOLD)
        else:
            cvr_colors.append(DEEP_NAVY)

    fig_wd_cvr = go.Figure()
    fig_wd_cvr.add_trace(go.Bar(
        x=wd_stats["wd_label"], y=wd_stats["avg_cvr"].round(2),
        name="Avg CVR (%)", marker_color=cvr_colors,
        error_y=dict(type="data", array=wd_stats["std_cvr"].round(2), visible=True, color="#94a3b8"),
        text=wd_stats["avg_cvr"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig_wd_cvr.update_layout(
        title="Avg CVR by Weekday - Amber = Best day",
        xaxis_title="Day of Week", yaxis_title="CVR (%)",
        height=340,
    )
    apply_theme(fig_wd_cvr)
    st.plotly_chart(fig_wd_cvr, use_container_width=True)

    # Chart C: Dwell by weekday
    wd_stats["avg_dwell_min"] = (wd_stats["avg_dwell"] / 60).round(1)
    dwell_colors = [GOLD if r else DEEP_NAVY for r in wd_stats["is_weekend"]]
    fig_wd_dwell = go.Figure()
    fig_wd_dwell.add_trace(go.Bar(
        x=wd_stats["wd_label"], y=wd_stats["avg_dwell_min"],
        name="Avg Dwell (min)", marker_color=dwell_colors,
        text=wd_stats["avg_dwell_min"].astype(str) + " min",
        textposition="outside",
    ))
    fig_wd_dwell.update_layout(
        title="Avg Dwell Time by Weekday",
        xaxis_title="Day of Week", yaxis_title="Avg Dwell (min)",
        height=320,
    )
    apply_theme(fig_wd_dwell)
    st.plotly_chart(fig_wd_dwell, use_container_width=True)

    # Chart D: Consistency (CV%)
    st.markdown("##### Day Consistency")
    cv_colors = []
    for cv in wd_stats["cv_visitors"]:
        if cv < 15:
            cv_colors.append("#22c55e")
        elif cv < 30:
            cv_colors.append(AMBER)
        else:
            cv_colors.append("#ef4444")

    fig_wd_cv = go.Figure()
    fig_wd_cv.add_trace(go.Bar(
        x=wd_stats["wd_label"], y=wd_stats["cv_visitors"],
        name="CV (%) - Visitor variability", marker_color=cv_colors,
        text=wd_stats["cv_visitors"].astype(str) + "%",
        textposition="outside",
    ))
    fig_wd_cv.add_hline(y=15, line_dash="dot", line_color="#22c55e")
    fig_wd_cv.add_hline(y=30, line_dash="dot", line_color="#ef4444")
    fig_wd_cv.update_layout(
        title="Visitor Variability by Weekday (green < 15% = predictable)",
        xaxis_title="Day of Week", yaxis_title="CV (%)",
        height=300,
    )
    apply_theme(fig_wd_cv)
    st.plotly_chart(fig_wd_cv, use_container_width=True)

    # Detail Table
    with st.expander("Full Weekday Summary Table", expanded=False):
        tbl = wd_stats[[
            "wd_label", "days", "avg_fp", "avg_visitors", "avg_cvr", "avg_dwell_min", "cv_visitors"
        ]].copy()
        tbl["avg_fp"] = tbl["avg_fp"].round(0).astype(int)
        tbl["avg_visitors"] = tbl["avg_visitors"].round(1)
        tbl["avg_cvr"] = tbl["avg_cvr"].round(2)
        tbl = tbl.rename(columns={
            "wd_label": "Day", "days": "# Days", "avg_fp": "Avg FP",
            "avg_visitors": "Avg Visitors", "avg_cvr": "Avg CVR (%)",
            "avg_dwell_min": "Avg Dwell (min)", "cv_visitors": "Variability CV (%)",
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # AI Button
    if has_api_key():
        st.markdown("")
        if st.button("AI Weekday Pattern Analysis", key="adv_ai_weekday_btn"):
            _ai_weekday_analysis(space_name, wd_stats)


def _ai_weekday_analysis(space_name: str, wd_stats: pd.DataFrame) -> None:
    """Generate AI analysis for weekday patterns."""
    try:
        wd_summary_lines = []
        for _, row in wd_stats.iterrows():
            wd_summary_lines.append(
                f"  {row['wd_label']}: avg_visitors={row['avg_visitors']:.1f}, "
                f"avg_fp={row['avg_fp']:.0f}, avg_cvr={row['avg_cvr']:.2f}%, "
                f"avg_dwell={row['avg_dwell_min']:.1f}min, cv={row['cv_visitors']:.1f}%"
            )
        wd_text = "\n".join(wd_summary_lines)

        ai_prompt = f"""[Store: {space_name}]
Weekday performance data (averages across all weeks in period):
{wd_text}

Provide a concise analysis (4-5 bullet points) covering:
1. Most notable weekday pattern (e.g., weekend surge, midweek dip)
2. Best day for promotion timing (high CVR) and why
3. Staffing recommendation based on traffic + variability
4. One surprising or counter-intuitive finding from the data
5. One watch-out or risk (e.g., high variability day, low CVR despite high traffic)

Be specific with numbers. English only."""

        with st.spinner("Analyzing weekday patterns..."):
            result = call_claude(
                ai_prompt,
                system="You are a retail analytics expert. Give specific, data-driven insights for store operations. Concise bullet points only.",
                space_notes=st.session_state.get("current_space_notes", ""),
            )
        st.markdown(result)
    except Exception as e:
        st.warning(f"AI analysis error: {e}")


# ── Tab 2: Heatmaps ──────────────────────────────────────────────────────────

def _render_heatmap_analysis(
    space_name: str,
    daily_hourly: pd.DataFrame,
    daily_stats: pd.DataFrame,
    wd_en: dict,
) -> None:
    """Weekday x Hour heatmaps for FP and CVR."""
    st.markdown("#### Weekday x Hour Heatmap")
    _info("""
Rows = day of week, Columns = hour (0-23). Darker cell = higher average.

Use together with the charts above:
- FP heatmap: when does foot traffic arrive by day?
- CVR heatmap: which day + hour combo has the best conversion?
""")

    if daily_hourly.empty or daily_stats.empty:
        st.caption("No hourly data available.")
        return

    heatmap_fp = _cached_heatmap(daily_hourly, daily_stats, "floating_count")
    heatmap_cvr = _cached_heatmap(daily_hourly, daily_stats, "conversion_rate")

    tab_fp, tab_cvr = st.tabs(["FP Heatmap", "CVR Heatmap"])

    with tab_fp:
        if not heatmap_fp.empty:
            pivot_fp = pivot_heatmap(heatmap_fp)
            pivot_fp.index = [wd_en.get(i, str(i)) for i in pivot_fp.index]
            fig_hm_fp = px.imshow(
                pivot_fp,
                labels=dict(x="Hour", y="Weekday", color="Floating Pop"),
                aspect="auto",
                color_continuous_scale=["#1a2035", AMBER, "#ccd6f6"],
            )
            fig_hm_fp.update_layout(height=400, title="Floating Pop - Weekday x Hour (avg)")
            apply_theme(fig_hm_fp)
            st.plotly_chart(fig_hm_fp, use_container_width=True)
        else:
            st.caption("No floating data.")

    with tab_cvr:
        if not heatmap_cvr.empty:
            pivot_cvr = pivot_heatmap(heatmap_cvr)
            pivot_cvr.index = [wd_en.get(i, str(i)) for i in pivot_cvr.index]
            fig_hm_cvr = px.imshow(
                pivot_cvr,
                labels=dict(x="Hour", y="Weekday", color="CVR (%)"),
                aspect="auto",
                color_continuous_scale=["#1a2035", GOLD, "#ccd6f6"],
            )
            fig_hm_cvr.update_layout(height=400, title="CVR - Weekday x Hour (avg)")
            apply_theme(fig_hm_cvr)
            st.plotly_chart(fig_hm_cvr, use_container_width=True)
        else:
            st.caption("No CVR data.")

    # Insights
    _wd_hm = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    hm_insights = []
    if not heatmap_fp.empty:
        best_fp = heatmap_fp.loc[heatmap_fp["value"].idxmax()]
        hm_insights.append(
            f"Peak FP slot: {_wd_hm.get(int(best_fp['weekday']), '?')} "
            f"{int(best_fp['hour']):02d}:00 - avg {best_fp['value']:.0f} devices"
        )
    if not heatmap_cvr.empty:
        best_cvr = heatmap_cvr.loc[heatmap_cvr["value"].idxmax()]
        hm_insights.append(
            f"Best CVR slot: {_wd_hm.get(int(best_cvr['weekday']), '?')} "
            f"{int(best_cvr['hour']):02d}:00 - avg {best_cvr['value']:.1f}%"
        )
    if hm_insights:
        st.info(" | ".join(hm_insights))

    # AI Button
    if has_api_key() and (not heatmap_fp.empty or not heatmap_cvr.empty):
        if st.button("AI Heatmap Insight", key="adv_ai_heatmap_btn"):
            _ai_heatmap_analysis(space_name, heatmap_fp, heatmap_cvr, _wd_hm)


def _ai_heatmap_analysis(
    space_name: str,
    heatmap_fp: pd.DataFrame,
    heatmap_cvr: pd.DataFrame,
    wd_map: dict,
) -> None:
    """Generate AI analysis for heatmaps."""
    try:
        hm_lines = []
        if not heatmap_fp.empty:
            for _, hr in heatmap_fp.nlargest(12, "value").iterrows():
                hm_lines.append(f"FP  | {wd_map.get(int(hr['weekday']), '?')} {int(hr['hour']):02d}:00 -> {hr['value']:.0f}")
        if not heatmap_cvr.empty:
            for _, hr in heatmap_cvr.nlargest(12, "value").iterrows():
                hm_lines.append(f"CVR | {wd_map.get(int(hr['weekday']), '?')} {int(hr['hour']):02d}:00 -> {hr['value']:.1f}%")

        hm_prompt = (
            f"[Weekday x Hour heatmap - top slots - {space_name}]\n"
            + "\n".join(hm_lines)
            + "\n\nAnalyze:\n"
            "1. What time windows consistently drive the most foot traffic (FP)?\n"
            "2. When does CVR peak - does it align with FP or is there a mismatch?\n"
            "3. Recommend 2 concrete operational actions (staffing, promotions) based on these patterns.\n"
            "Keep response to 3 paragraphs max."
        )
        with st.spinner("Analyzing heatmap patterns..."):
            result = call_claude(
                hm_prompt,
                system="You are a retail analytics assistant. Provide actionable insights from weekday x hour traffic and CVR heatmap data. Be specific and concise.",
                space_notes=st.session_state.get("current_space_notes", ""),
            )
        st.markdown(result)
    except Exception as e:
        st.warning(f"Heatmap AI error: {e}")


# ── Tab 3: Weather Effect ────────────────────────────────────────────────────

def _render_weather_analysis(space_name: str, daily_stats: pd.DataFrame) -> None:
    """Weather impact analysis."""
    st.markdown("#### Weather Effect - Cause Analysis")
    _info("""
Weather as **cause**; visitors and CVR as **effect**.
Chart 1: visitors by weather. Chart 2: visitors vs precipitation (dual axis).
Table: summary by weather; CVR vs Sunny = pp diff vs sunny baseline.
""")

    has_weather = "weather" in daily_stats.columns and daily_stats["weather"].notna().any()
    if not has_weather:
        st.caption("Weather data not in cache. Re-run precompute to enable weather analysis.")
        return

    ds = daily_stats.copy()
    ds_sorted = ds.sort_values("date")

    # Daily visitors colored by weather
    bar_colors = [_weather_color(w) for w in ds_sorted["weather"]]
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=ds_sorted["date"], y=ds_sorted["visitor_count"],
        marker_color=bar_colors, name="Visitors",
    ))
    fig_daily.update_layout(
        title="Daily Visitors by Weather",
        xaxis_title="Date", yaxis_title="Visitors",
        height=340,
    )
    apply_theme(fig_daily)
    st.plotly_chart(fig_daily, use_container_width=True)

    # Weather Impact Summary
    weather_summary = ds.groupby("weather", as_index=False).agg(
        Days=("date", "count"),
        Avg_FP=("floating_unique", "mean"),
        Avg_Visitors=("visitor_count", "mean"),
        Avg_CVR=("conversion_rate", "mean"),
        Avg_Dwell_sec=("dwell_seconds_mean", "mean"),
        Avg_Precip=("precipitation", "mean"),
    ).sort_values("Avg_Visitors", ascending=False)

    weather_summary["Avg_FP"] = weather_summary["Avg_FP"].round(0).astype(int)
    weather_summary["Avg_Visitors"] = weather_summary["Avg_Visitors"].round(1)
    weather_summary["Avg_CVR"] = weather_summary["Avg_CVR"].round(2)
    weather_summary["Avg_Dwell_min"] = (weather_summary["Avg_Dwell_sec"] / 60).round(1)
    weather_summary["Avg_Precip"] = weather_summary["Avg_Precip"].fillna(0).round(1)

    x_labels = [
        f"{_WEATHER_ICON.get(w, w)}\n({int(d)} days)"
        for w, d in zip(weather_summary["weather"], weather_summary["Days"])
    ]

    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(
        x=x_labels, y=weather_summary["Avg_Visitors"],
        name="Avg Visitors",
        marker_color=[_weather_color(w) for w in weather_summary["weather"]],
        text=weather_summary["Avg_Visitors"].round(0).astype(int),
        textposition="outside",
    ))
    fig_impact.add_trace(go.Scatter(
        x=x_labels, y=weather_summary["Avg_CVR"],
        name="Avg CVR (%)", yaxis="y2",
        line=dict(color=AMBER, width=2.5),
        mode="lines+markers", marker=dict(size=10),
    ))
    fig_impact.update_layout(
        title="Weather Impact on Visitors & CVR",
        xaxis_title="Weather Condition",
        yaxis=dict(title="Avg Visitors / day"),
        yaxis2=dict(
            title="Avg CVR (%)", overlaying="y", side="right",
            showgrid=False,
            range=[0, max(weather_summary["Avg_CVR"].max() * 1.4, 5)],
        ),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_theme(fig_impact)
    st.plotly_chart(fig_impact, use_container_width=True)

    # Precipitation chart
    if "precipitation" in ds_sorted.columns:
        st.markdown("##### Precipitation vs Visitors")
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

    # Summary table
    with st.expander("Weather Summary Table", expanded=False):
        sunny_cvr = weather_summary.loc[weather_summary["weather"] == "Sunny", "Avg_CVR"]
        baseline_cvr = float(sunny_cvr.iloc[0]) if not sunny_cvr.empty else None
        weather_summary["CVR vs Sunny"] = (
            weather_summary["Avg_CVR"].apply(lambda v: f"{v - baseline_cvr:+.1f}pp")
            if baseline_cvr and baseline_cvr > 0 else "N/A"
        )
        st.dataframe(
            weather_summary[["weather", "Days", "Avg_FP", "Avg_Visitors", "Avg_CVR", "CVR vs Sunny", "Avg_Dwell_min", "Avg_Precip"]].rename(columns={
                "weather": "Weather",
                "Avg_FP": "Avg FP", "Avg_Visitors": "Avg Visitors",
                "Avg_CVR": "Avg CVR (%)", "Avg_Dwell_min": "Avg Dwell (min)", "Avg_Precip": "Avg Rain (mm)",
            }),
            use_container_width=True, hide_index=True,
        )

    # AI Button
    if has_api_key():
        if st.button("AI Weather Impact Analysis", key="adv_ai_weather_btn"):
            _ai_weather_analysis(space_name, weather_summary)


def _ai_weather_analysis(space_name: str, weather_summary: pd.DataFrame) -> None:
    """Generate AI analysis for weather impact."""
    try:
        ws_text = weather_summary[
            ["weather", "Days", "Avg_FP", "Avg_Visitors", "Avg_CVR", "Avg_Dwell_min", "Avg_Precip"]
        ].to_string(index=False)
        w_prompt = (
            f"[Weather impact summary - {space_name}]\n{ws_text}\n\n"
            "Analyze:\n"
            "1. How significantly does weather affect visitor traffic and CVR?\n"
            "2. What promotional or operational strategies should be applied on poor-weather days?\n"
            "3. Are there any unexpected patterns (e.g., rain days with higher CVR)?\n"
            "Keep response to 3 paragraphs."
        )
        with st.spinner("Analyzing weather patterns..."):
            result = call_claude(
                w_prompt,
                system="You are a retail analytics assistant. Analyze weather impact on store performance with actionable recommendations. Be concise.",
                space_notes=st.session_state.get("current_space_notes", ""),
            )
        st.markdown(result)
    except Exception as e:
        st.warning(f"Weather AI error: {e}")


# ── Tab 4: Dwell Detail ──────────────────────────────────────────────────────

def _render_dwell_detail(
    space_name: str,
    sessions: pd.DataFrame,
    daily_stats: pd.DataFrame,
) -> None:
    """Dwell time distribution and analysis."""
    st.markdown("#### Dwell Intelligence")
    _info("""
Dwell time in 3 segments: **Short** (<3 min), **Medium** (3-10 min), **Long** (10+ min).
High short = quick visits or low engagement; high long = high-engagement customers.
Medium-dominant = typical browsing; promotions tend to work.
""")

    if sessions.empty or "dwell_seconds" not in sessions.columns:
        st.caption("No session data for dwell analysis.")
        return

    dist = _cached_dwell_dist(sessions)
    if dist.empty:
        st.caption("Could not compute dwell distribution.")
        return

    fig_dwell = px.bar(
        dist, x="segment", y="count", color="ratio",
        color_continuous_scale=["#1a2035", GOLD, "#ccd6f6"],
    )
    fig_dwell.update_layout(
        height=320, xaxis_title="Segment",
        yaxis_title="Count", title="Dwell time distribution",
    )
    apply_theme(fig_dwell)
    st.plotly_chart(fig_dwell, use_container_width=True)

    st.dataframe(
        dist.rename(columns={"segment": "Segment", "count": "Count", "ratio": "Ratio (%)"}),
        use_container_width=True, hide_index=True,
    )

    # Insights
    dominant = dist.loc[dist["count"].idxmax()]
    dom_seg = dominant["segment"]
    dom_ratio = float(dominant["ratio"])
    short_ratio = float(dist.loc[dist["segment"] == "Short", "ratio"].iloc[0]) if "Short" in dist["segment"].values else 0.0
    long_ratio = float(dist.loc[dist["segment"] == "Long", "ratio"].iloc[0]) if "Long" in dist["segment"].values else 0.0

    insights = [f"Dominant segment: {dom_seg} ({dom_ratio:.0f}% of sessions)"]
    if short_ratio > 50:
        insights.append("Over 50% Short-dwell (<3 min) - high bounce rate. Investigate entrance friction.")
    if long_ratio > 30:
        insights.append("Significant Long-dwell segment (10+ min) - prime candidates for loyalty programs.")

    st.info(" | ".join(insights))

    # Uplift chart
    st.markdown("##### Traffic Uplift vs Weekday Average")
    uplift_df = _cached_uplift(daily_stats)
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
        template="plotly_white", height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_theme(fig_uplift)
    st.plotly_chart(fig_uplift, use_container_width=True)

    # AI Button
    if has_api_key():
        if st.button("AI Dwell Analysis", key="adv_ai_dwell_btn"):
            _ai_dwell_analysis(space_name, dist)


def _ai_dwell_analysis(space_name: str, dist: pd.DataFrame) -> None:
    """Generate AI analysis for dwell patterns."""
    try:
        dwell_text = dist.to_string(index=False)
        d_prompt = (
            f"[Dwell time distribution - {space_name}]\n{dwell_text}\n\n"
            "Analyze:\n"
            "1. What does this dwell distribution reveal about customer engagement level?\n"
            "2. What are 2 specific actions to improve dwell time and conversion?\n"
            "3. Is there anything surprising or concerning?\n"
            "Keep response to 3 paragraphs."
        )
        with st.spinner("Analyzing dwell patterns..."):
            result = call_claude(
                d_prompt,
                system="You are a retail analytics assistant specializing in customer behaviour and dwell time analysis. Be concise and actionable.",
                space_notes=st.session_state.get("current_space_notes", ""),
            )
        st.markdown(result)
    except Exception as e:
        st.warning(f"Dwell AI error: {e}")


# ── Tab 5: Device Mix ────────────────────────────────────────────────────────

def _render_device_analysis(space_name: str, device_mix: pd.DataFrame) -> None:
    """iOS vs Android device mix analysis."""
    st.markdown("#### Device Mix - iOS vs Android")
    _info("""
Visitor sessions by iOS (Apple) vs Android. Track ratio changes over time.
Note: iOS MAC randomization may affect counts; focus on trends rather than absolute share.
""")

    if device_mix.empty:
        st.caption("No session/device data.")
        return

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

    # Insight
    ios_pct = mix["apple_ratio"] * 100
    adr_pct = mix["android_ratio"] * 100
    dominant = "iOS" if ios_pct >= adr_pct else "Android"
    if abs(ios_pct - adr_pct) > 20:
        st.info(f"Strong {dominant} dominance ({max(ios_pct, adr_pct):.0f}%)")
    elif abs(ios_pct - adr_pct) <= 10:
        st.info("Balanced device mix - near-equal iOS/Android split.")


# ── Tab 6: AI Tools ──────────────────────────────────────────────────────────

def _render_ai_tools(
    space_name: str,
    loader: CacheLoader,
    daily_stats: pd.DataFrame,
) -> None:
    """AI Anomaly Detection and Cause Analysis Chat."""
    st.markdown("#### AI Tools")

    if not has_api_key():
        st.info("Set ANTHROPIC_API_KEY in environment or secrets to use AI features.")
        return

    # Anomaly Detection
    st.markdown("##### Anomaly Detection")
    st.caption("Find dates outside mean +/- 2 sigma and interpret causes.")

    metric_choice = st.selectbox(
        "Metric for anomaly (mean +/- 2 sigma)",
        options=["conversion_rate", "visitor_count", "floating_unique", "dwell_seconds_mean"],
        format_func=lambda x: {
            "conversion_rate": "CVR (%)",
            "visitor_count": "Visitors",
            "floating_unique": "Floating Pop (FP)",
            "dwell_seconds_mean": "Avg dwell (sec)",
        }.get(x, x),
        key="adv_anomaly_metric",
    )

    if st.button("Run Anomaly Analysis", key="adv_ai_anomaly_btn"):
        _run_anomaly_analysis(space_name, loader, daily_stats, metric_choice)

    st.markdown("---")

    # Cause Analysis Chat
    st.markdown("##### Cause Analysis Chat")
    st.caption("Ask questions based on the data. e.g. 'Why is CVR lower on rainy days?'")

    user_question = st.text_input(
        "Ask a question",
        key="adv_ai_chat_input",
        placeholder="e.g. Why are weekend visitors lower than weekday?",
    )

    if user_question:
        _run_cause_analysis_chat(space_name, loader, user_question)


def _run_anomaly_analysis(
    space_name: str,
    loader: CacheLoader,
    daily_stats: pd.DataFrame,
    metric_choice: str,
) -> None:
    """Run AI anomaly analysis."""
    try:
        anom_ds = _ensure_day_type(loader)
        col = metric_choice

        if col not in anom_ds.columns or anom_ds[col].dropna().empty:
            st.warning("No data for selected metric.")
            return

        mu = anom_ds[col].mean()
        sig = anom_ds[col].std()
        anom_rows = anom_ds[((anom_ds[col] - mu).abs() > 2 * sig)].copy() if sig > 0 else anom_ds.head(0)

        if anom_rows.empty:
            st.info("No dates outside 2 sigma. Data is stable.")
            anom_summary = "(no anomalies)"
        else:
            disp_cols = [c for c in ["date", "day_type", "weather", col, "visitor_count", "floating_unique", "conversion_rate"] if c in anom_rows.columns]
            st.dataframe(anom_rows[disp_cols].sort_values("date"), use_container_width=True, hide_index=True)
            anom_summary = anom_rows[disp_cols].to_dict(orient="records")

        metric_label = {
            "conversion_rate": "CVR (%)",
            "visitor_count": "Visitors",
            "floating_unique": "Floating Pop (FP)",
            "dwell_seconds_mean": "Avg dwell (sec)",
        }.get(col, col)

        anom_prompt = f"""[Daily data - {space_name}]
Mean: {mu:.2f}, Std: {sig:.2f}

[Anomaly dates - {metric_label}, outside mean +/- 2 sigma]
{anom_summary}

Analyze:
1. For each anomaly date, explain possible causes (weather, holiday, day type) with evidence.
2. Two operational insights from the anomaly pattern.
If no anomalies, assess stability and notable features."""

        with st.spinner("Analyzing anomalies..."):
            result = call_claude(
                anom_prompt,
                system="You are a Hermes anomaly analyst. Interpret statistical anomalies with Cause-Effect and give actionable insights in English.",
                space_notes=st.session_state.get("current_space_notes", ""),
            )
        st.markdown(result)

    except Exception as e:
        st.warning(f"Anomaly analysis error: {e}")


def _run_cause_analysis_chat(
    space_name: str,
    loader: CacheLoader,
    question: str,
) -> None:
    """Run cause analysis chat with AI."""
    try:
        ce_daily = loader.get_daily_stats()
        ce_daily_with_type = add_day_type_to_daily_stats(ce_daily) if not ce_daily.empty else ce_daily

        ctx_cols = [c for c in [
            "date", "floating_unique", "visitor_count",
            "conversion_rate", "dwell_seconds_mean", "day_type", "weather",
        ] if c in ce_daily_with_type.columns]

        ctx_df = ce_daily_with_type[ctx_cols].tail(30)
        ctx_summary = ctx_df.describe().to_dict()

        system_prompt = (
            "You are a Hermes cause-effect analyst. Answer in English based only on the given data; "
            "mark speculation as 'Beyond data:'. Focus on weather, day type, holiday vs traffic metrics."
        )
        user_prompt = f"""[Store data - {space_name}]
Recent 30 days summary: {ctx_summary}

[Question]
{question}

Answer concisely based on the data. Mark any speculation as "Beyond data:"
"""

        with st.spinner("Analyzing..."):
            result = call_claude(
                user_prompt,
                system=system_prompt,
                space_notes=st.session_state.get("current_space_notes", ""),
            )

        st.session_state["adv_ai_last_response"] = result
        st.markdown(result)

    except Exception as e:
        st.warning(f"AI error: {e}")
