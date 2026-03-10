"""
Hermes Dashboard — Phase 1: Understand (Overview).
Core metrics, intraday timeseries, daily detail table, cumulative visitors.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.day_type import add_day_type_to_daily_stats
from src.analytics.uplift import compute_week_over_week
from src.analytics.device_craft import device_mix_by_date
from src.ui.helpers import (
    DEEP_NAVY, GOLD, AMBER, SLATE_GRAY, TEAL,
    _WEATHER_ICON, ensure_day_type, info, has_api_key,
    apply_theme,
)
from src.ai import call_claude


def render_overview(space_name: str, loader: CacheLoader) -> None:
    st.subheader("Phase 1 — Understand")
    st.markdown(
        "View store status with **precisely defined standard metrics**. "
        "See each section below for measurement and calculation methods."
    )

    daily_results = loader.get_daily_results()
    daily_stats = loader.get_daily_stats()
    device_mix = loader.get_device_mix()
    daily_timeseries = loader.get_daily_timeseries()
    daily_hourly = loader.get_daily_hourly()

    if not daily_results or daily_stats.empty:
        st.info("No daily results in cache. Run precompute first.")
        return

    dr = loader.get_date_range()
    if dr:
        st.caption(f"Period: {dr[0]} ~ {dr[-1]} ({len(dr)} days)")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    info("""
**Core Metric Snapshot**

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Floating Population (FP)** | Foot traffic near store | Unique MACs at entrance S-Ward with RSSI >= -80 dBm. Daily dedup. |
| **Verified Visitors (V)** | In-store visitors | Sessions passing **Strict Entry** (>=3 hits in 1 min + all RSSI >= -80 dBm) at inside S-Ward. |
| **Conversion Rate (CVR)** | FP -> visitor conversion | Visitors / Floating Pop x 100 (%). |
| **Avg Dwell Time** | Mean dwell | Entry to exit. Exit back-dated to last signal. Android 120s / Apple 180s buffer. |

> **Strict Entry**: Only counts as a visit when **>=3 hits in 1 min** and all RSSI strong enough. Filters passers-by.
""")

    wow = compute_week_over_week(daily_stats) if len(daily_stats) >= 7 else {}
    tw = wow.get("this_week", {})
    pw = wow.get("prev_week", {})
    d = wow.get("delta", {})

    has_quality = "quality_cvr" in daily_stats.columns and "quality_visitor_count" in daily_stats.columns
    has_median = "dwell_median_seconds" in daily_stats.columns

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = tw.get("floating_unique", daily_stats["floating_unique"].iloc[-1] if len(daily_stats) else 0)
        delta_str = f"{d.get('floating_pct', 0):+.1f}% vs prev week" if d else "Period avg"
        st.metric("Floating Pop (FP)", f"{val:,.0f}", delta_str, delta_color="normal")
    with col2:
        val = tw.get("quality_visitor_count", daily_stats["visitor_count"].iloc[-1] if len(daily_stats) else 0)
        lbl = "Quality Visitors" if has_quality else "Verified Visitors (V)"
        delta_str = f"{d.get('quality_visitor_pct', 0):+.1f}% vs prev week" if d else "Period avg"
        st.metric(lbl, f"{val:,.0f}", delta_str, delta_color="normal")
    with col3:
        val = tw.get("quality_cvr", daily_stats["conversion_rate"].iloc[-1] if len(daily_stats) else 0.0)
        lbl = "Quality CVR" if has_quality else "Conversion Rate (CVR)"
        delta_str = f"{d.get('quality_cvr_pp', 0):+.1f}%p vs prev week" if d else "Period avg"
        st.metric(lbl, f"{val:.1f}%", delta_str, delta_color="normal")
    with col4:
        dm = tw.get("dwell_median_seconds", 0) if has_median else (daily_stats["dwell_seconds_mean"].iloc[-1] if len(daily_stats) else 0)
        mm, ss = int(dm) // 60, int(dm) % 60
        val_str = f"{mm}m {ss}s" if has_median else f"{dm / 60:.1f} min"
        delta_val = d.get("dwell_median", 0)
        delta_str = f"{delta_val:+.0f}s vs prev week" if d and delta_val != 0 else ("Period avg" if not d else "Same")
        st.metric("Median Dwell" if has_median else "Avg Dwell Time", val_str, delta_str, delta_color="normal")

    st.markdown("---")

    # ── Daily trend chart ─────────────────────────────────────────────────────
    st.markdown("#### Daily FP vs Visitors")
    info("""
- **Floating Pop (navy)**: Daily unique devices at entrance. Patterns vary by weekend/holiday/weather.
- **Visitors (gold)**: Sessions passing Strict Entry. Tighter gap vs FP = higher CVR.
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

    # ── Cumulative Visitors (신규) ────────────────────────────────────────────
    st.markdown("#### Cumulative Visitors")
    info("""
**Cumulative Visitors**: Running total of daily visitors over the period.
Shows the overall growth trajectory. Steeper slope = busier period.
Dotted line: cumulative floating population for reference.
""", label="📖 Cumulative")

    ds = daily_stats_sorted.copy()
    ds["cum_visitors"] = ds["visitor_count"].cumsum()
    ds["cum_floating"] = ds["floating_unique"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=ds["date"], y=ds["cum_floating"],
        name="Cumulative FP",
        line=dict(color=DEEP_NAVY, width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.05)",
    ))
    fig_cum.add_trace(go.Scatter(
        x=ds["date"], y=ds["cum_visitors"],
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

    # Summary metrics
    total_fp = int(ds["cum_floating"].iloc[-1]) if len(ds) else 0
    total_v = int(ds["cum_visitors"].iloc[-1]) if len(ds) else 0
    avg_daily_v = total_v / len(ds) if len(ds) else 0
    period_cvr = total_v / total_fp * 100 if total_fp else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total FP", f"{total_fp:,}")
    c2.metric("Total Visitors", f"{total_v:,}")
    c3.metric("Avg Daily Visitors", f"{avg_daily_v:.0f}")
    c4.metric("Period CVR", f"{period_cvr:.1f}%")

    # ── 체류 퍼널 스택 차트 ───────────────────────────────────────────────────
    if all(c in daily_stats.columns for c in ["short_dwell_count", "medium_dwell_count", "long_dwell_count", "quality_cvr"]):
        st.markdown("---")
        st.markdown("#### Dwell Funnel — Short / Medium / Long")
        info("""
| Segment | Dwell | Meaning |
|---------|-------|---------|
| **Short** | <3 min | Browse, pass-through |
| **Medium** | 3-10 min | Interested, exploring |
| **Long** | 10+ min | Purchase intent |

**Quality CVR** = (medium + long) / FP x 100.
""", label="📖 Dwell funnel")
        ds_f = daily_stats.sort_values("date")
        fig_funnel = go.Figure()
        fig_funnel.add_trace(go.Bar(x=ds_f["date"], y=ds_f["short_dwell_count"], name="Short (<3min)", marker_color=SLATE_GRAY))
        fig_funnel.add_trace(go.Bar(x=ds_f["date"], y=ds_f["medium_dwell_count"], name="Medium (3-10min)", marker_color=DEEP_NAVY))
        fig_funnel.add_trace(go.Bar(x=ds_f["date"], y=ds_f["long_dwell_count"], name="Long (10min+)", marker_color=GOLD))
        fig_funnel.update_layout(barmode="stack", height=320, xaxis_title="Date", yaxis_title="Count")
        fig_funnel.add_trace(go.Scatter(
            x=ds_f["date"], y=ds_f["quality_cvr"],
            name="Quality CVR (%)", yaxis="y2", line=dict(color=AMBER, width=2, dash="dot"),
        ))
        fig_funnel.update_layout(
            yaxis2=dict(overlaying="y", side="right", range=[0, max(ds_f["quality_cvr"].max() * 1.2, 10)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        apply_theme(fig_funnel)
        st.plotly_chart(fig_funnel, use_container_width=True)

        total_v_sum = daily_stats["visitor_count"].sum()
        qv = daily_stats["quality_visitor_count"].sum() if "quality_visitor_count" in daily_stats.columns else total_v_sum
        long_ratio = (daily_stats["long_dwell_count"].sum() / total_v_sum * 100) if total_v_sum else 0
        qcvr = (qv / daily_stats["floating_unique"].sum() * 100) if daily_stats["floating_unique"].sum() else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality visitor ratio", f"{qv / total_v_sum * 100:.1f}%" if total_v_sum else "—", "Medium+Long / total")
        c2.metric("Long dwell ratio", f"{long_ratio:.1f}%", "10min+")
        c3.metric("Quality CVR (period)", f"{qcvr:.1f}%", "Quality visitors / FP")

    st.markdown("---")

    # ── iOS vs Android Daily Trend (신규) ─────────────────────────────────────
    st.markdown("#### iOS vs Android — Daily Trend")
    info("""
Daily iOS and Android visitor share (%). Track device composition changes over time.
Note: iOS MAC randomization may affect counts; focus on trends.
""", label="📖 Device trend")

    if not device_mix.empty:
        mix_daily = device_mix_by_date(device_mix)
        if not mix_daily.empty:
            fig_dev = go.Figure()
            fig_dev.add_trace(go.Scatter(
                x=mix_daily["date"], y=mix_daily["ios_pct"],
                name="iOS (%)", line=dict(color=DEEP_NAVY, width=2),
                fill="tozeroy", fillcolor="rgba(15,23,42,0.06)",
            ))
            fig_dev.add_trace(go.Scatter(
                x=mix_daily["date"], y=mix_daily["android_pct"],
                name="Android (%)", line=dict(color=GOLD, width=2),
            ))
            fig_dev.update_layout(
                title="Daily iOS vs Android Share",
                xaxis_title="Date", yaxis_title="Share (%)",
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            apply_theme(fig_dev)
            st.plotly_chart(fig_dev, use_container_width=True)

            # Summary
            avg_ios = mix_daily["ios_pct"].mean()
            avg_android = mix_daily["android_pct"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Avg iOS Share", f"{avg_ios:.1f}%")
            c2.metric("Avg Android Share", f"{avg_android:.1f}%")

    st.markdown("---")

    # ── Intraday timeseries ───────────────────────────────────────────────────
    st.markdown("#### Intraday Traffic — Minute-level flow")
    info("""
| Chart | Meaning |
|-------|---------|
| **Floating Population** (navy) | Unique MACs per minute at entrance. |
| **Active Visitors** (gold) | In-store sessions per minute (occupancy). |

**Resolution**: 1 min = fine peaks; 5 min = overall flow.
""")

    if not daily_timeseries.empty and dr:
        ts_date = st.selectbox("Date (Intraday)", options=dr, key="overview_ts_date")
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
    info("""
| Column | Description |
|--------|-------------|
| **Floating Pop** | Daily unique devices at entrance |
| **Visitors** | Sessions passing Strict Entry |
| **CVR (%)** | Visitors / FP x 100 |
| **Avg Dwell (min)** | Mean dwell (entry to last signal) |
| **Day Type** | weekday / weekend / holiday |
| **Weather** | Open-Meteo tag |
""")

    daily_with_type = add_day_type_to_daily_stats(daily_stats)
    detail = daily_with_type[[
        "date", "floating_unique", "visitor_count",
        "conversion_rate", "dwell_seconds_mean", "day_type", "weekday"
    ]].copy()
    detail = detail.rename(columns={
        "floating_unique": "Floating Pop",
        "visitor_count": "Visitors",
        "conversion_rate": "CVR (%)",
        "dwell_seconds_mean": "Avg Dwell (sec)",
        "day_type": "Day Type",
    })
    detail["Avg Dwell (min)"] = (detail["Avg Dwell (sec)"] / 60).round(1)
    detail["CVR (%)"] = detail["CVR (%)"].round(1)

    if not device_mix.empty:
        mix_by_d = device_mix_by_date(device_mix)
        detail = detail.merge(mix_by_d[["date", "ios_pct", "android_pct"]], on="date", how="left")
        detail["Visitor iOS (%)"] = detail["ios_pct"].fillna(0).round(1)
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
        fp_device["FP iOS (%)"] = (100 * fp_device["floating_apple"] / fp_device["fp_total"].replace(0, 1)).round(1)
        fp_device["FP Android (%)"] = (100 * fp_device["floating_android"] / fp_device["fp_total"].replace(0, 1)).round(1)
        detail = detail.merge(fp_device[["date", "FP iOS (%)", "FP Android (%)"]], on="date", how="left")
        detail["FP iOS (%)"] = detail["FP iOS (%)"].fillna(0.0)
        detail["FP Android (%)"] = detail["FP Android (%)"].fillna(0.0)
        has_fp_device = True

    has_weather = "weather" in daily_with_type.columns
    if has_weather:
        detail["Weather"] = daily_with_type["weather"].map(lambda w: _WEATHER_ICON.get(w, w)).values
        detail["Rain (mm)"] = daily_with_type["precipitation"].fillna(0).round(1).values

    display_cols = ["date", "Floating Pop", "Visitors", "CVR (%)", "Avg Dwell (min)", "Day Type"]
    if has_weather:
        display_cols += ["Weather", "Rain (mm)"]
    display_cols += ["Visitor iOS (%)", "Visitor Android (%)"]
    if has_fp_device:
        display_cols += ["FP iOS (%)", "FP Android (%)"]

    st.dataframe(detail[display_cols], use_container_width=True, hide_index=True)

    # ── AI 일간 요약 ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### AI Daily Summary")
    st.caption("Claude AI summarizes notable patterns and action items from the metrics.")

    if not has_api_key():
        st.info("Set ANTHROPIC_API_KEY in `.streamlit/secrets.toml` or environment to use AI features.")
    else:
        if st.button("Generate AI insights", key="ai_overview_btn"):
            try:
                daily_with_type_ai = ensure_day_type(loader)
                avg_fp = float(daily_stats["floating_unique"].mean()) if "floating_unique" in daily_stats.columns else 0.0
                avg_v = float(daily_stats["visitor_count"].mean()) if "visitor_count" in daily_stats.columns else 0.0
                avg_cvr = float(daily_stats["conversion_rate"].mean()) if "conversion_rate" in daily_stats.columns else 0.0
                avg_dwell = float(daily_stats["dwell_seconds_mean"].mean()) / 60 if "dwell_seconds_mean" in daily_stats.columns else 0.0

                last = daily_results[-1] if daily_results else {}
                last_date = last.get("date", "N/A")
                last_fp = last.get("floating_unique", 0)
                last_v = last.get("visitor_count", 0)
                last_cvr = last.get("conversion_rate", 0.0)

                day_type_cvr_lines = []
                if "day_type" in daily_with_type_ai.columns and "conversion_rate" in daily_with_type_ai.columns:
                    dt_cvr = daily_with_type_ai.groupby("day_type")["conversion_rate"].mean().reset_index()
                    day_type_cvr_lines = [f"  - {r['day_type']}: {r['conversion_rate']:.1f}%" for r in dt_cvr.to_dict("records")]
                day_type_cvr_table = "\n".join(day_type_cvr_lines) if day_type_cvr_lines else "  (no data)"

                weather_section = ""
                if "weather" in daily_stats.columns and "conversion_rate" in daily_stats.columns:
                    w_cvr = daily_stats.groupby("weather")["conversion_rate"].mean().reset_index()
                    lines = [f"  - {r['weather']}: {r['conversion_rate']:.1f}%" for r in w_cvr.to_dict("records")]
                    weather_section = "\n[Avg CVR by Weather]\n" + "\n".join(lines)

                dr_list = loader.get_date_range() or []
                period_str = f"{dr_list[0]} ~ {dr_list[-1]}" if len(dr_list) >= 2 else (dr_list[0] if dr_list else "N/A")
                n_days = len(dr_list)

                system_prompt = "You are a Hermes retail insight analyst. Apply Cause-Effect and provide concise, data-driven insights in English. Include numbers and two actionable items for store operations."
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
1. 2-3 notable patterns (with numbers)
2. Two actionable items for store operations.
Keep it concise."""

                with st.spinner("Analyzing data..."):
                    result = call_claude(
                        user_prompt, system=system_prompt,
                        space_notes=st.session_state.get("current_space_notes", ""),
                    )
                st.markdown(result) if "⚠️" not in result else st.warning(result)

            except Exception as _exc:
                st.warning(f"AI analysis error: {_exc}")
