"""
Hermes — MAC Stitching Tab.
Educates on BLE MAC randomization, shows algorithm overview,
and provides before/after comparison using raw vs stitched aggregates.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.mac_stitching import stitching_evaluation, stitching_daily_summary
from src.analytics.dwell_intelligence import dwell_distribution
from src.analytics.device_craft import device_mix_by_date
from src.ui.chart_theme import apply_theme

# Palette (dark theme)
DEEP_NAVY = "#4A90D9"
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct_change(old: float, new: float) -> str:
    """Format percentage change string."""
    if old == 0:
        return "N/A"
    delta = (new - old) / old * 100
    return f"{delta:+.1f}%"


def _info(text: str, label: str = "📖 About this") -> None:
    with st.expander(label, expanded=False):
        st.markdown(text)


# ── Main Render ───────────────────────────────────────────────────────────────

def render_stitching_tab(space_name: str, loader: CacheLoader) -> None:
    """MAC Stitching tab — education, algorithm overview, before/after comparison."""
    st.subheader("🔗 MAC Stitching — BLE Signal Correction")
    st.markdown(
        "Correct BLE MAC address randomization artifacts that inflate visitor counts "
        "and deflate dwell times. **All metrics in other tabs use stitched data.**"
    )

    # ── Section 1: Education ──────────────────────────────────────────────
    st.markdown("#### Understanding BLE MAC Randomization")
    _info("""
**Why MAC Stitching matters for BLE analytics**

Modern smartphones (Apple & Android) periodically rotate their BLE MAC address for privacy.
This creates a fundamental problem for visitor analytics:

| Behavior | Impact |
|----------|--------|
| 1 person visits for 60 min | Appears as 3-4 separate visitors with ~15 min dwell each |
| MAC rotates every ~15-20 min (Apple) | Each rotation creates a new "visitor session" |
| MAC rotates every ~3-9 min (Android) | Even more session fragmentation |

**Without correction**: Visitor counts are **inflated** and dwell times are **deflated**.

**Hermes MAC Stitching** uses a 2-level algorithm to detect and merge these fragmented sessions,
producing accurate visitor counts and realistic dwell times.
""", label="📖 Why MAC Stitching?")

    # ── Section 2: Algorithm Overview ─────────────────────────────────────
    st.markdown("#### Algorithm Overview")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("##### Level 1 — Raw Signal Stitching")
        st.markdown("""
**Pattern A: Co-existence Detection**
- Old & new MAC observed simultaneously at same time slot
- Same device type, RSSI difference ≤ threshold
- Catches ~33% of Apple MAC rotations

**Pattern B: Adjacent Swap Detection**
- Old MAC disappears → gap → new MAC appears
- No co-existence, but temporal + RSSI proximity
- Catches ~67% of rotations (the majority)

*Thresholds: Apple ≤ 3dBm, Android ≤ 5dBm, Gap ≤ 30s*
""")

    with col_l2:
        st.markdown("##### Level 2 — Session Post-Hoc")
        st.markdown("""
**Session-level merge** after L1:
- Session A ends → short gap → Session B starts
- Same device type, similar RSSI
- At least one is a "fragment" (< 5 min)
- No other active session during gap

*Gap limits: Apple ≤ 300s, Android ≤ 200s*
*RSSI threshold: ≤ 5dBm*

**Combined**: L1 catches signal-level swaps, L2 catches
sessions that slipped through L1 due to sensor gaps.
""")

    st.markdown("---")

    # ── Load Data ─────────────────────────────────────────────────────────
    daily_stats = loader.get_daily_stats()
    daily_stats_raw = loader.get_daily_stats_raw()
    sessions_all = loader.get_sessions_all()
    sessions_stitched = loader.get_sessions_stitched()

    if daily_stats_raw.empty or daily_stats.empty:
        st.warning(
            "⚠️ Raw comparison data not available. "
            "Re-run `python precompute.py --space {space} --force` with the latest code."
        )
        return

    # ── Section 3: KPI Comparison Cards ───────────────────────────────────
    st.markdown("#### Before & After — Key Metrics")

    raw_avg_visitors = daily_stats_raw["visitor_count"].mean()
    sti_avg_visitors = daily_stats["visitor_count"].mean()
    raw_avg_dwell = daily_stats_raw["dwell_seconds_mean"].mean()
    sti_avg_dwell = daily_stats["dwell_seconds_mean"].mean()
    raw_avg_cvr = daily_stats_raw["conversion_rate"].mean() if "conversion_rate" in daily_stats_raw.columns else 0
    sti_avg_cvr = daily_stats["conversion_rate"].mean() if "conversion_rate" in daily_stats.columns else 0
    raw_med_dwell = daily_stats_raw["dwell_median_seconds"].mean() if "dwell_median_seconds" in daily_stats_raw.columns else 0
    sti_med_dwell = daily_stats["dwell_median_seconds"].mean() if "dwell_median_seconds" in daily_stats.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Avg Daily Visitors",
        f"{sti_avg_visitors:,.0f}",
        f"{_pct_change(raw_avg_visitors, sti_avg_visitors)} (from {raw_avg_visitors:,.0f})",
    )
    k2.metric(
        "Avg CVR (%)",
        f"{sti_avg_cvr:.1f}%",
        f"{_pct_change(raw_avg_cvr, sti_avg_cvr)} (from {raw_avg_cvr:.1f}%)",
    )
    k3.metric(
        "Mean Dwell (sec)",
        f"{sti_avg_dwell:.0f}s",
        f"{_pct_change(raw_avg_dwell, sti_avg_dwell)} (from {raw_avg_dwell:.0f}s)",
    )
    k4.metric(
        "Median Dwell (sec)",
        f"{sti_med_dwell:.0f}s",
        f"{_pct_change(raw_med_dwell, sti_med_dwell)} (from {raw_med_dwell:.0f}s)",
    )

    st.markdown("---")

    # ── Section 4: Evaluation Dashboard ───────────────────────────────────
    st.markdown("#### Detailed Evaluation")

    if not sessions_all.empty and not sessions_stitched.empty:
        eval_data = stitching_evaluation(sessions_all, sessions_stitched, sessions_stitched)

        raw_n = eval_data.get("raw_sessions", 0)
        l2_n = eval_data.get("l2_sessions", 0)
        total_merge = eval_data.get("total_merge_pct", 0)
        dwell_improve = eval_data.get("dwell_improvement_pct", 0)
        raw_short = eval_data.get("raw_short", 0)
        l2_short = eval_data.get("l2_short", 0)

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Raw Sessions (Total)", f"{raw_n:,}")
        ec2.metric("After Stitching", f"{l2_n:,}", f"-{total_merge:.1f}%")
        ec3.metric(
            "Dwell Improvement",
            f"+{dwell_improve:.1f}%",
            f"{eval_data.get('raw_dwell_mean', 0):.0f}s → {eval_data.get('l2_dwell_mean', 0):.0f}s",
        )
        short_delta = l2_short - raw_short
        ec4.metric(
            "Short Sessions (<3min)",
            f"{l2_short:,}",
            f"{short_delta:+,} ({short_delta / max(raw_short, 1) * 100:+.0f}%)",
        )

        # Comparison table
        comp_data = {
            "Metric": [
                "Sessions", "Dwell Mean (s)", "Dwell Median (s)", "Dwell P90 (s)",
                "Apple Sessions", "Android Sessions",
                "Short (<3min)", "Medium (3-10min)", "Long (10min+)",
            ],
            "Raw": [
                f"{raw_n:,}",
                f"{eval_data.get('raw_dwell_mean', 0):.1f}",
                f"{eval_data.get('raw_dwell_median', 0):.1f}",
                f"{eval_data.get('raw_dwell_p90', 0):.1f}",
                f"{eval_data.get('raw_apple', 0):,}",
                f"{eval_data.get('raw_android', 0):,}",
                f"{raw_short:,}",
                f"{eval_data.get('raw_medium', 0):,}",
                f"{eval_data.get('raw_long', 0):,}",
            ],
            "After Stitching (L1+L2)": [
                f"{l2_n:,}",
                f"{eval_data.get('l2_dwell_mean', 0):.1f}",
                f"{eval_data.get('l2_dwell_median', 0):.1f}",
                f"{eval_data.get('l2_dwell_p90', 0):.1f}",
                f"{eval_data.get('l2_apple', 0):,}",
                f"{eval_data.get('l2_android', 0):,}",
                f"{l2_short:,}",
                f"{eval_data.get('l2_medium', 0):,}",
                f"{eval_data.get('l2_long', 0):,}",
            ],
            "Change": [
                f"-{total_merge:.1f}%",
                f"+{dwell_improve:.1f}%",
                _pct_change(eval_data.get("raw_dwell_median", 1), eval_data.get("l2_dwell_median", 0)),
                _pct_change(eval_data.get("raw_dwell_p90", 1), eval_data.get("l2_dwell_p90", 0)),
                _pct_change(eval_data.get("raw_apple", 1), eval_data.get("l2_apple", 0)),
                _pct_change(eval_data.get("raw_android", 1), eval_data.get("l2_android", 0)),
                f"{short_delta:+,}",
                f"{eval_data.get('l2_medium', 0) - eval_data.get('raw_medium', 0):+,}",
                f"{eval_data.get('l2_long', 0) - eval_data.get('raw_long', 0):+,}",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 5: Daily Comparison Charts ────────────────────────────────
    st.markdown("#### Daily Trends — Raw vs Stitched")

    if not sessions_all.empty and not sessions_stitched.empty:
        daily_comp = stitching_daily_summary(sessions_all, sessions_stitched, sessions_stitched)

        if not daily_comp.empty and "raw_sessions" in daily_comp.columns:
            # Sessions chart
            fig_sess = go.Figure()
            fig_sess.add_trace(go.Bar(
                x=daily_comp["date"], y=daily_comp["raw_sessions"],
                name="Raw Sessions", marker_color=SLATE_GRAY, opacity=0.6,
            ))
            fig_sess.add_trace(go.Bar(
                x=daily_comp["date"], y=daily_comp["l2_sessions"],
                name="Stitched Sessions", marker_color=GOLD,
            ))
            fig_sess.update_layout(
                title="Daily Sessions: Raw vs Stitched",
                xaxis_title="Date", yaxis_title="Sessions",
                barmode="group", height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            apply_theme(fig_sess)
            st.plotly_chart(fig_sess, use_container_width=True)

            # Dwell chart
            if "raw_dwell_mean" in daily_comp.columns and "l2_dwell_mean" in daily_comp.columns:
                fig_dwell = go.Figure()
                fig_dwell.add_trace(go.Scatter(
                    x=daily_comp["date"], y=daily_comp["raw_dwell_mean"] / 60,
                    name="Raw Dwell (min)", line=dict(color=SLATE_GRAY, width=2, dash="dot"),
                    mode="lines+markers",
                ))
                fig_dwell.add_trace(go.Scatter(
                    x=daily_comp["date"], y=daily_comp["l2_dwell_mean"] / 60,
                    name="Stitched Dwell (min)", line=dict(color=GOLD, width=2.5),
                    mode="lines+markers",
                ))
                fig_dwell.update_layout(
                    title="Daily Mean Dwell: Raw vs Stitched",
                    xaxis_title="Date", yaxis_title="Mean Dwell (min)",
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                apply_theme(fig_dwell)
                st.plotly_chart(fig_dwell, use_container_width=True)

    # ── Section 6: Visitors comparison from daily_stats ────────────────────
    st.markdown("#### Daily Visitors — Aggregate Comparison")

    if not daily_stats_raw.empty and not daily_stats.empty:
        merged = daily_stats_raw[["date", "visitor_count", "dwell_seconds_mean"]].rename(
            columns={"visitor_count": "raw_visitors", "dwell_seconds_mean": "raw_dwell"}
        ).merge(
            daily_stats[["date", "visitor_count", "dwell_seconds_mean"]].rename(
                columns={"visitor_count": "stitched_visitors", "dwell_seconds_mean": "stitched_dwell"}
            ),
            on="date",
            how="outer",
        )
        merged["visitor_Δ%"] = (
            (merged["stitched_visitors"] - merged["raw_visitors"]) / merged["raw_visitors"].replace(0, 1) * 100
        ).round(1)
        merged["dwell_Δ%"] = (
            (merged["stitched_dwell"] - merged["raw_dwell"]) / merged["raw_dwell"].replace(0, 1) * 100
        ).round(1)
        merged["raw_dwell"] = merged["raw_dwell"].round(1)
        merged["stitched_dwell"] = merged["stitched_dwell"].round(1)

        display_df = merged.rename(columns={
            "date": "Date",
            "raw_visitors": "Raw Visitors",
            "stitched_visitors": "Stitched Visitors",
            "visitor_Δ%": "Visitor Δ%",
            "raw_dwell": "Raw Dwell (s)",
            "stitched_dwell": "Stitched Dwell (s)",
            "dwell_Δ%": "Dwell Δ%",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Section 7: Dwell Distribution Comparison ──────────────────────────
    st.markdown("#### Dwell Distribution — Raw vs Stitched")

    if not sessions_all.empty and not sessions_stitched.empty:
        dist_raw = dwell_distribution(sessions_all)
        dist_stitched = dwell_distribution(sessions_stitched)

        if not dist_raw.empty and not dist_stitched.empty:
            col_dr, col_ds = st.columns(2)
            with col_dr:
                st.markdown("**Raw (Before Stitching)**")
                fig_dr = px.bar(
                    dist_raw, x="segment", y="count", color="ratio",
                    color_continuous_scale=["#1a2035", SLATE_GRAY, "#ccd6f6"],
                )
                fig_dr.update_layout(height=280, title="Raw Dwell Segments", showlegend=False)
                apply_theme(fig_dr)
                st.plotly_chart(fig_dr, use_container_width=True)
                st.dataframe(
                    dist_raw.rename(columns={"segment": "Segment", "count": "Count", "ratio": "Ratio (%)"}),
                    use_container_width=True, hide_index=True,
                )

            with col_ds:
                st.markdown("**After Stitching**")
                fig_ds = px.bar(
                    dist_stitched, x="segment", y="count", color="ratio",
                    color_continuous_scale=["#1a2035", GOLD, "#ccd6f6"],
                )
                fig_ds.update_layout(height=280, title="Stitched Dwell Segments", showlegend=False)
                apply_theme(fig_ds)
                st.plotly_chart(fig_ds, use_container_width=True)
                st.dataframe(
                    dist_stitched.rename(columns={"segment": "Segment", "count": "Count", "ratio": "Ratio (%)"}),
                    use_container_width=True, hide_index=True,
                )

    # ── Section 8: Note ───────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "📌 **All metrics in other tabs** (Understand, Learn, Intelligence, Report) "
        "**use stitched data.** This tab shows raw data for comparison purposes only."
    )
