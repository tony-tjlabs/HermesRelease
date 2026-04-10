"""
Hermes Dashboard v3 — AI-First Redesign.

Two modes in one file:
  - Daily Analysis: KPI -> Daily Trend -> Hourly -> Dwell -> AI (3-section)
  - Period Comparison: Summary Table -> Trend -> Dwell -> AI (3-section)

Simplified from v2: removed redundant sections, enhanced AI prompts
with advertising/location/operations perspectives.
Dark theme, metric-card/section-header/ai-comment CSS classes.

v3.1 (2026-04-05): Reactivity optimizations + AI full-data support
  - Expanded @st.cache_data coverage for hourly calculations
  - Reduced DataFrame.copy() calls to necessary cases only
  - AI now receives full available data (up to 30 days) instead of 14 days
"""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.uplift import compute_week_over_week
from src.analytics.hourly_analysis import hourly_stats_flexible, identify_peak_hours
from src.analytics.korean_calendar import get_korean_calendar_context
from src.ui.helpers import (
    has_api_key,
    render_metric_card, make_plotly_layout,
    render_section_header, render_ai_comment,
)
from src.ai import call_claude
from config import REGISTERED_SECTORS

# Maximum days to include in AI prompt (to avoid token overflow)
_AI_MAX_DAYS = 30


# -- Color Palette ---------------------------------------------------------

FP_COLOR = "#4A90D9"      # Blue for Floating Pop
VISITOR_COLOR = "#64ffda"  # Teal for Visitors
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"


_LOCATION_CONTEXT = {
    "Victor_Suwon_Starfield": (
        "[Store & Location Context]\n"
        "- Store: VICTOR (global badminton brand), showcase-type experiential retail store\n"
        "- Location: Starfield Suwon 6F (Sports & Home Furnishing floor)\n"
        "- Address: 175 Suseong-ro, Jangan-gu, Suwon-si, Gyeonggi-do\n"
        "- Mall: Starfield Suwon — 8-story large-scale complex (shopping + cinema + leisure + dining)\n"
        "- 6F neighbors: sports brands, outdoor gear, home furnishing, community zones\n"
        "- Operating hours: 10:00-22:00 (mall standard)\n"
        "- Opened: Jan 30, 2026\n"
        "\n"
        "[Location Characteristics]\n"
        "- Suburban mega-mall: visitors come by car (parking 3,600+), planned shopping trips\n"
        "- Weekend/holiday traffic 2-3x higher than weekday (leisure destination)\n"
        "- 6F is upper floor → visitors who reach here have strong intent (not casual walk-by)\n"
        "- Mall-wide events (seasonal sales, exhibitions, food festivals) strongly drive traffic to all floors\n"
        "- Surrounding: residential area (Gwanggyo new town), university (Sungkyunkwan), Suwon station nearby\n"
        "- Competition: other sports brands on same floor, online shopping\n"
        "- Key advantage: experiential/showcase store — try before buy, brand immersion\n"
        "- CVR interpretation: low CVR (1-3%) is normal for mall upper floors; quality of visit matters more"
    ),
    "GS25_Yeoksam": (
        "[Store & Location Context]\n"
        "- Store: GS25 convenience store (역삼홍인점)\n"
        "- Location: Bongeunsa-ro 30-gil 43, 1F, Yeoksam-dong, Gangnam-gu, Seoul\n"
        "- Side street off the main Bongeunsa-ro (major 6-lane road)\n"
        "- Operating hours: 24 hours\n"
        "\n"
        "[Location Characteristics]\n"
        "- Gangnam business district: one of Korea's densest office areas\n"
        "- 3 subway stations within walking distance: Gangnam, Yeoksam, Seolleung\n"
        "- Daytime floating population: ~200,000 (office workers + visitors)\n"
        "- Weekday >> Weekend traffic (office-driven demand)\n"
        "- Peak hours: morning commute (07-09h), lunch (12-13h), evening (18-20h)\n"
        "- Late-night demand (22-02h): significant for CVS in entertainment/office area\n"
        "- Surrounding: mixed residential + office buildings, restaurants, cafes\n"
        "- Side-street location: less walk-by traffic than main road, but loyal regular customers\n"
        "- Competition: multiple CVS within 200m radius (intense competition)\n"
        "- Key advantage: proximity to residential buildings → regular customers, late-night monopoly\n"
        "- CVR interpretation: 2-5% CVR is typical for side-street CVS; higher CVR than main-road stores"
    ),
}


def _get_sector_context(space_name: str) -> str:
    """Build location & store context for AI prompts."""
    if space_name in _LOCATION_CONTEXT:
        return _LOCATION_CONTEXT[space_name]
    meta = REGISTERED_SECTORS.get(space_name, {})
    desc = meta.get("description", "")
    loc = meta.get("location", "")
    stype = meta.get("store_type", "retail")
    if desc:
        return f"[Store Context: {desc} Location: {loc}. Type: {stype}]"
    return f"[Store: {space_name}]"


# -- Cached Wrappers -------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_wow(daily_stats: pd.DataFrame, days_per_week: int = 7, fp_col: str = "floating_unique") -> dict:
    return compute_week_over_week(daily_stats, days_per_week=days_per_week, fp_col=fp_col)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_hourly_stats(
    daily_hourly_json: str,
    sessions_json: str,
    daily_timeseries_json: str,
    dates: Tuple[str, ...],
    bin_minutes: int,
    daily_stats_json: str,
    fp_coverage: str = "medium",
) -> pd.DataFrame:
    """Cached wrapper for hourly_stats_flexible to avoid recomputation on reruns."""
    from io import StringIO
    daily_hourly = pd.read_json(StringIO(daily_hourly_json), orient="split") if daily_hourly_json else pd.DataFrame()
    sessions = pd.read_json(StringIO(sessions_json), orient="split") if sessions_json else pd.DataFrame()
    daily_timeseries = pd.read_json(StringIO(daily_timeseries_json), orient="split") if daily_timeseries_json else pd.DataFrame()
    daily_stats = pd.read_json(StringIO(daily_stats_json), orient="split") if daily_stats_json else None
    return hourly_stats_flexible(
        daily_hourly, sessions, daily_timeseries,
        list(dates), bin_minutes=bin_minutes,
        daily_stats=daily_stats, fp_coverage=fp_coverage,
    )


def _get_hourly_stats_cached(
    daily_hourly: pd.DataFrame,
    sessions: pd.DataFrame,
    daily_timeseries: pd.DataFrame,
    dates: list,
    bin_minutes: int,
    daily_stats: Optional[pd.DataFrame] = None,
    fp_coverage: str = "medium",
) -> pd.DataFrame:
    """Helper that serializes DataFrames for caching."""
    dh_json = daily_hourly.to_json(orient="split", date_format="iso") if not daily_hourly.empty else ""
    sess_json = sessions.to_json(orient="split", date_format="iso") if not sessions.empty else ""
    ts_json = daily_timeseries.to_json(orient="split", date_format="iso") if not daily_timeseries.empty else ""
    ds_json = daily_stats.to_json(orient="split", date_format="iso") if daily_stats is not None and not daily_stats.empty else ""
    return _cached_hourly_stats(dh_json, sess_json, ts_json, tuple(dates), bin_minutes, ds_json, fp_coverage)





@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ai_daily(metrics_json: str, space_notes: str = "", lang: str = "English") -> str:
    """Cache AI analysis result for daily mode."""
    metrics = json.loads(metrics_json)
    system_prompt = (
        "You are a Korean retail spatial intelligence expert specializing in urban convenience store analytics. "
        "You deeply understand Korean society, office-district consumption patterns, and the CVS (convenience store) industry.\n\n"
        "Technical context:\n"
        "- Data source: BLE S-Ward sensors, 10-second sampling intervals\n"
        "- FP (Floating Population): unique BLE MAC addresses detected near the store for ≥30 seconds; "
        "  affected by MAC randomization (iPhone every ~15-20min, Android every ~5-10min) — FP undercounts real people\n"
        "- Visitors: confirmed store entries passing 3-stage filter (session construction + min dwell 60s + 60% RSSI pass ratio)\n"
        "- CVR = Visitors ÷ FP × 100%. Industry benchmark for side-street CVS in Gangnam: 2–5%\n"
        "- Quality CVR = (visitors with 3min+ dwell) ÷ FP. Measures genuine engagement, not quick grab-and-go\n"
        "- Dwell time: entry-to-exit per session; sessions >15min may be underestimated due to MAC rotation\n\n"
        "Korean CVS expertise:\n"
        "- Gangnam Yeoksam: ~200,000 daily office workers within walking distance (Gangnam/Yeoksam/Seolleung stations)\n"
        "- Weekday demand peaks: 07-09h (commute coffee/breakfast), 12-13h (lunch meal purchase), 18-20h (return commute snack)\n"
        "- Weekend traffic 30-50% lower than weekday for office-district CVS\n"
        "- Korean public holidays = near-zero office traffic → drops to residential/tourist baseline\n"
        "- Rain/cold weather → hot food (라면, 온음료) impulse purchase → CVR increases despite lower FP\n"
        "- Warm sunny weather → people eat outside → CVS meal purchase drops, beverage purchase up\n"
        "- Side-street CVS advantage: loyal repeat customers (residents/fixed-desk workers), late-night demand\n"
        "- Seasonal: year-end (Dec) = office parties → late-night spike; spring exam season (Apr-May) = student traffic\n\n"
        "Response rules:\n"
        "- Respond in the language specified in the prompt (한국어 or English)\n"
        "- Structure output with the EXACT section labels requested\n"
        "- Always cite specific numbers from the data\n"
        "- Connect data patterns to real Korean social/behavioral context (not generic retail advice)\n"
        "- Be direct: state what is happening and why, then give one clear action per recommendation"
    )
    user_prompt = _build_daily_ai_prompt(metrics)
    return call_claude(user_prompt, system=system_prompt, space_notes=space_notes, lang=lang)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ai_comparison(metrics_json: str, space_notes: str = "", lang: str = "English") -> str:
    """Cache AI analysis result for comparison mode."""
    metrics = json.loads(metrics_json)
    system_prompt = (
        "You are a Korean retail spatial intelligence expert specializing in urban convenience store analytics. "
        "You deeply understand Korean society, office-district consumption patterns, and the CVS industry.\n\n"
        "Technical context:\n"
        "- BLE sensor data: FP = unique MACs ≥30s detection; Visitors = 3-stage filter (session+dwell+RSSI)\n"
        "- CVR industry benchmark for side-street Gangnam CVS: 2–5%\n"
        "- MAC randomization causes FP undercounting; trends and ratios are reliable, absolute numbers approximate\n"
        "- Day-type matters: Weekday (office workers) >> Weekend for Gangnam CVS\n\n"
        "Korean CVS expertise (same as daily mode — apply to period analysis):\n"
        "- Public holidays = near-zero office traffic; long weekends = compounding effect\n"
        "- Trend direction should be interpreted against Korean social calendar events\n"
        "- Week-over-week changes >10% in Gangnam CVS are meaningful (not noise)\n"
        "- FP trend reflects area foot traffic; CVR trend reflects store effectiveness\n"
        "- If FP drops but CVR rises: fewer people but more targeted visitors (bad weather, holiday, etc.)\n"
        "- If FP rises but CVR drops: more passersby but less store appeal (competing promotions, etc.)\n\n"
        "Response rules:\n"
        "- Respond in the language specified\n"
        "- Structure with EXACT section labels requested\n"
        "- Reference specific dates and numbers from the data table\n"
        "- Explain trends using Korean social context (not generic patterns)\n"
        "- Recommendations must be actionable for a single CVS store operator"
    )
    user_prompt = _build_comparison_ai_prompt(metrics)
    return call_claude(user_prompt, system=system_prompt, space_notes=space_notes, lang=lang)


# -- Main Entry Point ------------------------------------------------------

def render_dashboard(
    space_name: str,
    loader: CacheLoader,
    selected_date: Optional[str] = None,
    time_range: tuple[int, int] = (7, 23),
    mode: str = "daily",
    fp_coverage: str = "medium",
) -> None:
    """
    Render dashboard in daily or comparison mode.

    Parameters
    ----------
    space_name : str
    loader : CacheLoader
    selected_date : str | None
        Selected date for daily mode
    time_range : tuple[int, int]
        Hour range filter (start, end)
    mode : str
        "daily" or "comparison"
    fp_coverage : str
        "narrow" | "medium" | "wide" — selects floating population coverage column
    """
    if mode == "daily":
        _render_daily(space_name, loader, selected_date, time_range, fp_coverage=fp_coverage)
    else:
        _render_comparison(space_name, loader, time_range, fp_coverage=fp_coverage)


# ===========================================================================
#  DAILY ANALYSIS MODE
# ===========================================================================

def _fp_col(daily_stats: pd.DataFrame, fp_coverage: str) -> str:
    """Return the floating population column name based on coverage setting.
    Falls back to 'floating_unique' if coverage columns are not yet in cache.
    """
    col_map = {"narrow": "floating_narrow", "medium": "floating_medium", "wide": "floating_wide", "full": "floating_full"}
    col = col_map.get(fp_coverage, "floating_medium")
    if col in daily_stats.columns:
        return col
    return "floating_unique"


def _render_daily(
    space_name: str,
    loader: CacheLoader,
    selected_date: Optional[str],
    time_range: tuple[int, int],
    fp_coverage: str = "medium",
) -> None:
    """Single-date deep dive: KPI -> Trend -> Hourly -> Dwell -> AI."""

    daily_stats = loader.get_daily_stats()
    daily_hourly = loader.get_daily_hourly()
    daily_timeseries = loader.get_daily_timeseries()
    sessions = loader.get_sessions_stitched()
    if sessions.empty:
        sessions = loader.get_sessions_all()

    if daily_stats.empty:
        st.info("No data available. Run preprocessing first.")
        return

    available_dates = loader.get_date_range() or []
    if not available_dates:
        st.warning("No dates available.")
        return

    # Use selected_date or latest
    if selected_date and selected_date in available_dates:
        date_str = selected_date
    else:
        date_str = available_dates[-1]

    # Get daily row
    ds = daily_stats[daily_stats["date"].astype(str) == date_str]
    if ds.empty:
        st.warning(f"No data for {date_str}.")
        return

    row = ds.iloc[0]

    # Header with weather context
    header = f"## {date_str}"
    weather_badge = ""
    if "weather" in ds.columns:
        w = row.get("weather", "")
        w_icon = {"Sunny": "☀️", "Rain": "🌧️", "Snow": "❄️"}.get(str(w), "")
        temp_str = ""
        if "temp_max" in ds.columns and "temp_min" in ds.columns:
            tmax = row.get("temp_max", 0)
            tmin = row.get("temp_min", 0)
            temp_str = f" {tmin:.0f}~{tmax:.0f}°C"
        weather_badge = f"&nbsp;&nbsp;{w_icon} {w}{temp_str}"
    st.markdown(f"{header}{weather_badge}", unsafe_allow_html=True)

    # -- KPI Cards --
    has_quality = "quality_cvr" in daily_stats.columns
    has_median = "dwell_median_seconds" in daily_stats.columns

    fp_col = _fp_col(daily_stats, fp_coverage)
    fp_val = int(row.get(fp_col, 0))
    v_val = int(row.get("visitor_count", 0))
    cvr_val = (v_val / fp_val * 100.0) if fp_val > 0 else 0.0
    q_cvr_val = (float(row.get("quality_visitor_count", 0)) / fp_val * 100.0) if has_quality and fp_val > 0 else None
    if has_median:
        dwell_sec = row.get("dwell_median_seconds", 0)
    else:
        dwell_sec = row.get("dwell_seconds_mean", 0)
    mm, ss = int(dwell_sec) // 60, int(dwell_sec) % 60

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Floating Population", f"{fp_val:,}", "Passersby near store")
    with cols[1]:
        render_metric_card("Visitors", f"{v_val:,}", "Entered store")
    with cols[2]:
        cvr_sub = f"Quality CVR: {q_cvr_val:.1f}%" if q_cvr_val is not None else ""
        render_metric_card("CVR", f"{cvr_val:.1f}%", cvr_sub)
    with cols[3]:
        dwell_label = "Median Dwell" if has_median else "Avg Dwell"
        render_metric_card(dwell_label, f"{mm}m {ss}s", "Time inside store")

    # -- Metric Definitions --
    with st.expander("Metric Definitions", expanded=False):
        st.markdown("""
| Metric | Definition |
|--------|-----------|
| **FP (Floating Population)** | 매장 주변에서 BLE 신호가 감지된 고유 디바이스 수. 입구 센서 + 내부 센서 합집합. 지나가는 사람과 들어온 사람 모두 포함. |
| **Visitors** | 3중 필터를 통과한 실제 매장 방문자. ① 내부 센서에서 세션 구성 → ② 최소 체류시간 이상 → ③ 신호의 80% 이상이 RSSI 임계값 통과. |
| **CVR (Conversion Rate)** | Visitors ÷ FP × 100%. 유동인구 중 실제 매장에 들어온 비율. 매장의 유인력(Pull Power) 지표. |
| **Quality CVR** | (중기+장기 체류 방문자) ÷ FP × 100%. 짧은 방문(단기)을 제외하고 의미 있는 체류를 한 고객만 카운트. 매장의 진짜 매력도 지표. |
| **Dwell Time** | 방문 세션의 입장~퇴장 시간. 퇴장 시각은 마지막 신호 수신 기준 (back-dating). 체류 품질을 나타냄. |
| **Short / Medium / Long** | 체류시간 세그먼트. 매장 유형별 기준이 다름 (편의점: <2분/2~5분/5분+ / 스포츠매장: <3분/3~10분/10분+). |
""")


    # -- Intraday Traffic Pattern --
    render_section_header("Intraday Traffic Pattern")

    # Resolution selector
    bin_options = {"5 min": 5, "10 min": 10, "30 min": 30, "1 hour": 60}
    col_res, _ = st.columns([1, 3])
    with col_res:
        bin_label = st.selectbox(
            "Resolution",
            list(bin_options.keys()),
            index=2,  # default: 30 min
            key="dashboard_bin_resolution",
        )
    bin_minutes = bin_options[bin_label]

    if not daily_hourly.empty:
        h_df = _get_hourly_stats_cached(
            daily_hourly, sessions, daily_timeseries,
            [date_str], bin_minutes=bin_minutes,
            daily_stats=daily_stats, fp_coverage=fp_coverage,
        )

        if not h_df.empty:
            # Filter by time_range (no copy needed - filtering creates new DataFrame)
            if time_range:
                start_bin = time_range[0] * 60 // bin_minutes
                end_bin = time_range[1] * 60 // bin_minutes
                h_df_filtered = h_df[
                    (h_df["bin_idx"] >= start_bin)
                    & (h_df["bin_idx"] < end_bin)
                ]
            else:
                h_df_filtered = h_df

            # Peak hours
            peaks = identify_peak_hours(h_df_filtered, metric="visitor_count", top_n=3)
            if peaks:
                peak_cols = st.columns(3)
                for i, p in enumerate(peaks):
                    with peak_cols[i]:
                        render_metric_card(
                            f"Peak #{p['rank']}: {p['bin_label']}",
                            f"{int(p['visitors'])} visitors",
                            f"FP: {int(p['fp'])} | CVR: {p['cvr']:.1f}%",
                        )

            # Bar chart: FP + Visitors + CVR overlay
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=h_df_filtered["bin_label"], y=h_df_filtered["floating_count"],
                name="Floating Pop",
                marker_color=FP_COLOR, opacity=0.7,
            ))
            fig.add_trace(go.Bar(
                x=h_df_filtered["bin_label"], y=h_df_filtered["visitor_count"],
                name="Visitors",
                marker_color=VISITOR_COLOR,
            ))
            fig.add_trace(go.Scatter(
                x=h_df_filtered["bin_label"], y=h_df_filtered["cvr_pct"],
                name="CVR (%)", yaxis="y2",
                line=dict(color=AMBER, width=2.5),
                mode="lines+markers", marker=dict(size=3),
            ))
            chart_title = f"FP & Visitors ({bin_label} bins)"
            layout = make_plotly_layout(chart_title, height=380)
            layout["barmode"] = "group"
            layout["yaxis2"] = dict(
                title=dict(text="CVR (%)", font=dict(color=AMBER)),
                overlaying="y", side="right",
                range=[0, max(h_df_filtered["cvr_pct"].max() * 1.3, 15)] if not h_df_filtered.empty else [0, 15],
                gridcolor="#1a2035",
                tickfont=dict(color=AMBER),
            )
            fig.update_layout(**layout)
            # Show every Nth label to avoid overlap on 5-min bins
            tick_interval = max(1, len(h_df_filtered) // 24)
            fig.update_layout(xaxis=dict(
                tickangle=-45,
                dtick=tick_interval,
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"CVR in chart = visitors ÷ FP per time bin. "
                f"Daily CVR ({cvr_val:.1f}%) uses full-day unique FP ({fp_val:,}), "
                f"so per-bin CVR appears higher due to the same person being counted in multiple bins."
            )
    else:
        st.caption("No traffic data available for this date.")

    # -- Dwell Funnel (5-tier from sessions) --
    render_section_header("Dwell Time Distribution")

    # Compute 5-tier dwell from sessions directly
    s_day = sessions[sessions["date"].astype(str) == date_str] if "date" in sessions.columns else sessions
    if not s_day.empty and "dwell_seconds" in s_day.columns:
        d = s_day["dwell_seconds"]
        total_v = len(d) or 1
        dwell_tiers = [
            ("1~3 min",   int(((d >= 60) & (d < 180)).sum()),   "#64748b"),
            ("3~6 min",   int(((d >= 180) & (d < 360)).sum()),  FP_COLOR),
            ("6~10 min",  int(((d >= 360) & (d < 600)).sum()),  VISITOR_COLOR),
            ("10~15 min", int(((d >= 600) & (d < 900)).sum()),  GOLD),
            ("15+ min",   int((d >= 900).sum()),                AMBER),
        ]
        # Quality = 3min 이상 체류
        quality_count = sum(c for _, c, _ in dwell_tiers[1:])
        quality_pct = quality_count / total_v * 100

        # KPI cards
        fc2 = st.columns(5)
        for i, (label, count, _) in enumerate(dwell_tiers):
            with fc2[i]:
                pct = count / total_v * 100
                render_metric_card(label, f"{pct:.1f}%", f"{count}")

        st.caption(f"Quality Visitors (3min+): {quality_count} of {total_v} ({quality_pct:.1f}%)")

        # Bar chart
        fig = go.Figure()
        for label, count, color in dwell_tiers:
            pct = count / total_v * 100
            fig.add_trace(go.Bar(
                x=[label], y=[count],
                name=label, marker_color=color,
                text=[f"{count} ({pct:.0f}%)"],
                textposition="outside",
                textfont=dict(color="#ccd6f6", size=11),
            ))
        fig.update_layout(**make_plotly_layout("Dwell Categories", height=350))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Dwell funnel data not available.")

    # -- Device Mix (iPhone vs Android) --
    _render_device_mix_daily(loader, date_str)

    # -- Detail Table (expandable) --
    with st.expander("Daily Detail Table"):
        detail_cols = ["date", fp_col if fp_col != "floating_unique" else "floating_unique", "visitor_count", "conversion_rate"]
        if has_quality:
            detail_cols.append("quality_cvr")
        if has_median:
            detail_cols.append("dwell_median_seconds")
        else:
            detail_cols.append("dwell_seconds_mean")
        if "day_type" in daily_stats.columns:
            detail_cols.append("day_type")
        if "weather" in daily_stats.columns:
            detail_cols.append("weather")

        detail = daily_stats[[c for c in detail_cols if c in daily_stats.columns]].copy()
        detail = detail.sort_values("date", ascending=False)
        st.dataframe(detail, use_container_width=True, hide_index=True, height=400)

    # -- AI Analysis --
    st.markdown("---")
    render_section_header("AI Analysis")
    st.caption("AI analyzes store traffic patterns and provides actionable insights.")

    if not has_api_key():
        st.info("Set ANTHROPIC_API_KEY in environment or secrets to enable AI.")
        return

    col_btn, col_lang = st.columns([3, 1])
    with col_lang:
        ai_lang = st.selectbox("Language", ["English", "한국어"], key="daily_ai_lang", label_visibility="collapsed")
    with col_btn:
        if st.button("Generate AI Analysis", type="primary", use_container_width=True, key="daily_ai_btn"):
            _run_daily_ai(space_name, row, daily_stats, loader, date_str, time_range, lang=ai_lang, fp_coverage=fp_coverage)


def _render_ai_sections(text: str, section_labels: list[str]) -> None:
    """Render AI response as ai-comment boxes.

    Tries to split by section labels (PERFORMANCE, BEHAVIOR, etc.).
    Handles both English and Korean responses where model may output
    "한국어제목 (EnglishLabel)" or "## EnglishLabel" formats.
    Falls back to a single box if no labels found.
    """
    if not text or not text.strip():
        return

    # Build regex: match lines that contain any of the section labels
    # e.g. "PERFORMANCE:", "## Performance", "성과 분석 (PERFORMANCE)", "**PERFORMANCE**"
    label_pattern = "|".join(re.escape(lb) for lb in section_labels)
    # Match a line that contains a label (possibly with Korean prefix, ##, **, :, etc.)
    split_re = re.compile(
        rf"^.*?({label_pattern}).*?$",
        re.IGNORECASE | re.MULTILINE,
    )

    parts = split_re.split(text)
    # parts = [preamble, label1, content1, label2, content2, ...]

    sections: list[tuple[str, str]] = []
    if len(parts) >= 3:
        # Skip preamble (parts[0]), then pair (label, content)
        i = 1
        while i < len(parts) - 1:
            label = parts[i].strip()
            content = parts[i + 1].strip()
            # Clean up content: remove leading ), }, #, *, etc.
            content = re.sub(r"^[\)\}\]\*#\s:]+", "", content).strip()
            if content:
                sections.append((label.title(), content))
            i += 2

    if not sections:
        # Fallback: render as single block
        render_ai_comment("AI Insights", text.strip())
    else:
        for title, content in sections:
            render_ai_comment(title, content)


def _run_daily_ai(
    space_name: str,
    row: pd.Series,
    daily_stats: pd.DataFrame,
    loader: CacheLoader,
    date_str: str,
    time_range: tuple[int, int],
    lang: str = "English",
    fp_coverage: str = "medium",
) -> None:
    """Run AI analysis for daily mode — enriched with hourly pattern, trend direction, Korean calendar."""
    has_quality = "quality_cvr" in daily_stats.columns
    has_median = "dwell_median_seconds" in daily_stats.columns
    qc = "quality_cvr" if has_quality else "conversion_rate"
    dwell_col = "dwell_median_seconds" if has_median else "dwell_seconds_mean"
    fp_col_daily = _fp_col(daily_stats, fp_coverage)

    metrics = {
        "space_name": space_name,
        "date": date_str,
        "time_range": list(time_range),
        "fp": int(row.get(fp_col_daily, 0)),
        "visitors": int(row.get("visitor_count", 0)),
        "cvr": float(row.get(qc, 0)),
        "dwell_sec": float(row.get(dwell_col, 0)),
        "has_quality": has_quality,
    }

    # ── 한국 캘린더 컨텍스트 ──────────────────────────────────────────────
    cal = get_korean_calendar_context(date_str)
    if cal:
        metrics["korean_calendar"] = cal

    # ── 5-tier 체류 분포 ──────────────────────────────────────────────────
    sessions = loader.get_sessions_stitched()
    if sessions.empty:
        sessions = loader.get_sessions_all()
    s_day = sessions[sessions["date"].astype(str) == date_str] if not sessions.empty and "date" in sessions.columns else pd.DataFrame()
    if not s_day.empty and "dwell_seconds" in s_day.columns:
        d_ai = s_day["dwell_seconds"]
        total_v = len(d_ai) or 1
        metrics["dwell_1_3min_pct"]   = round(((d_ai >= 60)  & (d_ai < 180)).sum() / total_v * 100, 1)
        metrics["dwell_3_6min_pct"]   = round(((d_ai >= 180) & (d_ai < 360)).sum() / total_v * 100, 1)
        metrics["dwell_6_10min_pct"]  = round(((d_ai >= 360) & (d_ai < 600)).sum() / total_v * 100, 1)
        metrics["dwell_10_15min_pct"] = round(((d_ai >= 600) & (d_ai < 900)).sum() / total_v * 100, 1)
        metrics["dwell_15plus_pct"]   = round((d_ai >= 900).sum() / total_v * 100, 1)

    # ── 시간대별 패턴 (hourly breakdown for AI) ───────────────────────────
    daily_hourly = loader.get_daily_hourly()
    if not daily_hourly.empty:
        h_day = daily_hourly[daily_hourly["date"].astype(str) == date_str]
        if not h_day.empty:
            def _band_sum(h_df, h_start, h_end, col):
                return int(h_df[(h_df["hour"] >= h_start) & (h_df["hour"] < h_end)][col].sum())
            _fp_col_h_cov = f"floating_count_{fp_coverage}"
            fp_col_h = _fp_col_h_cov if _fp_col_h_cov in h_day.columns else "floating_count"
            metrics["hourly_bands"] = {
                "morning_07_09":   {"fp": _band_sum(h_day, 7, 9, fp_col_h),   "visitors": _band_sum(h_day, 7, 9, "visitor_count")},
                "lunch_12_14":     {"fp": _band_sum(h_day, 12, 14, fp_col_h), "visitors": _band_sum(h_day, 12, 14, "visitor_count")},
                "afternoon_14_18": {"fp": _band_sum(h_day, 14, 18, fp_col_h), "visitors": _band_sum(h_day, 14, 18, "visitor_count")},
                "evening_18_21":   {"fp": _band_sum(h_day, 18, 21, fp_col_h), "visitors": _band_sum(h_day, 18, 21, "visitor_count")},
                "night_21_02":     {"fp": _band_sum(h_day, 21, 24, fp_col_h) + _band_sum(h_day, 0, 2, fp_col_h),
                                    "visitors": _band_sum(h_day, 21, 24, "visitor_count") + _band_sum(h_day, 0, 2, "visitor_count")},
            }
            # Top-3 peak hours
            peak_rows = h_day.sort_values("visitor_count", ascending=False).head(3)
            metrics["peak_hours"] = [
                {"hour": int(r["hour"]), "visitors": int(r["visitor_count"]), "fp": int(r.get(fp_col_h, r.get("floating_count", 0)))}
                for _, r in peak_rows.iterrows()
            ]

    # ── 날씨 ──────────────────────────────────────────────────────────────
    for col in ("day_type", "weather", "temp_max", "temp_min", "precipitation"):
        if col in row.index:
            metrics[col] = float(row.get(col, 0)) if col in ("temp_max", "temp_min", "precipitation") else str(row.get(col, ""))

    # ── 트렌드 방향성 ─────────────────────────────────────────────────────
    sorted_stats = daily_stats.sort_values("date").reset_index(drop=True)
    n_total = len(sorted_stats)
    metrics["total_days_available"] = n_total
    _fp_avg_col = fp_col_daily if fp_col_daily in sorted_stats.columns else "floating_unique"
    metrics["all_period_avg_fp"]  = float(sorted_stats[_fp_avg_col].mean())
    metrics["all_period_avg_v"]   = float(sorted_stats["visitor_count"].mean())
    metrics["all_period_avg_cvr"] = float(sorted_stats[qc].mean())

    today_idx = sorted_stats[sorted_stats["date"].astype(str) == date_str].index
    if len(today_idx) > 0:
        idx = today_idx[0]
        # 전주 동요일 비교
        same_dow_rows = sorted_stats[sorted_stats["weekday"] == row.get("weekday", -1)] if "weekday" in sorted_stats.columns else pd.DataFrame()
        prev_same_dow = same_dow_rows[same_dow_rows.index < idx].tail(1)
        if not prev_same_dow.empty:
            pr = prev_same_dow.iloc[0]
            pr_fp = int(pr.get(_fp_avg_col, pr.get("floating_unique", 0)))
            metrics["vs_same_dow_last_week"] = {
                "date": str(pr["date"]),
                "fp_delta_pct": round((metrics["fp"] - pr_fp) / max(pr_fp, 1) * 100, 1),
                "visitor_delta_pct": round((metrics["visitors"] - int(pr.get("visitor_count", 0))) / max(int(pr.get("visitor_count", 1)), 1) * 100, 1),
            }
        # 최근 7일 트렌드 (선형 방향)
        recent7 = sorted_stats[sorted_stats.index < idx].tail(7)
        if len(recent7) >= 3:
            v_series = recent7["visitor_count"].values
            slope = float(v_series[-1] - v_series[0]) / max(len(v_series) - 1, 1)
            if slope > 5:
                trend = "increasing"
            elif slope < -5:
                trend = "decreasing"
            else:
                trend = "stable"
            metrics["recent_7day_trend"] = trend
            metrics["recent_7day_avg_v"] = round(float(v_series.mean()), 1)

    # ── Day-type / Weather 요약 ───────────────────────────────────────────
    if "day_type" in sorted_stats.columns:
        metrics["daytype_summary"] = sorted_stats.groupby("day_type").agg(
            {_fp_avg_col: "mean", "visitor_count": "mean", qc: "mean"}
        ).rename(columns={_fp_avg_col: "floating_unique"}).round(1).to_dict(orient="index")
    if "weather" in sorted_stats.columns:
        metrics["weather_summary"] = sorted_stats.groupby("weather").agg(
            {_fp_avg_col: "mean", "visitor_count": "mean", qc: "mean"}
        ).rename(columns={_fp_avg_col: "floating_unique"}).round(1).to_dict(orient="index")

    # ── 날짜별 히스토리 테이블 ────────────────────────────────────────────
    recent_days = sorted_stats.tail(_AI_MAX_DAYS)
    if len(recent_days) >= 2:
        recent_rows = []
        for _, r in recent_days.iterrows():
            entry = {
                "date": str(r["date"]),
                "fp": int(r.get(_fp_avg_col, r.get("floating_unique", 0))),
                "visitors": int(r.get("visitor_count", 0)),
                "cvr": round(float(r.get(qc, 0)), 1),
                "dwell_min": round(float(r.get(dwell_col, 0)) / 60, 1),
            }
            if "day_type" in r.index: entry["day_type"] = str(r.get("day_type", ""))
            if "weather" in r.index:  entry["weather"]  = str(r.get("weather", ""))
            if "temp_max" in r.index: entry["temp"] = f"{r.get('temp_min', 0):.0f}~{r.get('temp_max', 0):.0f}"
            recent_rows.append(entry)
        metrics["history_days"] = recent_rows
        metrics["history_count"] = len(recent_rows)
        metrics["period_avg_fp"]  = float(recent_days[_fp_avg_col].mean())
        metrics["period_avg_v"]   = float(recent_days["visitor_count"].mean())
        metrics["period_avg_cvr"] = float(recent_days[qc].mean())

    space_notes = st.session_state.get("current_space_notes", "")
    with st.spinner("AI is analyzing..."):
        result = _cached_ai_daily(json.dumps(metrics, default=str), space_notes=space_notes, lang=lang)

    if result and not result.startswith("\u26a0\ufe0f"):
        _render_ai_sections(result, ["PERFORMANCE", "BEHAVIOR", "RECOMMENDATIONS"])
    else:
        st.warning(result if result else "AI analysis failed.")


# ===========================================================================
#  PERIOD COMPARISON MODE
# ===========================================================================

def _render_comparison(
    space_name: str,
    loader: CacheLoader,
    time_range: tuple[int, int],
    fp_coverage: str = "medium",
) -> None:
    """Multi-date comparison: Summary -> Trend -> Dwell -> AI."""

    daily_stats = loader.get_daily_stats()
    if daily_stats.empty:
        st.info("No data available.")
        return

    available_dates = loader.get_date_range() or []
    if len(available_dates) < 2:
        st.warning("Need at least 2 dates for comparison.")
        return

    st.markdown("## Period Comparison Analysis")

    ds = daily_stats.sort_values("date").copy()
    has_quality = "quality_cvr" in ds.columns
    has_median = "dwell_median_seconds" in ds.columns
    qc = "quality_cvr" if has_quality else "conversion_rate"
    dwell_col = "dwell_median_seconds" if has_median else "dwell_seconds_mean"

    fp_col = _fp_col(ds, fp_coverage)

    # -- Build x-axis labels: "03-29(Sat)☀️" --
    _DOW = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    _WICON = {"Sunny": "☀️", "Rain": "🌧", "Snow": "❄️"}

    def _x_label(row):
        try:
            dt = pd.to_datetime(row["date"])
            dow = _DOW.get(dt.dayofweek, "")
            lbl = f"{dt.strftime('%m-%d')}({dow})"
            if "weather" in row.index:
                icon = _WICON.get(str(row.get("weather", "")), "")
                if icon:
                    lbl += icon
            return lbl
        except Exception:
            return str(row["date"])

    ds["x_label"] = ds.apply(_x_label, axis=1)

    # -- KPI summary cards (first) --
    avg_fp = ds[fp_col].mean()
    avg_v = ds["visitor_count"].mean()
    avg_cvr = (avg_v / avg_fp * 100.0) if avg_fp > 0 else 0.0
    avg_dwell = ds[dwell_col].mean()
    mm, ss = int(avg_dwell) // 60, int(avg_dwell) % 60

    st.caption(f"Period: {ds['date'].iloc[0]} ~ {ds['date'].iloc[-1]} ({len(ds)} days)")

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_metric_card("Avg Daily FP", f"{avg_fp:,.0f}", f"{len(ds)} days")
    with kpi_cols[1]:
        render_metric_card("Avg Daily Visitors", f"{avg_v:,.0f}", "")
    with kpi_cols[2]:
        q_cvr_str = f"Quality: {ds[qc].mean():.1f}%" if has_quality else ""
        render_metric_card("Avg CVR", f"{avg_cvr:.1f}%", q_cvr_str)
    with kpi_cols[3]:
        render_metric_card("Avg Dwell", f"{mm}m {ss}s", "")

    with st.expander("Metric Definitions", expanded=False):
        st.markdown("""
| Metric | Definition |
|--------|-----------|
| **FP (Floating Population)** | 매장 주변에서 감지된 고유 디바이스 수. 지나가는 사람 + 들어온 사람 모두 포함. |
| **Visitors** | 3중 필터(세션 구성 + 최소 체류 + 80% RSSI 통과)를 통과한 실제 방문자. |
| **CVR** | Visitors ÷ FP × 100%. 유동인구 대비 실제 입장 비율. |
| **Quality CVR** | (중기+장기 체류 방문자) ÷ FP × 100%. 의미 있는 체류를 한 고객만 카운트. |
| **Dwell Time** | 입장~퇴장 시간. 체류 품질을 나타내는 핵심 지표. |
""")

    # -- WoW Comparison --
    if len(ds) >= 14:
        render_section_header("Week-over-Week Comparison")
        wow = _cached_wow(ds, fp_col=fp_col)
        if wow:
            tw = wow.get("this_week", {})
            d = wow.get("delta", {})
            wow_cols = st.columns(4)
            with wow_cols[0]:
                st.metric("FP (This Week)", f"{tw.get('floating_unique', 0):,.0f}",
                          f"{d.get('floating_pct', 0):+.1f}% WoW")
            with wow_cols[1]:
                v_key = "quality_visitor_count" if has_quality else "visitor_count"
                st.metric("Visitors", f"{tw.get(v_key, 0):,.0f}",
                          f"{d.get('quality_visitor_pct' if has_quality else 'visitor_pct', 0):+.1f}% WoW")
            with wow_cols[2]:
                st.metric("CVR", f"{tw.get(qc, 0):.1f}%",
                          f"{d.get('quality_cvr_pp' if has_quality else 'cvr_pp', 0):+.1f}pp WoW")
            with wow_cols[3]:
                dm = tw.get("dwell_median_seconds", 0)
                st.metric("Dwell", f"{int(dm)//60}m {int(dm)%60}s",
                          f"{d.get('dwell_median', 0):+.0f}s WoW")

    # -- Traffic Trend --
    render_section_header("Daily Traffic Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ds["x_label"], y=ds[fp_col],
        name="Floating Pop",
        line=dict(color=FP_COLOR, width=2),
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=ds["x_label"], y=ds["visitor_count"],
        name="Visitors",
        line=dict(color=VISITOR_COLOR, width=2),
        mode="lines+markers",
        yaxis="y2",
    ))
    layout = make_plotly_layout("FP & Visitors Over Time")
    layout["yaxis"] = dict(
        title=dict(text="Floating Pop", font=dict(color=FP_COLOR)),
        gridcolor="#1a2035", zerolinecolor="#1a2035",
        tickfont=dict(color=FP_COLOR),
    )
    layout["yaxis2"] = dict(
        title=dict(text="Visitors", font=dict(color=VISITOR_COLOR)),
        overlaying="y", side="right",
        gridcolor="#1a2035", zerolinecolor="#1a2035",
        tickfont=dict(color=VISITOR_COLOR),
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # -- CVR Trend --
    render_section_header("CVR Trend")

    fig_cvr = go.Figure()
    # Recompute CVR dynamically from the selected coverage column so it responds to fp_coverage
    ds["cvr_dynamic"] = ds.apply(
        lambda r: round(r["visitor_count"] / r[fp_col] * 100.0, 2) if r.get(fp_col, 0) > 0 else 0.0,
        axis=1,
    )
    cvr_col = "cvr_dynamic"
    fig_cvr.add_trace(go.Scatter(
        x=ds["x_label"], y=ds[cvr_col],
        name="CVR (%)",
        line=dict(color=AMBER, width=2.5),
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(217,119,6,0.1)",
    ))
    # Also show Quality CVR if available
    if has_quality:
        fig_cvr.add_trace(go.Scatter(
            x=ds["x_label"], y=ds["quality_cvr"],
            name="Quality CVR (%)",
            line=dict(color=GOLD, width=1.5, dash="dash"),
            mode="lines+markers",
        ))
    fig_cvr.update_layout(**make_plotly_layout("CVR (%) — solid: All Visitors, dashed: Quality", height=300))
    st.plotly_chart(fig_cvr, use_container_width=True)

    # -- Device Mix Trend --
    _render_device_mix_comparison(loader)

    # -- Dwell Trend (5-tier from sessions) --
    sessions = loader.get_sessions_stitched()
    if sessions.empty:
        sessions = loader.get_sessions_all()

    if not sessions.empty and "dwell_seconds" in sessions.columns and "date" in sessions.columns:
        render_section_header("Dwell Distribution Over Time")

        _TIERS = [
            ("1~3min",   60,  180, "#64748b"),
            ("3~6min",  180,  360, FP_COLOR),
            ("6~10min", 360,  600, VISITOR_COLOR),
            ("10~15min",600,  900, GOLD),
            ("15+min",  900, 99999, AMBER),
        ]

        # Compute per-date tier counts
        dwell_rows = []
        for date_val in ds["date"].values:
            s_d = sessions[sessions["date"].astype(str) == str(date_val)]
            d = s_d["dwell_seconds"] if not s_d.empty else pd.Series(dtype=float)
            entry = {"date": str(date_val)}
            for label, lo, hi, _ in _TIERS:
                entry[label] = int(((d >= lo) & (d < hi)).sum())
            dwell_rows.append(entry)
        dwell_df = pd.DataFrame(dwell_rows)
        # Merge x_label
        dwell_df = dwell_df.merge(ds[["date", "x_label"]].astype(str), on="date", how="left")

        fig_dwell = go.Figure()
        for label, _, _, color in _TIERS:
            fig_dwell.add_trace(go.Bar(
                x=dwell_df["x_label"], y=dwell_df[label],
                name=label, marker_color=color,
            ))
        layout = make_plotly_layout("Dwell Breakdown (Visitors Only)", height=380)
        layout["barmode"] = "stack"
        layout["legend"] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#ccd6f6"),
            traceorder="normal",  # 1~3min first → 15+min last
        )
        fig_dwell.update_layout(**layout)
        st.plotly_chart(fig_dwell, use_container_width=True)

    # -- Day Type / Weather Pattern --
    if "day_type" in ds.columns:
        render_section_header("Weekday vs Weekend Pattern")
        dt_summary = ds.groupby("day_type").agg(
            Avg_FP=(fp_col, "mean"),
            Avg_Visitors=("visitor_count", "mean"),
            Avg_CVR=(qc, "mean"),
            Days=("date", "count"),
        ).round(1).reset_index()
        dt_summary = dt_summary.rename(columns={"day_type": "Day Type"})
        st.dataframe(dt_summary, use_container_width=True, hide_index=True)

    if "weather" in ds.columns:
        render_section_header("Weather Impact")
        w_summary = ds.groupby("weather").agg(
            Avg_FP=(fp_col, "mean"),
            Avg_Visitors=("visitor_count", "mean"),
            Avg_CVR=(qc, "mean"),
            Days=("date", "count"),
        ).round(1).reset_index()
        w_summary = w_summary.rename(columns={"weather": "Weather"})
        st.dataframe(w_summary, use_container_width=True, hide_index=True)

    # -- AI Comparison --
    st.markdown("---")
    render_section_header("AI Comparison Analysis")
    st.caption("AI analyzes traffic trends across the entire period.")

    if not has_api_key():
        st.info("Set ANTHROPIC_API_KEY to enable AI.")
        return

    col_btn2, col_lang2 = st.columns([3, 1])
    with col_lang2:
        ai_lang2 = st.selectbox("Language", ["English", "한국어"], key="comparison_ai_lang", label_visibility="collapsed")
    with col_btn2:
        if st.button("Generate Period Analysis", type="primary", use_container_width=True, key="comparison_ai_btn"):
            _run_comparison_ai(space_name, ds, loader, time_range, lang=ai_lang2, fp_coverage=fp_coverage)

    # -- Daily Detail Table (at bottom, collapsed) --
    st.markdown("---")
    with st.expander("Daily Detail Table", expanded=False):
        summary_df = ds[["date", fp_col, "visitor_count", qc, dwell_col]].copy()
        summary_df = summary_df.rename(columns={
            "date": "Date",
            fp_col: "FP",
            "visitor_count": "Visitors",
            qc: "CVR (%)",
            dwell_col: "Dwell (sec)",
        })
        if "Dwell (sec)" in summary_df.columns:
            summary_df["Dwell (min)"] = (summary_df["Dwell (sec)"] / 60).round(1)
            summary_df = summary_df.drop(columns=["Dwell (sec)"])
        if "CVR (%)" in summary_df.columns:
            summary_df["CVR (%)"] = summary_df["CVR (%)"].round(1)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _run_comparison_ai(
    space_name: str,
    ds: pd.DataFrame,
    loader: CacheLoader,
    time_range: tuple[int, int],
    lang: str = "English",
    fp_coverage: str = "medium",
) -> None:
    """Run AI analysis for comparison mode — enriched with trend direction and Korean calendar."""

    has_quality = "quality_cvr" in ds.columns
    qc = "quality_cvr" if has_quality else "conversion_rate"
    dwell_col = "dwell_median_seconds" if "dwell_median_seconds" in ds.columns else "dwell_seconds_mean"
    fp_col = _fp_col(ds, fp_coverage)
    fp_vals = ds[fp_col] if fp_col in ds.columns else ds["floating_unique"]

    ds_sorted = ds.sort_values("date").reset_index(drop=True)

    metrics = {
        "space_name": space_name,
        "n_days": len(ds_sorted),
        "date_range": f"{ds_sorted['date'].min()} ~ {ds_sorted['date'].max()}",
        "avg_fp": round(float(fp_vals.mean()), 0),
        "avg_visitors": round(float(ds_sorted["visitor_count"].mean()), 1),
        "avg_cvr": round(float(ds_sorted[qc].mean()), 1),
        "avg_dwell_sec": round(float(ds_sorted[dwell_col].mean()), 0),
        "total_fp": int(fp_vals.sum()),
        "total_visitors": int(ds_sorted["visitor_count"].sum()),
    }

    # Best / worst days
    best_day  = ds_sorted.loc[ds_sorted["visitor_count"].idxmax()]
    worst_day = ds_sorted.loc[ds_sorted["visitor_count"].idxmin()]
    metrics["best_day"]  = {"date": str(best_day["date"]),  "visitors": int(best_day["visitor_count"]),  "cvr": round(float(best_day.get(qc, 0)), 1)}
    metrics["worst_day"] = {"date": str(worst_day["date"]), "visitors": int(worst_day["visitor_count"]), "cvr": round(float(worst_day.get(qc, 0)), 1)}

    # ── 트렌드 방향 (전반부 vs 후반부) ───────────────────────────────────
    n = len(ds_sorted)
    if n >= 4:
        half = n // 2
        first_half_avg  = float(ds_sorted["visitor_count"].iloc[:half].mean())
        second_half_avg = float(ds_sorted["visitor_count"].iloc[half:].mean())
        delta_pct = round((second_half_avg - first_half_avg) / max(first_half_avg, 1) * 100, 1)
        metrics["trend_direction"] = "increasing" if delta_pct > 5 else ("decreasing" if delta_pct < -5 else "stable")
        metrics["trend_delta_pct"] = delta_pct
        metrics["first_half_avg_v"]  = round(first_half_avg, 1)
        metrics["second_half_avg_v"] = round(second_half_avg, 1)

    # ── 요일별 패턴 ────────────────────────────────────────────────────
    if "day_type" in ds_sorted.columns:
        dt_grp = ds_sorted.groupby("day_type").agg(
            avg_fp=   (fp_col if fp_col in ds_sorted.columns else "floating_unique", "mean"),
            avg_v=    ("visitor_count", "mean"),
            avg_cvr=  (qc, "mean"),
            days=     ("date", "count"),
        ).round(1)
        metrics["day_type_summary"] = dt_grp.to_dict(orient="index")

    # ── 날씨별 패턴 ────────────────────────────────────────────────────
    if "weather" in ds_sorted.columns:
        w_grp = ds_sorted.groupby("weather").agg(
            avg_v=   ("visitor_count", "mean"),
            avg_cvr= (qc, "mean"),
            days=    ("date", "count"),
        ).round(1)
        metrics["weather_summary"] = w_grp.to_dict(orient="index")

    if "temp_max" in ds_sorted.columns:
        metrics["temp_range"] = f"{ds_sorted['temp_min'].min():.1f}~{ds_sorted['temp_max'].max():.1f}°C"
        metrics["avg_temp"]   = f"{((ds_sorted['temp_max'] + ds_sorted['temp_min']) / 2).mean():.1f}°C"
    if "precipitation" in ds_sorted.columns:
        metrics["rainy_days"]          = int((ds_sorted["precipitation"] > 0).sum())
        metrics["total_precipitation"] = round(float(ds_sorted["precipitation"].sum()), 1)

    # ── 한국 캘린더 — 기간 내 공휴일 목록 ────────────────────────────────
    holiday_list = []
    for date_val in ds_sorted["date"].astype(str):
        cal = get_korean_calendar_context(date_val)
        if cal.get("is_holiday"):
            holiday_list.append({"date": date_val, "name": cal["holiday_name"]})
    if holiday_list:
        metrics["holidays_in_period"] = holiday_list

    # ── 날짜별 상세 테이블 ────────────────────────────────────────────
    detail_rows = []
    for _, r in ds_sorted.iterrows():
        entry = {
            "date":      str(r["date"]),
            "fp":        int(r.get(fp_col, r.get("floating_unique", 0))),
            "visitors":  int(r.get("visitor_count", 0)),
            "cvr":       round(float(r.get(qc, 0)), 1),
            "dwell_min": round(float(r.get(dwell_col, 0)) / 60, 1),
        }
        if "day_type" in r.index: entry["day_type"] = str(r.get("day_type", ""))
        if "weather"  in r.index: entry["weather"]  = str(r.get("weather",  ""))
        if "temp_max" in r.index: entry["temp"]     = f"{r.get('temp_min', 0):.0f}~{r.get('temp_max', 0):.0f}°C"
        detail_rows.append(entry)
    metrics["daily_table"] = detail_rows

    space_notes = st.session_state.get("current_space_notes", "")
    with st.spinner("AI is analyzing period data..."):
        result = _cached_ai_comparison(json.dumps(metrics, default=str), space_notes=space_notes, lang=lang)

    if result and not result.startswith("\u26a0\ufe0f"):
        _render_ai_sections(result, ["PERFORMANCE", "PATTERNS", "STRATEGY"])
    else:
        st.warning(result if result else "AI analysis failed.")


# -- Device Mix Helpers ----------------------------------------------------

IPHONE_COLOR = "#007AFF"
ANDROID_COLOR = "#3DDC84"

_DEVICE_LABELS = {1: "iPhone", 10: "Android"}


def _render_device_mix_daily(loader: CacheLoader, date_str: str) -> None:
    """Render iPhone vs Android donut chart + metrics for a single date."""
    device_mix = loader.get_device_mix()
    if device_mix.empty:
        return

    day_mix = device_mix[device_mix["date"].astype(str) == date_str]
    if day_mix.empty:
        return

    iphone_count = int(day_mix.loc[day_mix["device_type"] == 1, "count"].sum())
    android_count = int(day_mix.loc[day_mix["device_type"] == 10, "count"].sum())
    total_dev = iphone_count + android_count
    if total_dev == 0:
        return

    render_section_header("Device Mix (iPhone vs Android)")

    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=["iPhone", "Android"],
            values=[iphone_count, android_count],
            marker=dict(colors=[IPHONE_COLOR, ANDROID_COLOR]),
            textinfo="label+percent",
            textfont=dict(size=13),
            hole=0.45,
        ))
        layout = make_plotly_layout("", height=300)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(**layout)
        fig.update_layout(
            annotations=[dict(
                text=f"{total_dev:,}",
                x=0.5, y=0.5, font_size=20, font_color="#ccd6f6",
                showarrow=False,
            )],
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        iphone_pct = iphone_count / total_dev * 100
        android_pct = android_count / total_dev * 100
        render_metric_card("iPhone", f"{iphone_count:,}", f"{iphone_pct:.1f}%")
        render_metric_card("Android", f"{android_count:,}", f"{android_pct:.1f}%")
        render_metric_card("Total Devices", f"{total_dev:,}", "Visitors detected")


def _render_device_mix_comparison(loader: CacheLoader) -> None:
    """Render iPhone vs Android ratio trend line chart for period comparison."""
    device_mix = loader.get_device_mix()
    if device_mix.empty:
        return

    # Pivot: date x device_type -> count
    pivot = device_mix.pivot_table(
        index="date", columns="device_type", values="count", fill_value=0,
    ).reset_index()

    iphone_col = 1 if 1 in pivot.columns else None
    android_col = 10 if 10 in pivot.columns else None
    if iphone_col is None and android_col is None:
        return

    pivot["iphone"] = pivot.get(iphone_col, 0)
    pivot["android"] = pivot.get(android_col, 0)
    pivot["total"] = pivot["iphone"] + pivot["android"]
    pivot["iphone_pct"] = (pivot["iphone"] / pivot["total"].replace(0, 1) * 100).round(1)
    pivot["android_pct"] = (pivot["android"] / pivot["total"].replace(0, 1) * 100).round(1)
    pivot = pivot.sort_values("date")

    render_section_header("Device Mix Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot["date"].astype(str),
        y=pivot["iphone_pct"],
        name="iPhone %",
        line=dict(color=IPHONE_COLOR, width=2),
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=pivot["date"].astype(str),
        y=pivot["android_pct"],
        name="Android %",
        line=dict(color=ANDROID_COLOR, width=2),
        mode="lines+markers",
    ))
    layout = make_plotly_layout("iPhone vs Android Ratio (%)", height=320)
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# -- AI Prompt Builders ----------------------------------------------------

def _get_sector_behavior_context(space_name: str) -> str:
    """Return sector-specific behavioral patterns for AI prompts."""
    meta = REGISTERED_SECTORS.get(space_name, {})
    stype = meta.get("store_type", "retail")
    if stype == "convenience_store":
        return (
            "[Visitor Behavior Patterns — Convenience Store]\n"
            "- Traffic drivers: commute hours (07-09h, 18-20h), lunch rush (12-13h), late-night (22-02h)\n"
            "- Weather impact: rain → umbrella/hot drink impulse buy; cold → warm food (increased CVR)\n"
            "- Sunny warm weather → people eat outside, CVS traffic may drop\n"
            "- Weekday >> Weekend for office-area CVS (reversed from malls)\n"
            "- Dwell 1-3min: quick purchase (drinks, snacks, cigarettes) — majority of visits\n"
            "- Dwell 3-6min: browsing or meal purchase — high-value customer\n"
            "- Dwell 6min+: eating in-store or using services (ATM, delivery pickup)\n"
            "- MAC randomization note: iPhone MAC changes every 15-20min, Android every 5-10min\n"
            "  → dwell >15min may be underestimated. Focus on 1-10min range for accurate analysis.\n"
            "- Advertising insight: morning/lunch promos target office workers; night promos target residents"
        )
    elif stype == "sports_retail":
        return (
            "[Visitor Behavior Patterns — Sports Retail in Mall]\n"
            "- Traffic drivers: weekend leisure, mall events, seasonal sports (spring/fall best)\n"
            "- Weather impact: rain/snow → indoor mall traffic increases (positive for store)\n"
            "- Sunny weekend → outdoor activities compete with mall visits (negative)\n"
            "- Weekend traffic 2-3x weekday (mall characteristic)\n"
            "- Peak hours: 14-18h (afternoon shopping after lunch)\n"
            "- Dwell 1-3min: window shopping / passing through — low purchase intent\n"
            "- Dwell 3-6min: browsing specific items — medium intent\n"
            "- Dwell 6-10min: trying products, comparing — high intent\n"
            "- Dwell 10min+: serious buyer, likely to purchase or seek staff assistance\n"
            "- MAC randomization note: long browsing sessions (20min+) may split into multiple visits\n"
            "  → focus on 3-15min range for accurate purchase-intent analysis.\n"
            "- Advertising insight: Instagram/social ads before weekend; in-mall digital signage on lower floors\n"
            "- Upper floor (6F) location: visitors reaching here already passed 5 floors of alternatives\n"
            "  → high intent but smaller volume. Quality over quantity."
        )
    return ""


def _build_daily_ai_prompt(metrics: dict) -> str:
    """Build user prompt for daily AI analysis — enriched with hourly bands, trend, Korean calendar."""
    dwell_min = metrics.get("dwell_sec", 0) / 60
    space_name = metrics.get("space_name", "Unknown")
    lang = "한국어"  # GS25 Yeoksam is a Korean store — respond in Korean by default

    lines = [
        _get_sector_context(space_name),
        f"[분석 날짜: {metrics.get('date', 'N/A')}]",
        "",
    ]

    # ── 오늘 핵심 지표 ──────────────────────────────────────────────────────
    lines += [
        "[오늘 핵심 지표]",
        f"- 유동인구 (매장 근처 탐지): {metrics.get('fp', 0):,}명",
        f"- 방문자 (실제 입장 확인): {metrics.get('visitors', 0):,}명",
        f"- CVR (전환율): {metrics.get('cvr', 0):.1f}%  ← 강남 골목 CVS 벤치마크: 2~5%",
        f"- 체류시간 (중앙값): {dwell_min:.1f}분",
    ]

    # ── 한국 캘린더 컨텍스트 ───────────────────────────────────────────────
    cal = metrics.get("korean_calendar", {})
    if cal:
        cal_lines = ["", "[한국 사회 캘린더 컨텍스트]"]
        if cal.get("is_holiday"):
            cal_lines.append(f"- ⚠️  오늘은 공휴일: {cal['holiday_name']} → 강남 오피스 공실, 직장인 수요 급감")
        if cal.get("is_long_weekend") and not cal.get("is_holiday"):
            cal_lines.append("- ⚠️  연휴 기간: 직장인 이탈, 유동인구 감소 예상")
        if cal.get("days_to_next_holiday") == 1:
            cal_lines.append(f"- 내일 공휴일 ({cal['next_holiday_name']}): 오늘 저녁 사전 구매 수요 상승 가능")
        if cal.get("days_since_last_holiday") == 1:
            cal_lines.append(f"- 어제 공휴일 ({cal['last_holiday_name']}) 다음날: 직장인 복귀 → 오전 피크 급증")
        cal_lines.append(f"- 학기 현황: {cal.get('school_term', '')}")
        if cal.get("retail_calendar_note") and cal["retail_calendar_note"] != "일반 평일/주말 패턴":
            cal_lines.append(f"- 유통 시사점: {cal['retail_calendar_note']}")
        lines += cal_lines

    # ── 날씨/요일 컨텍스트 ─────────────────────────────────────────────────
    ctx = []
    if "day_type" in metrics:
        ctx.append(f"- 요일 유형: {metrics['day_type']}")
    if "weather" in metrics:
        w = metrics["weather"]
        if "temp_max" in metrics:
            w += f", 최고 {metrics['temp_max']:.1f}°C / 최저 {metrics['temp_min']:.1f}°C"
        if metrics.get("precipitation", 0) > 0:
            w += f", 강수 {metrics['precipitation']:.1f}mm"
        ctx.append(f"- 날씨: {w}")
    if ctx:
        lines += ["", "[날씨 & 요일]"] + ctx

    # ── 시간대별 패턴 ──────────────────────────────────────────────────────
    bands = metrics.get("hourly_bands", {})
    if bands:
        lines += ["", "[시간대별 유동인구 & 방문자 (오늘)]",
                  "시간대         | 유동인구 | 방문자",
                  "-------------- | -------- | ------"]
        band_labels = {
            "morning_07_09":   "출근(07-09시)",
            "lunch_12_14":     "점심(12-14시)",
            "afternoon_14_18": "오후(14-18시)",
            "evening_18_21":   "퇴근(18-21시)",
            "night_21_02":     "야간(21-02시)",
        }
        for key, label in band_labels.items():
            b = bands.get(key, {})
            lines.append(f"{label:14s} | {b.get('fp', 0):8,} | {b.get('visitors', 0):6,}")

    peaks = metrics.get("peak_hours", [])
    if peaks:
        lines += ["", "[오늘 TOP 3 피크 시간]"]
        for i, p in enumerate(peaks, 1):
            lines.append(f"  #{i}: {p['hour']:02d}시 — 방문자 {p['visitors']}명 (유동인구 {p['fp']}명)")

    # ── 체류 분포 ──────────────────────────────────────────────────────────
    if "dwell_1_3min_pct" in metrics:
        lines += [
            "", "[체류시간 분포 (방문자 기준, 최소 60초 필터 적용)]",
            f"- 1~3분 (즉시구매): {metrics['dwell_1_3min_pct']}%",
            f"- 3~6분 (탐색/식사): {metrics['dwell_3_6min_pct']}%",
            f"- 6~10분 (심층탐색): {metrics['dwell_6_10min_pct']}%",
            f"- 10~15분 (장시간 체류): {metrics['dwell_10_15min_pct']}%",
            f"- 15분+ (매장 이용/휴식): {metrics['dwell_15plus_pct']}%  ← MAC 로테이션으로 과소평가 가능",
        ]

    # ── 트렌드 방향 ────────────────────────────────────────────────────────
    if "recent_7day_trend" in metrics:
        trend_kr = {"increasing": "📈 상승 추세", "decreasing": "📉 하락 추세", "stable": "➡️ 보합"}.get(metrics["recent_7day_trend"], "")
        lines += ["", f"[최근 7일 추세] {trend_kr} (7일 평균 방문자: {metrics.get('recent_7day_avg_v', 0):.0f}명)"]
    if "vs_same_dow_last_week" in metrics:
        c = metrics["vs_same_dow_last_week"]
        lines.append(f"  - 전주 동요일({c['date']}) 대비: 유동인구 {c['fp_delta_pct']:+.1f}%, 방문자 {c['visitor_delta_pct']:+.1f}%")

    # ── 전체 기간 요약 ─────────────────────────────────────────────────────
    n_total = metrics.get("total_days_available", 0)
    if n_total:
        lines += [
            "", f"[전체 기간 평균 — {n_total}일]",
            f"- 평균 유동인구: {metrics.get('all_period_avg_fp', 0):.0f}명",
            f"- 평균 방문자: {metrics.get('all_period_avg_v', 0):.0f}명",
            f"- 평균 CVR: {metrics.get('all_period_avg_cvr', 0):.1f}%",
        ]

    daytype_summary = metrics.get("daytype_summary", {})
    if daytype_summary:
        lines += ["", "[요일 유형별 평균 (전체 기간)"]
        for dt, v in daytype_summary.items():
            lines.append(f"  - {dt}: 방문자 {v.get('visitor_count', 0):.0f}명, CVR {v.get('quality_cvr', v.get('conversion_rate', 0)):.1f}%")

    weather_summary = metrics.get("weather_summary", {})
    if weather_summary:
        lines += ["", "[날씨별 평균 (전체 기간)]"]
        for w, v in weather_summary.items():
            lines.append(f"  - {w}: 방문자 {v.get('visitor_count', 0):.0f}명, CVR {v.get('quality_cvr', v.get('conversion_rate', 0)):.1f}%")

    # ── 날짜별 히스토리 ────────────────────────────────────────────────────
    history_days = metrics.get("history_days", [])
    history_count = metrics.get("history_count", len(history_days))
    if history_days:
        lines += ["", f"[최근 {history_count}일 데이터 — 오늘과 비교]",
                  "날짜       | 요일유형  | 날씨    | 기온       | 유동인구 | 방문자 | CVR   | 체류",
                  "---------- | --------- | ------- | ---------- | -------- | ------ | ----- | ----"]
        for d in history_days:
            marker = " ← 오늘" if d["date"] == metrics.get("date") else ""
            lines.append(
                f"{d['date']} | {d.get('day_type',''):9s} | {d.get('weather',''):7s} | "
                f"{d.get('temp',''):10s} | {d['fp']:8,} | {d['visitors']:6,} | "
                f"{d['cvr']:5.1f}% | {d['dwell_min']:.1f}분{marker}"
            )
    if "period_avg_fp" in metrics:
        lines += ["", f"[{history_count}일 평균] 유동인구: {metrics['period_avg_fp']:.0f}명, "
                  f"방문자: {metrics['period_avg_v']:.0f}명, CVR: {metrics['period_avg_cvr']:.1f}%"]

    space_notes = st.session_state.get("current_space_notes", "")
    if space_notes:
        lines += ["", f"[매장 메모]: {space_notes}"]

    lines += [
        "",
        "위 데이터를 바탕으로 아래 정확히 3개 섹션으로 답하세요 (레이블 그대로 사용):",
        "",
        "PERFORMANCE:",
        "오늘 성과를 히스토리 테이블의 같은 요일 유형, 유사 날씨 날짜들과 비교하여 설명하세요. "
        "한국 사회 캘린더(공휴일, 날씨, 요일 특성)를 연결해 왜 높거나 낮은지 원인을 분석하세요. "
        "구체적인 날짜와 수치를 인용. 3~4문장.",
        "",
        "BEHAVIOR:",
        "시간대별 유동인구 패턴을 분석하세요 (출근/점심/퇴근/야간 중 어느 시간대가 강한지). "
        "체류시간 분포가 오늘 방문자의 구매 행동(즉시구매 vs 탐색 vs 장시간 이용)을 어떻게 설명하는지 서술. "
        "3~4문장.",
        "",
        "RECOMMENDATIONS:",
        "실제 실행 가능한 3가지 권고사항을 제시하세요:",
        "① 프로모션/광고 타이밍 — 언제, 어떤 채널, 왜 (데이터 근거 포함)",
        "② 매장 운영 최적화 — 인력 배치 또는 진열 변경 (구체적 시간대/요일 명시)",
        "③ 다음 동향 예측 — 가까운 공휴일/날씨/트렌드 기반으로 다음 주 대비사항",
        "",
        "총 350자 이내 (한국어). 반드시 숫자와 날짜를 포함할 것.",
    ]

    return "\n".join(lines)


def _build_comparison_ai_prompt(metrics: dict) -> str:
    """Build user prompt for comparison AI analysis — Korean, structured, calendar-aware."""
    space_name = metrics.get("space_name", "Unknown")
    n_days = metrics.get("n_days", 0)
    lines = [
        _get_sector_context(space_name),
        _get_sector_behavior_context(space_name),
        "",
        f"[분석 기간: {metrics.get('date_range', 'N/A')} ({n_days}일)]",
        "",
        "[기간 평균]",
        f"- 일 평균 유동인구: {metrics.get('avg_fp', 0):.0f}명",
        f"- 일 평균 방문자: {metrics.get('avg_visitors', 0):.0f}명",
        f"- 평균 CVR: {metrics.get('avg_cvr', 0):.1f}%",
        f"- 평균 체류시간: {metrics.get('avg_dwell_sec', 0) / 60:.1f}분",
        f"- 기간 합계 유동인구: {metrics.get('total_fp', 0):,}명",
        f"- 기간 합계 방문자: {metrics.get('total_visitors', 0):,}명",
    ]

    # 트렌드 방향
    if "trend_direction" in metrics:
        direction_kr = {"increasing": "상승", "decreasing": "하락", "stable": "보합"}.get(
            metrics["trend_direction"], metrics["trend_direction"]
        )
        lines.extend([
            "",
            "[트렌드 방향 (전반부 vs 후반부)]",
            f"- 추세: {direction_kr} ({metrics.get('trend_delta_pct', 0):+.1f}%)",
            f"- 전반부 일평균 방문자: {metrics.get('first_half_avg_v', 0):.0f}명",
            f"- 후반부 일평균 방문자: {metrics.get('second_half_avg_v', 0):.0f}명",
        ])

    # Best/Worst day
    best = metrics.get("best_day", {})
    worst = metrics.get("worst_day", {})
    if best or worst:
        lines.append("")
        lines.append("[최고/최저 날]")
        if best:
            lines.append(f"- 최고: {best.get('date')} — 방문자 {best.get('visitors')}명, CVR {best.get('cvr')}%")
        if worst:
            lines.append(f"- 최저: {worst.get('date')} — 방문자 {worst.get('visitors')}명, CVR {worst.get('cvr')}%")

    # 요일 유형별 패턴
    if "day_type_summary" in metrics:
        lines.append("")
        lines.append("[요일 유형별 평균]")
        for dt, v in metrics["day_type_summary"].items():
            lines.append(
                f"  - {dt}: 유동인구 {v.get('avg_fp', 0):.0f}명, "
                f"방문자 {v.get('avg_v', 0):.0f}명, CVR {v.get('avg_cvr', 0):.1f}% "
                f"({int(v.get('days', 0))}일)"
            )

    # 날씨 패턴
    if "weather_summary" in metrics:
        lines.append("")
        lines.append("[날씨별 평균]")
        for w, v in metrics["weather_summary"].items():
            lines.append(
                f"  - {w}: 방문자 {v.get('avg_v', 0):.0f}명, CVR {v.get('avg_cvr', 0):.1f}% "
                f"({int(v.get('days', 0))}일)"
            )
    if "temp_range" in metrics:
        lines.append(f"  - 기온 범위: {metrics['temp_range']}, 평균 {metrics.get('avg_temp', '')}")
    if "rainy_days" in metrics:
        lines.append(f"  - 우설일 수: {metrics['rainy_days']}/{n_days}일, 합계 강수 {metrics.get('total_precipitation', 0)}mm")

    # 기간 내 공휴일
    holidays = metrics.get("holidays_in_period", [])
    if holidays:
        lines.append("")
        lines.append("[기간 내 공휴일]")
        for h in holidays:
            lines.append(f"  - {h['date']}: {h['name']}")

    # 날짜별 상세 테이블
    daily = metrics.get("daily_table", [])
    if daily:
        lines.append("")
        lines.append("[일별 상세 데이터]")
        header = "날짜 | 유동인구 | 방문자 | CVR | 체류(분)"
        if daily and "day_type" in daily[0]:
            header += " | 요일유형"
        if daily and "weather" in daily[0]:
            header += " | 날씨"
        lines.append(header)
        for row in daily:
            r_line = (
                f"{row['date']} | {row['fp']:,} | {row['visitors']:,} | "
                f"{row['cvr']:.1f}% | {row['dwell_min']:.1f}"
            )
            if "day_type" in row:
                r_line += f" | {row['day_type']}"
            if "weather" in row:
                r_line += f" | {row['weather']}"
            if "temp" in row:
                r_line += f" {row['temp']}"
            lines.append(r_line)

    lines.extend([
        "",
        "─" * 50,
        "아래 3개 섹션을 정확한 라벨 그대로 한국어로 작성하라.",
        "",
        "PERFORMANCE:",
        "기간 전체 성과 요약: 트렌드 방향(전반부→후반부 변화율), 최고/최저 날 원인 추론,"
        " 핵심 지표(FP·CVR·체류) 변화. 수치 포함. 2~3문장.",
        "",
        "PATTERNS:",
        "반복 패턴 분석: 평일 vs 주말 차이, 날씨·공휴일 영향,"
        " 특정 요일·기간에 성과가 집중되는 이유(한국 사회 맥락 반영). 수치 포함. 2~3문장.",
        "",
        "STRATEGY:",
        "운영자가 다음 기간에 실행할 수 있는 구체적 제안 3가지:"
        " ① 프로모션 타이밍(어떤 요일/날씨에 집중할지),"
        " ② FP 대비 CVR 효율 평가(개선 여지),"
        " ③ 재고·인력 배치 최적화. 각 제안은 데이터 근거 포함.",
        "",
        "전체 350자 이내. 한국어로.",
    ])

    return "\n".join(lines)
