"""
Hermes Report View — Weekly traffic report generation + PDF download.

Dark theme v2 — charts use make_plotly_layout for dark preview.
PDF generation still uses chart_theme.apply_theme for printable output.
"""
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.analytics.day_type import add_day_type_to_daily_stats, get_day_context
from src.analytics.uplift import compute_week_over_week
from src.analytics.weekly_report import predict_next_week
from src.data.external_api import fetch_weather_forecast
from src.ui.chart_theme import apply_theme
from src.ui.helpers import make_plotly_layout, render_section_header, BG_COLOR
from src.ai import (
    generate_weekly_report_insight,
    generate_prediction_comment,
    generate_kpi_summary,
    generate_context_comment,
)

# Palette (dark theme)
DEEP_NAVY = "#0f172a"
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"
FP_COLOR = "#4A90D9"
VISITOR_COLOR = "#64ffda"

_WEATHER_ICON = {
    "Sunny": "sun Sunny",
    "Rain": "rain Rain",
    "Snow": "snow Snow",
    "Unknown": "-- Unknown",
}


def _apply_dark_theme(fig) -> None:
    """Apply dark theme to Plotly figure for dashboard preview."""
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#ccd6f6", size=11),
        xaxis=dict(gridcolor="#1a2035", zerolinecolor="#1a2035"),
        yaxis=dict(gridcolor="#1a2035", zerolinecolor="#1a2035"),
    )


# ── Cached wrappers ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_day_type(daily_stats: pd.DataFrame) -> pd.DataFrame:
    return add_day_type_to_daily_stats(daily_stats)


@st.cache_data(show_spinner=False)
def _cached_wow(daily_stats: pd.DataFrame, days_per_week: int = 7) -> dict:
    return compute_week_over_week(daily_stats, days_per_week=days_per_week)


def _ensure_day_type(loader: CacheLoader) -> pd.DataFrame:
    daily_stats = loader.get_daily_stats()
    if daily_stats.empty:
        return daily_stats
    return _cached_day_type(daily_stats)


# ── Main render function ─────────────────────────────────────────────────────

def render_report(space_name: str, loader: CacheLoader) -> None:
    """Render the Report view: config panel, generate button, preview, PDF download."""
    st.subheader("Report")
    st.markdown("Generate a weekly traffic report and download as PDF.")

    daily_stats = _ensure_day_type(loader)
    if daily_stats.empty:
        st.warning("No daily data in cache. Run preprocessing first.")
        return

    period, include = _render_report_config_panel(daily_stats)

    if st.button("Generate Report", type="primary", key="report_generate_btn"):
        if not period:
            st.warning("Select report period.")
        else:
            _generate_report(space_name, daily_stats, period, include, loader)

    # Show preview if report is ready
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
            label="Download PDF",
            data=pdf_data,
            file_name=fname,
            mime="application/pdf",
            type="primary",
            key="report_pdf_dl",
        )


# ── Config Panel ─────────────────────────────────────────────────────────────

def _render_report_config_panel(daily_stats: pd.DataFrame):
    """Render report period and options. Returns (period_tuple, include_options_dict)."""
    st.markdown("##### Weekly Traffic Report")

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
                if hasattr(x, "date"):
                    return x.date()
                return pd.Timestamp(str(x)).date()

            start = st.date_input(
                "Start date",
                value=_to_date(dr[-7]) if len(dr) >= 7 else _to_date(dr[0]),
                key="report_start",
            )
            end = st.date_input(
                "End date",
                value=_to_date(dr[-1]),
                key="report_end",
            )
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

    st.markdown("**Language**")
    report_lang = st.selectbox("Report language", ["English", "한국어"], key="report_lang", label_visibility="collapsed")
    return period, include


# ── Report Generation ────────────────────────────────────────────────────────

def _generate_report(
    space_name: str,
    daily_stats: pd.DataFrame,
    period: tuple,
    include: dict,
    loader: CacheLoader,
) -> None:
    """Generate report data and store in session state."""
    with st.spinner("Generating report..."):
        space_notes = st.session_state.get("current_space_notes", "")
        report_lang = st.session_state.get("report_lang", "English")
        report_data = _prepare_report_data(daily_stats, period, space_notes)

        if not report_data:
            st.warning("Cannot build data for selected period.")
            return

        # Generate AI content
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
                    this_week_for_ai, prev_week_for_ai, ctx, space_notes=space_notes, lang=report_lang
                )
            except Exception:
                report_data["kpi_summary"] = ""
            try:
                report_data["context_comment"] = generate_context_comment(
                    report_data.get("week_stats", {}),
                    report_data.get("holiday_info", {}),
                    ctx.get("season", ""),
                    space_notes=space_notes,
                    lang=report_lang,
                )
            except Exception:
                report_data["context_comment"] = ""

        # Weekly insight
        keep_cols = [
            c for c in ["date", "floating_unique", "quality_visitor_count", "quality_cvr", "dwell_median_seconds", "weather", "day_type"]
            if c in report_data["daily"].columns
        ]
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
                space_notes=space_notes, lang=report_lang,
            )

        if report_data.get("predictions") and include.get("ai"):
            try:
                report_data["prediction_comment"] = generate_prediction_comment(
                    report_data["predictions"], space_notes=space_notes, lang=report_lang
                )
            except Exception:
                pass

        # Build charts
        chart_figs = _build_report_charts(report_data)

        # Generate PDF
        try:
            from src.report import generate_weekly_report_pdf
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

        # Store in session state
        st.session_state["report_ready"] = True
        st.session_state["report_pdf"] = pdf_bytes
        st.session_state["report_data"] = report_data
        st.session_state["report_ai"] = ai_insight
        st.session_state["report_charts"] = chart_figs
        st.session_state["report_period"] = period
        st.success("Report ready. Check preview below and download PDF.")


def _prepare_report_data(daily_stats: pd.DataFrame, period: tuple, space_notes: str) -> dict:
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
    fp_total = df["floating_unique"].sum() or 1

    # 5분류 체류시간 집계
    d_1_3 = df["dwell_1_3min_count"].sum() if "dwell_1_3min_count" in df.columns else 0
    d_3_6 = df["dwell_3_6min_count"].sum() if "dwell_3_6min_count" in df.columns else 0
    d_6_10 = df["dwell_6_10min_count"].sum() if "dwell_6_10min_count" in df.columns else 0
    d_10_15 = df["dwell_10_15min_count"].sum() if "dwell_10_15min_count" in df.columns else 0
    d_15plus = df["dwell_15plus_count"].sum() if "dwell_15plus_count" in df.columns else 0

    # 3분류 하위호환 (deprecated columns)
    short = df["short_dwell_count"].sum() if "short_dwell_count" in df.columns else d_1_3
    medium = df["medium_dwell_count"].sum() if "medium_dwell_count" in df.columns else (d_3_6 + d_6_10)
    long = df["long_dwell_count"].sum() if "long_dwell_count" in df.columns else (d_10_15 + d_15plus)

    # Quality = 3분 이상 체류
    quality_v = d_3_6 + d_6_10 + d_10_15 + d_15plus
    if quality_v == 0:
        quality_v = df[qv].sum() if qv in df.columns else 0

    funnel = {
        # 5분류 (신규)
        "1_3min_pct": d_1_3 / total_v * 100 if total_v else 0,
        "3_6min_pct": d_3_6 / total_v * 100 if total_v else 0,
        "6_10min_pct": d_6_10 / total_v * 100 if total_v else 0,
        "10_15min_pct": d_10_15 / total_v * 100 if total_v else 0,
        "15plus_pct": d_15plus / total_v * 100 if total_v else 0,
        # 3분류 (하위호환)
        "short_pct": short / total_v * 100 if total_v else 0,
        "medium_pct": medium / total_v * 100 if total_v else 0,
        "long_pct": long / total_v * 100 if total_v else 0,
        # 공통 지표
        "quality_visitor_ratio": quality_v / total_v * 100 if total_v else 0,
        "long_ratio": long / total_v * 100 if total_v else 0,
        "quality_cvr": quality_v / fp_total * 100 if fp_total else 0,
    }

    first_row = df.iloc[0]
    daily_weather = [
        {
            "date": str(r["date"]),
            "weather": r.get("weather", "Unknown"),
            "temp_max": r.get("temp_max"),
            "temp_min": r.get("temp_min"),
        }
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

    return {
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


# ── Chart Building ───────────────────────────────────────────────────────────

def _build_dwell_funnel_chart(report_data: dict):
    """Donut chart: 5-tier dwell distribution for the week.

    Colors:
      - 1~3min: #64748b (slate gray)
      - 3~6min: #4A90D9 (blue)
      - 6~10min: #64ffda (teal)
      - 10~15min: #c49a3a (gold)
      - 15+min: #d97706 (amber)
    """
    funnel = report_data.get("funnel", {})
    d_1_3 = funnel.get("1_3min_pct", 0) or 0
    d_3_6 = funnel.get("3_6min_pct", 0) or 0
    d_6_10 = funnel.get("6_10min_pct", 0) or 0
    d_10_15 = funnel.get("10_15min_pct", 0) or 0
    d_15plus = funnel.get("15plus_pct", 0) or 0

    if d_1_3 == 0 and d_3_6 == 0 and d_6_10 == 0 and d_10_15 == 0 and d_15plus == 0:
        return None

    # Quality = 3분 이상 체류
    quality_pct = d_3_6 + d_6_10 + d_10_15 + d_15plus

    fig = go.Figure(go.Pie(
        labels=["1-3min", "3-6min", "6-10min", "10-15min", "15+min"],
        values=[d_1_3, d_3_6, d_6_10, d_10_15, d_15plus],
        hole=0.55,
        marker_colors=["#64748b", "#4A90D9", "#64ffda", "#c49a3a", "#d97706"],
        textinfo="label+percent",
        textfont_size=9,
    ))
    fig.add_annotation(
        text=f"Quality<br>{quality_pct:.1f}%",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#ccd6f6", family="Helvetica"),
    )
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        paper_bgcolor=BG_COLOR,
        font=dict(color="#ccd6f6"),
    )
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
    colors = [GOLD if (hasattr(d, "weekday") and d.weekday() >= 5) else FP_COLOR for d in dates]

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
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        showlegend=False,
        yaxis_title="Floating Pop",
        font=dict(size=10, color="#ccd6f6"),
        xaxis=dict(tickangle=0, gridcolor="#1a2035"),
        yaxis=dict(gridcolor="#1a2035"),
    )
    return fig


def _build_report_charts(report_data: dict) -> dict:
    """Build Plotly figures for traffic, funnel (stacked bar + donut), and prediction."""
    figs = {}
    df = report_data.get("daily", pd.DataFrame())
    qc = "quality_cvr" if "quality_cvr" in df.columns else "conversion_rate"
    anomaly = set(report_data.get("anomaly_dates", []))

    if not df.empty:
        fig_t = go.Figure()
        colors = [AMBER if str(d) in anomaly else FP_COLOR for d in df["date"]]
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
        # apply_theme already applies dark palette
        figs["traffic"] = fig_t

    # 5분류 스택 바 차트 (우선) or 3분류 하위호환
    has_5tier = all(c in df.columns for c in ["dwell_1_3min_count", "dwell_3_6min_count", "dwell_6_10min_count", "dwell_10_15min_count", "dwell_15plus_count"])
    has_3tier = all(c in df.columns for c in ["short_dwell_count", "medium_dwell_count", "long_dwell_count"])

    if not df.empty and (has_5tier or has_3tier) and qc in df.columns:
        fig_f = go.Figure()

        if has_5tier:
            # 5분류 스택 바 차트
            fig_f.add_trace(go.Bar(x=df["date"], y=df["dwell_1_3min_count"], name="1-3min", marker_color="#64748b"))
            fig_f.add_trace(go.Bar(x=df["date"], y=df["dwell_3_6min_count"], name="3-6min", marker_color="#4A90D9"))
            fig_f.add_trace(go.Bar(x=df["date"], y=df["dwell_6_10min_count"], name="6-10min", marker_color="#64ffda"))
            fig_f.add_trace(go.Bar(x=df["date"], y=df["dwell_10_15min_count"], name="10-15min", marker_color="#c49a3a"))
            fig_f.add_trace(go.Bar(x=df["date"], y=df["dwell_15plus_count"], name="15+min", marker_color="#d97706"))
        else:
            # 3분류 하위호환
            fig_f.add_trace(go.Bar(x=df["date"], y=df["short_dwell_count"], name="Short (<3min)", marker_color=SLATE_GRAY))
            fig_f.add_trace(go.Bar(x=df["date"], y=df["medium_dwell_count"], name="Medium (3-10min)", marker_color=FP_COLOR))
            fig_f.add_trace(go.Bar(x=df["date"], y=df["long_dwell_count"], name="Long (10min+)", marker_color=VISITOR_COLOR))

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
        # apply_theme already applies dark palette
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


# ── Report Preview ───────────────────────────────────────────────────────────

def _render_report_preview(report_data: dict, ai_insight: str, chart_figs: dict) -> None:
    """Render report preview with styled sections matching PDF layout."""
    st.markdown("---")
    st.markdown("#### Report Preview")

    kpi = report_data.get("kpi", {})
    tw = kpi.get("this_week", {})
    d = kpi.get("delta", {})

    # Section 1: Key Metrics
    st.markdown(
        '<div class="report-section-title">1. Key Metrics - This Week</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Floating Pop", f"{tw.get('floating_unique', 0):,.0f}",
        f"{d.get('floating_pct', 0):+.1f}% vs prev week" if d else "",
    )
    c2.metric(
        "Quality Visitors", f"{tw.get('quality_visitor_count', 0):,.0f}",
        f"{d.get('quality_visitor_pct', 0):+.1f}% vs prev week" if d else "",
    )
    c3.metric(
        "Quality CVR", f"{tw.get('quality_cvr', 0):.1f}%",
        f"{d.get('quality_cvr_pp', 0):+.1f}%p vs prev week" if d else "",
    )
    dm = tw.get("dwell_median_seconds", 0)
    c4.metric(
        "Median Dwell", f"{int(dm)//60}m {int(dm)%60}s",
        f"{d.get('dwell_median', 0):+.0f}s vs prev week" if d else "",
    )

    kpi_summary = report_data.get("kpi_summary", "")
    if kpi_summary:
        st.markdown(
            f'<div class="report-ai-box">{kpi_summary}</div>',
            unsafe_allow_html=True,
        )

    # Section 2: Weekly Traffic
    st.markdown(
        '<div class="report-section-title">2. Weekly Traffic Flow</div>',
        unsafe_allow_html=True,
    )
    if "traffic" in chart_figs:
        st.plotly_chart(chart_figs["traffic"], use_container_width=True)

    # Section 3: Dwell Funnel (5-tier)
    st.markdown(
        '<div class="report-section-title">3. Dwell Funnel (5-tier)</div>',
        unsafe_allow_html=True,
    )
    funnel = report_data.get("funnel", {})
    # 5분류 메트릭 표시 (2행 3열)
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("1-3min", f"{funnel.get('1_3min_pct', 0):.1f}%")
    fc2.metric("3-6min", f"{funnel.get('3_6min_pct', 0):.1f}%")
    fc3.metric("6-10min", f"{funnel.get('6_10min_pct', 0):.1f}%")
    fc4, fc5, fc6 = st.columns(3)
    fc4.metric("10-15min", f"{funnel.get('10_15min_pct', 0):.1f}%")
    fc5.metric("15+min", f"{funnel.get('15plus_pct', 0):.1f}%")
    fc6.metric("Quality CVR", f"{funnel.get('quality_cvr', 0):.1f}%")
    if "funnel" in chart_figs:
        st.plotly_chart(chart_figs["funnel"], use_container_width=True)
    if "dwell_funnel" in chart_figs:
        st.plotly_chart(chart_figs["dwell_funnel"], use_container_width=True)

    # Section 4: This Week Context
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
            rename_cols["temp_max"] = "High (C)"
        if "temp_min" in ctx_df.columns:
            rename_cols["temp_min"] = "Low (C)"
        st.dataframe(ctx_df.rename(columns=rename_cols), use_container_width=True, hide_index=True)

    st.caption(f"Season: {ctx.get('season', '-')} | Month: {ctx.get('month_label', '-')} | Holiday: {ctx.get('holiday_period', '-')}")

    context_comment = report_data.get("context_comment", "")
    if context_comment:
        st.markdown(
            f'<div class="report-ai-box">{context_comment}</div>',
            unsafe_allow_html=True,
        )

    if ctx.get("space_notes"):
        with st.expander("Space Notes", expanded=False):
            st.text(ctx["space_notes"])

    # Section 5: AI Insights
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

    # Section 6: Next Week Outlook
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
                "FP (predicted)": f"{p.get('floating_mean', 0):.0f} +/- {p.get('floating_std', 0):.0f}",
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
