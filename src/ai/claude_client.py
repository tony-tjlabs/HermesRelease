"""
Hermes AI — Claude API client (stdlib only, no requests).

API key priority:
  1. st.secrets["ANTHROPIC_API_KEY"]  (Streamlit Cloud)
  2. os.environ["ANTHROPIC_API_KEY"]  (local .env or shell export)

Failure policy: never raises; returns a warning string prefixed with ⚠️.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Any, Dict, List, Optional

_API_URL    = "https://api.anthropic.com/v1/messages"
_MODEL      = "claude-haiku-4-5-20251001"   # 저비용 권장
_MAX_TOKENS = 1024
_REPORT_MAX_TOKENS = 1500
_SYNTHESIS_MAX_TOKENS = 400
_TIMEOUT    = 30

HERMES_CONTEXT = """
[Project Hermes 개요]
Hermes는 소형 리테일 매장의 BLE 신호 데이터를 분석하는 Spatial Cause-Effect Intelligence Platform이다.
S-Ward 센서가 수집한 스마트폰 BLE 신호로 유동인구·방문객·체류시간·전환율을 정밀 측정한다.

[핵심 철학: Cause → Effect]
- Cause (원인): 요일, 공휴일, 날씨, 이벤트, 시간대 특성
- Effect (결과): 방문 전환율(CVR), 방문객 수, 체류시간, 트래픽 패턴 변화
- Hermes는 이 원인-결과 관계를 데이터로 학습하는 시스템이다.

[지표 정의]
- FP (Floating Population): 입구 S-Ward에서 RSSI >= -80dBm으로 수신된 unique MAC 수. 매장 외부 유동인구.
- V (Verified Visitors): 내부 S-Ward에서 Strict Entry 조건(1분 내 2회 이상 + 모든 신호 RSSI >= -80dBm)을 통과한 방문 세션 수.
- CVR (Conversion Rate): Visitors / Floating Population × 100(%). 매장의 유인력 핵심 효율 지표.
- Dwell Time: 방문 세션의 입장~퇴장 시간(초). 퇴장 시각은 마지막 신호 수신 기준 Back-dating 적용.

[Strict Entry 방문객 판별 기준]
- 내부 센서 기준으로 1분 윈도우 내 수신 2회 이상 AND 모든 RSSI >= -80dBm 이어야만 방문자로 인정.
- 세션 시작 후에는 Apple 180초 / Android 120초 유예(Hysteresis) 적용.
- 이 기준은 단순 통행자를 걸러내는 고정밀(Luxury-level precision) 필터다.

[매장 유형 분류]
- weekday: 평일 (월~금, 공휴일 제외)
- weekend: 주말 (토, 일)
- holiday: 한국 공휴일

[분석의 목적]
단순 숫자 보고가 아니라, 원인(날씨/요일/공휴일)이 방문 전환율과 체류 패턴에 어떤 영향을 주는지
데이터 기반으로 해석하고, 매장 운영에 실질적으로 활용 가능한 인사이트를 도출하는 것이다.
""".strip()


def _get_api_key() -> str:
    """Return API key from st.secrets or environment (.env). Never from UI input (deployment-safe)."""
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def call_claude(
    prompt: str,
    system: str = "",
    space_notes: str = "",
    max_tokens: Optional[int] = None,
) -> str:
    """
    Send a prompt to Claude and return the text response.

    Parameters
    ----------
    prompt : str
        User message.
    system : str
        Optional system prompt for role/context setting.
    space_notes : str
        Optional operator notes for the space (from sidebar Space Notes).
        Injected into the system prompt so Claude can reference store-specific context.

    Returns
    -------
    str
        Claude's response text, or a ⚠️-prefixed error message on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "⚠️ ANTHROPIC_API_KEY is not set.  \n"
            "Set it in `.streamlit/secrets.toml` or in the ANTHROPIC_API_KEY environment variable to use AI features."
        )

    payload: dict = {
        "model":      _MODEL,
        "max_tokens": max_tokens if max_tokens is not None else _MAX_TOKENS,
        "messages":   [{"role": "user", "content": prompt}],
    }
    # English-only first so PDF/render never get non-ASCII.
    full_system = (
        "You MUST respond ONLY in English. Do not use Korean, Chinese, or other non-ASCII characters. "
        "All analysis, headings, and recommendations must be in clear English.\n\n"
    )
    full_system += HERMES_CONTEXT
    if space_notes and space_notes.strip():
        full_system += "\n\n[Operator notes — for context]\n" + space_notes.strip()
    if system:
        full_system = full_system + "\n\n" + system
    payload["system"] = full_system

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        _API_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_ssl_context()) as resp:
            if resp.status != 200:
                return f"⚠️ API error (HTTP {resp.status})"
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            msg    = detail.get("error", {}).get("message", str(exc))
        except Exception:
            msg = str(exc)
        return f"⚠️ API error: {msg}"
    except urllib.error.URLError as exc:
        return f"⚠️ Network error: {exc.reason}"
    except TimeoutError:
        return "⚠️ Request timed out (30s). Check network."
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error: {exc}"


def generate_kpi_summary(
    this_week: Dict[str, Any],
    prev_week: Dict[str, Any],
    context: Dict[str, Any],
    space_notes: str = "",
) -> str:
    """Summarize the 4 KPIs in 1-2 sentences for store owners. max_tokens=100."""
    system = (
        "You are a retail analyst. Respond in English only. "
        "Write exactly 2 sentences (max 50 words) summarizing the 4 KPIs "
        "in plain language for a store owner. Focus on the most important signal."
    )
    user = (
        f"This week: Floating={this_week.get('floating', 0):,.0f} ({this_week.get('fp_delta', 0):+.1f}%), "
        f"Quality Visitors={this_week.get('quality_visitor', 0):,.0f} ({this_week.get('qv_delta', 0):+.1f}%), "
        f"Quality CVR={this_week.get('quality_cvr', 0):.1f}% ({this_week.get('cvr_delta', 0):+.1f}%), "
        f"Median Dwell={this_week.get('dwell_median_str', 'N/A')} ({this_week.get('dwell_delta_str', '')}).\n"
        f"Context: {context.get('holiday_period', '')}, Season: {context.get('season', '')}.\n"
        "IMPORTANT: English only. 2 sentences max."
    )
    return call_claude(user, system=system, space_notes=space_notes, max_tokens=100)


def generate_context_comment(
    week_stats: Dict[str, Any],
    holiday_info: Dict[str, Any],
    season: str,
    space_notes: str = "",
) -> str:
    """Explain this week's business context (holiday/weather/season) in 2-3 sentences. max_tokens=120."""
    system = (
        "You are a Korean retail market analyst. Respond in English only. "
        "Write 2-3 sentences explaining the business context of this week "
        "for a sports/lifestyle store in a Korean shopping mall. "
        "Be specific about how the holiday/weather affects shopping behavior."
    )
    user = (
        f"Week: {week_stats.get('date_range', '')}\n"
        f"Holiday: {holiday_info.get('period', 'None')} "
        f"({holiday_info.get('days', 0)} holiday days)\n"
        f"Weather: {week_stats.get('dominant_weather', 'Sunny')}\n"
        f"Season: {season}\n\n"
        "Explain in 2-3 sentences how this context (holiday + weather + season) "
        "typically affects foot traffic and conversion for a sports retail store "
        "in a Korean shopping mall. Be practical and specific.\n"
        "IMPORTANT: English only."
    )
    return call_claude(user, system=system, space_notes=space_notes, max_tokens=120)


def generate_weekly_report_insight(
    weekly_stats: List[Dict[str, Any]],
    prev_week_stats: List[Dict[str, Any]],
    daily_weather: List[Dict[str, Any]],
    day_context_list: List[Dict[str, Any]],
    space_notes: str = "",
) -> str:
    """
    Weekly synthesis: DIAGNOSIS (2-3 sentences), PATTERN (2 bullets), ACTIONS (2 bullets).
    Structured for PDF rendering. max_tokens=400.
    """
    import json
    system = (
        "You are a senior retail traffic analyst providing a weekly synthesis report. "
        "You MUST respond ONLY in English. "
        "Structure your response in exactly 3 parts:\n"
        "DIAGNOSIS: 2-3 sentences - the single most important insight this week.\n"
        "PATTERN: 2 bullet points - key pattern and its root cause.\n"
        "ACTIONS: 2 bullet points - specific, actionable recommendations for next week.\n"
        "Total response: under 200 words. No markdown headers. Use the labels "
        "DIAGNOSIS:, PATTERN:, ACTIONS: exactly."
    )
    user = (
        f"[This Week Data]\n{json.dumps(weekly_stats, ensure_ascii=False, indent=2)}\n\n"
        f"[Previous Week Data]\n{json.dumps(prev_week_stats, ensure_ascii=False, indent=2)}\n\n"
        f"[Daily Weather]\n{json.dumps(daily_weather, ensure_ascii=False, indent=2)}\n\n"
        f"[Calendar Context]\n{json.dumps(day_context_list, ensure_ascii=False, indent=2)}\n\n"
        f"[Space Notes]\n{space_notes or 'None'}\n\n"
        "Provide a synthesis of this week's performance. "
        "Focus on the 'so what' - what does this mean for the business?\n"
        "IMPORTANT: English only. Follow the DIAGNOSIS/PATTERN/ACTIONS structure exactly."
    )
    return call_claude(user, system=system, space_notes=space_notes, max_tokens=_SYNTHESIS_MAX_TOKENS)


def generate_prediction_comment(
    predictions: List[Dict[str, Any]],
    space_notes: str = "",
) -> str:
    """
    Generate 1-2 sentence operational comment for next week from prediction data.
    E.g. "Weekend expected to outperform. Consider extra staffing on Sat/Sun."
    """
    if not predictions:
        return ""
    lines = []
    for p in predictions[:7]:
        d = p.get("date_obj") or p.get("date")
        if hasattr(d, "strftime"):
            d_str = d.strftime("%a %b %d")
        else:
            d_str = str(d)[:10]
        w = p.get("weather", "?")
        fp = p.get("floating_mean", 0)
        cvr = p.get("quality_cvr_mean", 0)
        lines.append(f"{d_str}: {w} FP~{fp:,.0f} CVR~{cvr:.1f}%")
    pred_summary = "\n".join(lines)
    system = (
        "You are a retail operations advisor. "
        "You MUST respond ONLY in English. "
        "Write exactly 2 concise sentences (max 60 words total) "
        "with a concrete operational recommendation for next week."
    )
    user = (
        f"Next week forecast:\n{pred_summary}\n\n"
        "Give 2 sentences: (1) what to expect, (2) one specific action to take.\n"
        "IMPORTANT: English only."
    )
    return call_claude(user, system=system, space_notes=space_notes, max_tokens=120)
