"""
Korean social calendar context for retail AI analysis.

Provides holiday info, long weekends, school term/vacation periods,
and retail-relevant calendar events for a given date.
"""
from datetime import date, timedelta
from typing import Optional

# ── 고정 공휴일 (월-일 기준) ──────────────────────────────────────────────
_FIXED_HOLIDAYS: dict[str, str] = {
    "01-01": "신정(New Year's Day)",
    "03-01": "삼일절(Independence Movement Day)",
    "05-05": "어린이날(Children's Day)",
    "06-06": "현충일(Memorial Day)",
    "08-15": "광복절(Liberation Day)",
    "10-03": "개천절(National Foundation Day)",
    "10-09": "한글날(Hangul Proclamation Day)",
    "12-25": "크리스마스(Christmas)",
}

# ── 음력 기반 공휴일 (연도별 날짜 명시) ──────────────────────────────────
_LUNAR_HOLIDAYS: dict[str, str] = {
    # 2025 설날 연휴 (1/28~30)
    "2025-01-28": "설날 연휴",
    "2025-01-29": "설날",
    "2025-01-30": "설날 연휴",
    # 2025 부처님오신날
    "2025-05-05": "부처님오신날+어린이날 대체공휴일",
    # 2025 추석 연휴 (10/5~7, 대체 10/8)
    "2025-10-05": "추석 연휴",
    "2025-10-06": "추석",
    "2025-10-07": "추석 연휴",
    "2025-10-08": "추석 대체공휴일",
    # 2026 설날 연휴 (1/28~30)
    "2026-01-27": "설날 연휴(임시공휴일 예정)",
    "2026-01-28": "설날 연휴",
    "2026-01-29": "설날",
    "2026-01-30": "설날 연휴",
    # 2026 부처님오신날
    "2026-05-14": "부처님오신날",
    # 2026 추석 연휴 (9/24~26)
    "2026-09-24": "추석 연휴",
    "2026-09-25": "추석",
    "2026-09-26": "추석 연휴",
}

# ── 학기/방학 구분 (한국 기준) ────────────────────────────────────────────
# (월, 일) 범위로 정의
_SCHOOL_TERMS = [
    # 1학기: 3/2 ~ 6/30
    ((3, 2),  (6, 30),  "school_term_1",    "봄학기(Spring Semester)"),
    # 여름방학: 7/1 ~ 8/20
    ((7, 1),  (8, 20),  "summer_vacation",  "여름방학(Summer Vacation)"),
    # 2학기: 8/21 ~ 12/20
    ((8, 21), (12, 20), "school_term_2",    "가을학기(Fall Semester)"),
    # 겨울방학: 12/21 ~ 2/28(29)
    ((12, 21),(12, 31), "winter_vacation",  "겨울방학(Winter Vacation)"),
    ((1, 1),  (2, 28),  "winter_vacation",  "겨울방학(Winter Vacation)"),
]


def _is_fixed_holiday(d: date) -> Optional[str]:
    key = d.strftime("%m-%d")
    return _FIXED_HOLIDAYS.get(key)


def _is_lunar_holiday(d: date) -> Optional[str]:
    return _LUNAR_HOLIDAYS.get(d.isoformat())


def _is_holiday(d: date) -> Optional[str]:
    """Return holiday name if the date is a Korean public holiday, else None."""
    return _is_lunar_holiday(d) or _is_fixed_holiday(d)


def _get_school_term(d: date) -> str:
    md = (d.month, d.day)
    for (m1, d1), (m2, d2), key, label in _SCHOOL_TERMS:
        if (m1, d1) <= md <= (m2, d2):
            return label
    return "겨울방학(Winter Vacation)"  # fallback for edge dates


def _is_long_weekend(d: date) -> bool:
    """True if the date is part of a 3+ day consecutive holiday block."""
    # Check if any of [d-1, d, d+1] creates a 3+ block of holidays or weekends
    def is_off(x: date) -> bool:
        return x.weekday() >= 5 or _is_holiday(x) is not None

    # Look for a 3-day window containing d
    for offset in range(-2, 1):
        block = [d + timedelta(days=i) for i in range(offset, offset + 3)]
        if all(is_off(b) for b in block) and d in block:
            return True
    return False


def _days_to_next_holiday(d: date) -> Optional[tuple[int, str]]:
    """Return (days_away, holiday_name) for the nearest upcoming holiday within 7 days."""
    for offset in range(1, 8):
        candidate = d + timedelta(days=offset)
        name = _is_holiday(candidate)
        if name:
            return offset, name
    return None


def _days_since_last_holiday(d: date) -> Optional[tuple[int, str]]:
    """Return (days_ago, holiday_name) for the most recent holiday within 7 days."""
    for offset in range(1, 8):
        candidate = d - timedelta(days=offset)
        name = _is_holiday(candidate)
        if name:
            return offset, name
    return None


def get_korean_calendar_context(date_str: str) -> dict:
    """
    Return a dict of calendar context for the given date string (YYYY-MM-DD).

    Keys:
      is_holiday (bool), holiday_name (str|None)
      is_long_weekend (bool)
      school_term (str)
      days_to_next_holiday (int|None), next_holiday_name (str|None)
      days_since_last_holiday (int|None), last_holiday_name (str|None)
      retail_calendar_note (str) — human-readable summary for AI prompt
    """
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        return {}

    holiday_name = _is_holiday(d)
    is_holiday = holiday_name is not None
    long_weekend = _is_long_weekend(d)
    school_term = _get_school_term(d)

    next_hol = _days_to_next_holiday(d)
    last_hol = _days_since_last_holiday(d)

    notes = []
    if is_holiday:
        notes.append(f"공휴일: {holiday_name}")
    if long_weekend and not is_holiday:
        notes.append("연휴 기간 (장기 연휴)")
    if next_hol:
        days, name = next_hol
        notes.append(f"{days}일 후 공휴일: {name}")
    if last_hol:
        days, name = last_hol
        notes.append(f"{days}일 전 공휴일: {name} (연휴 후 복귀 효과 가능)")

    # Retail implications for Korean CVS
    retail_note_parts = []
    if is_holiday:
        retail_note_parts.append(
            "공휴일: 강남 오피스 공실 → 직장인 수요 급감, 거주자/관광객 위주로 전환"
        )
    if long_weekend and not is_holiday:
        retail_note_parts.append(
            "연휴 기간: 직장인 이탈 → 유동인구 감소 예상, 전날 '귀성 전 구매' 수요 주목"
        )
    if next_hol and next_hol[0] == 1:
        retail_note_parts.append(
            f"내일 공휴일({next_hol[1]}): 오늘 저녁 사전 구매 수요 상승 가능"
        )
    if last_hol and last_hol[0] == 1:
        retail_note_parts.append(
            f"어제 공휴일({last_hol[1]}) 다음날: 직장인 복귀 → 오전 피크 급증 예상"
        )
    if "winter_vacation" in school_term or "summer_vacation" in school_term:
        retail_note_parts.append(
            "방학 기간: 학생 유동인구 증가, 낮 시간대 트래픽 분포 변화"
        )

    retail_note = "; ".join(retail_note_parts) if retail_note_parts else "일반 평일/주말 패턴"

    return {
        "is_holiday": is_holiday,
        "holiday_name": holiday_name,
        "is_long_weekend": long_weekend,
        "school_term": school_term,
        "days_to_next_holiday": next_hol[0] if next_hol else None,
        "next_holiday_name": next_hol[1] if next_hol else None,
        "days_since_last_holiday": last_hol[0] if last_hol else None,
        "last_holiday_name": last_hol[1] if last_hol else None,
        "retail_calendar_note": retail_note,
        "calendar_notes": notes,
    }
