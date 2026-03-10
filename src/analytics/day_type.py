"""
Day-type classification for Hermes Cause–Effect analysis.

Primary source: `holidays` library (KR) — covers all years automatically.
Fallback: hand-curated KR_HOLIDAYS_2026 set (used when the library is unavailable).

Public API:
  get_day_type(date_str)              → "weekday" | "weekend" | "holiday"
  get_day_context(date_str, ...)     → dict (holiday_period, season, month_label 등)
  add_day_type_to_daily_stats(df)    → df with 'day_type' column
  add_day_context_to_daily_stats(df) → df with 한국 달력 컨텍스트 컬럼
  weekday_names_en()                 → {0: "Mon", …}
  weekday_names_kr()                 → {0: "월", …}
"""
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Fallback holiday set (used only when the `holidays` package is unavailable)
# ---------------------------------------------------------------------------
KR_HOLIDAYS_2026: Set[str] = {
    "2026-01-01",                                                    # New Year's Day
    "2026-02-16", "2026-02-17", "2026-02-18",                       # Lunar New Year
    "2026-03-01",                                                    # Independence Movement Day
    "2026-04-05",                                                    # Arbor Day
    "2026-05-05",                                                    # Children's Day
    "2026-06-06",                                                    # Memorial Day
    "2026-08-15",                                                    # Liberation Day
    "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-27",         # Chuseok
    "2026-10-03",                                                    # National Foundation Day
    "2026-10-09",                                                    # Hangeul Day
    "2026-12-25",                                                    # Christmas
}


def _build_kr_holidays(year: int):
    """
    Return a dict-like object of KR public holidays for *year*.
    Uses the `holidays` library when available; otherwise falls back to
    KR_HOLIDAYS_2026 (only accurate for 2026).
    """
    try:
        import holidays as hol_lib
        return hol_lib.country_holidays("KR", years=year)
    except ImportError:
        # Convert the fallback strings to date objects for consistent key type
        from datetime import date
        result = {}
        for s in KR_HOLIDAYS_2026:
            try:
                d = date.fromisoformat(s)
                if d.year == year:
                    result[d] = "Holiday"
            except ValueError:
                pass
        return result


def get_day_type(date_str: str, holidays: Optional[Set[str]] = None) -> str:
    """
    Classify a date string ("YYYY-MM-DD") as "weekday" | "weekend" | "holiday".

    When `holidays` argument is None (default), the `holidays` library is used
    for automatic detection; the hand-curated set is the fallback.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Allow callers to pass an explicit set (backward compatibility)
    if holidays is not None:
        if date_str in holidays:
            return "holiday"
        return "weekend" if dt.weekday() >= 5 else "weekday"

    # Auto-detect via library
    kr = _build_kr_holidays(dt.year)
    if dt.date() in kr:
        return "holiday"
    return "weekend" if dt.weekday() >= 5 else "weekday"


def add_day_type_to_daily_stats(daily_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'day_type' column to daily_stats if it is missing.

    Uses get_day_type() per row, which internally calls the `holidays` library.
    If 'day_type' already exists (e.g. populated by enrich_external in runner.py)
    the existing column is preserved unchanged.
    """
    df = daily_stats.copy()
    if "date" not in df.columns:
        return df
    if "day_type" not in df.columns:
        df["day_type"] = df["date"].astype(str).map(get_day_type)
    return df


# ---------------------------------------------------------------------------
# 한국 달력 컨텍스트 (연휴, 계절, 월별 특성)
# ---------------------------------------------------------------------------

_MONTH_LABELS: Dict[Tuple[int, bool], str] = {
    (1, False): "January (New Year)",
    (2, True):  "February (Lunar New Year)",
    (2, False): "February (Low season)",
    (3, False): "March (Spring start)",
    (4, False): "April (Spring peak)",
    (5, False): "May (Family month)",
    (6, False): "June (Summer start)",
    (7, False): "July (Summer peak)",
    (8, False): "August (Summer peak)",
    (9, True):  "September (Chuseok)",
    (9, False): "September (Autumn start)",
    (10, False): "October (Autumn peak)",
    (11, False): "November (Pre-winter)",
    (12, False): "December (Year-end peak)",
}


def _holiday_name_en(hol_name: str) -> str:
    """Map library holiday name to English for display."""
    s = (hol_name or "").strip()
    if not s:
        return ""
    h = s.lower()
    if "설" in s or "lunar" in h or "new year" in h:
        return "Lunar New Year"
    if "추석" in s or "chuseok" in h:
        return "Chuseok"
    if "christmas" in h or "크리스마스" in s:
        return "Christmas"
    if "independence" in h or "3·1" in s or "삼일" in s:
        return "Independence Movement Day"
    if "arbor" in h or "식목" in s:
        return "Arbor Day"
    if "children" in h or "어린이" in s:
        return "Children's Day"
    if "memorial" in h or "현충" in s:
        return "Memorial Day"
    if "liberation" in h or "광복" in s:
        return "Liberation Day"
    if "foundation" in h or "개천" in s:
        return "National Foundation Day"
    if "hangeul" in h or "한글" in s:
        return "Hangeul Day"
    return s


def _holiday_period_name(hol_name: str, _month: int) -> str:
    """Holiday name -> holiday period block label (English)."""
    s = hol_name or ""
    h = s.lower()
    if "설" in s or "lunar" in h or "new year" in h:
        return "Lunar New Year Holiday"
    if "추석" in s or "chuseok" in h:
        return "Chuseok Holiday"
    if "christmas" in h or "크리스마스" in s:
        return "Year-End Holiday"
    return s or "Holiday"


def _get_consecutive_holiday_blocks(year: int) -> List[Tuple[date, date, str]]:
    """
    해당 연도 공휴일을 연속 블록으로 묶음.
    Returns: [(start_date, end_date, period_name), ...]
    """
    kr = _build_kr_holidays(year)
    if not kr:
        return []
    hol_dates = sorted([d for d in kr.keys() if isinstance(d, date)])
    if not hol_dates:
        return []

    blocks: List[Tuple[date, date, str]] = []
    i = 0
    while i < len(hol_dates):
        start = hol_dates[i]
        first_name = str(kr.get(start, "Holiday"))
        name = _holiday_period_name(first_name, start.month)
        j = i
        while j + 1 < len(hol_dates) and (hol_dates[j + 1] - hol_dates[j]).days <= 2:
            j += 1
        end = hol_dates[j]
        blocks.append((start, end, name))
        i = j + 1
    return blocks


def get_day_context(
    date_str: str,
    holiday_info: Optional[Dict[date, str]] = None,
) -> Dict[str, Any]:
    """
    한국 달력 컨텍스트 레이어 생성.

    Parameters
    ----------
    date_str : "YYYY-MM-DD"
    holiday_info : date -> holiday_name (None이면 holidays 라이브러리 사용)

    Returns
    -------
    dict with: day_type, holiday_name, holiday_period, holiday_sequence,
               season, month_label, is_long_weekend
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    d = dt.date()
    month = d.month
    wd = d.weekday()  # 0=Mon ... 6=Sun

    # day_type, holiday_name (English for PDF/UI)
    day_type_val = get_day_type(date_str)
    kr = holiday_info if holiday_info is not None else _build_kr_holidays(d.year)
    hol_raw = str(kr.get(d, "")) if d in kr else ""
    hol_name = _holiday_name_en(hol_raw)

    # season (English)
    if month in (12, 1, 2):
        season = "Winter"
    elif month in (3, 4, 5):
        season = "Spring"
    elif month in (6, 7, 8):
        season = "Summer"
    else:
        season = "Autumn"

    # month_label (Lunar New Year Feb, Chuseok Sep)
    blocks = _get_consecutive_holiday_blocks(d.year)
    has_seollal = any(b[2] == "Lunar New Year Holiday" for b in blocks)
    has_chuseok = any(b[2] == "Chuseok Holiday" for b in blocks)
    month_label = _MONTH_LABELS.get(
        (month, (month == 2 and has_seollal) or (month == 9 and has_chuseok)),
        _MONTH_LABELS.get((month, False), f"Month {month}"),
    )

    # holiday_period, holiday_sequence, is_long_weekend
    holiday_period = ""
    holiday_sequence = ""
    is_long_weekend = False

    blocks = _get_consecutive_holiday_blocks(d.year)
    for start, end, period_name in blocks:
        if start <= d <= end:
            holiday_period = period_name
            seq = (d - start).days + 1
            holiday_sequence = f"Day {seq} of holiday"
            # 블록 내 주말 포함 여부
            cur = start
            while cur <= end:
                if cur.weekday() >= 5:  # Sat or Sun
                    is_long_weekend = True
                    break
                cur += timedelta(days=1)
            if not is_long_weekend:
                # 블록 직전 금요일 또는 직후 월요일
                if start.weekday() == 4 or end.weekday() == 0:
                    is_long_weekend = True
            break

    return {
        "day_type": day_type_val,
        "holiday_name": hol_name,
        "holiday_period": holiday_period,
        "holiday_sequence": holiday_sequence,
        "season": season,
        "month_label": month_label,
        "is_long_weekend": is_long_weekend,
    }


def add_day_context_to_daily_stats(daily_stats: pd.DataFrame) -> pd.DataFrame:
    """
    daily_stats에 get_day_context() 결과 컬럼 추가.
    기존 day_type, holiday_name이 있으면 유지하고 나머지만 추가.
    """
    if daily_stats.empty or "date" not in daily_stats.columns:
        return daily_stats
    df = daily_stats.copy()
    years = pd.to_datetime(df["date"], errors="coerce").dt.year.dropna().unique().astype(int).tolist()
    kr_holidays = {}
    for y in years:
        kr = _build_kr_holidays(y)
        if kr:
            kr_holidays.update({k: str(v) for k, v in kr.items() if isinstance(k, date)})

    rows = []
    for _, row in df.iterrows():
        ctx = get_day_context(str(row["date"]), holiday_info=kr_holidays if kr_holidays else None)
        rows.append({**row.to_dict(), **ctx})
    out = pd.DataFrame(rows)
    return out


def weekday_names_kr() -> dict:
    return {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


def weekday_names_en() -> dict:
    return {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
