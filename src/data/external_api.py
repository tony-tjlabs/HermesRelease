"""
External data collection for Hermes pipeline.

Provides two capabilities:
  1. Korean public holiday detection via the `holidays` library (no API key needed).
  2. Historical daily weather data via the Open-Meteo Archive API (free, no API key needed).

All functions are safe to call even if the library or API is unavailable:
  - Missing `holidays` package → falls back to an empty set (no holidays flagged).
  - Network / API errors    → returns NaN / default values, logs the issue.

Default location: Suwon Starfield (lat=37.285, lon=126.990).
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Location constant (Suwon Starfield)
# ---------------------------------------------------------------------------
DEFAULT_LATITUDE: float = 37.285
DEFAULT_LONGITUDE: float = 126.990

# ---------------------------------------------------------------------------
# 1.  Korean public holiday detection
# ---------------------------------------------------------------------------

def _load_kr_holidays(years: List[int]):
    """
    Return a holidays.HolidayBase object for Korea covering the given years.
    Falls back to an empty dict if the `holidays` package is not installed.
    """
    try:
        import holidays as hol_lib
        kr = hol_lib.country_holidays("KR", years=years)
        return kr
    except ImportError:
        logger.warning(
            "`holidays` package not installed. "
            "Run `pip install holidays` to enable automatic holiday detection. "
            "Falling back to empty holiday set."
        )
        return {}
    except Exception as exc:
        logger.warning("Failed to load KR holidays: %s. Falling back to empty set.", exc)
        return {}


def enrich_holidays(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add holiday / weekday columns to a DataFrame that has a date column.

    Added columns
    -------------
    is_holiday : bool   – True if the date is a Korean public holiday.
    holiday_name: str   – Name of the holiday (empty string if not a holiday).
    weekday    : int    – ISO weekday number (0=Mon … 6=Sun).
    weekday_name: str   – Short English name (Mon, Tue, …).
    day_type   : str    – "holiday" | "weekend" | "weekday"

    Parameters
    ----------
    df       : DataFrame that contains *date_col*.
    date_col : Name of the date column (values must be parseable by pd.to_datetime).

    Returns
    -------
    A new DataFrame with the extra columns appended.
    """
    if df.empty or date_col not in df.columns:
        return df

    out = df.copy()
    dates_parsed: pd.Series = pd.to_datetime(out[date_col], errors="coerce")

    # Derive the years present in the dataset so the library covers them all.
    years: List[int] = sorted(dates_parsed.dt.year.dropna().unique().astype(int).tolist())
    kr_holidays = _load_kr_holidays(years)

    _wd_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    is_holiday_list: List[bool] = []
    holiday_name_list: List[str] = []
    weekday_list: List[int] = []
    weekday_name_list: List[str] = []
    day_type_list: List[str] = []

    for dt in dates_parsed:
        if pd.isna(dt):
            is_holiday_list.append(False)
            holiday_name_list.append("")
            weekday_list.append(-1)
            weekday_name_list.append("")
            day_type_list.append("unknown")
            continue

        d = dt.date()
        hol_name = kr_holidays.get(d, "")
        is_hol = bool(hol_name)
        wd = d.weekday()          # 0=Mon … 6=Sun

        is_holiday_list.append(is_hol)
        holiday_name_list.append(hol_name if isinstance(hol_name, str) else str(hol_name))
        weekday_list.append(wd)
        weekday_name_list.append(_wd_names.get(wd, ""))

        if is_hol:
            day_type_list.append("holiday")
        elif wd >= 5:
            day_type_list.append("weekend")
        else:
            day_type_list.append("weekday")

    out["is_holiday"] = is_holiday_list
    out["holiday_name"] = holiday_name_list
    out["weekday"] = weekday_list
    out["weekday_name"] = weekday_name_list
    out["day_type"] = day_type_list

    return out


# ---------------------------------------------------------------------------
# 2.  Open-Meteo Archive API — daily weather
# ---------------------------------------------------------------------------

# Open-Meteo endpoints (free, no API key)
_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def _weather_tag(precipitation: float, snowfall: float) -> str:
    """
    Derive a simple weather label from daily precipitation / snowfall totals.

      - snowfall > 0 mm         → "Snow"
      - precipitation > 0 mm    → "Rain"
      - otherwise               → "Sunny"
    """
    if pd.isna(precipitation) and pd.isna(snowfall):
        return "Unknown"
    if not pd.isna(snowfall) and snowfall > 0:
        return "Snow"
    if not pd.isna(precipitation) and precipitation > 0:
        return "Rain"
    return "Sunny"


def fetch_weather(
    start_date: str,
    end_date: str,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> pd.DataFrame:
    """
    Fetch daily weather from the Open-Meteo Archive API (stdlib only, no requests).

    Implementation notes
    --------------------
    - Uses urllib.request (stdlib) — no external dependencies.
    - timeout=10s on the HTTP call to avoid blocking the dashboard.
    - Returns an empty DataFrame (correct columns, no rows) on ANY failure:
        network error, HTTP error, malformed JSON, empty "daily" array, future date.
    - Callers should check `result.empty` before merging.

    Parameters
    ----------
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"  (must be today or earlier for archive API)
    latitude   : float  (default: Suwon Starfield 37.285)
    longitude  : float  (default: Suwon Starfield 126.990)

    Returns
    -------
    DataFrame columns:
        date (str "YYYY-MM-DD"), precipitation (mm), snowfall (cm),
        temp_max (°C), temp_min (°C), weather ("Sunny"|"Rain"|"Snow"|"Unknown")
    """
    import json
    import ssl
    import urllib.parse
    import urllib.request

    _empty = pd.DataFrame(columns=["date", "precipitation", "snowfall", "temp_max", "temp_min", "weather"])

    # Build SSL context — use certifi bundle if available (fixes macOS venv SSL errors),
    # otherwise fall back to the default context.
    def _ssl_context() -> ssl.SSLContext:
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Seoul",
        }
        url = _OPEN_METEO_ARCHIVE_URL + "?" + urllib.parse.urlencode(params, doseq=True)
        logger.info("Fetching weather: %s", url)

        with urllib.request.urlopen(url, timeout=10, context=_ssl_context()) as resp:
            if resp.status != 200:
                logger.warning("Weather API returned HTTP %s — skipping.", resp.status)
                return _empty
            data = json.loads(resp.read().decode("utf-8"))

        # API-level error (e.g. future date beyond archive range)
        if "error" in data:
            logger.warning("Weather API error: %s", data.get("reason", data["error"]))
            return _empty

        daily = data.get("daily", {})
        dates = daily.get("time", [])

        # Empty array = no data for the requested range (future date, etc.)
        if not dates:
            logger.warning(
                "Weather API returned 0 days for %s → %s. "
                "This can happen for future dates or unsupported ranges.",
                start_date, end_date,
            )
            return _empty

        precip = daily.get("precipitation_sum", [None] * len(dates))
        snow   = daily.get("snowfall_sum",      [None] * len(dates))
        tmax   = daily.get("temperature_2m_max",[None] * len(dates))
        tmin   = daily.get("temperature_2m_min",[None] * len(dates))

        def _safe_float(v) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        rows = []
        for d, p, s, tx, tn in zip(dates, precip, snow, tmax, tmin):
            p_val  = _safe_float(p)
            s_val  = _safe_float(s)
            tx_val = _safe_float(tx)
            tn_val = _safe_float(tn)
            rows.append({
                "date":          d,
                "precipitation": round(p_val,  1) if not pd.isna(p_val)  else float("nan"),
                "snowfall":      round(s_val,  1) if not pd.isna(s_val)  else float("nan"),
                "temp_max":      round(tx_val, 1) if not pd.isna(tx_val) else float("nan"),
                "temp_min":      round(tn_val, 1) if not pd.isna(tn_val) else float("nan"),
                "weather":       _weather_tag(p_val, s_val),
            })

        result = pd.DataFrame(rows)
        logger.info("Weather fetched: %d days (%s → %s)", len(result), start_date, end_date)
        return result

    except Exception as exc:
        logger.warning(
            "Weather fetch failed (%s → %s): %s — daily_stats will have NaN weather columns.",
            start_date, end_date, exc,
        )
        return _empty


def fetch_weather_forecast(
    days: int = 7,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo Forecast API (stdlib urllib).

    Parameters
    ----------
    days : int
        Number of days to forecast (default 7).
    latitude, longitude : float
        Location (default: Suwon Starfield).

    Returns
    -------
    DataFrame with columns: date, precipitation, snowfall, temp_max, temp_min, weather
    Empty DataFrame on failure.
    """
    import json
    import ssl
    import urllib.parse
    import urllib.request

    _empty = pd.DataFrame(columns=["date", "precipitation", "snowfall", "temp_max", "temp_min", "weather"])

    def _ssl_context() -> ssl.SSLContext:
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Seoul",
            "forecast_days": min(days, 16),
        }
        url = _OPEN_METEO_FORECAST_URL + "?" + urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes/1.0"})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
            if resp.status != 200:
                return _empty
            data = json.loads(resp.read().decode("utf-8"))
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return _empty
        precip = daily.get("precipitation_sum", [None] * len(dates))
        snow = daily.get("snowfall_sum", [None] * len(dates))
        tmax = daily.get("temperature_2m_max", [None] * len(dates))
        tmin = daily.get("temperature_2m_min", [None] * len(dates))
        valid_weather = {"Sunny", "Rain", "Snow", "Unknown"}
        rows = []
        for d, p, s, tx, tn in zip(dates, precip, snow, tmax, tmin):
            try:
                p_val = float(p) if p is not None else float("nan")
            except (TypeError, ValueError):
                p_val = float("nan")
            try:
                s_val = float(s) if s is not None else float("nan")
            except (TypeError, ValueError):
                s_val = float("nan")
            try:
                tx_val = float(tx) if tx is not None else float("nan")
            except (TypeError, ValueError):
                tx_val = float("nan")
            try:
                tn_val = float(tn) if tn is not None else float("nan")
            except (TypeError, ValueError):
                tn_val = float("nan")
            w = _weather_tag(p_val, s_val)
            if w not in valid_weather or not w or len(str(w)) > 20:
                w = "Unknown"
            date_str = str(d)[:10] if d else ""
            rows.append({
                "date": date_str,
                "precipitation": round(p_val, 1) if not pd.isna(p_val) else float("nan"),
                "snowfall": round(s_val, 1) if not pd.isna(s_val) else float("nan"),
                "temp_max": round(tx_val, 1) if not pd.isna(tx_val) else float("nan"),
                "temp_min": round(tn_val, 1) if not pd.isna(tn_val) else float("nan"),
                "weather": w,
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("Weather forecast fetch failed: %s", exc)
        return _empty


def enrich_weather(
    df: pd.DataFrame,
    date_col: str = "date",
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> pd.DataFrame:
    """
    Fetch weather for the date range present in *df* and left-join it.

    Added columns: precipitation, snowfall, temp_max, temp_min, weather.

    If the API call fails, those columns are filled with NaN / "Unknown".

    Parameters
    ----------
    df       : DataFrame containing *date_col*.
    date_col : Name of the date column.
    latitude / longitude : Location for the weather query.

    Returns
    -------
    A new DataFrame with weather columns merged in.
    """
    if df.empty or date_col not in df.columns:
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return df

    start_date = dates.min().strftime("%Y-%m-%d")
    end_date   = dates.max().strftime("%Y-%m-%d")

    weather_df = fetch_weather(start_date, end_date, latitude, longitude)

    out = df.copy()
    if weather_df.empty:
        out["precipitation"] = float("nan")
        out["snowfall"]      = float("nan")
        out["temp_max"]      = float("nan")
        out["temp_min"]      = float("nan")
        out["weather"]       = "Unknown"
        return out

    # Normalise date column type for merge
    weather_df["date"] = weather_df["date"].astype(str)
    out[date_col] = out[date_col].astype(str)

    out = out.merge(weather_df, left_on=date_col, right_on="date", how="left", suffixes=("", "_w"))
    # Drop duplicated date column produced by merge when date_col != "date"
    if date_col != "date" and "date" in out.columns:
        out = out.drop(columns=["date"])

    for col in ["precipitation", "snowfall", "temp_max", "temp_min"]:
        if col not in out.columns:
            out[col] = float("nan")
    if "weather" not in out.columns:
        out["weather"] = "Unknown"

    return out


# ---------------------------------------------------------------------------
# 3.  Convenience: enrich both holiday and weather in one call
# ---------------------------------------------------------------------------

def enrich_external(
    df: pd.DataFrame,
    date_col: str = "date",
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> pd.DataFrame:
    """
    Apply holiday enrichment then weather enrichment to *df*.

    This is the single entry-point called by `runner.py`.

    Added columns
    -------------
    is_holiday, holiday_name, weekday, weekday_name, day_type  (from enrich_holidays)
    precipitation, snowfall, temp_max, temp_min, weather        (from enrich_weather)

    All additions are fail-safe: API / library errors produce NaN / default values.
    """
    df = enrich_holidays(df, date_col=date_col)
    df = enrich_weather(df, date_col=date_col, latitude=latitude, longitude=longitude)
    return df
