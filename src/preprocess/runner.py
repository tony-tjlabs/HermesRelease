"""
Hermes preprocessing runner: Raw → BLE sessions & aggregates → Parquet/JSON cache.
Timezone: Asia/Seoul (UTC+9). Sessions include entry_time, exit_time (back-dated).

Architecture: Stitching-First Pipeline
  engine.run_daily() → RAW sessions → L1+L2 Stitching → STITCHED sessions → aggregates
  RAW aggregates are stored separately (*_raw.parquet) for MAC Stitching comparison tab.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.data.space_loader import load_sward_config, load_raw_date, get_available_dates
from src.data.external_api import enrich_external
from src.analytics.day_type import add_day_context_to_daily_stats
from src.analytics.ble_engine import HermesBLEEngine
from src.analytics.mac_stitching import (
    detect_mac_swaps,
    detect_mac_swaps_v2,
    stitch_visitor_sessions,
    stitch_sessions_posthoc,
)
from src.cache.cache_io import CacheWriter
from src.config.paths import get_cache_path
from src.config.constants import (
    DEVICE_TYPE_APPLE,
    DEVICE_TYPE_ANDROID,
    TIME_UNIT_SECONDS,
    DWELL_SHORT_MAX,
    DWELL_MEDIUM_MAX,
    SECONDS_PER_HOUR,
    STORE_OPEN_HOUR,
    STORE_CLOSE_HOUR,
)

logger = logging.getLogger(__name__)


def _compute_dwell_funnel(sessions_df: pd.DataFrame, floating_unique: int) -> dict:
    """
    세션 dwell_seconds 기준으로 퍼널 집계.
    단기(<3분), 중기(3~10분), 장기(10분+), quality_visitor, quality_cvr, dwell_median.
    """
    if sessions_df.empty or "dwell_seconds" not in sessions_df.columns:
        return {
            "short_dwell_count": 0,
            "medium_dwell_count": 0,
            "long_dwell_count": 0,
            "short_dwell_ratio": 0.0,
            "medium_dwell_ratio": 0.0,
            "long_dwell_ratio": 0.0,
            "quality_visitor_count": 0,
            "quality_cvr": 0.0,
            "dwell_median_seconds": 0.0,
        }
    dwell = sessions_df["dwell_seconds"]
    n = len(dwell)
    short = int((dwell < DWELL_SHORT_MAX).sum())
    medium = int(((dwell >= DWELL_SHORT_MAX) & (dwell < DWELL_MEDIUM_MAX)).sum())
    long_ = int((dwell >= DWELL_MEDIUM_MAX).sum())
    quality = medium + long_
    return {
        "short_dwell_count": short,
        "medium_dwell_count": medium,
        "long_dwell_count": long_,
        "short_dwell_ratio": round(short / n * 100, 2) if n else 0.0,
        "medium_dwell_ratio": round(medium / n * 100, 2) if n else 0.0,
        "long_dwell_ratio": round(long_ / n * 100, 2) if n else 0.0,
        "quality_visitor_count": quality,
        "quality_cvr": round(quality / floating_unique * 100, 2) if floating_unique else 0.0,
        "dwell_median_seconds": round(float(dwell.median()), 2),
    }


# Asia/Seoul: format entry/exit as HH:MM:SS+09:00 on given date
def _time_index_to_iso_time(date_str: str, time_index: int) -> str:
    sec = time_index * TIME_UNIT_SECONDS
    h = sec // SECONDS_PER_HOUR
    m = (sec % SECONDS_PER_HOUR) // 60
    s = sec % 60
    return f"{date_str} {h:02d}:{m:02d}:{s:02d}+09:00"


def _compute_hourly_visitor_metrics(sessions_df: pd.DataFrame) -> Dict[int, dict]:
    """Compute per-hour visitor metrics from a sessions DataFrame.

    Returns dict keyed by hour (0-23) with visitor_count, dwell_seconds_mean,
    visitor_apple, visitor_android.
    """
    result: Dict[int, dict] = {}
    for h in range(24):
        if sessions_df.empty:
            result[h] = {
                "visitor_count": 0,
                "dwell_seconds_mean": 0.0,
                "visitor_apple": 0,
                "visitor_android": 0,
            }
        else:
            h_sess = sessions_df[sessions_df["hour"] == h]
            result[h] = {
                "visitor_count": len(h_sess),
                "dwell_seconds_mean": round(h_sess["dwell_seconds"].mean(), 2) if len(h_sess) > 0 else 0.0,
                "visitor_apple": int((h_sess["device_type"] == DEVICE_TYPE_APPLE).sum()),
                "visitor_android": int((h_sess["device_type"] == DEVICE_TYPE_ANDROID).sum()),
            }
    return result


def _build_daily_stats_row(
    date_str: str,
    floating_unique: int,
    sessions_df: pd.DataFrame,
) -> dict:
    """Build a daily_stats row from a sessions DataFrame."""
    date_obj = pd.to_datetime(date_str)
    visitor_count = len(sessions_df) if not sessions_df.empty else 0
    dwell_mean = float(sessions_df["dwell_seconds"].mean()) if not sessions_df.empty else 0.0
    cvr = (visitor_count / floating_unique * 100.0) if floating_unique > 0 else 0.0
    funnel = _compute_dwell_funnel(sessions_df, floating_unique)
    return {
        "date": date_str,
        "floating_unique": floating_unique,
        "visitor_count": visitor_count,
        "conversion_rate": round(cvr, 2),
        "dwell_seconds_mean": round(dwell_mean, 2),
        "weekday": date_obj.weekday(),
        "weekday_name": date_obj.strftime("%A"),
        **funnel,
    }


def _build_daily_result_row(
    date_str: str,
    floating_unique: int,
    sessions_df: pd.DataFrame,
) -> dict:
    """Build a daily_results entry from a sessions DataFrame."""
    visitor_count = len(sessions_df) if not sessions_df.empty else 0
    dwell_mean = float(sessions_df["dwell_seconds"].mean()) if not sessions_df.empty else 0.0
    cvr = (visitor_count / floating_unique * 100.0) if floating_unique > 0 else 0.0
    return {
        "date": date_str,
        "floating_unique": floating_unique,
        "visitor_count": visitor_count,
        "conversion_rate": round(cvr, 2),
        "dwell_seconds_mean": round(dwell_mean, 2),
    }


def _build_hourly_rows(
    date_str: str,
    floating_base: Dict[int, dict],
    visitor_metrics: Dict[int, dict],
) -> List[dict]:
    """Merge floating-population base with visitor metrics into hourly rows."""
    rows = []
    for h in range(24):
        fb = floating_base.get(h, {"floating_count": 0, "floating_apple": 0, "floating_android": 0})
        vm = visitor_metrics.get(h, {"visitor_count": 0, "dwell_seconds_mean": 0.0, "visitor_apple": 0, "visitor_android": 0})
        fp = fb["floating_count"]
        vc = vm["visitor_count"]
        cvr_h = (vc / fp * 100.0) if fp > 0 else 0.0
        rows.append({
            "date": date_str,
            "hour": h,
            "floating_count": fp,
            "visitor_count": vc,
            "conversion_rate": round(cvr_h, 2),
            "dwell_seconds_mean": vm["dwell_seconds_mean"],
            "floating_apple": fb["floating_apple"],
            "floating_android": fb["floating_android"],
            "visitor_apple": vm["visitor_apple"],
            "visitor_android": vm["visitor_android"],
        })
    return rows


def _collect_device_mix(sessions_df: pd.DataFrame, date_str: str) -> List[dict]:
    """Collect device type counts from sessions for device_mix."""
    if sessions_df.empty:
        return []
    counts = []
    for dtype, cnt in sessions_df["device_type"].value_counts().items():
        counts.append({
            "date": date_str,
            "device_type": int(dtype),
            "count": int(cnt),
        })
    return counts


def run_preprocess_space(
    space_name: str,
    date_range: Optional[List[str]] = None,
    force: bool = False,
    max_dates: Optional[int] = None,
) -> bool:
    """
    Run full pipeline for one space: load raw → BLE engine → MAC Stitching → aggregates.

    Architecture: Stitching-First
      All primary aggregates (daily_stats, daily_hourly, device_mix, daily_timeseries)
      are computed from STITCHED sessions. Raw aggregates stored separately for comparison.

    Parameters
    ----------
    force : bool
        If False (default) and a valid cache already exists, skip.
        If True, always regenerate.
    """
    # ── Cache existence check ─────────────────────────────────────────────
    cache_marker = get_cache_path(space_name) / "metadata.json"
    if not force and cache_marker.exists():
        logger.info(
            "Cache already exists for '%s' — skipping. Use --force to regenerate.",
            space_name,
        )
        return True

    try:
        logger.info("Loading sward config for %s", space_name)
        sward_config = load_sward_config(space_name)
        dates = date_range or get_available_dates(space_name)
        if not dates:
            logger.warning("No dates found for %s", space_name)
            return False
        if max_dates is not None and max_dates > 0:
            dates = dates[:max_dates]
        logger.info("Processing %d dates for %s", len(dates), space_name)

        engine = HermesBLEEngine(sward_config)
        writer = CacheWriter(space_name)

        # PRIMARY accumulators (stitched-based)
        daily_results: List[dict] = []
        daily_hourly_rows: List[dict] = []
        daily_stats_rows: List[dict] = []
        device_type_counts: List[dict] = []
        timeseries_dfs: List[pd.DataFrame] = []
        sessions_dfs: List[pd.DataFrame] = []
        sessions_stitched_dfs: List[pd.DataFrame] = []

        # RAW accumulators (for comparison tab)
        raw_daily_results: List[dict] = []
        raw_daily_hourly_rows: List[dict] = []
        raw_daily_stats_rows: List[dict] = []
        raw_device_type_counts: List[dict] = []

        for idx, date_str in enumerate(dates):
            logger.info("  [%d/%d] %s", idx + 1, len(dates), date_str)
            df = load_raw_date(space_name, date_str)
            if df.empty:
                logger.warning("    Empty raw data, skip")
                continue

            # ── Phase A: Run BLE engine (produces RAW sessions) ───────────
            out = engine.run_daily(df)
            sessions_df = out["sessions_df"]
            hf = out["hourly_floating"]

            # Ensure sessions_df has "hour" column
            if not sessions_df.empty and "hour" not in sessions_df.columns:
                sessions_df["hour"] = (
                    sessions_df["entry_time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR
                ).astype(int).clip(0, 23)

            # ── Phase B: Compute Floating-Population Hourly Base ──────────
            # FP is BLE-scan-based, unaffected by session stitching
            if engine.entrance_swards:
                float_df = df[df["sward_name"].isin(engine.entrance_swards) & (df["rssi"] >= engine.floating_rssi)]
            else:
                float_df = df[df["sward_name"].isin(engine.inside_swards) & (df["rssi"] >= -90)]
            float_df = float_df.copy()
            float_df["hour"] = (float_df["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int).clip(0, 23)
            mac_type_per_hour = float_df.groupby(["hour", "mac_address"])["type"].first().reset_index()

            floating_base: Dict[int, dict] = {}
            for h in range(24):
                fp = hf[hf["hour"] == h]["floating_count"].sum() if not hf.empty else 0
                h_mac = mac_type_per_hour[mac_type_per_hour["hour"] == h]
                floating_base[h] = {
                    "floating_count": int(fp),
                    "floating_apple": int((h_mac["type"] == DEVICE_TYPE_APPLE).sum()),
                    "floating_android": int((h_mac["type"] == DEVICE_TYPE_ANDROID).sum()),
                }

            # ── Phase C: Collect RAW aggregates (for comparison) ──────────
            # 운영 시간(10:00–22:00) 세션만 집계에 사용 — 영업 외 시간 방문은 CVR 분자에서 제외
            sessions_raw_ops = (
                sessions_df[sessions_df["hour"].between(STORE_OPEN_HOUR, STORE_CLOSE_HOUR - 1)]
                if not sessions_df.empty else sessions_df
            )
            raw_daily_results.append(
                _build_daily_result_row(date_str, out["floating_unique"], sessions_raw_ops)
            )
            raw_daily_stats_rows.append(
                _build_daily_stats_row(date_str, out["floating_unique"], sessions_raw_ops)
            )
            raw_visitor_metrics = _compute_hourly_visitor_metrics(sessions_raw_ops)
            raw_daily_hourly_rows.extend(
                _build_hourly_rows(date_str, floating_base, raw_visitor_metrics)
            )
            raw_device_type_counts.extend(_collect_device_mix(sessions_raw_ops, date_str))

            # ── Phase D: MAC Stitching L1+L2 (Visitor sessions only) ─────
            # Level 1: Raw signal (Pattern A co-existence + Pattern B adjacent swap)
            swap_map = detect_mac_swaps_v2(df, inside_swards=engine.inside_swards)
            if not sessions_df.empty and swap_map:
                sessions_df_stitched = stitch_visitor_sessions(sessions_df, swap_map)
                logger.info(
                    "    L1 stitching: %d swap(s) → sessions %d→%d",
                    len(swap_map), len(sessions_df), len(sessions_df_stitched),
                )
            else:
                sessions_df_stitched = sessions_df.copy() if not sessions_df.empty else pd.DataFrame()
                if not sessions_df_stitched.empty:
                    sessions_df_stitched["stitched"] = False
                    sessions_df_stitched["raw_session_count"] = 1
                    sessions_df_stitched["original_macs"] = sessions_df_stitched["mac_address"].astype(str)

            # Level 2: Session-level post-hoc stitching
            if not sessions_df_stitched.empty:
                n_before_l2 = len(sessions_df_stitched)
                sessions_df_stitched = stitch_sessions_posthoc(sessions_df_stitched)
                n_after_l2 = len(sessions_df_stitched)
                if n_before_l2 != n_after_l2:
                    logger.info(
                        "    L2 stitching: sessions %d→%d",
                        n_before_l2, n_after_l2,
                    )

            # Ensure stitched has "hour" column (may have been lost in stitching)
            if not sessions_df_stitched.empty and "hour" not in sessions_df_stitched.columns:
                sessions_df_stitched["hour"] = (
                    sessions_df_stitched["entry_time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR
                ).astype(int).clip(0, 23)

            # ── Phase E: Compute PRIMARY aggregates (from stitched) ───────
            # 운영 시간(10:00–22:00) 세션만 집계 — CVR 분자에 영업 외 세션 포함 방지
            sessions_stitched_ops = (
                sessions_df_stitched[sessions_df_stitched["hour"].between(STORE_OPEN_HOUR, STORE_CLOSE_HOUR - 1)]
                if not sessions_df_stitched.empty else sessions_df_stitched
            )
            daily_results.append(
                _build_daily_result_row(date_str, out["floating_unique"], sessions_stitched_ops)
            )
            daily_stats_rows.append(
                _build_daily_stats_row(date_str, out["floating_unique"], sessions_stitched_ops)
            )
            stitched_visitor_metrics = _compute_hourly_visitor_metrics(sessions_stitched_ops)
            daily_hourly_rows.extend(
                _build_hourly_rows(date_str, floating_base, stitched_visitor_metrics)
            )
            device_type_counts.extend(_collect_device_mix(sessions_stitched_ops, date_str))

            # ── Phase F: Timeseries from stitched sessions ────────────────
            ts_df = engine.build_timeseries(
                df, sessions_df_stitched if not sessions_df_stitched.empty else pd.DataFrame()
            )
            if not ts_df.empty:
                ts_df = ts_df.copy()
                ts_df["date"] = date_str
                timeseries_dfs.append(ts_df)

            # ── Phase G: Accumulate session DataFrames ────────────────────
            if not sessions_df.empty:
                sessions_df = sessions_df.copy()
                sessions_df["date"] = date_str
                sessions_df["entry_time"] = sessions_df["entry_time_index"].map(
                    lambda t: _time_index_to_iso_time(date_str, int(t))
                )
                sessions_df["exit_time"] = sessions_df["exit_time_index"].map(
                    lambda t: _time_index_to_iso_time(date_str, int(t))
                )
                sessions_dfs.append(sessions_df)

            if not sessions_df_stitched.empty:
                sessions_df_stitched = sessions_df_stitched.copy()
                sessions_df_stitched["date"] = date_str
                sessions_stitched_dfs.append(sessions_df_stitched)

        # ── Build final DataFrames ────────────────────────────────────────
        # PRIMARY (stitched-based)
        daily_hourly = pd.DataFrame(daily_hourly_rows)
        daily_stats = pd.DataFrame(daily_stats_rows)
        device_mix = pd.DataFrame(device_type_counts) if device_type_counts else pd.DataFrame()
        daily_timeseries = (
            pd.concat(timeseries_dfs, ignore_index=True) if timeseries_dfs else pd.DataFrame()
        )
        sessions_all = pd.concat(sessions_dfs, ignore_index=True) if sessions_dfs else pd.DataFrame()
        sessions_all_stitched = (
            pd.concat(sessions_stitched_dfs, ignore_index=True)
            if sessions_stitched_dfs else pd.DataFrame()
        )

        # RAW (for comparison tab)
        raw_daily_hourly = pd.DataFrame(raw_daily_hourly_rows)
        raw_daily_stats = pd.DataFrame(raw_daily_stats_rows)
        raw_device_mix = pd.DataFrame(raw_device_type_counts) if raw_device_type_counts else pd.DataFrame()

        # ── Enrich daily_stats with holidays + weather (both versions) ────
        for label, ds in [("primary (stitched)", daily_stats), ("raw", raw_daily_stats)]:
            if ds.empty:
                continue
            try:
                ds_enriched = enrich_external(ds, date_col="date")
                ds_enriched = add_day_context_to_daily_stats(ds_enriched)
                if label.startswith("primary"):
                    daily_stats = ds_enriched
                else:
                    raw_daily_stats = ds_enriched
                logger.info("Enriched %s daily_stats with external data.", label)
            except Exception as exc:
                logger.warning("External enrichment failed for %s (non-fatal): %s", label, exc)

        # ── Write cache ───────────────────────────────────────────────────
        logger.info("Writing cache to %s", writer.cache_dir)
        writer.write(
            date_range=dates,
            # PRIMARY (stitched-based)
            daily_results=daily_results,
            daily_hourly=daily_hourly,
            daily_stats=daily_stats,
            sessions_all=sessions_all if not sessions_all.empty else None,
            device_mix=device_mix if not device_mix.empty else None,
            daily_timeseries=daily_timeseries if not daily_timeseries.empty else None,
            sessions_stitched=sessions_all_stitched if not sessions_all_stitched.empty else None,
            # RAW (for MAC Stitching comparison tab)
            daily_stats_raw=raw_daily_stats if not raw_daily_stats.empty else None,
            daily_hourly_raw=raw_daily_hourly if not raw_daily_hourly.empty else None,
            device_mix_raw=raw_device_mix if not raw_device_mix.empty else None,
            daily_results_raw=raw_daily_results if raw_daily_results else None,
        )
        logger.info("Preprocess complete for %s (%d dates)", space_name, len(dates))
        return True
    except Exception as e:
        logger.exception("Preprocess failed for %s", space_name)
        raise RuntimeError(f"Preprocess failed for {space_name}: {e}") from e
