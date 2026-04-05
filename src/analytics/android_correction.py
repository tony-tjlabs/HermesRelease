"""
Android Floating-Population 보정 모듈.

문제: Android BLE Tx Power가 iPhone 대비 ~3-5 dB 낮아서
약신호 구간에서 Android 기기가 감지되지 않음 → 유동인구 과소 카운트.

해결: 강신호 구간(iPhone·Android 모두 잘 잡히는 영역)의 Mix Ratio를
      기준으로 약신호 구간의 누락 Android 수를 추정하여 보정.

적용 범위: Floating(유동인구)만. Visitor(방문자)는 보정하지 않음.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.constants import (
    CORRECTION_TH_IPH,
    CORRECTION_TH_AND_MIN,
    CORRECTION_TH_AND_MAX,
    CORRECTION_SELF_CONSISTENCY_SKIP,
    CORRECTION_SELF_CONSISTENCY_OK,
    CORRECTION_MIN_STRONG_COUNT,
    DEVICE_TYPE_APPLE,
    DEVICE_TYPE_ANDROID,
    TIME_UNIT_SECONDS,
    SECONDS_PER_HOUR,
    STORE_OPEN_HOUR,
    STORE_CLOSE_HOUR,
)

logger = logging.getLogger(__name__)


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class CorrectionCalibration:
    """Self-consistency 탐색을 통한 Th_and 캘리브레이션 결과."""

    th_iph: int                     # iPhone 강신호 경계 (고정)
    th_and: int                     # 최적 Android 강신호 경계
    self_consistency_diff: float    # |expected% - actual%| 최솟값
    correction_needed: bool         # diff < SKIP 임계값 여부
    mix_ratio_global: float         # 캘리브레이션 데이터 전체 Mix Ratio
    calibration_date_count: int     # 캘리브레이션에 사용된 날짜 수

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CorrectionCalibration":
        return cls(**d)


# ── 캘리브레이션 ─────────────────────────────────────────────────────────────

def calibrate_th_and(
    raw_df: pd.DataFrame,
    entrance_swards: List[str],
    inside_swards: List[str],
    floating_rssi: int,
    th_iph: int = CORRECTION_TH_IPH,
) -> CorrectionCalibration:
    """
    Self-consistency 탐색으로 최적 Th_and를 결정한다.

    알고리즘:
    1. Floating 센서 기준으로 필터링
    2. MAC별 best RSSI 추출 (시간대 무관, 전체 캘리브레이션 기간)
    3. Th_and를 -66 ~ -95 범위에서 1dB 단위 탐색
       - iPhone_strong: best_rssi >= th_iph
       - Android_strong: best_rssi >= th_and
       - iPhone_weak: best_rssi < th_iph
       - Mix Ratio = Android_strong / iPhone_strong
       - Expected Android% = Mix Ratio / (1 + Mix Ratio) × 100
       - Missing Android = iPhone_weak × Mix Ratio
       - Corrected Android = Measured Android + Missing Android
       - Actual Android% = Corrected Android / (iPhone_total + Corrected Android) × 100
       - diff = |Expected% - Actual%|
    4. diff 최소인 Th_and 선택
    5. min_diff > SKIP 임계값 → correction_needed = False
    """
    # 1. 유동 센서 필터링
    fp_df = _filter_floating_data(raw_df, entrance_swards, inside_swards, floating_rssi)
    if fp_df.empty:
        logger.warning("No floating data for calibration")
        return _no_correction_result(th_iph)

    # 2. MAC별 best RSSI + 디바이스 타입
    mac_best = (
        fp_df.groupby("mac_address")
        .agg(best_rssi=("rssi", "max"), device_type=("type", "first"))
        .reset_index()
    )

    iphone_macs = mac_best[mac_best["device_type"] == DEVICE_TYPE_APPLE]
    android_macs = mac_best[mac_best["device_type"] == DEVICE_TYPE_ANDROID]

    if iphone_macs.empty or android_macs.empty:
        logger.warning("Insufficient device types for calibration (iPhone=%d, Android=%d)",
                       len(iphone_macs), len(android_macs))
        return _no_correction_result(th_iph)

    # iPhone 전체 / Android 전체
    iph_total = len(iphone_macs)
    and_total = len(android_macs)

    # iPhone strong (fixed threshold)
    iph_strong = int((iphone_macs["best_rssi"] >= th_iph).sum())
    iph_weak = iph_total - iph_strong

    if iph_strong < CORRECTION_MIN_STRONG_COUNT:
        logger.warning("Insufficient iPhone strong-signal MACs: %d", iph_strong)
        return _no_correction_result(th_iph)

    # 3. Th_and 탐색
    best_diff = float("inf")
    best_th_and = CORRECTION_TH_AND_MIN
    best_mix_ratio = 0.0

    for th_and in range(CORRECTION_TH_AND_MAX, CORRECTION_TH_AND_MIN - 1, -1):
        and_strong = int((android_macs["best_rssi"] >= th_and).sum())
        if and_strong < 1:
            continue

        mix_ratio = and_strong / iph_strong
        expected_pct = mix_ratio / (1.0 + mix_ratio) * 100.0

        missing_android = iph_weak * mix_ratio
        corrected_android = and_total + missing_android
        total_corrected = iph_total + corrected_android
        actual_pct = corrected_android / total_corrected * 100.0 if total_corrected > 0 else 0.0

        diff = abs(expected_pct - actual_pct)
        if diff < best_diff:
            best_diff = diff
            best_th_and = th_and
            best_mix_ratio = mix_ratio

    correction_needed = best_diff < CORRECTION_SELF_CONSISTENCY_SKIP
    date_count = fp_df["date"].nunique() if "date" in fp_df.columns else 1

    logger.info(
        "Calibration result: Th_and=%d, diff=%.2f%%, mix_ratio=%.3f, needed=%s (dates=%d)",
        best_th_and, best_diff, best_mix_ratio, correction_needed, date_count,
    )

    return CorrectionCalibration(
        th_iph=th_iph,
        th_and=best_th_and,
        self_consistency_diff=round(best_diff, 4),
        correction_needed=correction_needed,
        mix_ratio_global=round(best_mix_ratio, 4),
        calibration_date_count=date_count,
    )


# ── 시간대별 보정 ────────────────────────────────────────────────────────────

def correct_hourly_floating(
    raw_df: pd.DataFrame,
    entrance_swards: List[str],
    inside_swards: List[str],
    floating_rssi: int,
    calibration: CorrectionCalibration,
) -> Dict[int, dict]:
    """
    시간대별(0~23) Android FP 보정 적용.

    각 시간대에서:
    1. MAC별 best_rssi 계산
    2. 강/약 신호 분류 (th_iph, calibration.th_and 사용)
    3. Mix Ratio = Android_strong / iPhone_strong
       - 강신호 부족 시 calibration.mix_ratio_global 폴백
    4. Missing Android = iPhone_weak × Mix Ratio
    5. Corrected Total = Raw Total + Missing Android

    Returns:
        dict[hour → {
            floating_count_raw, floating_apple, floating_android_raw,
            floating_count_corrected, floating_android_corrected,
            missing_android, mix_ratio, correction_applied
        }]
    """
    fp_df = _filter_floating_data(raw_df, entrance_swards, inside_swards, floating_rssi)
    result: Dict[int, dict] = {}

    if fp_df.empty:
        for h in range(24):
            result[h] = _empty_hourly_correction()
        return result

    fp_df = fp_df.copy()
    fp_df["hour"] = (fp_df["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int).clip(0, 23)

    th_iph = calibration.th_iph
    th_and = calibration.th_and

    for h in range(24):
        h_df = fp_df[fp_df["hour"] == h]
        if h_df.empty:
            result[h] = _empty_hourly_correction()
            continue

        # MAC별 best RSSI + device type
        mac_best = (
            h_df.groupby("mac_address")
            .agg(best_rssi=("rssi", "max"), device_type=("type", "first"))
            .reset_index()
        )

        iph_mask = mac_best["device_type"] == DEVICE_TYPE_APPLE
        and_mask = mac_best["device_type"] == DEVICE_TYPE_ANDROID

        iph_all = int(iph_mask.sum())
        and_all = int(and_mask.sum())
        total_raw = iph_all + and_all

        if iph_all == 0:
            result[h] = {
                "floating_count_raw": total_raw,
                "floating_apple": iph_all,
                "floating_android_raw": and_all,
                "floating_count_corrected": total_raw,
                "floating_android_corrected": and_all,
                "missing_android": 0,
                "mix_ratio": 0.0,
                "correction_applied": False,
            }
            continue

        iph_strong = int((mac_best[iph_mask]["best_rssi"] >= th_iph).sum())
        and_strong = int((mac_best[and_mask]["best_rssi"] >= th_and).sum()) if and_all > 0 else 0
        iph_weak = iph_all - iph_strong

        # Mix Ratio 결정: 시간대별 우선, 부족 시 global 폴백
        if iph_strong >= CORRECTION_MIN_STRONG_COUNT and and_strong >= 1:
            mix_ratio = and_strong / iph_strong
        else:
            mix_ratio = calibration.mix_ratio_global

        missing_android = round(iph_weak * mix_ratio)
        corrected_android = and_all + missing_android
        corrected_total = iph_all + corrected_android

        result[h] = {
            "floating_count_raw": total_raw,
            "floating_apple": iph_all,
            "floating_android_raw": and_all,
            "floating_count_corrected": corrected_total,
            "floating_android_corrected": corrected_android,
            "missing_android": missing_android,
            "mix_ratio": round(mix_ratio, 4),
            "correction_applied": missing_android > 0,
        }

    return result


# ── 일별 보정 ────────────────────────────────────────────────────────────────

def correct_daily_floating(
    hourly_correction: Dict[int, dict],
    raw_daily_unique: int,
    store_open_hour: int = STORE_OPEN_HOUR,
    store_close_hour: int = STORE_CLOSE_HOUR,
) -> dict:
    """
    시간대별 보정을 집계하여 일별 보정값 산출.

    daily_unique는 dedup된 값이므로 시간대별 합산과 다를 수 있다.
    따라서 scaling factor 적용:
        corrected = raw_daily_unique × (sum_hourly_corrected / sum_hourly_raw)

    Parameters
    ----------
    store_open_hour : int
        영업 시작 시간 (inclusive). 기본값 STORE_OPEN_HOUR (10)
    store_close_hour : int
        영업 종료 시간 (exclusive). 기본값 STORE_CLOSE_HOUR (22)
        24시간 영업 시 open=0, close=24 사용.

    Returns:
        {
            "floating_unique_raw": int,
            "floating_unique_corrected": int,
            "total_missing_android": int,
            "correction_pct": float,  # 보정 비율 (%)
            "correction_applied": bool,
        }
    """
    # 운영 시간만 집계 (24시간 영업 시 0~24)
    sum_raw = sum(
        hourly_correction.get(h, {}).get("floating_count_raw", 0)
        for h in range(store_open_hour, store_close_hour)
    )
    sum_corrected = sum(
        hourly_correction.get(h, {}).get("floating_count_corrected", 0)
        for h in range(store_open_hour, store_close_hour)
    )
    total_missing = sum(
        hourly_correction.get(h, {}).get("missing_android", 0)
        for h in range(store_open_hour, store_close_hour)
    )

    if sum_raw == 0 or total_missing == 0:
        return {
            "floating_unique_raw": raw_daily_unique,
            "floating_unique_corrected": raw_daily_unique,
            "total_missing_android": 0,
            "correction_pct": 0.0,
            "correction_applied": False,
        }

    scaling_factor = sum_corrected / sum_raw
    corrected_unique = round(raw_daily_unique * scaling_factor)
    correction_pct = round((corrected_unique - raw_daily_unique) / raw_daily_unique * 100, 2)

    return {
        "floating_unique_raw": raw_daily_unique,
        "floating_unique_corrected": corrected_unique,
        "total_missing_android": total_missing,
        "correction_pct": correction_pct,
        "correction_applied": True,
    }


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _filter_floating_data(
    raw_df: pd.DataFrame,
    entrance_swards: List[str],
    inside_swards: List[str],
    floating_rssi: int,
) -> pd.DataFrame:
    """유동인구 계산에 사용하는 raw 데이터 필터링 (센서 + RSSI + 디바이스 타입)."""
    if raw_df.empty:
        return pd.DataFrame()

    # iPhone/Android만 (type 1, 10)
    df = raw_df[raw_df["type"].isin([DEVICE_TYPE_APPLE, DEVICE_TYPE_ANDROID])].copy()
    if df.empty:
        return pd.DataFrame()

    # 센서 선택: 입구 센서 우선, 없으면 내부 센서
    if entrance_swards:
        df = df[df["sward_name"].isin(entrance_swards)]
        th = floating_rssi
    else:
        df = df[df["sward_name"].isin(inside_swards)]
        th = -90  # FALLBACK_FLOATING_RSSI

    # RSSI 필터링 — 기본 임계값 적용 (강/약 분류는 보정 알고리즘에서 별도 처리)
    df = df[df["rssi"] >= th]

    return df


def _no_correction_result(th_iph: int) -> CorrectionCalibration:
    """캘리브레이션 불가 시 기본 결과 (보정 안 함)."""
    return CorrectionCalibration(
        th_iph=th_iph,
        th_and=th_iph,
        self_consistency_diff=999.0,
        correction_needed=False,
        mix_ratio_global=0.0,
        calibration_date_count=0,
    )


def _empty_hourly_correction() -> dict:
    """데이터 없는 시간대의 보정 결과."""
    return {
        "floating_count_raw": 0,
        "floating_apple": 0,
        "floating_android_raw": 0,
        "floating_count_corrected": 0,
        "floating_android_corrected": 0,
        "missing_android": 0,
        "mix_ratio": 0.0,
        "correction_applied": False,
    }
