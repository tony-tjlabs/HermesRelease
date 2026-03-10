"""
Hermes MAC Stitching — BLE MAC Address Rotation Compensation.

BLE 기기(Apple/Android)는 신호 송출 중 MAC 주소를 자동 교체한다.
이 모듈은 Raw 내부 센서 데이터에서 MAC 교체를 탐지하고,
Visitor 세션의 분절된 세션을 하나로 연결(stitch)하여 체류시간·방문자 수를 보정한다.

v2 (2026-03): 실증 분석 기반 개선
  - Pattern A (공존): 기존 T/T+1/T+2 패턴 + 갭 허용 + 디바이스 타입 일치
  - Pattern B (인접 교체): 공존 없이 old MAC 소멸 → new MAC 출현 (전환의 67% 차지)
  - Level 2: 세션 레벨 post-hoc stitching
  - 평가 프레임워크: Raw → L1 → L2 비교

설계 원칙:
  - Raw mac_address 컬럼은 절대 수정하지 않음.
  - Visitor 세션에만 적용. 유동인구(FP)에는 미적용.
  - 날짜 경계를 넘는 stitching 없음 — 날짜별 독립 처리.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.constants import (
    TIME_UNIT_SECONDS,
    DEVICE_TYPE_APPLE,
    DEVICE_TYPE_ANDROID,
    STITCH_GAP_MAX_SLOTS,
    STITCH_RSSI_DIFF_APPLE,
    STITCH_RSSI_DIFF_ANDROID,
    STITCH_LOOKAHEAD_SLOTS,
    SESSION_STITCH_GAP_SEC_APPLE,
    SESSION_STITCH_GAP_SEC_ANDROID,
    SESSION_STITCH_SHORT_THRESHOLD,
    SESSION_STITCH_RSSI_DIFF,
    SESSION_STITCH_MAX_CHAIN,
)


def detect_mac_swaps(
    df: pd.DataFrame,
    inside_swards: List[str],
    max_rssi_diff: int = 10,
) -> Dict[str, str]:
    """
    Raw 내부 센서 데이터에서 MAC 교체 이벤트를 탐지한다.

    Apple 기기의 MAC 교체 패턴:
      T   : MAC-A만 존재
      T+1 : MAC-A + MAC-B 동시 존재 (RSSI 유사)
      T+2 : MAC-B만 존재 (MAC-A 소멸)

    탐지 조건 (세 가지 모두 만족):
      1. 소멸 조건: MAC-A가 T+1에 존재하지만 T+2에는 없음
      2. 출현 조건: MAC-B가 T+1에 처음 등장 (T 이전에 없었던 MAC)
      3. RSSI 유사성: T+1 시점 MAC-A와 MAC-B의 RSSI 평균 차이 <= max_rssi_diff

    후보가 여러 개일 때 → RSSI 차이가 가장 작은 것 선택.

    Notes
    -----
    Greedy 방식이므로 모든 교체를 탐지하지 못할 수 있음.
    False Positive(오탐) 최소화를 우선하며, max_rssi_diff로 민감도 조정 가능.

    Parameters
    ----------
    df : Raw DataFrame (단일 날짜, inside + entrance 센서 모두 포함 가능)
    inside_swards : 내부 센서 이름 리스트 (entrance 데이터는 무시됨)
    max_rssi_diff : T+1 시점 RSSI 차이 허용치 (dBm, 기본 10)

    Returns
    -------
    swap_map : {old_mac: new_mac} 딕셔너리 (1:1 매핑)
    """
    swap_map: Dict[str, str] = {}

    # 내부 센서 데이터만 사용
    inside_df = df[df["sward_name"].isin(inside_swards)].copy()
    if inside_df.empty:
        return swap_map

    # time_index별 존재 MAC 집합 및 RSSI 평균
    slot_macs: Dict[int, set] = {}
    slot_rssi: Dict[int, Dict[str, float]] = {}
    for ti, grp in inside_df.groupby("time_index"):
        ti = int(ti)
        slot_macs[ti] = set(grp["mac_address"].unique())
        slot_rssi[ti] = grp.groupby("mac_address")["rssi"].mean().to_dict()

    time_indices = sorted(slot_macs.keys())
    ti_set = set(time_indices)

    used_as_new: set = set()   # 이미 new_mac으로 할당된 MAC
    used_as_old: set = set()   # 이미 old_mac으로 처리된 MAC

    for t in time_indices:
        t1 = t + 1
        t2 = t + 2
        if t1 not in ti_set:
            continue

        macs_t   = slot_macs[t]
        macs_t1  = slot_macs[t1]
        macs_t2  = slot_macs.get(t2, set())

        # 소멸 후보: T+1에 있지만 T+2에 없는 MAC-A
        vanishing = macs_t1 - macs_t2
        # 출현 후보: T+1에 처음 등장 (T에 없었던) MAC-B
        appearing = macs_t1 - macs_t

        if not vanishing or not appearing:
            continue

        rssi_t1 = slot_rssi[t1]

        for old_mac in vanishing:
            if old_mac in used_as_old or old_mac in used_as_new:
                continue
            if old_mac not in rssi_t1:
                continue

            rssi_a = rssi_t1[old_mac]
            best_new = None
            best_diff = float("inf")

            for new_mac in appearing:
                if new_mac in used_as_new or new_mac in used_as_old:
                    continue
                if new_mac not in rssi_t1:
                    continue
                diff = abs(rssi_a - rssi_t1[new_mac])
                if diff <= max_rssi_diff and diff < best_diff:
                    best_diff = diff
                    best_new = new_mac

            if best_new is not None:
                swap_map[old_mac] = best_new
                used_as_old.add(old_mac)
                used_as_new.add(best_new)

    return swap_map


# ═══════════════════════════════════════════════════════════════════
# Level 1 v2: Pattern A (공존) + Pattern B (인접 교체)
# ═══════════════════════════════════════════════════════════════════

def detect_mac_swaps_v2(
    df: pd.DataFrame,
    inside_swards: List[str],
    gap_max: int = STITCH_GAP_MAX_SLOTS,
    rssi_diff_apple: int = STITCH_RSSI_DIFF_APPLE,
    rssi_diff_android: int = STITCH_RSSI_DIFF_ANDROID,
    lookahead: int = STITCH_LOOKAHEAD_SLOTS,
) -> Dict[str, str]:
    """
    실증 기반 MAC 교체 탐지 v2.

    직원 데이터 분석 결과:
      - Apple 전환의 33%만 공존(같은 time_index), 67%는 공존 없이 교체
      - RSSI 차이: Apple 71.5%가 3dBm 이내, Android 77.1%가 3dBm 이내
      - 갭: 대부분 0~3 time_index (0~30초)

    Pattern A (공존): old/new MAC이 동일 time_index에 공존 + 이후 old 소멸
    Pattern B (인접): old MAC 소멸 → gap ≤ gap_max → new MAC 출현 (공존 없음)

    Returns
    -------
    swap_map : {old_mac: new_mac}
    """
    swap_map: Dict[str, str] = {}

    inside_df = df[df["sward_name"].isin(inside_swards)].copy()
    if inside_df.empty:
        return swap_map

    # ── MAC별 메타데이터 사전 계산 (dict 조회로 O(1)) ─────────
    mac_stats = inside_df.groupby("mac_address").agg(
        first_ti=("time_index", "min"),
        last_ti=("time_index", "max"),
        avg_rssi=("rssi", "mean"),
        device_type=("type", "first"),
        hit_count=("time_index", "count"),
    )
    mac_first: Dict[str, int] = mac_stats["first_ti"].astype(int).to_dict()
    mac_last: Dict[str, int] = mac_stats["last_ti"].astype(int).to_dict()
    mac_device: Dict[str, int] = mac_stats["device_type"].to_dict()
    mac_stats = mac_stats.reset_index()

    # time_index별 MAC 집합 및 RSSI
    slot_macs: Dict[int, set] = {}
    slot_rssi: Dict[int, Dict[str, float]] = {}
    for ti, grp in inside_df.groupby("time_index"):
        ti = int(ti)
        slot_macs[ti] = set(grp["mac_address"].unique())
        slot_rssi[ti] = grp.groupby("mac_address")["rssi"].mean().to_dict()

    time_indices = sorted(slot_macs.keys())
    ti_set = set(time_indices)

    used_as_new: set = set()
    used_as_old: set = set()

    def _get_rssi_threshold(device_type: int) -> int:
        return rssi_diff_apple if device_type == DEVICE_TYPE_APPLE else rssi_diff_android

    # ── Pattern A: 공존 탐지 (개선) ────────────────────────────
    # old MAC과 new MAC이 같은 time_index에 공존하고,
    # old MAC이 이후 gap_max 슬롯 내에 소멸하는 패턴.
    for t in time_indices:
        macs_t = slot_macs[t]
        for old_mac in macs_t:
            if old_mac in used_as_old or old_mac in used_as_new:
                continue
            if mac_last.get(old_mac) != t:
                continue  # 이 MAC은 t 이후에도 보이므로 여기서 소멸 아님

            old_rssi = slot_rssi[t].get(old_mac)
            if old_rssi is None:
                continue
            old_dtype = mac_device.get(old_mac)
            rssi_thresh = _get_rssi_threshold(old_dtype)

            best_new = None
            best_cost = float("inf")

            for new_mac in macs_t:
                if new_mac == old_mac:
                    continue
                if new_mac in used_as_new or new_mac in used_as_old:
                    continue
                if mac_device.get(new_mac) != old_dtype:
                    continue
                if mac_first.get(new_mac) != t:
                    continue

                new_rssi = slot_rssi[t].get(new_mac)
                if new_rssi is None:
                    continue
                diff = abs(old_rssi - new_rssi)
                if diff <= rssi_thresh and diff < best_cost:
                    best_cost = diff
                    best_new = new_mac

            if best_new is not None:
                swap_map[old_mac] = best_new
                used_as_old.add(old_mac)
                used_as_new.add(best_new)

    # ── Pattern B: 인접 교체 탐지 (신규) ──────────────────────
    # old MAC 소멸 → gap(1~gap_max) → new MAC 출현 (공존 없음)
    # 실증: 전환의 67%가 이 패턴 (4-6초 전송 중단으로 인해)

    # 소멸/출현 이벤트 인덱스 구축 (dict 기반, O(n))
    ending_macs: Dict[int, List[str]] = {}  # {last_ti: [mac, ...]}
    starting_macs: Dict[int, List[str]] = {}  # {first_ti: [mac, ...]}
    for mac, last_ti in mac_last.items():
        if mac in used_as_old or mac in used_as_new:
            continue
        ending_macs.setdefault(last_ti, []).append(mac)
    for mac, first_ti in mac_first.items():
        if mac in used_as_old or mac in used_as_new:
            continue
        starting_macs.setdefault(first_ti, []).append(mac)

    for t in time_indices:
        enders = ending_macs.get(t, [])
        if not enders:
            continue

        for old_mac in enders:
            if old_mac in used_as_old or old_mac in used_as_new:
                continue

            old_dtype = mac_device.get(old_mac)
            old_rssi_val = slot_rssi.get(t, {}).get(old_mac)
            if old_rssi_val is None:
                continue
            rssi_thresh = _get_rssi_threshold(old_dtype)

            # 재출현 확인: old_mac이 lookahead 내에 다시 보이면 소멸 아님
            # (이미 last_ti == t 로 필터했으므로 불필요하지만 안전장치)

            best_new = None
            best_cost = float("inf")

            for gap in range(1, gap_max + 1):
                t_new = t + gap
                starters = starting_macs.get(t_new, [])
                for new_mac in starters:
                    if new_mac in used_as_new or new_mac in used_as_old:
                        continue
                    new_dtype = mac_device.get(new_mac)
                    if new_dtype != old_dtype:
                        continue

                    new_rssi_val = slot_rssi.get(t_new, {}).get(new_mac)
                    if new_rssi_val is None:
                        continue
                    diff = abs(old_rssi_val - new_rssi_val)
                    if diff > rssi_thresh:
                        continue

                    # 비용: RSSI 차이(정규화) + 시간 갭(정규화)
                    cost = 0.6 * (diff / max(rssi_thresh, 1)) + 0.4 * (gap / gap_max)
                    if cost < best_cost:
                        best_cost = cost
                        best_new = new_mac

            if best_new is not None:
                swap_map[old_mac] = best_new
                used_as_old.add(old_mac)
                used_as_new.add(best_new)

    return swap_map


def build_stitch_chains(swap_map: Dict[str, str]) -> Dict[str, str]:
    """
    swap_map에서 연쇄 교체를 처리한다.

    예: A→B, B→C 모두 탐지된 경우 → A, B, C 모두 root A로 매핑.

    Union-Find 방식으로 구현.

    Parameters
    ----------
    swap_map : {old_mac: new_mac}

    Returns
    -------
    canonical_map : {any_mac: root_mac}
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])   # path compression
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra   # b의 root를 a의 root로 합침

    for old_mac, new_mac in swap_map.items():
        union(old_mac, new_mac)

    # 모든 관련 MAC에 대해 canonical root 매핑
    all_macs = set(swap_map.keys()) | set(swap_map.values())
    return {mac: find(mac) for mac in all_macs}


def stitch_visitor_sessions(
    sessions_df: pd.DataFrame,
    swap_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Visitor 세션에 MAC stitching을 적용한다.

    동일 canonical MAC을 가진 세션들을 date별로 merge:
      - entry_time_index: 최솟값
      - exit_time_index: 최댓값
      - dwell_seconds: (exit - entry) * TIME_UNIT_SECONDS 재계산
      - device_type: 첫 번째 세션 값
      - hour: entry_time_index 기준 재계산 (0–23 clamp)

    추가 컬럼:
      - stitched: bool — merge가 일어난 세션이면 True
      - raw_session_count: int — merge된 원본 세션 수
      - original_macs: str — ","로 join된 원본 MAC 목록

    Parameters
    ----------
    sessions_df : build_sessions()가 반환한 원본 Visitor 세션 DataFrame
    swap_map : detect_mac_swaps()가 반환한 {old_mac: new_mac}

    Returns
    -------
    새 DataFrame (원본 수정 없음)
    """
    if sessions_df.empty or not swap_map:
        out = sessions_df.copy()
        out["stitched"] = False
        out["raw_session_count"] = 1
        out["original_macs"] = out["mac_address"].astype(str)
        return out

    canonical_map = build_stitch_chains(swap_map)
    df = sessions_df.copy()

    # canonical MAC 컬럼 추가
    df["canonical_mac"] = df["mac_address"].map(canonical_map).fillna(df["mac_address"])

    # date 컬럼이 없으면 임시 생성
    has_date = "date" in df.columns
    if not has_date:
        df["date"] = "unknown"

    group_keys = ["canonical_mac", "date"]
    rows = []

    for (canon, date), grp in df.groupby(group_keys, sort=False):
        entry_ti = int(grp["entry_time_index"].min())
        exit_ti  = int(grp["exit_time_index"].max())
        dwell    = (exit_ti - entry_ti) * TIME_UNIT_SECONDS
        hour     = min(23, max(0, entry_ti * TIME_UNIT_SECONDS // 3600))
        n        = len(grp)

        row: dict = {
            "mac_address":       canon,
            "device_type":       int(grp["device_type"].iloc[0]),
            "entry_time_index":  entry_ti,
            "exit_time_index":   exit_ti,
            "dwell_seconds":     dwell,
            "hour":              hour,
            "date":              date,
            "stitched":          n > 1,
            "raw_session_count": n,
            "original_macs":     ",".join(grp["mac_address"].astype(str).unique()),
        }
        # 원본에 entry_time / exit_time(ISO 문자열) 이 있으면 첫 번째 값 유지
        for col in ("entry_time", "exit_time"):
            if col in grp.columns:
                row[col] = grp[col].iloc[0]

        rows.append(row)

    if not rows:
        out = sessions_df.copy()
        out["stitched"] = False
        out["raw_session_count"] = 1
        out["original_macs"] = out["mac_address"].astype(str)
        return out

    result = pd.DataFrame(rows)

    # date 컬럼이 원래 없었으면 제거
    if not has_date:
        result = result.drop(columns=["date"], errors="ignore")

    return result


def stitching_summary(
    original_df: pd.DataFrame,
    stitched_df: pd.DataFrame,
) -> dict:
    """
    Stitching 전후 주요 지표 비교.

    Parameters
    ----------
    original_df : stitching 전 원본 세션 DataFrame
    stitched_df : stitching 적용 후 세션 DataFrame

    Returns
    -------
    dict with keys:
      sessions_before       : 원본 세션 수
      sessions_after        : stitching 후 세션 수
      sessions_merged       : 실제로 merge된 원본 세션 수 (stitched=True인 것들의 raw_session_count 합)
      dwell_mean_before_min : 원본 평균 체류시간 (분)
      dwell_mean_after_min  : stitching 후 평균 체류시간 (분)
      dwell_gain_pct        : 체류시간 개선률 (%)
      visitor_reduction_pct : 방문자 수 감소율 (%)
    """
    n_before = len(original_df)
    n_after  = len(stitched_df)

    dwell_col = "dwell_seconds"
    dwell_before = (
        float(original_df[dwell_col].mean()) / 60
        if n_before and dwell_col in original_df.columns else 0.0
    )
    dwell_after = (
        float(stitched_df[dwell_col].mean()) / 60
        if n_after and dwell_col in stitched_df.columns else 0.0
    )

    sessions_merged = 0
    if "stitched" in stitched_df.columns and "raw_session_count" in stitched_df.columns:
        merged_rows = stitched_df[stitched_df["stitched"] == True]
        sessions_merged = int(merged_rows["raw_session_count"].sum()) if not merged_rows.empty else 0

    dwell_gain_pct = (
        (dwell_after - dwell_before) / dwell_before * 100
        if dwell_before > 0 else 0.0
    )
    visitor_reduction_pct = (
        (n_before - n_after) / n_before * 100
        if n_before > 0 else 0.0
    )

    return {
        "sessions_before":        n_before,
        "sessions_after":         n_after,
        "sessions_merged":        sessions_merged,
        "dwell_mean_before_min":  round(dwell_before, 1),
        "dwell_mean_after_min":   round(dwell_after, 1),
        "dwell_gain_pct":         round(dwell_gain_pct, 1),
        "visitor_reduction_pct":  round(visitor_reduction_pct, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Level 2: Session-Level Post-Hoc Stitching
# ═══════════════════════════════════════════════════════════════════

def stitch_sessions_posthoc(
    sessions_df: pd.DataFrame,
    gap_sec_apple: int = SESSION_STITCH_GAP_SEC_APPLE,
    gap_sec_android: int = SESSION_STITCH_GAP_SEC_ANDROID,
    short_threshold: int = SESSION_STITCH_SHORT_THRESHOLD,
    rssi_diff_max: float = SESSION_STITCH_RSSI_DIFF,
    max_chain: int = SESSION_STITCH_MAX_CHAIN,
) -> pd.DataFrame:
    """
    Level 2: 세션 레벨에서 MAC 교체를 추가 탐지하여 병합.

    Level 1이 raw signal 레벨에서 놓친 교체를 세션의 시간적 인접성으로 찾는다.
    "세션 A 종료 → 짧은 갭 → 세션 B 시작" 패턴을 병합.

    매칭 조건 (모두 충족):
      1. 동일 device_type
      2. 세션 간 갭 ≤ gap_sec_apple(Apple) / gap_sec_android(Android)
      3. 최소 하나가 단편 세션 (short_threshold 미만)
      4. 평균 RSSI 차이 ≤ rssi_diff_max
      5. 갭 구간에 동일 device_type의 다른 활성 세션이 없어야 함 (배타성)
    """
    if sessions_df.empty:
        return sessions_df.copy()

    required_cols = {"mac_address", "device_type", "entry_time_index", "exit_time_index", "dwell_seconds"}
    if not required_cols.issubset(sessions_df.columns):
        return sessions_df.copy()

    df = sessions_df.copy()

    # 평균 RSSI가 없으면 계산 불가 → 그냥 반환
    has_rssi = "avg_rssi" in df.columns
    if not has_rssi:
        # avg_rssi가 없으면 RSSI 조건 무시하고 시간 기반만 사용
        df["avg_rssi"] = 0.0

    # 날짜별 독립 처리
    if "date" not in df.columns:
        df["date"] = "unknown"

    result_dfs = []

    for date_val, day_df in df.groupby("date", sort=False):
        merged_map = _posthoc_match_day(
            day_df, gap_sec_apple, gap_sec_android,
            short_threshold, rssi_diff_max, max_chain, has_rssi,
        )
        if not merged_map:
            result_dfs.append(day_df)
            continue

        # Union-Find로 체인 처리
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        for old_mac, new_mac in merged_map.items():
            ra, rb = find(old_mac), find(new_mac)
            if ra != rb:
                parent[rb] = ra

        # canonical MAC 매핑
        all_macs = set(merged_map.keys()) | set(merged_map.values())
        canonical = {m: find(m) for m in all_macs}

        day_df = day_df.copy()
        day_df["_canon"] = day_df["mac_address"].map(canonical).fillna(day_df["mac_address"])

        merged_rows = []
        for canon, grp in day_df.groupby("_canon", sort=False):
            if len(grp) == 1:
                row = grp.iloc[0].to_dict()
                row.setdefault("l2_stitched", False)
                row.setdefault("l2_session_count", 1)
                merged_rows.append(row)
                continue

            entry_ti = int(grp["entry_time_index"].min())
            exit_ti = int(grp["exit_time_index"].max())
            dwell = (exit_ti - entry_ti) * TIME_UNIT_SECONDS
            hour = min(23, max(0, entry_ti * TIME_UNIT_SECONDS // 3600))

            row = {
                "mac_address": canon,
                "device_type": int(grp["device_type"].iloc[0]),
                "entry_time_index": entry_ti,
                "exit_time_index": exit_ti,
                "dwell_seconds": dwell,
                "hour": hour,
                "date": date_val,
                "l2_stitched": True,
                "l2_session_count": len(grp),
            }
            # 기존 stitching 정보 보존
            if "stitched" in grp.columns:
                row["stitched"] = bool(grp["stitched"].any())
            if "raw_session_count" in grp.columns:
                row["raw_session_count"] = int(grp["raw_session_count"].sum())
            if "original_macs" in grp.columns:
                all_orig = ",".join(grp["original_macs"].dropna().astype(str))
                row["original_macs"] = all_orig
            if "avg_rssi" in grp.columns:
                row["avg_rssi"] = float(grp["avg_rssi"].mean())
            for col in ("entry_time", "exit_time"):
                if col in grp.columns:
                    row[col] = grp.sort_values("entry_time_index").iloc[0 if col == "entry_time" else -1][col]

            merged_rows.append(row)

        result_dfs.append(pd.DataFrame(merged_rows))

    if not result_dfs:
        return sessions_df.copy()

    result = pd.concat(result_dfs, ignore_index=True)
    # _canon 컬럼 정리
    result = result.drop(columns=["_canon"], errors="ignore")

    # l2 플래그가 없는 행 채우기
    if "l2_stitched" not in result.columns:
        result["l2_stitched"] = False
    if "l2_session_count" not in result.columns:
        result["l2_session_count"] = 1
    result["l2_stitched"] = result["l2_stitched"].fillna(False)
    result["l2_session_count"] = result["l2_session_count"].fillna(1).astype(int)

    if not has_rssi:
        result = result.drop(columns=["avg_rssi"], errors="ignore")

    return result


def _posthoc_match_day(
    day_df: pd.DataFrame,
    gap_sec_apple: int,
    gap_sec_android: int,
    short_threshold: int,
    rssi_diff_max: float,
    max_chain: int,
    use_rssi: bool,
) -> Dict[str, str]:
    """하루 세션에서 post-hoc 매칭 수행. {old_mac: new_mac} 반환."""
    merge_map: Dict[str, str] = {}

    for dtype in (DEVICE_TYPE_APPLE, DEVICE_TYPE_ANDROID):
        gap_max_sec = gap_sec_apple if dtype == DEVICE_TYPE_APPLE else gap_sec_android
        dtype_df = day_df[day_df["device_type"] == dtype].copy()
        if len(dtype_df) < 2:
            continue

        dtype_df = dtype_df.sort_values("entry_time_index").reset_index(drop=True)

        # Pre-extract arrays for vectorized access
        _exit_sec = (dtype_df["exit_time_index"].values * TIME_UNIT_SECONDS).astype(int)
        _entry_sec = (dtype_df["entry_time_index"].values * TIME_UNIT_SECONDS).astype(int)
        _dwell = dtype_df["dwell_seconds"].values.astype(float)
        _rssi = dtype_df["avg_rssi"].values.astype(float) if "avg_rssi" in dtype_df.columns else np.zeros(len(dtype_df))
        _exit_ti = dtype_df["exit_time_index"].values.astype(int)
        _entry_ti = dtype_df["entry_time_index"].values.astype(int)
        n_sess = len(dtype_df)

        # 비용 매트릭스: (종료 세션, 시작 세션) 쌍
        candidates: List[Tuple[float, int, int]] = []  # (cost, idx_a, idx_b)

        for i in range(n_sess):
            exit_sec_a = _exit_sec[i]
            dwell_a = _dwell[i]
            rssi_a = _rssi[i]

            for j in range(i + 1, n_sess):
                gap_sec = _entry_sec[j] - exit_sec_a

                if gap_sec <= 0:
                    continue
                if gap_sec > gap_max_sec:
                    break  # sorted by entry_time, 이후는 더 멀어짐

                dwell_b = _dwell[j]

                # 조건 3: 최소 하나가 단편 세션
                if dwell_a >= short_threshold and dwell_b >= short_threshold:
                    continue

                # 조건 4: RSSI 유사성
                if use_rssi:
                    rssi_diff = abs(rssi_a - _rssi[j])
                    if rssi_diff > rssi_diff_max:
                        continue
                else:
                    rssi_diff = 0.0

                # 조건 5: 배타성 — 갭 구간에 다른 활성 세션 없어야 함 (벡터화)
                exit_ti_a = _exit_ti[i]
                entry_ti_b = _entry_ti[j]
                # 세션 k가 갭 구간에 겹치려면: entry < entry_ti_b AND exit > exit_ti_a
                overlap = bool(np.any(
                    (_entry_ti < entry_ti_b) & (_exit_ti > exit_ti_a)
                    & (np.arange(n_sess) != i) & (np.arange(n_sess) != j)
                ))
                if overlap:
                    continue

                # 비용 계산
                cost = (
                    0.4 * (gap_sec / max(gap_max_sec, 1))
                    + 0.4 * (rssi_diff / max(rssi_diff_max, 1))
                )
                # 양쪽 모두 단편이면 보너스
                if dwell_a < short_threshold and dwell_b < short_threshold:
                    cost -= 0.2

                candidates.append((cost, i, j))

        # Greedy 매칭: 최저 비용부터
        candidates.sort()
        used_i: set = set()
        used_j: set = set()
        _macs = dtype_df["mac_address"].values

        for cost, i, j in candidates:
            if i in used_i or j in used_j:
                continue
            merge_map[_macs[i]] = _macs[j]
            used_i.add(i)
            used_j.add(j)

    return merge_map


# ═══════════════════════════════════════════════════════════════════
# Level 3: 평가 프레임워크
# ═══════════════════════════════════════════════════════════════════

def stitching_evaluation(
    raw_sessions: pd.DataFrame,
    l1_sessions: pd.DataFrame,
    l2_sessions: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Raw → Level 1 → Level 2 3단계 비교 리포트.

    Returns dict with stage-by-stage metrics.
    """
    def _stage_metrics(sdf: pd.DataFrame, label: str) -> Dict[str, Any]:
        if sdf.empty:
            return {f"{label}_sessions": 0}
        dwell = sdf["dwell_seconds"] if "dwell_seconds" in sdf.columns else pd.Series(dtype=float)
        n = len(sdf)
        apple_n = int((sdf["device_type"] == DEVICE_TYPE_APPLE).sum()) if "device_type" in sdf.columns else 0
        android_n = int((sdf["device_type"] == DEVICE_TYPE_ANDROID).sum()) if "device_type" in sdf.columns else 0

        from src.config.constants import DWELL_SHORT_MAX, DWELL_MEDIUM_MAX
        short = int((dwell < DWELL_SHORT_MAX).sum()) if not dwell.empty else 0
        medium = int(((dwell >= DWELL_SHORT_MAX) & (dwell < DWELL_MEDIUM_MAX)).sum()) if not dwell.empty else 0
        long_ = int((dwell >= DWELL_MEDIUM_MAX).sum()) if not dwell.empty else 0

        return {
            f"{label}_sessions": n,
            f"{label}_dwell_mean": round(float(dwell.mean()), 1) if not dwell.empty else 0.0,
            f"{label}_dwell_median": round(float(dwell.median()), 1) if not dwell.empty else 0.0,
            f"{label}_dwell_p90": round(float(dwell.quantile(0.9)), 1) if not dwell.empty else 0.0,
            f"{label}_apple": apple_n,
            f"{label}_android": android_n,
            f"{label}_short": short,
            f"{label}_medium": medium,
            f"{label}_long": long_,
        }

    raw_m = _stage_metrics(raw_sessions, "raw")
    l1_m = _stage_metrics(l1_sessions, "l1")
    l2_m = _stage_metrics(l2_sessions, "l2")

    # 병합률 계산
    raw_n = raw_m.get("raw_sessions", 0)
    l1_n = l1_m.get("l1_sessions", 0)
    l2_n = l2_m.get("l2_sessions", 0)

    result = {**raw_m, **l1_m, **l2_m}
    result["l1_merge_pct"] = round((raw_n - l1_n) / raw_n * 100, 1) if raw_n else 0.0
    result["l2_merge_pct"] = round((l1_n - l2_n) / l1_n * 100, 1) if l1_n else 0.0
    result["total_merge_pct"] = round((raw_n - l2_n) / raw_n * 100, 1) if raw_n else 0.0

    # 체류시간 개선
    raw_dwell = raw_m.get("raw_dwell_mean", 0)
    l2_dwell = l2_m.get("l2_dwell_mean", 0)
    result["dwell_improvement_pct"] = (
        round((l2_dwell - raw_dwell) / raw_dwell * 100, 1) if raw_dwell else 0.0
    )

    return result


def stitching_daily_summary(
    raw_sessions: pd.DataFrame,
    l1_sessions: pd.DataFrame,
    l2_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    날짜별 Raw → L1 → L2 비교 테이블.
    """
    def _daily_agg(sdf: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if sdf.empty or "date" not in sdf.columns:
            return pd.DataFrame()
        grp = sdf.groupby("date").agg(
            sessions=("mac_address", "count"),
            dwell_mean=("dwell_seconds", "mean"),
            dwell_median=("dwell_seconds", "median"),
        ).reset_index()
        grp.columns = ["date", f"{prefix}_sessions", f"{prefix}_dwell_mean", f"{prefix}_dwell_median"]
        grp[f"{prefix}_dwell_mean"] = grp[f"{prefix}_dwell_mean"].round(1)
        grp[f"{prefix}_dwell_median"] = grp[f"{prefix}_dwell_median"].round(1)
        return grp

    raw_d = _daily_agg(raw_sessions, "raw")
    l1_d = _daily_agg(l1_sessions, "l1")
    l2_d = _daily_agg(l2_sessions, "l2")

    if raw_d.empty:
        return pd.DataFrame()

    result = raw_d.copy()
    if not l1_d.empty:
        result = result.merge(l1_d, on="date", how="left")
    if not l2_d.empty:
        result = result.merge(l2_d, on="date", how="left")

    return result
