"""
Hermes BLE Session Analysis Engine.

- Floating: Entrance sensor, RSSI >= threshold (config). Unique MAC count.
- Visitor: Inside sensor only. Strict Entry = "latest 1 min" (current + previous 5 time_index = 6 slots)
  with (1) count >= MIN_HITS_PER_MIN, (2) all RSSI >= -80.
- Session: Hysteresis buffer (Apple 180s=18 slots, Android 120s=12 slots). Exit time = back-dated to last signal.
- Timezone: Asia/Seoul (UTC+9). time_index: 10 sec units (1 min = 6 slots).
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.constants import (
    TIME_UNIT_SECONDS,
    SLOTS_PER_MINUTE,
    MIN_HITS_PER_MIN,
    EXIT_BUFFER_SEC,
    APPLE_EXIT_BUFFER_SEC,
    DEFAULT_RSSI_THRESHOLD,
    FALLBACK_FLOATING_RSSI,
    ANDROID_RSSI_OFFSET,
    DEVICE_TYPE_APPLE,
    DEVICE_TYPE_ANDROID,
    INSTALL_INSIDE,
    INSTALL_ENTRANCE,
    SECONDS_PER_HOUR,
    MINUTES_PER_DAY,
    STORE_OPEN_HOUR,
    STORE_CLOSE_HOUR,
)


class HermesBLEEngine:
    """
    S-Ward config–driven BLE engine: floating population, strict visitor entry, session with hysteresis & back-dating.
    """

    def __init__(
        self,
        sward_config: pd.DataFrame,
        rssi_threshold_override: Optional[int] = None,
        store_open_hour: int = STORE_OPEN_HOUR,
        store_close_hour: int = STORE_CLOSE_HOUR,
        min_hits_per_min: int = MIN_HITS_PER_MIN,
        min_dwell_seconds: int = 60,
        rssi_pass_ratio: float = 0.80,
    ):
        self.sward_config = sward_config.copy()
        self.sward_config["sward_name"] = self.sward_config["sward_name"].astype(str).str.strip()

        # Store operating hours (for CVR calculation)
        self.store_open_hour = store_open_hour
        self.store_close_hour = store_close_hour
        self.min_hits_per_min = min_hits_per_min
        self.min_dwell_seconds = min_dwell_seconds
        self.rssi_pass_ratio = rssi_pass_ratio

        self.entrance_swards: List[str] = self.sward_config[
            self.sward_config["install_location"] == INSTALL_ENTRANCE
        ]["sward_name"].tolist()
        self.inside_swards: List[str] = self.sward_config[
            self.sward_config["install_location"] == INSTALL_INSIDE
        ]["sward_name"].tolist()

        self._rssi_by_sward: Dict[str, int] = dict(zip(
            self.sward_config["sward_name"],
            self.sward_config.get("rssi_threshold", pd.Series(DEFAULT_RSSI_THRESHOLD, index=self.sward_config.index)).astype(int),
        ))
        if rssi_threshold_override is not None:
            for k in self._rssi_by_sward:
                self._rssi_by_sward[k] = rssi_threshold_override

        self.inside_rssi = (
            int(self.sward_config[self.sward_config["install_location"] == INSTALL_INSIDE]["rssi_threshold"].iloc[0])
            if len(self.inside_swards) > 0
            else DEFAULT_RSSI_THRESHOLD
        )
        # Android는 Apple보다 신호가 약하므로 별도 임계값 적용 (Apple 기준 + ANDROID_RSSI_OFFSET)
        self.inside_rssi_android = self.inside_rssi + ANDROID_RSSI_OFFSET
        if rssi_threshold_override is not None:
            self.inside_rssi = rssi_threshold_override
            self.inside_rssi_android = rssi_threshold_override + ANDROID_RSSI_OFFSET

        if self.entrance_swards:
            self.floating_rssi = self._rssi_by_sward.get(
                self.entrance_swards[0], DEFAULT_RSSI_THRESHOLD
            )
        else:
            self.floating_rssi = FALLBACK_FLOATING_RSSI
        if rssi_threshold_override is not None:
            self.floating_rssi = rssi_threshold_override

    def _exit_buffer_slots(self, device_type: int) -> int:
        """Exit buffer in time_index units (Apple 18, Android 12)."""
        sec = APPLE_EXIT_BUFFER_SEC if device_type == DEVICE_TYPE_APPLE else EXIT_BUFFER_SEC
        return sec // TIME_UNIT_SECONDS

    def floating_per_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        WARNING — DO NOT use for hourly aggregation.
        This method returns one row per time_index (10 s), each containing
        the unique-MAC count inside a 1-min rolling window.  Summing those
        rows over an hour inflates the count by up to 360×, which causes
        hourly CVR to appear near-zero.  This was a confirmed bug (fixed).

        For hourly floating population, use the direct approach instead:
            fp_raw.groupby("hour")["mac_address"].nunique()
        which is now done inside run_daily().

        This method is preserved for future per-time-index / real-time
        streaming use cases only.
        ---
        Per–time_index floating count (1-min window: [max(1,t-5), t] unique MAC).
        """
        if df.empty:
            return pd.DataFrame(columns=["time_index", "floating_count"])

        if self.entrance_swards:
            use_df = df[df["sward_name"].isin(self.entrance_swards)].copy()
            th = self.floating_rssi
        else:
            use_df = df[df["sward_name"].isin(self.inside_swards)].copy()
            th = FALLBACK_FLOATING_RSSI
        use_df = use_df[use_df["rssi"] >= th]

        ti_min = use_df["time_index"].min()
        ti_max = use_df["time_index"].max()
        rows = []
        for t in range(int(ti_min), int(ti_max) + 1):
            start = max(1, t - (SLOTS_PER_MINUTE - 1))
            window = use_df[(use_df["time_index"] >= start) & (use_df["time_index"] <= t)]
            rows.append({"time_index": t, "floating_count": window["mac_address"].nunique()})
        return pd.DataFrame(rows)

    def floating_daily_unique(self, df: pd.DataFrame) -> int:
        """Daily floating population: unique MACs from ALL sensors (entrance + inside).

        FP = 유동인구 = 매장 근처에서 감지된 모든 unique MAC.
        - entrance 센서: 매장 앞 지나가는 사람 + 들어오는 사람
        - inside 센서: 매장 안에 들어온 사람

        FP는 반드시 Visitors를 포함해야 하므로 (CVR = Visitors / FP ≤ 100%),
        entrance만이 아니라 **모든 센서**에서 감지된 MAC을 합산한다.

        각 센서별 RSSI 임계값 적용:
        - entrance: self.floating_rssi
        - inside: FALLBACK_FLOATING_RSSI (-90dBm, 더 관대한 기준)

        Note: store_open_hour, store_close_hour은 인스턴스 생성 시 지정.
              24시간 영업(open=0, close=24)이면 필터링 없음.
        """
        if df.empty:
            return 0

        # ── Entrance 센서 MAC 수집 ─────────────────────────────────────────
        entrance_macs = set()
        if self.entrance_swards:
            ent_df = df[df["sward_name"].isin(self.entrance_swards)]
            ent_df = ent_df[ent_df["rssi"] >= self.floating_rssi]
            if not (self.store_open_hour == 0 and self.store_close_hour == 24):
                hour_s = (ent_df["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int)
                ent_df = ent_df[(hour_s >= self.store_open_hour) & (hour_s < self.store_close_hour)]
            entrance_macs = set(ent_df["mac_address"].unique())

        # ── Inside 센서 MAC 수집 ───────────────────────────────────────────
        inside_macs = set()
        if self.inside_swards:
            ins_df = df[df["sward_name"].isin(self.inside_swards)]
            ins_df = ins_df[ins_df["rssi"] >= FALLBACK_FLOATING_RSSI]
            if not (self.store_open_hour == 0 and self.store_close_hour == 24):
                hour_s = (ins_df["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int)
                ins_df = ins_df[(hour_s >= self.store_open_hour) & (hour_s < self.store_close_hour)]
            inside_macs = set(ins_df["mac_address"].unique())

        # ── 합집합 = 전체 유동인구 ─────────────────────────────────────────
        all_macs = entrance_macs | inside_macs
        return len(all_macs)

    def build_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        3-Stage Visitor Detection (inspired by TheHyundaiSeoul).

        Stage 1 — Session Construction (all signals, RSSI agnostic):
          MAC별로 inside 센서의 모든 신호를 시간순 정렬.
          갭 > hysteresis buffer이면 세션 분리.
          exit_time_index = 마지막 실제 신호 (back-dating).

        Stage 2 — Minimum Dwell Filter:
          세션 체류 시간 ≥ min_dwell_seconds 이상만 유효.

        Stage 3 — RSSI Pass-Ratio Filter:
          세션 내 전체 신호 중 rssi_pass_ratio (80%) 이상이
          디바이스별 임계값(Apple -75 / Android -85)을 통과해야 방문자 인정.

        This replaces the previous "Strict Entry" 1-min-window approach.
        """
        inside = df[df["sward_name"].isin(self.inside_swards)].copy()
        if inside.empty:
            return pd.DataFrame(columns=[
                "mac_address", "device_type", "entry_time_index", "exit_time_index",
                "dwell_seconds",
            ])

        # ── 영업시간 필터 (24시간 영업이 아닌 경우) ────────────────────────
        if not (self.store_open_hour == 0 and self.store_close_hour == 24):
            hour_s = (inside["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int)
            inside = inside[(hour_s >= self.store_open_hour) & (hour_s < self.store_close_hour)]
            if inside.empty:
                return pd.DataFrame(columns=[
                    "mac_address", "device_type", "entry_time_index", "exit_time_index",
                    "dwell_seconds",
                ])

        if "device_type" not in inside.columns and "type" in inside.columns:
            inside["device_type"] = inside["type"]

        type_col = "device_type"

        sessions = []
        for mac, mac_group in inside.groupby("mac_address"):
            mac_group = mac_group.sort_values("time_index")
            device_type = int(mac_group[type_col].iloc[0])
            buffer_slots = self._exit_buffer_slots(device_type)

            # RSSI 임계값 (디바이스별)
            rssi_th = (
                self.inside_rssi_android
                if device_type == DEVICE_TYPE_ANDROID
                else self.inside_rssi
            )

            ti = mac_group["time_index"].values
            rssi_vals = mac_group["rssi"].values

            # ── Stage 1: Session construction (gap-based split) ──────────
            seg_start = 0
            for i in range(1, len(ti)):
                if int(ti[i]) > int(ti[i - 1]) + buffer_slots:
                    # Close previous segment
                    self._evaluate_and_add_session(
                        sessions, mac, device_type, rssi_th,
                        ti[seg_start:i], rssi_vals[seg_start:i],
                    )
                    seg_start = i
            # Last segment
            self._evaluate_and_add_session(
                sessions, mac, device_type, rssi_th,
                ti[seg_start:], rssi_vals[seg_start:],
            )

        return pd.DataFrame(sessions) if sessions else pd.DataFrame(columns=[
            "mac_address", "device_type", "entry_time_index", "exit_time_index",
            "dwell_seconds",
        ])

    def _evaluate_and_add_session(
        self,
        sessions: list,
        mac: str,
        device_type: int,
        rssi_th: int,
        ti_arr: np.ndarray,
        rssi_arr: np.ndarray,
    ) -> None:
        """Stage 2 + 3: minimum dwell + RSSI pass-ratio filter."""
        if len(ti_arr) == 0:
            return

        entry_ti = int(ti_arr[0])
        exit_ti = int(ti_arr[-1])
        dwell_sec = (exit_ti - entry_ti) * TIME_UNIT_SECONDS

        # ── Stage 2: Minimum dwell ───────────────────────────────────────
        if dwell_sec < self.min_dwell_seconds:
            return

        # ── Stage 3: RSSI pass-ratio ─────────────────────────────────────
        total_signals = len(rssi_arr)
        pass_count = int(np.sum(rssi_arr >= rssi_th))
        pass_ratio = pass_count / total_signals if total_signals > 0 else 0

        if pass_ratio < self.rssi_pass_ratio:
            return

        sessions.append({
            "mac_address": mac,
            "device_type": device_type,
            "entry_time_index": entry_ti,
            "exit_time_index": exit_ti,
            "dwell_seconds": dwell_sec,
        })

    def run_daily(self, df: pd.DataFrame) -> Dict:
        """
        Run full daily pipeline: floating, strict-entry visitors, sessions, hourly aggregates.
        Returns floating_unique, visitor_count, conversion_rate, dwell_seconds_mean,
        sessions_df, hourly_floating, hourly_visitor.
        """
        if df.empty:
            return {
                "floating_unique": 0,
                "visitor_count": 0,
                "conversion_rate": 0.0,
                "dwell_seconds_mean": 0.0,
                "sessions_df": pd.DataFrame(),
                "hourly_floating": pd.DataFrame(),
                "hourly_visitor": pd.DataFrame(),
            }

        floating_unique = self.floating_daily_unique(df)
        sessions_df = self.build_sessions(df)
        visitor_count = len(sessions_df)
        cvr = (visitor_count / floating_unique * 100.0) if floating_unique > 0 else 0.0
        dwell_mean = float(sessions_df["dwell_seconds"].mean()) if len(sessions_df) > 0 else 0.0

        # Hourly floating: unique MACs per hour from ALL sensors (entrance + inside).
        # Same logic as floating_daily_unique() but grouped by hour.
        fp_parts = []
        if self.entrance_swards:
            ent = df[df["sward_name"].isin(self.entrance_swards)].copy()
            ent = ent[ent["rssi"] >= self.floating_rssi]
            if not ent.empty:
                ent["hour"] = (ent["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int).clip(0, 23)
                fp_parts.append(ent[["hour", "mac_address"]])
        if self.inside_swards:
            ins = df[df["sward_name"].isin(self.inside_swards)].copy()
            ins = ins[ins["rssi"] >= FALLBACK_FLOATING_RSSI]
            if not ins.empty:
                ins["hour"] = (ins["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int).clip(0, 23)
                fp_parts.append(ins[["hour", "mac_address"]])

        if fp_parts:
            fp_all = pd.concat(fp_parts, ignore_index=True)
            hourly_floating = (
                fp_all.groupby("hour")["mac_address"].nunique()
                .reset_index()
                .rename(columns={"mac_address": "floating_count"})
            )
        else:
            hourly_floating = pd.DataFrame(columns=["hour", "floating_count"])

        if not sessions_df.empty:
            sessions_df = sessions_df.copy()
            # Use entry_time_index: "when did the visitor arrive?" (not when they left)
            sessions_df["hour"] = (
                sessions_df["entry_time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR
            ).astype(int).clip(0, 23)
            hourly_visitor = sessions_df.groupby("hour", as_index=False).size()
            hourly_visitor.columns = ["hour", "visitor_count"]
        else:
            hourly_visitor = pd.DataFrame(columns=["hour", "visitor_count"])

        return {
            "floating_unique": floating_unique,
            "visitor_count": visitor_count,
            "conversion_rate": cvr,
            "dwell_seconds_mean": dwell_mean,
            "sessions_df": sessions_df,
            "hourly_floating": hourly_floating,
            "hourly_visitor": hourly_visitor,
        }

    def build_timeseries(
        self, df: pd.DataFrame, sessions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Per-minute time series for one day (1 440 rows, one per minute 0-1439).

        Columns
        -------
        minute          : int 0-1439
        floating_count  : unique MAC addresses detected at entrance/inside per minute
                          (no rolling window — simply unique MACs whose any signal
                           falls within that minute's 6 time_index slots)
        active_visitors : number of visitor sessions that are "in store" at this minute
                          (entry_minute <= minute <= exit_minute, i.e. occupancy)

        Notes
        -----
        - floating_count is NOT the rolling-window count used for daily/hourly CVR.
          It is a raw per-minute snapshot for visualisation only.
        - active_visitors shows occupancy (how many people are present), not arrivals.
        """
        minutes_range = range(MINUTES_PER_DAY)

        # ── Floating population: unique MACs per minute (all sensors) ────────
        fp_parts = []
        if self.entrance_swards:
            ent = df[
                df["sward_name"].isin(self.entrance_swards)
                & (df["rssi"] >= self.floating_rssi)
            ]
            if not ent.empty:
                fp_parts.append(ent[["time_index", "mac_address"]])
        if self.inside_swards:
            ins = df[
                df["sward_name"].isin(self.inside_swards)
                & (df["rssi"] >= FALLBACK_FLOATING_RSSI)
            ]
            if not ins.empty:
                fp_parts.append(ins[["time_index", "mac_address"]])

        fp_raw = pd.concat(fp_parts, ignore_index=True) if fp_parts else pd.DataFrame()

        if not fp_raw.empty:
            fp_raw["minute"] = (
                (fp_raw["time_index"] * TIME_UNIT_SECONDS) // 60
            ).clip(0, MINUTES_PER_DAY - 1).astype(int)
            floating_by_min = (
                fp_raw.groupby("minute")["mac_address"]
                .nunique()
                .reindex(minutes_range, fill_value=0)
            )
        else:
            floating_by_min = pd.Series(0, index=minutes_range)

        # ── Active visitors: occupancy per minute ────────────────────────────
        active_arr = np.zeros(MINUTES_PER_DAY, dtype=np.int32)
        if not sessions_df.empty and "entry_time_index" in sessions_df.columns:
            entry_mins = (
                (sessions_df["entry_time_index"] * TIME_UNIT_SECONDS / 60)
                .astype(int)
                .clip(0, MINUTES_PER_DAY - 1)
                .values
            )
            exit_mins = (
                (sessions_df["exit_time_index"] * TIME_UNIT_SECONDS / 60)
                .astype(int)
                .clip(0, MINUTES_PER_DAY - 1)
                .values
            )
            for em, xm in zip(entry_mins, exit_mins):
                active_arr[em : xm + 1] += 1

        return pd.DataFrame({
            "minute": list(minutes_range),
            "floating_count": floating_by_min.values,
            "active_visitors": active_arr,
        })
