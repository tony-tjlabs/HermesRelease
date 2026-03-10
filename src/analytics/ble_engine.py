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
    ):
        self.sward_config = sward_config.copy()
        self.sward_config["sward_name"] = self.sward_config["sward_name"].astype(str).str.strip()

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
        """Daily floating: unique MAC during store operating hours (STORE_OPEN_HOUR ~ STORE_CLOSE_HOUR).

        CVR(방문율) = Visitors / FP 이므로, 영업 외 시간(10시 이전, 22시 이후) MAC은
        분모에서 제외해야 정확한 방문율을 계산할 수 있다.
        """
        if df.empty:
            return 0
        if self.entrance_swards:
            use_df = df[df["sward_name"].isin(self.entrance_swards)]
            th = self.floating_rssi
        else:
            use_df = df[df["sward_name"].isin(self.inside_swards)]
            th = FALLBACK_FLOATING_RSSI
        use_df = use_df[use_df["rssi"] >= th]
        # ── 운영 시간 필터: 10:00–22:00 (exclusive) ──────────────────────────
        hour_series = (use_df["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int)
        use_df = use_df[(hour_series >= STORE_OPEN_HOUR) & (hour_series < STORE_CLOSE_HOUR)]
        return use_df["mac_address"].nunique()

    def _visitor_strict_entry_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strict Entry (Hermes Standard): inside sensor only.
        For each MAC, at least one window of "latest 1 min" (current time_index + previous 5 = 6 slots)
        such that: (1) hit count >= MIN_HITS_PER_MIN, (2) all RSSI in that window >= threshold.
        Window for time_index t: [max(1, t-5), t] inclusive.

        RSSI threshold is device-type-aware:
          - Apple  (type=1) : self.inside_rssi          (sward_config 값 그대로)
          - Android (type=10): self.inside_rssi_android  (= inside_rssi + ANDROID_RSSI_OFFSET)
        """
        inside = df[df["sward_name"].isin(self.inside_swards)].copy()
        if inside.empty:
            return pd.DataFrame()

        candidates = set()
        type_col = "device_type" if "device_type" in inside.columns else "type"
        for mac, grp in inside.groupby("mac_address"):
            grp = grp.sort_values("time_index")
            ti   = grp["time_index"].values
            rssi = grp["rssi"].values

            # 디바이스 타입별 RSSI 임계값 분기
            device_type_val = (
                int(grp[type_col].iloc[0]) if type_col in grp.columns else DEVICE_TYPE_APPLE
            )
            th = (
                self.inside_rssi_android
                if device_type_val == DEVICE_TYPE_ANDROID
                else self.inside_rssi
            )

            for i in range(len(ti)):
                t = int(ti[i])
                start = max(1, t - (SLOTS_PER_MINUTE - 1))
                mask = (ti >= start) & (ti <= t)
                if mask.sum() >= MIN_HITS_PER_MIN and np.all(rssi[mask] >= th):
                    candidates.add(mac)
                    break
        return inside[inside["mac_address"].isin(candidates)]

    def build_sessions(
        self,
        df: pd.DataFrame,
        visitor_macs: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        Build visitor sessions from inside-sensor data for Strict Entry candidates only.

        Design intent — two-stage gate:
        ┌─ Stage 1: _visitor_strict_entry_candidates() ─────────────────────────┐
        │  A MAC must pass the Strict Entry test ONCE:                           │
        │    · ≥ MIN_HITS_PER_MIN (2) signals within any 1-min window           │
        │    · ALL those signals have RSSI ≥ inside_rssi (−80 dBm)             │
        │  Only MACs that pass this gate are handed to build_sessions().         │
        └────────────────────────────────────────────────────────────────────────┘
        ┌─ Stage 2: build_sessions() (this method) ──────────────────────────────┐
        │  Once admitted, RSSI is no longer checked — any subsequent signal      │
        │  from the same MAC (regardless of strength) extends the session        │
        │  via Hysteresis, until the buffer expires.                             │
        │                                                                        │
        │  Rationale: inside the store, signal strength fluctuates due to        │
        │  multipath, body blocking, and sensor range.  Requiring RSSI ≥ −80   │
        │  throughout would prematurely end valid sessions.  The Strict Entry    │
        │  gate already ensures the device is genuinely inside; Hysteresis        │
        │  handles the signal gaps that follow.                                  │
        └────────────────────────────────────────────────────────────────────────┘

        Session logic:
        - Hysteresis: session continues if re-detected within buffer
          (Apple 180 s = 18 slots, Android 120 s = 12 slots).
        - Back-dating: exit_time_index = last actual signal received
          (not the end of the Hysteresis buffer).
        - Multiple sessions per MAC: if a gap exceeds the buffer, a new
          session starts at the first signal after the gap.
        """
        inside = df[df["sward_name"].isin(self.inside_swards)].copy()
        if inside.empty:
            return pd.DataFrame(columns=[
                "mac_address", "device_type", "entry_time_index", "exit_time_index",
                "dwell_seconds",
            ])

        if visitor_macs is None:
            entry_candidates = self._visitor_strict_entry_candidates(df)
            visitor_macs = set(entry_candidates["mac_address"].unique())

        if "device_type" not in inside.columns and "type" in inside.columns:
            inside["device_type"] = inside["type"]

        sessions = []
        for mac in visitor_macs:
            mac_df = inside[inside["mac_address"] == mac].sort_values("time_index")
            if mac_df.empty:
                continue
            device_type = int(mac_df["device_type"].iloc[0])
            buffer_slots = self._exit_buffer_slots(device_type)

            ti = mac_df["time_index"].values
            entry_ti = int(ti[0])
            last_ti = int(ti[0])
            for t in ti[1:]:
                t = int(t)
                if t <= last_ti + buffer_slots:
                    last_ti = t
                else:
                    sessions.append({
                        "mac_address": mac,
                        "device_type": device_type,
                        "entry_time_index": entry_ti,
                        "exit_time_index": last_ti,
                        "dwell_seconds": (last_ti - entry_ti) * TIME_UNIT_SECONDS,
                    })
                    entry_ti = t
                    last_ti = t
            sessions.append({
                "mac_address": mac,
                "device_type": device_type,
                "entry_time_index": entry_ti,
                "exit_time_index": last_ti,
                "dwell_seconds": (last_ti - entry_ti) * TIME_UNIT_SECONDS,
            })

        return pd.DataFrame(sessions)

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
        entry_candidates = self._visitor_strict_entry_candidates(df)
        visitor_macs = set(entry_candidates["mac_address"].unique())
        sessions_df = self.build_sessions(df, visitor_macs=visitor_macs)
        visitor_count = len(sessions_df)
        cvr = (visitor_count / floating_unique * 100.0) if floating_unique > 0 else 0.0
        dwell_mean = float(sessions_df["dwell_seconds"].mean()) if len(sessions_df) > 0 else 0.0

        # Hourly floating: unique MACs per hour — must use nunique(), not sum() of rolling-window rows.
        # floating_per_time_index() produces one row per time_index with a 1-min rolling window count.
        # Summing those rows over an hour inflates the count by ~360x, making hourly CVR near zero.
        # Correct method: same filter as floating_daily_unique(), then nunique() per hour.
        if self.entrance_swards:
            fp_raw = df[df["sward_name"].isin(self.entrance_swards)].copy()
            fp_th = self.floating_rssi
        else:
            fp_raw = df[df["sward_name"].isin(self.inside_swards)].copy()
            fp_th = FALLBACK_FLOATING_RSSI
        fp_raw = fp_raw[fp_raw["rssi"] >= fp_th].copy()
        if not fp_raw.empty:
            fp_raw["hour"] = (fp_raw["time_index"] * TIME_UNIT_SECONDS // SECONDS_PER_HOUR).astype(int).clip(0, 23)
            hourly_floating = (
                fp_raw.groupby("hour")["mac_address"].nunique()
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

        # ── Floating population: unique MACs per minute ─────────────────────
        if self.entrance_swards:
            fp_raw = df[
                df["sward_name"].isin(self.entrance_swards)
                & (df["rssi"] >= self.floating_rssi)
            ].copy()
        else:
            fp_raw = df[
                df["sward_name"].isin(self.inside_swards)
                & (df["rssi"] >= FALLBACK_FLOATING_RSSI)
            ].copy()

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
