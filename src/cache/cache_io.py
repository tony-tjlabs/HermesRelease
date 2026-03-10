"""
Hermes cache I/O: write preprocessed Parquet/JSON under Datafile/{space_name}/cache/.

Architecture: Stitching-First (v2.0)
  Primary files (daily_stats, daily_hourly, device_mix, daily_results) are computed
  from STITCHED sessions. Raw (pre-stitching) versions stored as *_raw files for
  the MAC Stitching comparison tab.

Output files (timezone Asia/Seoul):
- sessions_all.parquet     : Raw sessions (pre-stitching)
- sessions_stitched.parquet: MAC Stitching 적용 세션 (L1+L2)
- daily_stats.parquet      : Stitched 기반 일별 통계
- daily_stats_raw.parquet  : Raw 기반 일별 통계 (비교용)
- daily_hourly.parquet     : Stitched 기반 시간대별 통계
- daily_hourly_raw.parquet : Raw 기반 시간대별 통계 (비교용)
- device_mix.parquet       : Stitched 기반 디바이스 믹스
- device_mix_raw.parquet   : Raw 기반 디바이스 믹스 (비교용)
- daily_timeseries.parquet : Stitched 기반 분 단위 시계열
- daily_results.json       : Stitched 기반 일별 요약
- daily_results_raw.json   : Raw 기반 일별 요약 (비교용)
- metadata.json            : cache_version, space_name, created_at, date_range
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import get_cache_path


CACHE_VERSION = "2.0"


def _serialize_df(df: pd.DataFrame) -> List[Dict]:
    """DataFrame을 JSON 저장 가능한 리스트로."""
    if df is None or df.empty:
        return []
    return df.replace({float("nan"): None}).to_dict(orient="records")


def _deserialize_df(data: List[Dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


class CacheWriter:
    """공간별 전처리 결과를 cache/ 에 저장."""

    def __init__(self, space_name: str):
        self.space_name = space_name
        self.cache_dir = get_cache_path(space_name)

    def write(
        self,
        date_range: List[str],
        daily_results: List[Dict],
        daily_hourly: pd.DataFrame,
        daily_stats: pd.DataFrame,
        sessions_all: Optional[pd.DataFrame] = None,
        device_mix: Optional[pd.DataFrame] = None,
        daily_timeseries: Optional[pd.DataFrame] = None,
        sessions_stitched: Optional[pd.DataFrame] = None,
        metadata_extra: Optional[Dict] = None,
        # RAW versions for comparison tab
        daily_stats_raw: Optional[pd.DataFrame] = None,
        daily_hourly_raw: Optional[pd.DataFrame] = None,
        device_mix_raw: Optional[pd.DataFrame] = None,
        daily_results_raw: Optional[List[Dict]] = None,
    ) -> None:
        """Write all cache files (Parquet + JSON). Timezone for session times: Asia/Seoul (+09:00)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "cache_version": CACHE_VERSION,
            "space_name": self.space_name,
            "created_at": datetime.now().isoformat(),
            "date_range": date_range,
            **(metadata_extra or {}),
        }
        with open(self.cache_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # PRIMARY (stitched-based)
        with open(self.cache_dir / "daily_results.json", "w", encoding="utf-8") as f:
            json.dump(daily_results, f, indent=2, ensure_ascii=False)

        daily_hourly.to_parquet(self.cache_dir / "daily_hourly.parquet", index=False)
        daily_stats.to_parquet(self.cache_dir / "daily_stats.parquet", index=False)

        if sessions_all is not None and not sessions_all.empty:
            sessions_all.to_parquet(self.cache_dir / "sessions_all.parquet", index=False)
        if device_mix is not None and not device_mix.empty:
            device_mix.to_parquet(self.cache_dir / "device_mix.parquet", index=False)
        if daily_timeseries is not None and not daily_timeseries.empty:
            daily_timeseries.to_parquet(self.cache_dir / "daily_timeseries.parquet", index=False)
        if sessions_stitched is not None and not sessions_stitched.empty:
            sessions_stitched.to_parquet(self.cache_dir / "sessions_stitched.parquet", index=False)

        # RAW (for MAC Stitching comparison tab)
        if daily_stats_raw is not None and not daily_stats_raw.empty:
            daily_stats_raw.to_parquet(self.cache_dir / "daily_stats_raw.parquet", index=False)
        if daily_hourly_raw is not None and not daily_hourly_raw.empty:
            daily_hourly_raw.to_parquet(self.cache_dir / "daily_hourly_raw.parquet", index=False)
        if device_mix_raw is not None and not device_mix_raw.empty:
            device_mix_raw.to_parquet(self.cache_dir / "device_mix_raw.parquet", index=False)
        if daily_results_raw is not None:
            with open(self.cache_dir / "daily_results_raw.json", "w", encoding="utf-8") as f:
                json.dump(daily_results_raw, f, indent=2, ensure_ascii=False)


class CacheLoader:
    """공간별 캐시 로드 (대시보드용)."""

    def __init__(self, space_name: str):
        self.space_name = space_name
        self.cache_dir = get_cache_path(space_name)
        # PRIMARY (stitched-based)
        self._daily_results: Optional[List[Dict]] = None
        self._daily_hourly: Optional[pd.DataFrame] = None
        self._daily_stats: Optional[pd.DataFrame] = None
        self._sessions_all: Optional[pd.DataFrame] = None
        self._sessions_stitched: Optional[pd.DataFrame] = None
        self._device_mix: Optional[pd.DataFrame] = None
        self._daily_timeseries: Optional[pd.DataFrame] = None
        self._metadata: Optional[Dict] = None
        # RAW (for comparison tab)
        self._daily_stats_raw: Optional[pd.DataFrame] = None
        self._daily_hourly_raw: Optional[pd.DataFrame] = None
        self._device_mix_raw: Optional[pd.DataFrame] = None
        self._daily_results_raw: Optional[List[Dict]] = None

    def is_available(self) -> bool:
        return (self.cache_dir / "metadata.json").exists()

    def get_metadata(self) -> Dict:
        if self._metadata is None:
            path = self.cache_dir / "metadata.json"
            if not path.exists():
                self._metadata = {}
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
        return self._metadata

    # ── PRIMARY getters (stitched-based) ──────────────────────────────────

    def get_daily_results(self) -> List[Dict]:
        if self._daily_results is None:
            path = self.cache_dir / "daily_results.json"
            if not path.exists():
                self._daily_results = []
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self._daily_results = json.load(f)
        return self._daily_results

    def get_daily_hourly(self) -> pd.DataFrame:
        if self._daily_hourly is None:
            path = self.cache_dir / "daily_hourly.parquet"
            self._daily_hourly = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._daily_hourly

    def get_daily_stats(self) -> pd.DataFrame:
        if self._daily_stats is None:
            path = self.cache_dir / "daily_stats.parquet"
            self._daily_stats = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._daily_stats

    def get_sessions_all(self) -> pd.DataFrame:
        if self._sessions_all is None:
            path = self.cache_dir / "sessions_all.parquet"
            self._sessions_all = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._sessions_all

    def get_device_mix(self) -> pd.DataFrame:
        if self._device_mix is None:
            path = self.cache_dir / "device_mix.parquet"
            self._device_mix = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._device_mix

    def get_sessions_stitched(self) -> pd.DataFrame:
        if self._sessions_stitched is None:
            path = self.cache_dir / "sessions_stitched.parquet"
            self._sessions_stitched = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._sessions_stitched

    def get_daily_timeseries(self) -> pd.DataFrame:
        if self._daily_timeseries is None:
            path = self.cache_dir / "daily_timeseries.parquet"
            self._daily_timeseries = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._daily_timeseries

    def get_date_range(self) -> List[str]:
        return self.get_metadata().get("date_range", [])

    # ── RAW getters (for MAC Stitching comparison tab) ────────────────────

    def get_daily_stats_raw(self) -> pd.DataFrame:
        if self._daily_stats_raw is None:
            path = self.cache_dir / "daily_stats_raw.parquet"
            self._daily_stats_raw = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._daily_stats_raw

    def get_daily_hourly_raw(self) -> pd.DataFrame:
        if self._daily_hourly_raw is None:
            path = self.cache_dir / "daily_hourly_raw.parquet"
            self._daily_hourly_raw = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._daily_hourly_raw

    def get_device_mix_raw(self) -> pd.DataFrame:
        if self._device_mix_raw is None:
            path = self.cache_dir / "device_mix_raw.parquet"
            self._device_mix_raw = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return self._device_mix_raw

    def get_daily_results_raw(self) -> List[Dict]:
        if self._daily_results_raw is None:
            path = self.cache_dir / "daily_results_raw.json"
            if not path.exists():
                self._daily_results_raw = []
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self._daily_results_raw = json.load(f)
        return self._daily_results_raw

    # ── Space notes (independent of precompute) ───────────────────────────

    def get_space_notes(self) -> str:
        """저장된 공간 메모를 반환한다. 없으면 빈 문자열."""
        path = self.cache_dir / "space_notes.json"
        if not path.exists():
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("notes", "")
        except Exception:
            return ""

    def save_space_notes(self, notes: str) -> bool:
        """공간 메모를 저장한다. 성공 시 True 반환."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / "space_notes.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"notes": notes, "updated_at": datetime.now().isoformat()},
                    f, indent=2, ensure_ascii=False,
                )
            return True
        except Exception:
            return False
