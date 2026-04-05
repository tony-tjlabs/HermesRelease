"""
공간(Space) 단위 데이터 로더.
- 공간 탐색, S-Ward 설정 로드, Raw 일별/기간 로드.
- store_config.json 로드: Sector별 운영 파라미터 (영업시간, 체류시간 세그먼트 등)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.config.paths import (
    DATAFILE_ROOT,
    get_space_path,
    get_rawdata_path,
    get_sward_config_path,
)
from src.config.constants import (
    STORE_OPEN_HOUR,
    STORE_CLOSE_HOUR,
    MIN_HITS_PER_MIN,
    DWELL_SHORT_MAX,
    DWELL_MEDIUM_MAX,
)

logger = logging.getLogger(__name__)


# ── Store Configuration ─────────────────────────────────────────────────────

@dataclass
class StoreConfig:
    """Sector별 운영 파라미터. store_config.json에서 로드하거나 기본값 사용."""

    store_open_hour: int = STORE_OPEN_HOUR       # 영업 시작 (inclusive)
    store_close_hour: int = STORE_CLOSE_HOUR     # 영업 종료 (exclusive)
    min_hits_per_min: int = MIN_HITS_PER_MIN     # 방문객 진입 최소 신호 수
    min_dwell_seconds: int = 60                   # 최소 체류 시간 (초)
    rssi_threshold_apple: int = -75               # Apple RSSI 임계값
    rssi_threshold_android: int = -85             # Android RSSI 임계값
    rssi_pass_ratio: float = 0.80                 # RSSI 통과율 (3중 필터)
    dwell_short_max: int = DWELL_SHORT_MAX       # 단기 체류 상한 (초)
    dwell_medium_max: int = DWELL_MEDIUM_MAX     # 중기 체류 상한 (초)
    description: str = ""                         # 설명 (선택)

    def is_24h(self) -> bool:
        """24시간 영업 여부."""
        return self.store_open_hour == 0 and self.store_close_hour == 24


def get_store_config_path(space_name: str) -> Path:
    """공간별 store_config.json 파일 경로 반환."""
    return get_space_path(space_name) / "sward_configuration" / "store_config.json"


def load_store_config(space_name: str) -> StoreConfig:
    """
    공간의 store_config.json 로드.

    store_config.json이 없으면 constants.py의 기본값을 사용하여 StoreConfig 반환.
    일부 필드만 있는 경우 나머지는 기본값 사용.
    """
    path = get_store_config_path(space_name)
    if not path.exists():
        logger.debug("store_config.json not found for %s, using defaults", space_name)
        return StoreConfig()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 기본값을 먼저 설정하고, JSON에 있는 값으로 덮어쓰기
        config = StoreConfig(
            store_open_hour=data.get("store_open_hour", STORE_OPEN_HOUR),
            store_close_hour=data.get("store_close_hour", STORE_CLOSE_HOUR),
            min_hits_per_min=data.get("min_hits_per_min", MIN_HITS_PER_MIN),
            min_dwell_seconds=data.get("min_dwell_seconds", 60),
            rssi_threshold_apple=data.get("rssi_threshold_apple", -75),
            rssi_threshold_android=data.get("rssi_threshold_android", -85),
            rssi_pass_ratio=data.get("rssi_pass_ratio", 0.80),
            dwell_short_max=data.get("dwell_short_max", DWELL_SHORT_MAX),
            dwell_medium_max=data.get("dwell_medium_max", DWELL_MEDIUM_MAX),
            description=data.get("description", ""),
        )
        logger.info(
            "Loaded store_config for %s: open=%d, close=%d, 24h=%s",
            space_name, config.store_open_hour, config.store_close_hour, config.is_24h(),
        )
        return config
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse store_config.json for %s: %s, using defaults", space_name, e)
        return StoreConfig()
    except Exception as e:
        logger.warning("Error loading store_config.json for %s: %s, using defaults", space_name, e)
        return StoreConfig()


def discover_spaces() -> List[str]:
    """
    Datafile/ 아래 유효한 공간 폴더 목록 반환.

    우선 조건 (전처리 환경):
      rawdata/ 존재 + sward_configuration/sward_config.csv 존재 + 날짜 CSV 1개 이상

    폴백 조건 (배포/대시보드 전용 환경 — rawdata 없음):
      cache/metadata.json 존재 + sward_configuration/sward_config.csv 존재
    """
    if not DATAFILE_ROOT.exists():
        return []
    spaces = []
    for folder in DATAFILE_ROOT.iterdir():
        if not folder.is_dir():
            continue
        config_path = folder / "sward_configuration" / "sward_config.csv"
        if not config_path.exists():
            continue

        raw_dir = folder / "rawdata"
        cache_meta = folder / "cache" / "metadata.json"

        if raw_dir.exists():
            # 전처리 환경: rawdata에 날짜 CSV가 있어야 유효
            has_csv = any(
                f.suffix.lower() == ".csv"
                and _parse_date_from_stem(f.stem) is not None
                for f in raw_dir.iterdir()
            )
            if has_csv:
                spaces.append(folder.name)
        elif cache_meta.exists():
            # 배포 환경: rawdata 없어도 캐시가 있으면 대시보드용으로 유효
            spaces.append(folder.name)

    return sorted(spaces)


def _parse_date_from_stem(stem: str) -> Optional[str]:
    """파일명 stem이 YYYY-MM-DD 형식이면 반환, 아니면 None."""
    try:
        datetime.strptime(stem, "%Y-%m-%d")
        return stem
    except ValueError:
        return None


def load_sward_config(space_name: str) -> pd.DataFrame:
    """
    공간의 sward_config.csv 로드.
    컬럼: sward_name, install_location, rssi_threshold, min_dwell_time
    """
    path = get_sward_config_path(space_name)
    if not path.exists():
        raise FileNotFoundError(f"S-Ward config not found: {path}")
    df = pd.read_csv(path)
    df["sward_name"] = df["sward_name"].astype(str).str.strip()
    return df


def get_available_dates(space_name: str) -> List[str]:
    """공간의 rawdata 내 사용 가능한 날짜(YYYY-MM-DD) 목록."""
    raw_dir = get_rawdata_path(space_name)
    if not raw_dir.exists():
        return []
    dates = []
    for f in raw_dir.iterdir():
        if f.suffix.lower() != ".csv":
            continue
        d = _parse_date_from_stem(f.stem)
        if d:
            dates.append(d)
    return sorted(dates)


def load_raw_date(space_name: str, date_str: str) -> pd.DataFrame:
    """
    특정 공간·특정 일자의 raw CSV 로드.
    insert_datetime 파싱, sward_name 문자열, type 유지.
    """
    raw_dir = get_rawdata_path(space_name)
    path = raw_dir / f"{date_str}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    if "insert_datetime" in df.columns:
        df["insert_datetime"] = pd.to_datetime(df["insert_datetime"], utc=False)
    df["sward_name"] = df["sward_name"].astype(str).str.strip()
    df["date"] = date_str
    return df


def load_raw_date_range(
    space_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """공간의 start_date ~ end_date 구간 raw 데이터 통합 로드."""
    dates = get_available_dates(space_name)
    target = [d for d in dates if start_date <= d <= end_date]
    if not target:
        return pd.DataFrame()
    dfs = []
    for d in target:
        try:
            dfs.append(load_raw_date(space_name, d))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_raw_all_dates(space_name: str) -> pd.DataFrame:
    """해당 공간의 모든 일자 raw 데이터 통합 로드."""
    dates = get_available_dates(space_name)
    if not dates:
        return pd.DataFrame()
    return load_raw_date_range(space_name, dates[0], dates[-1])
