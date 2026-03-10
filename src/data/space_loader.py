"""
공간(Space) 단위 데이터 로더.
- 공간 탐색, S-Ward 설정 로드, Raw 일별/기간 로드.
"""
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
