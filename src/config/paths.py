"""
Hermes 경로 설정.
루트 기준 경로를 사용해 Datafile, 공간별 cache 경로를 반환.
"""
from pathlib import Path

# Hermes 프로젝트 루트 (main.py가 있는 디렉터리)
HERMES_ROOT = Path(__file__).resolve().parent.parent.parent
DATAFILE_ROOT = HERMES_ROOT / "Datafile"


def get_space_path(space_name: str) -> Path:
    """공간 이름으로 Datafile 내 해당 공간 디렉터리 경로 반환."""
    return DATAFILE_ROOT / space_name


def get_cache_path(space_name: str) -> Path:
    """공간별 캐시 디렉터리 경로 반환."""
    return get_space_path(space_name) / "cache"


def get_rawdata_path(space_name: str) -> Path:
    """공간별 rawdata 디렉터리 경로 반환."""
    return get_space_path(space_name) / "rawdata"


def get_sward_config_path(space_name: str) -> Path:
    """공간별 sward_config 파일 경로 반환."""
    return get_space_path(space_name) / "sward_configuration" / "sward_config.csv"
