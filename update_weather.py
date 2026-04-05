"""
update_weather.py — 기존 캐시를 유지하면서 날씨 데이터만 갱신합니다.

전체 precompute를 재실행할 필요 없이,
daily_stats.parquet의 날씨 컬럼(precipitation, snowfall, temp_max, temp_min, weather)만
Open-Meteo Archive API로 새로 채웁니다.

사용법:
    python update_weather.py                         # 자동 탐색된 모든 공간
    python update_weather.py --space Victor_Suwon_Starfield
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config.paths import get_cache_path, DATAFILE_ROOT
from src.data.space_loader import discover_spaces
from src.data.external_api import enrich_weather, DEFAULT_LATITUDE, DEFAULT_LONGITUDE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 공간별 위치 오버라이드 (필요 시 추가)
SPACE_LOCATIONS: dict = {
    "Victor_Suwon_Starfield": (DEFAULT_LATITUDE, DEFAULT_LONGITUDE),
}

WEATHER_COLS = ["precipitation", "snowfall", "temp_max", "temp_min", "weather"]


def update_space_weather(space_name: str) -> bool:
    cache_dir = get_cache_path(space_name)
    parquet_path = cache_dir / "daily_stats.parquet"

    if not parquet_path.exists():
        logger.warning("[%s] daily_stats.parquet not found — run precompute first.", space_name)
        return False

    df = pd.read_parquet(parquet_path)
    if "date" not in df.columns or df.empty:
        logger.warning("[%s] daily_stats is empty or missing 'date' column.", space_name)
        return False

    # 기존 날씨 컬럼 제거 후 새로 붙이기
    df = df.drop(columns=[c for c in WEATHER_COLS if c in df.columns], errors="ignore")

    lat, lon = SPACE_LOCATIONS.get(space_name, (DEFAULT_LATITUDE, DEFAULT_LONGITUDE))
    logger.info("[%s] Fetching weather (lat=%.3f, lon=%.3f) …", space_name, lat, lon)

    df = enrich_weather(df, date_col="date", latitude=lat, longitude=lon)

    # 결과 확인
    if "weather" in df.columns and df["weather"].eq("Unknown").all():
        logger.error("[%s] ❌ Weather fetch failed — all values are Unknown. Check network.", space_name)
        return False

    df.to_parquet(parquet_path, index=False)
    logger.info("[%s] ✅ daily_stats.parquet updated with weather.", space_name)

    # 결과 출력
    print(f"\n{'날짜':<12} {'강수(mm)':>8} {'눈(cm)':>7} {'최고°C':>7} {'최저°C':>7}  날씨")
    print("-" * 58)
    for _, row in df.sort_values("date").iterrows():
        icon = {"Sunny": "☀️ Sunny", "Rain": "🌧 Rain", "Snow": "❄️ Snow"}.get(
            row.get("weather", ""), f"— {row.get('weather','')}"
        )
        print(
            f"{str(row['date']):<12}"
            f" {row.get('precipitation', 0) or 0:>8.1f}"
            f" {row.get('snowfall', 0) or 0:>7.1f}"
            f" {row.get('temp_max', 0) or 0:>7.1f}"
            f" {row.get('temp_min', 0) or 0:>7.1f}"
            f"  {icon}"
        )
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Update weather columns in daily_stats.parquet")
    parser.add_argument("--space", type=str, default=None, help="Space name (default: all)")
    args = parser.parse_args()

    spaces = [args.space] if args.space else discover_spaces()
    if not spaces:
        logger.error("No spaces found under %s", DATAFILE_ROOT)
        sys.exit(1)

    logger.info("Updating weather for: %s", spaces)
    success = 0
    for sp in spaces:
        if update_space_weather(sp):
            success += 1

    logger.info("Done. %d/%d space(s) updated successfully.", success, len(spaces))
    if success < len(spaces):
        sys.exit(1)


if __name__ == "__main__":
    main()
