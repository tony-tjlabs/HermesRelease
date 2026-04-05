"""
Hermes preprocessing entrypoint.

Reads raw BLE data, runs strict visitor/session pipeline, writes cache (Parquet + JSON)
under Datafile/{space_name}/cache/ for dashboard consumption.

Usage:
  python precompute.py --space Victor_Suwon_Starfield
  python precompute.py --all
  python precompute.py --space Victor_Suwon_Starfield --max-dates 3
"""
import argparse
import logging
import sys

from src.data.space_loader import discover_spaces, get_available_dates
from src.preprocess.runner import run_preprocess_space

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("hermes.precompute")


def main():
    parser = argparse.ArgumentParser(
        description="Hermes: Preprocess raw BLE data → sessions & aggregates → Parquet/JSON cache."
    )
    parser.add_argument(
        "--space",
        type=str,
        default=None,
        help="Space name (e.g. Victor_Suwon_Starfield). Required unless --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run preprocessing for all discovered spaces.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache and regenerate.",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Max number of dates per space (for testing).",
    )
    args = parser.parse_args()

    if args.all:
        spaces = discover_spaces()
        if not spaces:
            logger.warning("No valid spaces found under Datafile/.")
            return
        for sp in spaces:
            logger.info("Preprocessing space: %s", sp)
            try:
                run_preprocess_space(sp, force=args.force, max_dates=args.max_dates)
                logger.info("  OK %s", sp)
            except Exception as e:
                logger.error("  FAIL %s: %s", sp, e)
        return

    space = args.space
    if not space:
        logger.error("Specify --space SPACE_NAME or --all.")
        return

    if space not in discover_spaces():
        logger.error("Space '%s' not found. Available: %s", space, discover_spaces())
        return

    logger.info("Preprocessing space: %s", space)
    run_preprocess_space(space, force=args.force, max_dates=args.max_dates)
    logger.info("Done.")


if __name__ == "__main__":
    main()
