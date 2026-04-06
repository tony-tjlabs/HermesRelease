"""
Hermes Pipeline Tab — Cloud deployment (preprocessing disabled).
"""

import streamlit as st
from datetime import datetime

from src.config.paths import get_cache_path


# ── 캐시 현황 조회 ─────────────────────────────────────────────────────────

def _get_pipeline_status(space_name: str) -> dict:
    import json

    cache_dir = get_cache_path(space_name)
    meta_path = cache_dir / "metadata.json"

    if not meta_path.exists():
        return {
            "has_cache": False,
            "cached_dates": [],
            "created_at": None,
            "cache_version": None,
        }

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached_dates  = meta.get("date_range", [])
        created_at    = meta.get("created_at", None)
        cache_version = meta.get("cache_version", "?")
    except Exception:
        cached_dates  = []
        created_at    = None
        cache_version = None

    return {
        "has_cache":     True,
        "cached_dates":  cached_dates,
        "created_at":    created_at,
        "cache_version": cache_version,
    }


# ── Pipeline 탭 메인 ───────────────────────────────────────────────────────

def render_pipeline_tab(space_name: str) -> None:
    """Pipeline 탭: 캐시 현황 표시 (전처리 비활성화)."""
    st.markdown("### Pipeline")
    st.caption(f"Space: `{space_name}`")

    status = _get_pipeline_status(space_name)

    if status["has_cache"]:
        cached_dates = status["cached_dates"]
        date_range_str = (
            f"{cached_dates[0]} ~ {cached_dates[-1]}" if cached_dates else "—"
        )
        created = ""
        if status["created_at"]:
            try:
                dt = datetime.fromisoformat(status["created_at"])
                created = f" | Built {dt.strftime('%Y-%m-%d %H:%M')}"
            except Exception:
                pass

        st.success(
            f"Cache loaded: {len(cached_dates)} dates ({date_range_str}){created}"
        )
    else:
        st.warning("No cache found for this space.")

    st.info("Pipeline preprocessing is not available in cloud deployment.")
