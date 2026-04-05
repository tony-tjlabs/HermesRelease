"""
Hermes Pipeline Tab — Cloud deployment (read-only).

In the release build, preprocessing is not available.
Only cache status is displayed.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from src.data.space_loader import get_available_dates
from src.config.paths import get_cache_path


# -- Cache status --

def _get_pipeline_status(space_name: str) -> dict:
    import json

    raw_dates = get_available_dates(space_name)
    cache_dir = get_cache_path(space_name)
    meta_path = cache_dir / "metadata.json"

    if not meta_path.exists():
        return {
            "has_cache": False,
            "cached_dates": [],
            "raw_dates": raw_dates,
            "new_dates": raw_dates,
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

    cached_set = set(cached_dates)
    new_dates  = [d for d in raw_dates if d not in cached_set]

    return {
        "has_cache":     True,
        "cached_dates":  cached_dates,
        "raw_dates":     raw_dates,
        "new_dates":     new_dates,
        "created_at":    created_at,
        "cache_version": cache_version,
    }


# -- Status cards --

def _render_status_cards(status: dict) -> None:
    c1, c2, c3 = st.columns(3)

    total_raw = len(status["raw_dates"])
    cached_n  = len(status["cached_dates"])
    new_n     = len(status["new_dates"])

    with c1:
        st.metric("Raw Dates", total_raw, help="rawdata/ folder date CSV count")
    with c2:
        if status["has_cache"]:
            st.metric("Cached Dates", cached_n)
        else:
            st.metric("Cached Dates", "---")
    with c3:
        if new_n > 0:
            st.metric("New / Pending", new_n, delta=f"+{new_n}", delta_color="inverse")
        else:
            st.metric("New / Pending", 0)


def _render_status_banner(status: dict) -> None:
    if not status["has_cache"]:
        st.warning(
            f"Cache not found. "
            f"{len(status['raw_dates'])} raw date(s) detected."
        )
        return

    cached_dates = status["cached_dates"]
    date_range_str = (
        f"{cached_dates[0]} ~ {cached_dates[-1]}" if cached_dates else "---"
    )
    created = ""
    if status["created_at"]:
        try:
            dt = datetime.fromisoformat(status["created_at"])
            created = f" | Built {dt.strftime('%Y-%m-%d %H:%M')}"
        except Exception:
            pass

    if status["new_dates"]:
        st.info(
            f"Cached {len(status['cached_dates'])} dates ({date_range_str}){created}. "
            f"**{len(status['new_dates'])} new date(s)** not yet in cache."
        )
    else:
        st.success(
            f"Cache up-to-date. "
            f"{len(status['cached_dates'])} dates ({date_range_str}){created}."
        )


# -- Pipeline tab main --

def render_pipeline_tab(space_name: str) -> None:
    """Pipeline tab: cache status only (cloud deployment)."""
    from src.auth import is_admin

    st.markdown("### Pipeline")
    st.caption(f"Space: `{space_name}`")

    status = _get_pipeline_status(space_name)

    _render_status_cards(status)
    st.markdown("")
    _render_status_banner(status)

    # Raw date list expander
    if status["raw_dates"]:
        with st.expander(f"Raw dates ({len(status['raw_dates'])})", expanded=False):
            cols = st.columns(4)
            for i, d in enumerate(status["raw_dates"]):
                is_new    = d in status["new_dates"]
                is_cached = d in status["cached_dates"]
                icon = "NEW" if is_new else ("OK" if is_cached else "-")
                cols[i % 4].markdown(f"`[{icon}]` `{d}`")

    # Cloud deployment notice
    st.markdown("---")
    st.info(
        "Pipeline preprocessing is not available in cloud deployment. "
        "Cache data is pre-built and included in the deployment package. "
        "Contact the development team to update cache data."
    )
