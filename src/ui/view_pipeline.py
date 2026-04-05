"""
Hermes — Pipeline & Data View.

Wrapper module combining:
- Preprocessing Pipeline (cache management, raw data processing)
- MAC Stitching (BLE signal correction, before/after comparison)

Admin-only view for data infrastructure management.
"""
import streamlit as st

from src.cache.cache_io import CacheLoader
from src.ui.page_pipeline import render_pipeline_tab
from src.ui.page_stitching import render_stitching_tab


def render_pipeline_view(space_name: str, loader: CacheLoader) -> None:
    """
    Pipeline & Data view: combines preprocessing + MAC stitching.

    Parameters
    ----------
    space_name : str
        Current space/sector name
    loader : CacheLoader
        Cache loader instance for the space
    """
    st.subheader("Pipeline & Data")
    st.caption(
        "Data infrastructure management: preprocessing pipeline and BLE signal correction."
    )

    # Two sub-tabs
    tab_pipeline, tab_stitching = st.tabs([
        "Preprocessing",
        "MAC Stitching"
    ])

    with tab_pipeline:
        render_pipeline_tab(space_name)

    with tab_stitching:
        render_stitching_tab(space_name, loader)
