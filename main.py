"""
Project Hermes v2 — Spatial Intelligence Dashboard (Dark Theme).

UI redesigned to match TheHyundaiSeoul simplicity:
  - Dark theme (#0E1117)
  - Sidebar: Sector + Date + Mode radio + Time slider
  - Single-scroll content per mode
  - AI analysis at the bottom (1 button)

Modes
-----
- Daily Analysis : Single-date deep dive (KPI + Hourly + Dwell + AI)
- Period Comparison : Multi-date trend comparison + AI
- Report : Weekly PDF report generation
- Admin (Admin only) : Pipeline + MAC Stitching + Space Notes
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

st.set_page_config(
    page_title="Hermes",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.auth import (
    require_login, is_admin,
    get_current_sector, set_current_sector, get_allowed_sectors,
    logout,
)
from src.cache.cache_io import CacheLoader
from src.data.space_loader import load_store_config
from src.ui import get_custom_css
from src.ui.view_dashboard import render_dashboard
from src.ui.view_report import render_report
from src.ui.view_pipeline import render_pipeline_view
from src.ui.page_pipeline import render_pipeline_tab


# -- Cache Helpers ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_loader(space_name: str) -> CacheLoader:
    """CacheLoader per Sector, held in memory."""
    return CacheLoader(space_name)


# -- Space Notes (Admin) ---------------------------------------------------

def _render_space_notes(space_name: str, loader: CacheLoader) -> None:
    """Render space notes editor in sidebar (Admin only)."""
    with st.expander("Space Notes", expanded=False):
        saved_notes = loader.get_space_notes()
        from config import REGISTERED_SECTORS
        sector_meta = REGISTERED_SECTORS.get(space_name, {})
        stype = sector_meta.get("store_type", "")
        if stype == "sports_retail":
            placeholder_hint = "e.g. Seasonal sale, mall event, sport season opening..."
        elif stype == "convenience_store":
            placeholder_hint = "e.g. Nearby construction, office building closure, promotion campaign..."
        else:
            placeholder_hint = "e.g. Holiday hours, renovation, special events..."
        notes_input = st.text_area(
            label="Operator notes / remarks",
            value=saved_notes,
            height=100,
            key=f"space_notes_{space_name}",
            placeholder=placeholder_hint,
            label_visibility="collapsed",
        )
        if st.button("Save", key="save_notes_btn"):
            ok = loader.save_space_notes(notes_input)
            if ok:
                st.success("Saved")
            else:
                st.error("Save failed")
        st.session_state["current_space_notes"] = notes_input


# -- Sidebar ---------------------------------------------------------------

def _render_sidebar(loader: CacheLoader | None = None) -> tuple[str | None, str, str | None, tuple[int, int]]:
    """
    Render sidebar. Returns (space_name, view_mode, selected_date, time_range).

    Parameters
    ----------
    loader : CacheLoader | None
        Loader for space notes and date discovery.

    Returns
    -------
    tuple
        (space_name, view_mode, selected_date, time_range)
    """
    from src.auth import _get_sector_registry

    # Default time range (overridden once space_name is known)
    fallback_time = (7, 23)

    with st.sidebar:
        st.title("Hermes")
        st.caption("Spatial Intelligence")
        st.divider()

        # -- Sector Selection --
        sector_reg = _get_sector_registry()

        if is_admin():
            allowed = get_allowed_sectors()
            if not allowed:
                st.warning("No sectors found in Datafile/.")
                if st.button("Logout", use_container_width=True):
                    logout()
                return None, "Daily Analysis", None, fallback_time

            sector_labels = {
                sid: f"{sector_reg[sid]['icon']}  {sector_reg[sid]['label']}"
                for sid in allowed if sid in sector_reg
            }
            current = get_current_sector()

            selected = st.radio(
                "Sector",
                options=allowed,
                format_func=lambda x: sector_labels.get(x, x),
                index=allowed.index(current) if current in allowed else 0,
                key="hermes_sector_radio",
            )

            if selected != current:
                set_current_sector(selected)
                st.cache_data.clear()
                st.rerun()

            space_name = selected
        else:
            space_name = get_current_sector()
            if space_name and space_name in sector_reg:
                info = sector_reg[space_name]
                st.markdown(
                    f"**{info['icon']} {info['label']}**",
                    help="Your assigned sector",
                )
            elif space_name:
                st.markdown(f"**{space_name}**")

        if not space_name:
            if st.button("Logout", use_container_width=True):
                logout()
            return None, "Daily Analysis", None, fallback_time

        st.divider()

        # -- Date Selection --
        selected_date = None
        if loader is not None and loader.is_available():
            date_range_list = loader.get_date_range() or []
            if date_range_list:
                selected_date = st.selectbox(
                    "Date",
                    date_range_list,
                    index=len(date_range_list) - 1,
                )

        st.divider()

        # -- View Mode --
        views = ["Daily Analysis", "Period Comparison", "Report"]
        if is_admin():
            views.append("Admin")

        view_mode = st.radio(
            "Mode",
            views,
            index=0,
            key="hermes_view_mode",
        )

        st.divider()

        # -- Time Filter --
        st.markdown("**Filter**")
        # Load store_config for sector-specific defaults
        store_cfg = load_store_config(space_name)
        default_start = max(0, store_cfg.store_open_hour - 1)
        default_end = min(24, store_cfg.store_close_hour + 1)
        time_range = st.slider(
            "Hours",
            min_value=0, max_value=24, value=(default_start, default_end),
            help="Time range filter for analysis",
        )

        # -- Space Notes (Admin) --
        if is_admin() and loader is not None and loader.is_available():
            st.divider()
            _render_space_notes(space_name, loader)

        st.divider()

        # -- Sensor Info --
        st.markdown(
            '<div style="color:#8892b0; font-size:0.75rem;">'
            "BLE S-Ward sensor data<br>"
            "10-second sampling interval"
            "</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("Logout", use_container_width=True, key="hermes_logout"):
            logout()

    return space_name, view_mode, selected_date, time_range


# -- No Cache Handler ------------------------------------------------------

def _handle_no_cache(space_name: str) -> None:
    """Handle case when no preprocessed cache exists."""
    st.markdown(f"## {space_name}")
    st.warning(
        f"No preprocessed cache found for **{space_name}**. "
        + (
            "Switch to Admin mode and run Pipeline preprocessing."
            if is_admin()
            else "Contact your administrator to initialize the data."
        )
    )
    if is_admin():
        render_pipeline_tab(space_name)


# -- Main ------------------------------------------------------------------

def main():
    # 1. Auth gate
    require_login()

    # 2. CSS (dark theme)
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # 3. Determine loader before sidebar
    space_name_pre = get_current_sector()
    loader = _get_loader(space_name_pre) if space_name_pre else None
    if loader and not loader.is_available():
        loader = None

    # 4. Render sidebar
    space_name, view_mode, selected_date, time_range = _render_sidebar(loader=loader)
    if not space_name:
        st.info("Please select a sector.")
        return

    # 5. Re-check loader if sector changed
    if space_name != space_name_pre:
        loader = _get_loader(space_name)

    # 6. Cache check
    if loader is None or not loader.is_available():
        loader = _get_loader(space_name)
        if not loader.is_available():
            _handle_no_cache(space_name)
            return

    # 7. View routing
    if view_mode == "Daily Analysis":
        render_dashboard(space_name, loader, selected_date, time_range, mode="daily")
    elif view_mode == "Period Comparison":
        render_dashboard(space_name, loader, selected_date, time_range, mode="comparison")
    elif view_mode == "Report":
        render_report(space_name, loader)
    elif view_mode == "Admin" and is_admin():
        render_pipeline_view(space_name, loader)
    else:
        render_dashboard(space_name, loader, selected_date, time_range, mode="daily")


if __name__ == "__main__":
    main()
