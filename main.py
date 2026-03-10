"""
Project Hermes — Spatial Cause–Effect Intelligence Dashboard.
Entry: streamlit run main.py

API key: .env (ANTHROPIC_API_KEY) or .streamlit/secrets.toml. Never commit keys.
"""
from pathlib import Path

# Load .env into os.environ before any code that reads ANTHROPIC_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import plotly.io as pio

st.set_page_config(
    page_title="Hermes — Spatial Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates.default = "plotly_white"

from src.data.space_loader import discover_spaces
from src.cache.cache_io import CacheLoader
from src.ui import get_custom_css, render_stitching_tab, render_overview, render_hourly, render_patterns, render_report_tab


# ── Performance: cache expensive objects across reruns ────────────────────────

@st.cache_resource(show_spinner=False)
def _get_loader(space_name: str) -> CacheLoader:
    """Cache CacheLoader per space — Parquet/JSON files are read once and kept in memory."""
    return CacheLoader(space_name)


@st.cache_data(ttl=60, show_spinner=False)
def _discover_spaces() -> list:
    """Cache space discovery — filesystem scan runs at most once per minute."""
    return discover_spaces()


# ── 비밀번호 인증 ──────────────────────────────────────────────────────────────
def _check_password() -> bool:
    """st.secrets의 password와 입력값을 비교한다. 일치하면 True."""

    def _on_submit():
        try:
            correct = st.secrets.get("password", "")
        except Exception:
            correct = ""
        if st.session_state.get("_pw_input") == correct and correct:
            st.session_state["_pw_ok"] = True
            st.session_state.pop("_pw_input", None)
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok"):
        return True

    # 로그인 화면 렌더링
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center; color:#0f172a;'>◈ Hermes</h2>"
            "<p style='text-align:center; color:#64748b;'>"
            "Spatial Intelligence for Real-World Influence</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.text_input(
            "Password",
            type="password",
            key="_pw_input",
            placeholder="Enter password…",
            on_change=_on_submit,
        )
        st.button("Enter", on_click=_on_submit, use_container_width=True)
        if "_pw_ok" in st.session_state and not st.session_state["_pw_ok"]:
            st.error("Incorrect password.")
    return False


def main():
    if not _check_password():
        return

    st.markdown(get_custom_css(), unsafe_allow_html=True)

    spaces = _discover_spaces()
    if not spaces:
        st.error("No valid space under **Datafile/**. Ensure each space has **rawdata/** and **sward_configuration/sward_config.csv**.")
        return

    with st.sidebar:
        st.title("◈ Hermes")
        st.markdown("*Spatial Intelligence for Real-World Influence*")
        st.markdown("---")
        space_name = st.selectbox(
            "Select space",
            options=spaces,
            key="hermes_space",
        )
        st.markdown("---")

    if not space_name:
        st.info("Please select a space.")
        return

    loader = _get_loader(space_name)
    if not loader.is_available():
        st.warning(f"No cache for **{space_name}**. Run precompute first:")
        st.code(f"python precompute.py --space {space_name}", language="bash")
        return

    # ── Sidebar Space Notes ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("##### 📝 Space Notes")
        saved_notes = loader.get_space_notes()

        notes_input = st.text_area(
            label="Operator notes / remarks",
            value=saved_notes,
            height=150,
            key=f"space_notes_{space_name}",
            placeholder=(
                "e.g. Holiday hours, store renovation 2/10–2/15\n"
                "Promotion: Winter sports 20% off"
            ),
            label_visibility="collapsed",
        )

        col_save, col_status = st.columns([2, 3])
        with col_save:
            save_clicked = st.button("💾 Save", key="save_notes_btn", use_container_width=True)

        if save_clicked:
            ok = loader.save_space_notes(notes_input)
            with col_status:
                if ok:
                    st.success("Saved ✅")
                else:
                    st.error("Save failed")

        # Keep notes in session for AI context
        st.session_state["current_space_notes"] = notes_input

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview 매장현황",
        "⏰ Hourly 시간대분석",
        "📈 Patterns 패턴분석",
        "📋 Report 리포트",
        "🔗 Data Correction 보정",
    ])
    with tab0:
        render_overview(space_name, loader)
    with tab1:
        render_hourly(space_name, loader)
    with tab2:
        render_patterns(space_name, loader)
    with tab3:
        render_report_tab(space_name, loader)
    with tab4:
        render_stitching_tab(space_name, loader)


if __name__ == "__main__":
    main()
