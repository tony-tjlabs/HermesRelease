"""
Hermes Platform — Authentication & Session Management.

Login flow
----------
1. 로그인 화면: Sector 선택(=계정) + 비밀번호 입력
2. Admin   → 로그인 후 사이드바에서 Sector 자유 전환 (전체 접근)
3. Client  → 로그인 시 선택한 Sector에 고정

Session state keys
------------------
hermes_logged_in      : bool
hermes_user_id        : str
hermes_user_role      : "admin" | "client"
hermes_user_label     : str
hermes_current_sector : str | None
"""

import hmac
import logging
from datetime import datetime, timedelta

import streamlit as st
import config as cfg

logger = logging.getLogger("hermes.auth")

_KEY_LOGGED_IN      = "hermes_logged_in"
_KEY_USER_ID        = "hermes_user_id"
_KEY_USER_ROLE      = "hermes_user_role"
_KEY_USER_LABEL     = "hermes_user_label"
_KEY_CURRENT_SECTOR = "hermes_current_sector"
_KEY_LOGIN_ERROR    = "_hermes_login_error"


# ── 동적 Registry 캐시 (파일시스템 I/O를 매 rerun마다 반복하지 않도록) ────────

@st.cache_data(ttl=30, show_spinner=False)
def _get_sector_registry() -> dict:
    """30초 TTL로 Sector Registry 캐싱. 새 Datafile/ 폴더 추가 시 자동 갱신."""
    return cfg.build_sector_registry()


@st.cache_data(ttl=30, show_spinner=False)
def _get_user_registry() -> dict:
    """30초 TTL로 User Registry 캐싱."""
    return cfg.build_user_registry()


# ── Public helpers ─────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return bool(st.session_state.get(_KEY_LOGGED_IN))


def is_admin() -> bool:
    return st.session_state.get(_KEY_USER_ROLE) == "admin"


def get_current_sector() -> str | None:
    return st.session_state.get(_KEY_CURRENT_SECTOR)


def set_current_sector(sector_id: str) -> None:
    st.session_state[_KEY_CURRENT_SECTOR] = sector_id


def get_allowed_sectors() -> list[str]:
    user_id    = st.session_state.get(_KEY_USER_ID, "")
    user_reg   = _get_user_registry()
    sector_reg = _get_sector_registry()
    user       = user_reg.get(user_id, {})
    allowed    = user.get("sectors")
    if allowed is None:
        return list(sector_reg.keys())
    return [s for s in allowed if s in sector_reg]


def logout() -> None:
    for key in [_KEY_LOGGED_IN, _KEY_USER_ID, _KEY_USER_ROLE,
                _KEY_USER_LABEL, _KEY_CURRENT_SECTOR, _KEY_LOGIN_ERROR]:
        st.session_state.pop(key, None)
    st.rerun()


def require_login() -> None:
    """인증 게이트. 미로그인 시 로그인 화면 렌더링 후 st.stop()."""
    if not is_logged_in():
        _render_login_page()
        st.stop()


# ── Private: login logic ───────────────────────────────────────────────────

_KEY_LOGIN_ATTEMPTS  = "_hermes_login_attempts"
_KEY_LOCKOUT_UNTIL   = "_hermes_lockout_until"
_MAX_ATTEMPTS        = 5
_LOCKOUT_SECONDS     = 300  # 5 minutes


def _do_login(user_id: str, password: str) -> None:
    # ── Rate limiting (H-01) ──────────────────────────────────────────────
    lockout_until = st.session_state.get(_KEY_LOCKOUT_UNTIL)
    if lockout_until and datetime.now() < lockout_until:
        remaining = int((lockout_until - datetime.now()).total_seconds())
        st.session_state[_KEY_LOGIN_ERROR] = (
            f"Too many failed attempts. Try again in {remaining // 60}m {remaining % 60}s."
        )
        logger.warning(f"Login blocked (lockout): user_id={user_id}")
        return

    user_reg  = _get_user_registry()
    user_info = user_reg.get(user_id, {})
    if not user_info:
        st.session_state[_KEY_LOGIN_ERROR] = "Account not found."
        logger.warning(f"Login failed: user_id={user_id}, reason=not_found")
        return

    # ── Timing-safe password comparison (C-02) ────────────────────────────
    expected = user_info.get("password", "")
    if not hmac.compare_digest(password, expected):
        attempts = st.session_state.get(_KEY_LOGIN_ATTEMPTS, 0) + 1
        st.session_state[_KEY_LOGIN_ATTEMPTS] = attempts
        if attempts >= _MAX_ATTEMPTS:
            st.session_state[_KEY_LOCKOUT_UNTIL] = datetime.now() + timedelta(seconds=_LOCKOUT_SECONDS)
            st.session_state[_KEY_LOGIN_ERROR] = (
                "Too many failed attempts. Locked for 5 minutes."
            )
            logger.warning(f"Login lockout triggered: user_id={user_id}, attempts={attempts}")
        else:
            st.session_state[_KEY_LOGIN_ERROR] = "Incorrect password."
            logger.warning(f"Login failed: user_id={user_id}, attempts={attempts}")
        return

    # ── Success ───────────────────────────────────────────────────────────
    st.session_state[_KEY_LOGGED_IN]  = True
    st.session_state[_KEY_USER_ID]    = user_id
    st.session_state[_KEY_USER_ROLE]  = user_info["role"]
    st.session_state[_KEY_USER_LABEL] = user_info["label"]
    st.session_state.pop(_KEY_LOGIN_ERROR, None)
    st.session_state.pop(_KEY_LOGIN_ATTEMPTS, None)
    st.session_state.pop(_KEY_LOCKOUT_UNTIL, None)

    logger.info(f"Login success: user_id={user_id}, role={user_info['role']}")

    # 기본 Sector: 허용된 첫 번째
    allowed = get_allowed_sectors()
    st.session_state[_KEY_CURRENT_SECTOR] = allowed[0] if allowed else None

    st.rerun()


# ── Private: login page UI ─────────────────────────────────────────────────

def _render_login_page() -> None:
    from src.ui.styles import get_custom_css
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center; color:#ccd6f6; letter-spacing:1px;'>◈ Hermes</h2>"
            "<p style='text-align:center; color:#a8b2d1; margin-top:-8px; font-size:14px;'>"
            "Spatial Intelligence for Real-World Influence</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        user_reg  = _get_user_registry()
        user_ids  = list(user_reg.keys())
        user_labels = {uid: f"{info['icon']}  {info['label']}" for uid, info in user_reg.items()}

        selected_user = st.selectbox(
            "Sector / Account",
            options=user_ids,
            format_func=lambda x: user_labels[x],
            key="_hermes_login_user",
        )

        _render_sector_preview(selected_user)

        pw = st.text_input(
            "Password",
            type="password",
            key="_hermes_login_pw",
            placeholder="Enter password…",
        )

        if st.button("Login", use_container_width=True, type="primary"):
            _do_login(selected_user, pw)

        if err := st.session_state.get(_KEY_LOGIN_ERROR):
            st.error(err)


def _render_sector_preview(user_id: str) -> None:
    """선택된 계정이 접근 가능한 Sector 목록을 카드로 표시."""
    user_reg   = _get_user_registry()
    sector_reg = _get_sector_registry()
    user       = user_reg.get(user_id, {})
    allowed_s  = user.get("sectors")  # None = admin

    if not sector_reg:
        st.caption("No sectors found in Datafile/")
        return

    lines = []
    for sid, info in sector_reg.items():
        has_access = (allowed_s is None) or (sid in (allowed_s or []))
        mark, color = ("✓", "#22c55e") if has_access else ("—", "#5a6785")
        lines.append(
            f"<span style='color:{color}; font-weight:600; margin-right:8px;'>{mark}</span>"
            f"<span style='color:#a8b2d1;'>{info['icon']} {info['label']}</span>"
        )

    st.markdown(
        "<div style='background:#1a1f36; border:1px solid #2d3456; "
        "border-radius:8px; padding:10px 14px; margin:10px 0 14px; font-size:13px; "
        "line-height:1.9;'>"
        + "<br>".join(lines)
        + "</div>",
        unsafe_allow_html=True,
    )
