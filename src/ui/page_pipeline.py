"""
Hermes Pipeline Tab — 대시보드 내 전처리 실행 & 캐시 현황.

기능
----
- Datafile/ 폴더 자동 탐지 → 전처리 상태 표시
- 캐시 없음: [▶ Run Preprocessing] 버튼
- 캐시 있음 + 새 날짜: [▶ Update Cache] + [🔄 Force Rebuild] 버튼
- 캐시 최신: [🔄 Force Rebuild] 버튼만
- 실행 중: 날짜별 실시간 프로그레스 바 + 로그
- Admin만 실행 버튼 표시; Client는 상태만 열람
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from src.data.space_loader import get_available_dates
from src.config.paths import get_cache_path


# ── 캐시 현황 조회 ─────────────────────────────────────────────────────────

def _get_pipeline_status(space_name: str) -> dict:
    """
    Returns:
        has_cache   : bool
        cached_dates: list[str]  — metadata.json의 date_range
        raw_dates   : list[str]  — rawdata/ 내 CSV 날짜 목록
        new_dates   : list[str]  — raw에 있지만 캐시에 없는 날짜
        created_at  : str | None
        cache_version: str | None
    """
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


# ── 상태 카드 렌더링 ───────────────────────────────────────────────────────

def _render_status_cards(status: dict) -> None:
    c1, c2, c3 = st.columns(3)

    total_raw = len(status["raw_dates"])
    cached_n  = len(status["cached_dates"])
    new_n     = len(status["new_dates"])

    with c1:
        st.metric("Raw Dates", total_raw, help="rawdata/ 폴더의 날짜 CSV 총 수")
    with c2:
        if status["has_cache"]:
            st.metric("Cached Dates", cached_n)
        else:
            st.metric("Cached Dates", "—")
    with c3:
        if new_n > 0:
            st.metric("New / Pending", new_n, delta=f"+{new_n}", delta_color="inverse")
        else:
            st.metric("New / Pending", 0)


def _render_status_banner(status: dict) -> None:
    if not status["has_cache"]:
        st.warning(
            f"⚠️ **Cache not found.** "
            f"{len(status['raw_dates'])} raw date(s) are available and ready to process."
        )
        return

    cached_dates = status["cached_dates"]
    date_range_str = (
        f"{cached_dates[0]} ~ {cached_dates[-1]}" if cached_dates else "—"
    )
    created = ""
    if status["created_at"]:
        try:
            dt = datetime.fromisoformat(status["created_at"])
            created = f" · Built {dt.strftime('%Y-%m-%d %H:%M')}"
        except Exception:
            pass

    if status["new_dates"]:
        st.info(
            f"🔄 **New data available.** "
            f"Cached {len(status['cached_dates'])} dates ({date_range_str}){created}. "
            f"**{len(status['new_dates'])} new date(s)** not yet in cache."
        )
    else:
        st.success(
            f"✅ **Cache up-to-date.** "
            f"{len(status['cached_dates'])} dates ({date_range_str}){created}."
        )


# ── 전처리 실행 ─────────────────────────────────────────────────────────────

def _run_preprocessing(space_name: str, force: bool) -> None:
    """전처리 실행 + 실시간 진행 상황 표시."""
    from src.preprocess.runner import run_preprocess_space

    raw_dates = get_available_dates(space_name)
    if not raw_dates:
        st.error("No raw dates found in rawdata/ folder.")
        return

    st.markdown("---")
    st.markdown(f"**Processing {len(raw_dates)} date(s)…**")

    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    log_container = st.empty()
    logs: list[str] = []

    def on_progress(idx: int, total: int, date_str: str) -> None:
        pct = idx / total
        progress_bar.progress(pct)
        status_text.markdown(
            f"`[{idx + 1}/{total}]` Processing **{date_str}**…"
        )
        logs.append(f"[{idx + 1}/{total}] {date_str}")
        log_container.code("\n".join(logs[-12:]), language=None)

    try:
        ok = run_preprocess_space(space_name, force=force, on_progress=on_progress)
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Preprocessing failed: {exc}")
        return

    progress_bar.progress(1.0)
    status_text.empty()
    log_container.empty()

    if ok:
        st.success(f"✅ Preprocessing complete — {len(raw_dates)} date(s) processed.")
        # 캐시 무효화 후 재실행
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Preprocessing returned False — check logs for details.")


# ── Pipeline 탭 메인 ───────────────────────────────────────────────────────

def render_pipeline_tab(space_name: str) -> None:
    """Pipeline 탭: 캐시 현황 + 전처리 실행."""
    from src.auth import is_admin

    st.markdown("### ⚙️ Pipeline")
    st.caption(f"Space: `{space_name}`")

    status = _get_pipeline_status(space_name)

    _render_status_cards(status)
    st.markdown("")
    _render_status_banner(status)

    # Raw 날짜 목록 expander
    if status["raw_dates"]:
        with st.expander(f"📅 Raw dates ({len(status['raw_dates'])})", expanded=False):
            cols = st.columns(4)
            for i, d in enumerate(status["raw_dates"]):
                is_new    = d in status["new_dates"]
                is_cached = d in status["cached_dates"]
                icon = "🆕" if is_new else ("✅" if is_cached else "📄")
                cols[i % 4].markdown(f"{icon} `{d}`")

    # ── 실행 버튼 (Admin 전용) ──────────────────────────────────────────────
    if not is_admin():
        st.markdown("---")
        st.info("Contact administrator to run or update preprocessing.")
        return

    st.markdown("---")
    st.markdown("##### Run Preprocessing")

    has_cache = status["has_cache"]
    has_new   = len(status["new_dates"]) > 0

    btn_col1, btn_col2, _ = st.columns([2, 2, 3])

    run_main  = False
    run_force = False

    with btn_col1:
        if not has_cache:
            if st.button("▶ Run Preprocessing", type="primary", use_container_width=True):
                run_main = True
        elif has_new:
            if st.button("▶ Update Cache", type="primary", use_container_width=True,
                         help=f"Process {len(status['new_dates'])} new date(s) and rebuild cache"):
                run_main = True

    with btn_col2:
        if has_cache:
            if st.button("🔄 Force Rebuild", use_container_width=True,
                         help="Re-process all raw dates and overwrite existing cache"):
                run_force = True

    if run_main:
        _run_preprocessing(space_name, force=True)
    elif run_force:
        _run_preprocessing(space_name, force=True)
