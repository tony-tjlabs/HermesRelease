"""
Hermes Platform — Sector & User Registry (Dynamic).

Sector는 Datafile/ 폴더를 자동 탐지하여 동적으로 생성된다.
REGISTERED_SECTORS에 등록된 Sector는 사람이 읽기 좋은 레이블·아이콘이 적용된다.
등록되지 않은 Sector도 폴더가 존재하면 자동으로 추가된다.

User Registry 역시 자동으로 생성된다:
  - administrator: 항상 존재, 모든 Sector 접근 가능
  - [sector_id]: Sector당 1개 자동 생성, 해당 Sector만 접근 가능
"""
import os


def _get_secret(key: str, fallback: str = "") -> str:
    """Retrieve secret from st.secrets, then os.environ, then fallback."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, fallback)


# Passwords: prefer st.secrets / env vars, fallback to defaults for local dev
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", "wonderful2!!")

# ── 알려진 Sector 메타데이터 ────────────────────────────────────────────────
# key : Datafile/ 하위 폴더명 (space_name)
# 폴더가 존재할 때만 활성화된다. 폴더 없으면 무시된다.
REGISTERED_SECTORS: dict[str, dict] = {
    "Victor_Suwon_Starfield": {
        "label":    "Victor Suwon Starfield",
        "brand":    "Victor",
        "icon":     "🏋️",
        "password": _get_secret("PW_VICTOR_SUWON", "wonderful2$"),
        "description": "Sports & lifestyle retail store located on the 6th floor of Starfield Suwon, a large-scale entertainment complex (shopping mall + cinema + leisure). Visitors typically arrive with shopping intent; foot traffic is heavily influenced by mall-wide events, weekends, and seasonal sports trends.",
        "store_type": "sports_retail",
        "location": "Starfield Suwon 6F, Suwon, Gyeonggi-do",
    },
    "GS25_Yeoksam": {
        "label":    "GS25 역삼홍인점",
        "brand":    "GS25",
        "icon":     "🏪",
        "password": _get_secret("PW_GS25_YEOKSAM", "wonderful3@"),
        "description": "Convenience store (CVS) on a side street off the main road in Yeoksam-dong, Gangnam, Seoul. Surrounded by residential buildings but adjacent to a major commercial road. High weekday foot traffic from office workers and residents. Visit patterns driven by commute hours, lunch breaks, and late-night demand. Sensor 2100029C at entrance captures street-level foot traffic; 210000FD near POS captures actual in-store visitors.",
        "store_type": "convenience_store",
        "location": "봉은사로30길 43 1F, Yeoksam-dong, Gangnam-gu, Seoul",
    },
}


# ── 동적 Registry 빌더 ─────────────────────────────────────────────────────

def build_sector_registry() -> dict[str, dict]:
    """
    Datafile/ 폴더 스캔 + REGISTERED_SECTORS 메타데이터를 병합해 Sector Registry 생성.

    - REGISTERED_SECTORS에 등록된 폴더: 등록된 레이블·아이콘 사용
    - 미등록 폴더: 폴더명 기반 자동 레이블 + 기본 아이콘
    - 폴더가 없으면 REGISTERED_SECTORS에 있어도 제외됨
    """
    from src.data.space_loader import discover_spaces
    registry: dict[str, dict] = {}
    for space_name in discover_spaces():
        meta = REGISTERED_SECTORS.get(space_name, {})
        registry[space_name] = {
            "label":    meta.get("label", space_name.replace("_", " ")),
            "brand":    meta.get("brand", space_name.split("_")[0]),
            "icon":     meta.get("icon", "🏬"),
            "password": meta.get("password", ADMIN_PASSWORD),
            "status":   "active",
        }
    return registry


def build_user_registry() -> dict[str, dict]:
    """
    Sector Registry 기반으로 User Registry 동적 생성.

    administrator: sectors=None → 모든 active Sector 접근
    [sector_id]  : sectors=[sector_id] → 해당 Sector만 접근
    """
    sector_reg = build_sector_registry()
    users: dict[str, dict] = {
        "administrator": {
            "label":    "Administrator",
            "icon":     "⚙️",
            "role":     "admin",
            "password": ADMIN_PASSWORD,
            "sectors":  None,
        }
    }
    for sector_id, info in sector_reg.items():
        users[sector_id] = {
            "label":    info["label"],
            "icon":     info["icon"],
            "role":     "client",
            "password": info["password"],
            "sectors":  [sector_id],
        }
    return users


def get_allowed_sectors_for_user(user_id: str) -> list[str]:
    """유저가 접근 가능한 Sector 목록 반환 (동적 레지스트리 기반)."""
    user_reg   = build_user_registry()
    sector_reg = build_sector_registry()
    user       = user_reg.get(user_id, {})
    allowed    = user.get("sectors")   # None = admin (전체)
    if allowed is None:
        return list(sector_reg.keys())
    return [s for s in allowed if s in sector_reg]
