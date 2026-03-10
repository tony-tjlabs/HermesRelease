"""
Hermes 분석 상수.
BLE Traffic Analysis Logic (S-Ward Specific) 기준.
"""
# 시간: time_index 1 = 10초
TIME_UNIT_SECONDS = 10
# 1분 = 6 slots
SLOTS_PER_MINUTE = 6

# 방문객 진입: 1분 윈도우 내 최소 신호 수
MIN_HITS_PER_MIN = 3
# 퇴장 유예 (초): 이 시간 동안 재감지 없으면 세션 종료
EXIT_BUFFER_SEC = 120      # Android (type == 10)
APPLE_EXIT_BUFFER_SEC = 180  # Apple (type == 1)

# RSSI (sward_config 사용, 없을 때 기본값)
DEFAULT_RSSI_THRESHOLD = -80
# 입구 센서가 없고 내부만 있을 때 유동인구용 RSSI
FALLBACK_FLOATING_RSSI = -90

# Android는 Apple보다 신호세기가 약 10dBm 낮으므로 별도 오프셋 적용
# Apple threshold : sward_config.csv의 rssi_threshold 값 그대로
# Android threshold: rssi_threshold + ANDROID_RSSI_OFFSET
ANDROID_RSSI_OFFSET = -10

# 디바이스 타입 (raw 컬럼 type)
DEVICE_TYPE_APPLE = 1
DEVICE_TYPE_ANDROID = 10

# sward_config install_location 값
INSTALL_INSIDE = "inside_of_store"
INSTALL_ENTRANCE = "entrance_of_store"

# 시간 변환
SECONDS_PER_HOUR = 3600
MINUTES_PER_DAY = 1440

# 체류 시간 세그먼트 경계 (초)
DWELL_SHORT_MAX = 180    # 3분 미만 → Short
DWELL_MEDIUM_MAX = 600   # 10분 미만 → Medium, 이상 → Long

# ── MAC Stitching v2 ──────────────────────────────────────────────
# Level 1: Raw-Signal Stitching (실증 기반 — 직원 MAC 교체 1,800건 분석)
STITCH_GAP_MAX_SLOTS = 3           # 최대 갭 허용 (30초)
STITCH_RSSI_DIFF_APPLE = 3         # Apple RSSI 임계값 (실증: 71.5%가 3dBm 이내)
STITCH_RSSI_DIFF_ANDROID = 5       # Android RSSI 임계값 (실증: 77.1%가 3dBm 이내)
STITCH_LOOKAHEAD_SLOTS = 30        # MAC-A 재출현 확인 윈도우 (5분)

# Level 2: Session Post-Hoc Stitching
SESSION_STITCH_GAP_SEC_APPLE = 300    # 세션 간 최대 갭 (Apple, 5분)
SESSION_STITCH_GAP_SEC_ANDROID = 200  # 세션 간 최대 갭 (Android, 3.3분)
SESSION_STITCH_SHORT_THRESHOLD = 300  # 단편 세션 기준 (5분)
SESSION_STITCH_RSSI_DIFF = 5         # 세션 RSSI 차이 임계값 (실증 기반)
SESSION_STITCH_MAX_CHAIN = 5          # 최대 체인 깊이

# ── 매장 운영 시간 ──────────────────────────────────────────────────
# CVR(방문율) 계산 기준: 이 범위 외 FP·방문자는 분모/분자에서 제외
STORE_OPEN_HOUR  = 10   # 오전 10시 (포함, inclusive)
STORE_CLOSE_HOUR = 22   # 오후 10시 (미포함, exclusive) → hour 10~21 유효

# 타임존
TIMEZONE_STR = "Asia/Seoul"
UTC_OFFSET = "+09:00"
