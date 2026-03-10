"""
Hermes Dashboard — Polished light theme with refined typography and spacing.
Design: clean white surfaces, navy/gold accents, subtle shadows.
"""
DEEP_NAVY = "#0f172a"
NAVY_MID = "#1e293b"
TEXT_DARK = "#1e293b"
TEXT_MID = "#334155"
SLATE_GRAY = "#64748b"
SLATE_LIGHT = "#94a3b8"
GOLD = "#c49a3a"
AMBER = "#d97706"
WHITE = "#ffffff"
BG_SUBTLE = "#f8fafc"
BG_SIDEBAR = "#f1f5f9"
BORDER = "#e2e8f0"
BORDER_LIGHT = "#f1f5f9"
ACCENT_BLUE = "#3b82f6"

CUSTOM_CSS = f"""
<style>
/* ── Global ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {{
    background-color: {BG_SUBTLE};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.main .block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
    background-color: {WHITE};
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}}

/* ── Typography ─────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_DARK} !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}
h1 {{
    font-size: 1.75rem !important;
    border-bottom: 2px solid {GOLD};
    padding-bottom: 0.5rem;
    margin-bottom: 1rem !important;
}}
h2 {{
    font-size: 1.35rem !important;
    margin-top: 1.5rem !important;
    color: {TEXT_DARK} !important;
}}
h3 {{
    font-size: 1.1rem !important;
    color: {TEXT_DARK} !important;
}}
h4 {{
    font-size: 1rem !important;
    color: {TEXT_MID} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em !important;
    margin-top: 1.2rem !important;
}}
p, li, span, div, label {{
    color: {TEXT_DARK} !important;
    line-height: 1.6 !important;
}}

/* ── Metric cards — refined ─────────────────────────────────── */
[data-testid="stMetric"] {{
    background-color: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: box-shadow 0.15s ease;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {DEEP_NAVY} !important;
    letter-spacing: -0.02em !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.78rem !important;
    color: {SLATE_GRAY} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
    color: {TEXT_MID} !important;
}}

/* ── Sidebar — cleaner ──────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {TEXT_DARK} !important;
}}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {{
    color: {TEXT_DARK} !important;
}}
[data-testid="stSidebar"] .stSelectbox label {{
    color: {TEXT_DARK} !important;
}}

/* Sidebar Selectbox */
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="popover"],
[data-testid="stSidebar"] [data-baseweb="popover"] > div,
[data-testid="stSidebar"] [data-baseweb="menu"],
[data-testid="stSidebar"] [data-baseweb="menu"] ul,
[data-testid="stSidebar"] [data-baseweb="menu"] li,
[data-testid="stSidebar"] [data-baseweb="list"] li {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
}}

/* ── DataFrames / Tables ────────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] *,
.stDataFrame,
.stDataFrame * {{
    color: {TEXT_DARK} !important;
}}
[data-testid="stDataFrame"] {{
    background-color: {WHITE} !important;
    border-radius: 8px !important;
    overflow: hidden;
}}
[data-testid="stDataFrame"] > div {{
    background-color: {WHITE} !important;
}}
[data-testid="stDataFrame"] table,
.stDataFrame table,
div[data-testid="stDataFrame"] table {{
    background-color: {WHITE} !important;
}}
[data-testid="stDataFrame"] thead tr,
[data-testid="stDataFrame"] thead th,
.stDataFrame thead tr,
.stDataFrame thead th {{
    background-color: {BG_SUBTLE} !important;
    color: {TEXT_DARK} !important;
    border-color: {BORDER} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
[data-testid="stDataFrame"] tbody tr,
[data-testid="stDataFrame"] tbody td,
.stDataFrame tbody tr,
.stDataFrame tbody td {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
    border-color: {BORDER_LIGHT} !important;
    font-size: 0.88rem !important;
}}
[data-testid="stDataFrame"] tbody tr:hover td {{
    background-color: #f8fafc !important;
    color: {TEXT_DARK} !important;
}}

/* ── Selectbox, Radio, Inputs ───────────────────────────────── */
.stSelectbox label, .stRadio label, .stMultiSelect label {{
    color: {TEXT_DARK} !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}}
[data-baseweb="select"],
[data-baseweb="select"] > div {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
}}
[data-baseweb="input"] {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
}}

/* ── Dropdown menu (popover) — GLOBAL ───────────────────────── */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
div[data-baseweb="popover"],
div[data-baseweb="popover"] div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
}}
[data-baseweb="menu"],
[data-baseweb="menu"] *,
[data-baseweb="menu"] div,
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
}}
[data-baseweb="list"],
[data-baseweb="list"] *,
[data-baseweb="list-item"],
[data-baseweb="list-item"] *,
li[role="option"],
div[role="listbox"],
div[role="listbox"] * {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
}}
[data-baseweb="list-item"]:hover,
[data-baseweb="menu"] li:hover,
div[role="listbox"] li:hover {{
    background-color: {BG_SUBTLE} !important;
    color: {TEXT_DARK} !important;
}}
section[data-baseweb="popover"],
section[data-baseweb="popover"] *,
body > div [data-baseweb="popover"],
body > div [data-baseweb="popover"] * {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
}}

/* ── Expanders ──────────────────────────────────────────────── */
.stCaption {{
    color: {SLATE_GRAY} !important;
    font-size: 0.82rem !important;
}}
.streamlit-expanderHeader {{
    background-color: {WHITE} !important;
    color: {TEXT_DARK} !important;
    border: 1px solid {BORDER};
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}}

/* ── Buttons — refined ──────────────────────────────────────── */
.stButton > button {{
    background-color: {DEEP_NAVY} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
}}
.stButton > button *, .stButton > button p {{
    color: {WHITE} !important;
}}
.stButton > button:hover {{
    background-color: {NAVY_MID} !important;
    color: {WHITE} !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.15) !important;
    transform: translateY(-1px);
}}
.stButton > button:hover * {{
    color: {WHITE} !important;
}}
.stButton > button:focus, .stButton > button:active {{
    color: {WHITE} !important;
}}

/* ── Tabs — gold accent ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
    border-bottom: 2px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    color: {SLATE_GRAY} !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent;
    transition: all 0.15s ease;
}}
.stTabs [aria-selected="true"] {{
    color: {DEEP_NAVY} !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {GOLD} !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_DARK} !important;
}}

/* ── Alerts ─────────────────────────────────────────────────── */
.stAlert {{
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
}}

/* ── Dividers ───────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 1.5rem 0 !important;
}}

/* ── Toggle / Checkbox refinement ───────────────────────────── */
.stToggle label, .stCheckbox label {{
    font-size: 0.88rem !important;
    color: {TEXT_DARK} !important;
}}

/* ── Plotly chart container spacing ─────────────────────────── */
[data-testid="stPlotlyChart"] {{
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Download button — gold accent ──────────────────────────── */
[data-testid="stDownloadButton"] > button {{
    background-color: {GOLD} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background-color: {AMBER} !important;
    box-shadow: 0 2px 8px rgba(196,154,58,0.2) !important;
}}
[data-testid="stDownloadButton"] > button *,
[data-testid="stDownloadButton"] > button p {{
    color: {WHITE} !important;
}}

/* ── Scrollbar — subtle ─────────────────────────────────────── */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: {SLATE_LIGHT};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {SLATE_GRAY};
}}

/* ── Report Preview — PDF-like sections ───────────────────── */
.report-section-title {{
    background: {DEEP_NAVY};
    color: {WHITE} !important;
    padding: 8px 14px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 18px;
    margin-bottom: 10px;
    letter-spacing: 0.01em;
}}
.report-section-title * {{
    color: {WHITE} !important;
}}
.report-ai-box {{
    background: #fcf9f0;
    border-left: 3px solid {GOLD};
    padding: 12px 16px;
    margin: 8px 0 14px 0;
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: {TEXT_DARK} !important;
}}
.report-ai-box * {{
    color: {TEXT_DARK} !important;
}}
.report-header {{
    background: {DEEP_NAVY};
    color: {WHITE} !important;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 16px;
}}
.report-header * {{
    color: {WHITE} !important;
}}
</style>
"""


def get_custom_css() -> str:
    return CUSTOM_CSS
