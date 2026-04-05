"""
Hermes Dashboard — Dark theme inspired by TheHyundaiSeoul.
Design: dark surfaces (#0E1117), gradient cards, teal accents.
"""

BG_COLOR = "#0E1117"
CARD_BG = "#1E2130"
CARD_GRADIENT_START = "#1a1f36"
CARD_GRADIENT_END = "#252b48"
CARD_BORDER = "#2d3456"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#ccd6f6"
TEXT_MUTED = "#a8b2d1"
TEXT_DIMMED = "#8892b0"
GRID_COLOR = "#1a2035"
ACCENT_TEAL = "#64ffda"
ACCENT_GOLD = "#c49a3a"
ACCENT_AMBER = "#d97706"
DEEP_NAVY = "#0f172a"
GOLD = "#c49a3a"
AMBER = "#d97706"
SLATE_GRAY = "#64748b"

CUSTOM_CSS = f"""
<style>
/* -- Global ------------------------------------------------------------ */
.stApp {{
    background-color: {BG_COLOR};
}}

/* -- Metric Card (custom HTML) ----------------------------------------- */
.metric-card {{
    background: linear-gradient(135deg, {CARD_GRADIENT_START} 0%, {CARD_GRADIENT_END} 100%);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid {CARD_BORDER};
}}
.metric-value {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin: 4px 0;
}}
.metric-label {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.metric-sub {{
    font-size: 0.75rem;
    color: {TEXT_DIMMED};
    margin-top: 4px;
}}

/* -- Section Header ---------------------------------------------------- */
.section-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid {CARD_BORDER};
}}

/* -- AI Comment Box ---------------------------------------------------- */
.ai-comment {{
    background: linear-gradient(135deg, #1a2332 0%, #1e2d3d 100%);
    border-left: 3px solid {ACCENT_TEAL};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #b8c9e0;
    line-height: 1.6;
}}

/* -- Global text color override ---------------------------------------- */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {{
    color: {TEXT_MUTED};
}}

/* -- Typography -------------------------------------------------------- */
h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_SECONDARY} !important;
    font-weight: 600 !important;
}}
.stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown label, .stMarkdown div {{
    color: {TEXT_MUTED} !important;
}}

/* -- Streamlit captions & info text ------------------------------------ */
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: {TEXT_DIMMED} !important;
    font-size: 0.82rem !important;
}}

/* -- Streamlit expander header ----------------------------------------- */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
.streamlit-expanderHeader,
.streamlit-expanderHeader * {{
    color: {TEXT_SECONDARY} !important;
    background-color: {CARD_BG} !important;
}}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {{
    color: {TEXT_MUTED} !important;
}}

/* -- Streamlit info/warning/error boxes -------------------------------- */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
.stAlert p, .stAlert span {{
    color: {TEXT_SECONDARY} !important;
}}

/* -- Streamlit Metric (built-in) --------------------------------------- */
[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {CARD_GRADIENT_START} 0%, {CARD_GRADIENT_END} 100%);
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 14px 16px !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: {TEXT_PRIMARY} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.78rem !important;
    color: {TEXT_MUTED} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
}}

/* -- Sidebar ----------------------------------------------------------- */
[data-testid="stSidebar"] {{
    background-color: #0a0e1a !important;
    border-right: 1px solid {CARD_BORDER} !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {TEXT_SECONDARY} !important;
}}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: {TEXT_MUTED} !important;
}}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label {{
    color: {TEXT_SECONDARY} !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{
    color: {TEXT_SECONDARY} !important;
}}

/* -- Selectbox / Radio / Inputs (dark) --------------------------------- */
.stSelectbox label, .stRadio label, .stMultiSelect label {{
    color: {TEXT_SECONDARY} !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}}
[data-baseweb="select"],
[data-baseweb="select"] > div {{
    background-color: #161b2e !important;
    color: {TEXT_SECONDARY} !important;
    border-radius: 8px !important;
    border-color: {CARD_BORDER} !important;
}}
[data-baseweb="input"] {{
    background-color: #161b2e !important;
    color: {TEXT_SECONDARY} !important;
    border-radius: 8px !important;
}}

/* -- Dropdown menu (popover) ------------------------------------------- */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
[data-baseweb="list"],
[data-baseweb="list"] *,
[data-baseweb="list-item"],
[data-baseweb="list-item"] *,
li[role="option"],
div[role="listbox"],
div[role="listbox"] * {{
    background-color: #161b2e !important;
    color: {TEXT_SECONDARY} !important;
}}
[data-baseweb="list-item"]:hover,
[data-baseweb="menu"] li:hover,
div[role="listbox"] li:hover {{
    background-color: {CARD_GRADIENT_END} !important;
}}

/* -- DataFrames / Tables ----------------------------------------------- */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] * {{
    color: {TEXT_SECONDARY} !important;
}}
[data-testid="stDataFrame"] {{
    background-color: {CARD_BG} !important;
    border-radius: 8px !important;
    overflow: hidden;
}}
[data-testid="stDataFrame"] thead tr,
[data-testid="stDataFrame"] thead th {{
    background-color: {CARD_GRADIENT_START} !important;
    color: {TEXT_SECONDARY} !important;
    border-color: {CARD_BORDER} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}
[data-testid="stDataFrame"] tbody tr,
[data-testid="stDataFrame"] tbody td {{
    background-color: {CARD_BG} !important;
    color: {TEXT_MUTED} !important;
    border-color: {CARD_BORDER} !important;
}}

/* -- Buttons ----------------------------------------------------------- */
.stButton > button {{
    background-color: {DEEP_NAVY} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
}}
.stButton > button *,
.stButton > button p {{
    color: {TEXT_PRIMARY} !important;
}}
.stButton > button:hover {{
    background-color: #1e293b !important;
    border-color: {ACCENT_TEAL} !important;
}}

/* -- Expanders (border styling) ---------------------------------------- */
[data-testid="stExpander"] {{
    border: 1px solid {CARD_BORDER};
    border-radius: 8px !important;
}}

/* -- Tabs (gold accent) ------------------------------------------------ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
    border-bottom: 2px solid {CARD_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_MUTED} !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {TEXT_SECONDARY} !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {ACCENT_TEAL} !important;
}}

/* -- Alerts ------------------------------------------------------------ */
.stAlert {{
    color: {TEXT_SECONDARY} !important;
    border-radius: 8px !important;
}}

/* -- Dividers ---------------------------------------------------------- */
hr {{
    border: none !important;
    border-top: 1px solid {CARD_BORDER} !important;
    margin: 1.5rem 0 !important;
}}

/* -- Slider ------------------------------------------------------------ */
.stSlider label {{
    color: {TEXT_SECONDARY} !important;
}}

/* -- Download button --------------------------------------------------- */
[data-testid="stDownloadButton"] > button {{
    background-color: {ACCENT_GOLD} !important;
    color: {TEXT_PRIMARY} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background-color: {AMBER} !important;
}}
[data-testid="stDownloadButton"] > button *,
[data-testid="stDownloadButton"] > button p {{
    color: {TEXT_PRIMARY} !important;
}}

/* -- Scrollbar --------------------------------------------------------- */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: {TEXT_DIMMED};
    border-radius: 3px;
}}

/* -- Report sections (preserved for view_report.py) -------------------- */
.report-section-title {{
    background: {DEEP_NAVY};
    color: {TEXT_PRIMARY} !important;
    padding: 8px 14px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 18px;
    margin-bottom: 10px;
}}
.report-section-title * {{
    color: {TEXT_PRIMARY} !important;
}}
.report-ai-box {{
    background: linear-gradient(135deg, #1a2332 0%, #1e2d3d 100%);
    border-left: 3px solid {ACCENT_GOLD};
    padding: 12px 16px;
    margin: 8px 0 14px 0;
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #b8c9e0 !important;
}}
.report-ai-box * {{
    color: #b8c9e0 !important;
}}
.report-header {{
    background: {DEEP_NAVY};
    color: {TEXT_PRIMARY} !important;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 16px;
}}
.report-header * {{
    color: {TEXT_PRIMARY} !important;
}}
</style>
"""


def get_custom_css() -> str:
    return CUSTOM_CSS
