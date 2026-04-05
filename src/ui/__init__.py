"""
Hermes UI package — v2 (Dark Theme).

View modules:
- view_dashboard : Daily Analysis + Period Comparison (single-scroll)
- view_report    : PDF report generation
- view_pipeline  : Preprocessing + MAC Stitching (Admin only)

Utilities:
- styles       : Dark theme CSS (TheHyundaiSeoul-inspired)
- helpers      : render_metric_card, make_plotly_layout, etc.
- chart_theme  : Plotly theme (apply_theme for backward compat)
"""
from .styles import get_custom_css
from .chart_theme import apply_theme, apply_theme_light

# View modules
from .view_dashboard import render_dashboard
from .view_report import render_report
from .view_pipeline import render_pipeline_view

# Page modules (used by view_pipeline internally)
from .page_stitching import render_stitching_tab
from .page_pipeline import render_pipeline_tab

__all__ = [
    # Core
    "get_custom_css",
    "apply_theme",
    "apply_theme_light",
    # Views
    "render_dashboard",
    "render_report",
    "render_pipeline_view",
    # Page modules
    "render_stitching_tab",
    "render_pipeline_tab",
]
