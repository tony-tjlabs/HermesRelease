"""
Plotly chart theme — dark variant for Hermes dashboard.

On-screen charts use the dark palette (#0E1117 background).
PDF/print charts use apply_theme_light() for white backgrounds.
"""
# -- Dark palette (dashboard) --
BG_COLOR = "#0E1117"
TEXT_SECONDARY = "#ccd6f6"
TEXT_MUTED = "#8892b0"
GRID_COLOR = "#1a2035"

PLOTLY_LAYOUT = {
    "paper_bgcolor": BG_COLOR,
    "plot_bgcolor": BG_COLOR,
    "font": {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color": TEXT_MUTED,
        "size": 12,
    },
    "title": {
        "font": {"color": TEXT_SECONDARY, "size": 14, "weight": 600},
        "x": 0.02,
        "xanchor": "left",
    },
    "xaxis": {
        "tickfont": {"color": TEXT_MUTED, "size": 11},
        "title_font": {"color": TEXT_MUTED, "size": 12},
        "gridcolor": GRID_COLOR,
        "zerolinecolor": GRID_COLOR,
        "linecolor": GRID_COLOR,
        "showline": True,
        "linewidth": 1,
    },
    "yaxis": {
        "tickfont": {"color": TEXT_MUTED, "size": 11},
        "title_font": {"color": TEXT_MUTED, "size": 12},
        "gridcolor": GRID_COLOR,
        "zerolinecolor": GRID_COLOR,
        "gridwidth": 1,
    },
    "legend": {
        "font": {"color": TEXT_MUTED, "size": 11},
        "bgcolor": "rgba(14,17,23,0.8)",
        "bordercolor": "rgba(0,0,0,0)",
    },
    "coloraxis": {
        "colorbar": {
            "tickfont": {"color": TEXT_MUTED},
            "title_font": {"color": TEXT_MUTED},
        },
    },
    "margin": {"l": 50, "r": 30, "t": 50, "b": 40},
    "hoverlabel": {
        "bgcolor": "#1E2130",
        "font_size": 12,
        "font_color": TEXT_SECONDARY,
        "bordercolor": GRID_COLOR,
    },
}

# -- Light palette (PDF / print) --
_TEXT_DARK = "#1e293b"
_TEXT_MID = "#334155"
_GRID_LIGHT = "#f1f5f9"
_GRID_ZERO_LIGHT = "#e2e8f0"

PLOTLY_LAYOUT_LIGHT = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color": _TEXT_DARK,
        "size": 12,
    },
    "title": {
        "font": {"color": _TEXT_DARK, "size": 14, "weight": 600},
        "x": 0.02,
        "xanchor": "left",
    },
    "xaxis": {
        "tickfont": {"color": _TEXT_MID, "size": 11},
        "title_font": {"color": _TEXT_MID, "size": 12},
        "gridcolor": _GRID_LIGHT,
        "zerolinecolor": _GRID_ZERO_LIGHT,
        "linecolor": _GRID_ZERO_LIGHT,
        "showline": True,
        "linewidth": 1,
    },
    "yaxis": {
        "tickfont": {"color": _TEXT_MID, "size": 11},
        "title_font": {"color": _TEXT_MID, "size": 12},
        "gridcolor": _GRID_LIGHT,
        "zerolinecolor": _GRID_ZERO_LIGHT,
        "gridwidth": 1,
    },
    "legend": {
        "font": {"color": _TEXT_MID, "size": 11},
        "bgcolor": "rgba(255,255,255,0.9)",
        "bordercolor": "rgba(0,0,0,0)",
    },
    "coloraxis": {
        "colorbar": {
            "tickfont": {"color": _TEXT_MID},
            "title_font": {"color": _TEXT_MID},
        },
    },
    "margin": {"l": 50, "r": 30, "t": 50, "b": 40},
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 12,
        "font_color": _TEXT_DARK,
        "bordercolor": _GRID_ZERO_LIGHT,
    },
}


def apply_theme(fig):
    """Apply dark Hermes theme to a Plotly figure (dashboard display)."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def apply_theme_light(fig):
    """Apply light theme to a Plotly figure (PDF / print output)."""
    fig.update_layout(**PLOTLY_LAYOUT_LIGHT)
    return fig
