"""
Plotly chart theme: clean white background, refined typography, subtle grid.
"""
TEXT_DARK = "#1e293b"
TEXT_MID = "#334155"
SLATE_LIGHT = "#94a3b8"
GRID = "#f1f5f9"
GRID_ZERO = "#e2e8f0"

PLOTLY_LAYOUT = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color": TEXT_DARK,
        "size": 12,
    },
    "title": {
        "font": {"color": TEXT_DARK, "size": 14, "weight": 600},
        "x": 0.02,
        "xanchor": "left",
    },
    "xaxis": {
        "tickfont": {"color": TEXT_MID, "size": 11},
        "title_font": {"color": TEXT_MID, "size": 12},
        "gridcolor": GRID,
        "zerolinecolor": GRID_ZERO,
        "linecolor": GRID_ZERO,
        "showline": True,
        "linewidth": 1,
    },
    "yaxis": {
        "tickfont": {"color": TEXT_MID, "size": 11},
        "title_font": {"color": TEXT_MID, "size": 12},
        "gridcolor": GRID,
        "zerolinecolor": GRID_ZERO,
        "gridwidth": 1,
    },
    "legend": {
        "font": {"color": TEXT_MID, "size": 11},
        "bgcolor": "rgba(255,255,255,0.9)",
        "bordercolor": "rgba(0,0,0,0)",
    },
    "coloraxis": {
        "colorbar": {
            "tickfont": {"color": TEXT_MID},
            "title_font": {"color": TEXT_MID},
        },
    },
    "margin": {"l": 50, "r": 30, "t": 50, "b": 40},
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 12,
        "font_color": TEXT_DARK,
        "bordercolor": GRID_ZERO,
    },
}


def apply_theme(fig):
    """Apply polished Hermes theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
