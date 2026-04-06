"""
MAC Stitching — stub for cloud deployment.
Core stitching logic is not included in the release build.
"""

import pandas as pd


def stitching_evaluation(sessions_all, sessions_stitched, sessions_ref):
    """Return empty evaluation dict (stitching not available in cloud)."""
    return {
        "total_sessions": 0,
        "stitched_sessions": 0,
        "stitching_rate": 0.0,
        "note": "Stitching evaluation not available in cloud deployment.",
    }


def stitching_daily_summary(sessions_all, sessions_stitched, sessions_ref):
    """Return empty DataFrame (stitching not available in cloud)."""
    return pd.DataFrame()
