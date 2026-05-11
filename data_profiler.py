"""
data_profiler.py
================
Lightweight data profiling module for the Astrikos framework.

Generates a JSON-serializable summary of a DataFrame that can be
sent to the LLM layer for use-case inference and pipeline generation.
"""

from __future__ import annotations

import pandas as pd


def generate_profile(df: pd.DataFrame) -> dict:
    """Return a lightweight, JSON-serializable profile of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to profile.

    Returns
    -------
    dict
        A dictionary with the following structure::

            {
                "row_count": int,
                "column_count": int,
                "columns": [
                    {
                        "name": str,
                        "dtype": str,
                        "null_count": int,
                        "null_pct": float,
                    },
                    ...
                ]
            }
    """
    print("[PROFILER] generate_profile  ->  Profiling DataFrame...")

    columns_info = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        columns_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) > 0 else 0.0,
        })

    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
    }

    print(f"[PROFILER] generate_profile  ->  {profile['row_count']} rows, {profile['column_count']} columns")
    for col in columns_info:
        flag = f"  ** {col['null_count']} NaNs **" if col["null_count"] > 0 else ""
        print(f"  - {col['name']:20s}  {col['dtype']:15s}  nulls: {col['null_count']}{flag}")

    return profile


# ── CLI quick test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    df = pd.read_csv("data/dummy_electricity_data.csv", parse_dates=["timestamp"])
    profile = generate_profile(df)

    print("\n[JSON output]")
    print(json.dumps(profile, indent=2))
