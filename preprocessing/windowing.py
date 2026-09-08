from __future__ import annotations

import pandas as pd

def add_time_windows(
    df: pd.DataFrame,
    timestamp_col: str = "network.timestamp",
    window_seconds: int = 5,
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")

    if out[timestamp_col].isna().all():
        raise ValueError("All timestamps are invalid or missing")

    out = out.sort_values(timestamp_col).reset_index(drop=True)

    min_ts = out[timestamp_col].min()
    delta_seconds = (out[timestamp_col] - min_ts).dt.total_seconds()

    out["network.window_id"] = (delta_seconds // window_seconds).astype("Int64")
    out["network.window_start"] = min_ts + pd.to_timedelta(
        out["network.window_id"] * window_seconds, unit="s"
    )
    out["network.window_end"] = out["network.window_start"] + pd.to_timedelta(
        window_seconds, unit="s"
    )
    return out