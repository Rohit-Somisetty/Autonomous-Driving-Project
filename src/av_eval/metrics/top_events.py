"""Helpers for exporting the highest-severity safety events."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .events import EventTable

NEAR_MISS_MIN_TTC = 1e-6


def _output_columns(frames: pd.DataFrame) -> List[str]:
    columns = [
        "event_name",
        "trip_id",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "severity",
        "weather",
        "time_of_day",
        "traffic_density",
    ]
    if "driver_version" in frames.columns:
        columns.append("driver_version")
    return columns


def _compute_severity_scores(df: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(df["severity"], errors="coerce")
    scores = pd.Series(0.0, index=df.index, dtype=float)

    near_mask = df["event_name"] == "near_miss"
    if near_mask.any():
        clipped = np.clip(raw[near_mask], NEAR_MISS_MIN_TTC, None)
        scores.loc[near_mask] = 1.0 / clipped

    brake_mask = df["event_name"] == "hard_braking"
    if brake_mask.any():
        scores.loc[brake_mask] = raw[brake_mask].abs()

    lane_mask = df["event_name"] == "lane_deviation"
    if lane_mask.any():
        scores.loc[lane_mask] = raw[lane_mask].abs()

    other_mask = ~(near_mask | brake_mask | lane_mask)
    if other_mask.any():
        durations = pd.to_numeric(df.loc[other_mask, "duration_s"], errors="coerce").fillna(0.0)
        scores.loc[other_mask] = durations

    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scores


def rank_top_events(
    frames: pd.DataFrame,
    event_tables: Dict[str, EventTable],
    top_n: int,
) -> pd.DataFrame:
    """Return the highest-severity events with contextual metadata."""

    collected: List[pd.DataFrame] = []
    for table in event_tables.values():
        if table.empty:
            continue
        work = table.copy()
        scores = _compute_severity_scores(work)
        work["severity_score"] = scores
        if "severity" in work.columns:
            work = work.drop(columns=["severity"])
        collected.append(work)

    if not collected:
        return pd.DataFrame(columns=_output_columns(frames))

    combined = pd.concat(collected, ignore_index=True)
    meta_cols = ["trip_id", "weather", "time_of_day", "traffic_density"]
    if "driver_version" in frames.columns:
        meta_cols.append("driver_version")
    trip_meta = frames[meta_cols].drop_duplicates("trip_id")
    merged = combined.merge(trip_meta, on="trip_id", how="left")
    merged = merged.rename(columns={"severity_score": "severity"})

    order_cols = ["severity", "duration_s"]
    merged = merged.sort_values(order_cols, ascending=[False, False])
    desired_cols = _output_columns(frames)
    return merged[desired_cols].head(top_n).reset_index(drop=True)
