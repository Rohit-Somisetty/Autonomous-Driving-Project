"""Safety event detectors."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ..config import EventThresholds

EventTable = pd.DataFrame
SeverityFunc = Callable[[pd.DataFrame], float]

EVENT_COLUMNS = [
    "event_name",
    "trip_id",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "severity",
]


def _as_event_df(records: List[dict]) -> EventTable:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return df.reindex(columns=EVENT_COLUMNS)


def _finalize_runs(
    trip_df: pd.DataFrame,
    condition: np.ndarray,
    min_duration_s: float,
    event_name: str,
    severity_fn: SeverityFunc | None = None,
) -> List[dict]:
    """Return event dicts for contiguous True runs meeting duration criteria."""

    timestamps = trip_df["timestamp_s"].to_numpy()
    dt = trip_df["dt_s"].to_numpy()
    events: List[dict] = []
    start_idx: int | None = None

    for idx, active in enumerate(condition):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            end_idx = idx - 1
            _maybe_append_event(
                events,
                trip_df,
                start_idx,
                end_idx,
                timestamps,
                dt,
                min_duration_s,
                event_name,
                severity_fn,
            )
            start_idx = None

    if start_idx is not None:
        _maybe_append_event(
            events,
            trip_df,
            start_idx,
            len(condition) - 1,
            timestamps,
            dt,
            min_duration_s,
            event_name,
            severity_fn,
        )
    return events


def _maybe_append_event(
    events: List[dict],
    trip_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    timestamps: np.ndarray,
    dt: np.ndarray,
    min_duration_s: float,
    event_name: str,
    severity_fn: SeverityFunc | None,
) -> None:
    duration = timestamps[end_idx] - timestamps[start_idx] + dt[end_idx]
    if duration < min_duration_s:
        return

    event_slice = trip_df.iloc[start_idx : end_idx + 1]
    events.append(
        {
            "event_name": event_name,
            "trip_id": trip_df["trip_id"].iat[0],
            "start_time_s": float(timestamps[start_idx]),
            "end_time_s": float(timestamps[end_idx] + dt[end_idx]),
            "duration_s": float(duration),
            "severity": float(severity_fn(event_slice)) if severity_fn else None,
        }
    )


def detect_disengagements(frames: pd.DataFrame) -> EventTable:
    """Detect disengagement segments (engaged False)."""

    events: List[dict] = []
    for _, trip_df in frames.groupby("trip_id"):
        trip_df = trip_df.sort_values("timestamp_s")
        disengaged_mask = ~trip_df["engaged"].to_numpy()
        events.extend(
            _finalize_runs(
                trip_df,
                disengaged_mask,
                min_duration_s=trip_df["dt_s"].min(),
                event_name="disengagement",
                severity_fn=lambda rows: float(
                    rows["timestamp_s"].iloc[-1] - rows["timestamp_s"].iloc[0]
                ),
            )
        )
    return _as_event_df(events)


def detect_hard_braking(
    frames: pd.DataFrame, thresholds: EventThresholds
) -> EventTable:
    """Detect hard braking events below the configured acceleration threshold."""

    events: List[dict] = []
    for _, trip_df in frames.groupby("trip_id"):
        trip_df = trip_df.sort_values("timestamp_s")
        condition = trip_df["ego_accel_mps2"].to_numpy() < thresholds.hard_brake_accel_mps2
        events.extend(
            _finalize_runs(
                trip_df,
                condition,
                min_duration_s=thresholds.hard_brake_min_duration_s,
                event_name="hard_braking",
                severity_fn=lambda rows: float(rows["ego_accel_mps2"].min()),
            )
        )
    return _as_event_df(events)


def detect_lane_deviation(
    frames: pd.DataFrame, thresholds: EventThresholds
) -> EventTable:
    """Detect sustained lateral deviations."""

    events: List[dict] = []
    for _, trip_df in frames.groupby("trip_id"):
        trip_df = trip_df.sort_values("timestamp_s")
        condition = (
            np.abs(trip_df["lane_offset_m"].to_numpy()) > thresholds.lane_deviation_m
        )
        events.extend(
            _finalize_runs(
                trip_df,
                condition,
                min_duration_s=thresholds.lane_deviation_min_duration_s,
                event_name="lane_deviation",
                severity_fn=lambda rows: float(rows["lane_offset_m"].abs().max()),
            )
        )
    return _as_event_df(events)


def detect_near_miss(frames: pd.DataFrame, thresholds: EventThresholds) -> EventTable:
    """Detect near-miss TTC drops."""

    events: List[dict] = []
    for _, trip_df in frames.groupby("trip_id"):
        trip_df = trip_df.sort_values("timestamp_s")
        condition = trip_df["ttc_s"].to_numpy() < thresholds.near_miss_ttc_s
        events.extend(
            _finalize_runs(
                trip_df,
                condition,
                min_duration_s=thresholds.near_miss_min_duration_s,
                event_name="near_miss",
                severity_fn=lambda rows: float(rows["ttc_s"].min()),
            )
        )
    return _as_event_df(events)


def detect_all_events(
    frames: pd.DataFrame, thresholds: EventThresholds
) -> Dict[str, EventTable]:
    """Run all detectors and return a mapping from event name to table."""

    return {
        "disengagement": detect_disengagements(frames),
        "hard_braking": detect_hard_braking(frames, thresholds),
        "lane_deviation": detect_lane_deviation(frames, thresholds),
        "near_miss": detect_near_miss(frames, thresholds),
    }
