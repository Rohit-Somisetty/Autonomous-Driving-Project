"""Unit tests for event detectors."""

from __future__ import annotations

import pandas as pd

from av_eval.config import EventThresholds
from av_eval.metrics import events as detectors


def _build_frame_df() -> pd.DataFrame:
    timestamps = [i * 0.1 for i in range(20)]
    data = {
        "trip_id": ["trip_test"] * 20,
        "timestamp_s": timestamps,
        "dt_s": [0.1] * 20,
        "ego_speed_mps": [10.0] * 20,
        "ego_accel_mps2": [0.0] * 5 + [-4.0] * 5 + [0.0] * 10,
        "lane_offset_m": [0.2] * 5 + [0.8] * 10 + [0.1] * 5,
        "ttc_s": [3.0] * 10 + [1.0] * 4 + [3.0] * 6,
        "engaged": [True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        "weather": ["clear"] * 20,
        "time_of_day": ["day"] * 20,
        "traffic_density": ["light"] * 20,
        "distance_m": [100.0] * 20,
        "distance_miles": [100.0 / 1609.34] * 20,
        "duration_s": [timestamps[-1] + 0.1] * 20,
        "log_date": [pd.Timestamp("2025-01-01")] * 20,
    }
    return pd.DataFrame(data)


def test_detect_disengagements() -> None:
    df = _build_frame_df()
    events = detectors.detect_disengagements(df)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["duration_s"] > 0.1


def test_detect_hard_braking() -> None:
    df = _build_frame_df()
    thresholds = EventThresholds()
    events = detectors.detect_hard_braking(df, thresholds)
    assert len(events) == 1
    assert events.iloc[0]["severity"] <= thresholds.hard_brake_accel_mps2


def test_detect_lane_deviation() -> None:
    df = _build_frame_df()
    thresholds = EventThresholds()
    events = detectors.detect_lane_deviation(df, thresholds)
    assert len(events) == 1
    assert events.iloc[0]["duration_s"] >= thresholds.lane_deviation_min_duration_s


def test_detect_near_miss() -> None:
    df = _build_frame_df()
    thresholds = EventThresholds()
    events = detectors.detect_near_miss(df, thresholds)
    assert len(events) == 1
    assert events.iloc[0]["severity"] < thresholds.near_miss_ttc_s
