"""Synthetic log generator for autonomous driving evaluation."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import SyntheticGeneratorConfig

METERS_PER_MILE = 1609.34


def _simulate_engagement(rng: np.random.Generator, steps: int) -> np.ndarray:
    """Return a boolean engagement mask with rare disengagement streaks."""

    engaged = np.ones(steps, dtype=bool)
    idx = 0
    while idx < steps:
        if rng.random() < 0.002:
            disengage_len = int(rng.integers(5, 30))
            engaged[idx : min(steps, idx + disengage_len)] = False
            idx += disengage_len
        idx += 1
    return engaged


def _generate_trip(
    rng: np.random.Generator,
    trip_id: str,
    cfg: SyntheticGeneratorConfig,
    weather: str,
    time_of_day: str,
    traffic: str,
    version_tag: str,
    log_date: pd.Timestamp,
    driver_version: str,
) -> pd.DataFrame:
    duration = float(rng.uniform(cfg.min_duration_s, cfg.max_duration_s))
    steps = max(3, int(duration / cfg.dt_s))
    timestamps = np.arange(steps, dtype=float) * cfg.dt_s

    is_candidate = driver_version == "candidate"
    accel_scale = cfg.candidate_accel_scale if is_candidate else 1.0
    risk_scale = cfg.candidate_risk_scale if is_candidate else 1.0

    accel_noise = rng.normal(0, 0.6 * accel_scale, size=steps)
    base_speed = rng.uniform(5, 35)
    speed = np.clip(base_speed + np.cumsum(accel_noise) * cfg.dt_s, 0, 45)
    accel = np.gradient(speed, cfg.dt_s)

    lane_drift = rng.normal(0, 0.02)
    lane_noise = rng.normal(0, 0.1, size=steps)
    lane_offset = np.cumsum(rng.normal(lane_drift, 0.02, size=steps)) * 0.2 + lane_noise

    ttc = rng.normal(8, 2.0, size=steps)
    base_near_miss = max(1, steps // 40)
    near_miss_count = max(1, int(round(base_near_miss * risk_scale)))
    near_miss_count = min(steps, near_miss_count)
    near_miss_indices = rng.choice(steps, size=near_miss_count, replace=False)
    low, high = (0.8, 1.2) if is_candidate else (0.4, 1.2)
    ttc[near_miss_indices] = rng.uniform(low, high, size=near_miss_indices.size)
    ttc = np.clip(ttc, 0.1, None)

    engaged = _simulate_engagement(rng, steps)

    df = pd.DataFrame(
        {
            "trip_id": trip_id,
            "timestamp_s": timestamps,
            "dt_s": cfg.dt_s,
            "ego_speed_mps": speed,
            "ego_accel_mps2": accel,
            "lane_offset_m": lane_offset,
            "ttc_s": ttc,
            "engaged": engaged,
            "weather": weather,
            "time_of_day": time_of_day,
            "traffic_density": traffic,
            "log_version": version_tag,
            "log_date": log_date,
            "driver_version": driver_version,
        }
    )

    df["distance_increment_m"] = df["ego_speed_mps"] * df["dt_s"]
    trip_distance_m = df["distance_increment_m"].sum()
    duration_s = timestamps[-1] + cfg.dt_s
    df["distance_m"] = trip_distance_m
    df["distance_miles"] = trip_distance_m / METERS_PER_MILE
    df["duration_s"] = duration_s
    return df.drop(columns=["distance_increment_m"])


def generate_synthetic_logs(cfg: SyntheticGeneratorConfig) -> pd.DataFrame:
    """Generate a synthetic dataset following the canonical schema."""

    rng = np.random.default_rng(cfg.seed)
    frames: List[pd.DataFrame] = []
    base_date = pd.Timestamp("2025-01-01")

    num_candidate = int(round(cfg.num_trips * cfg.candidate_share))
    num_candidate = min(cfg.num_trips, max(0, num_candidate))
    driver_assignments = ["candidate"] * num_candidate + [
        "baseline"
    ] * (cfg.num_trips - num_candidate)
    rng.shuffle(driver_assignments)

    for idx in range(cfg.num_trips):
        trip_id = f"trip_{idx:05d}"
        weather = rng.choice(cfg.weather_options)
        time_of_day = rng.choice(cfg.time_of_day_options)
        traffic = rng.choice(cfg.traffic_density_options)
        version_tag = rng.choice(cfg.version_tags)
        log_date = base_date + pd.to_timedelta(int(rng.integers(0, 90)), unit="D")
        driver_version = driver_assignments[idx] if driver_assignments else "baseline"

        trip_df = _generate_trip(
            rng,
            trip_id,
            cfg,
            weather,
            time_of_day,
            traffic,
            version_tag,
            log_date,
            driver_version,
        )
        frames.append(trip_df)

    df = pd.concat(frames, ignore_index=True)
    return df
