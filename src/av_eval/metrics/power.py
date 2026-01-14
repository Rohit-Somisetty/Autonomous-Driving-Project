"""Power-analysis helpers for rare event detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


def required_exposure_miles(
    baseline_rate_per_mile: float,
    relative_lift: float,
    alpha: float,
    power: float,
) -> float:
    """Return miles per arm needed to detect a relative rate change.

    Parameters
    ----------
    baseline_rate_per_mile: float
        Baseline event rate expressed per mile (>0).
    relative_lift: float
        Fractional change to detect (candidate vs baseline). Negative values
        indicate improvements (lower rate). Must not be zero.
    alpha: float
        Two-sided significance level (0 < alpha < 1).
    power: float
        Desired statistical power (0 < power < 1).
    """

    if baseline_rate_per_mile <= 0:
        raise ValueError("Baseline rate must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if not (0 < power < 1):
        raise ValueError("power must be in (0,1)")
    if np.isclose(relative_lift, 0.0):
        raise ValueError("relative_lift must be non-zero")

    candidate_rate = baseline_rate_per_mile * (1.0 + relative_lift)
    if candidate_rate <= 0:
        candidate_rate = EPS
    delta = abs(baseline_rate_per_mile - candidate_rate)
    if delta <= 0:
        raise ValueError("Rates must differ for power calculation")

    z_alpha = stats.norm.ppf(1 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    numerator = (z_alpha + z_beta) ** 2 * (baseline_rate_per_mile + candidate_rate)
    exposure = numerator / (delta**2)
    return float(exposure)


def build_power_requirements_table(
    driver_metrics: pd.DataFrame,
    relative_lift: float,
    alpha: float,
    power: float,
) -> pd.DataFrame:
    """Return required exposure miles per arm for each baseline event rate."""

    required_cols = {"event_name", "driver_version", "rate_per_1k_miles"}
    if driver_metrics.empty or not required_cols.issubset(driver_metrics.columns):
        return pd.DataFrame(
            columns=[
                "event_name",
                "baseline_rate_per_1k_miles",
                "target_rate_per_1k_miles",
                "required_miles_per_arm",
            ]
        )

    baseline = (
        driver_metrics[driver_metrics["driver_version"] == "baseline"][
            ["event_name", "rate_per_1k_miles"]
        ]
        .rename(columns={"rate_per_1k_miles": "baseline_rate_per_1k_miles"})
        .copy()
    )
    if baseline.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "baseline_rate_per_1k_miles",
                "target_rate_per_1k_miles",
                "required_miles_per_arm",
            ]
        )

    def _row_calc(row: pd.Series) -> float:
        baseline_rate_per_mile = row["baseline_rate_per_1k_miles"] / 1000.0
        return required_exposure_miles(baseline_rate_per_mile, relative_lift, alpha, power)

    baseline["target_rate_per_1k_miles"] = (
        baseline["baseline_rate_per_1k_miles"] * (1.0 + relative_lift)
    ).clip(lower=0.0)
    baseline["required_miles_per_arm"] = baseline.apply(_row_calc, axis=1)
    baseline = baseline.sort_values("required_miles_per_arm", ascending=False).reset_index(drop=True)
    baseline["relative_lift_pct"] = relative_lift * 100.0
    baseline["alpha"] = alpha
    baseline["power"] = power
    return baseline
