"""Driver-version A/B comparison utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import integrate, stats

from ..config import EvaluationConfig
from .events import EventTable
from .slicing import build_slice_metrics

PRIOR_ALPHA = 0.5
PRIOR_BETA = 0.5

AB_VALUE_COLUMNS = [
    "baseline_event_count",
    "baseline_exposure_miles",
    "baseline_rate_per_1k_miles",
    "candidate_event_count",
    "candidate_exposure_miles",
    "candidate_rate_per_1k_miles",
    "delta_rate_per_1k_miles",
    "pct_change",
    "prob_candidate_better",
    "delta_ci_low",
    "delta_ci_high",
    "z_score",
    "p_value",
    "interpretation_note",
]
AB_OVERALL_COLUMNS = ["event_name", *AB_VALUE_COLUMNS]
AB_SLICE_COLUMNS = ["event_name", "slice_name", "slice_value", *AB_VALUE_COLUMNS]


def _empty_ab_df(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def probability_candidate_better(
    alpha_baseline: float,
    beta_baseline: float,
    alpha_candidate: float,
    beta_candidate: float,
) -> float:
    """Return P(candidate rate < baseline rate) under Gamma posteriors."""

    if min(alpha_baseline, beta_baseline, alpha_candidate, beta_candidate) <= 0:
        raise ValueError("Posterior parameters must be positive")

    if np.isclose(beta_baseline, beta_candidate):
        return float(stats.beta.cdf(0.5, alpha_candidate, alpha_baseline))

    def integrand(rate: float) -> float:
        return stats.gamma.pdf(rate, alpha_candidate, scale=1 / beta_candidate) * stats.gamma.cdf(
            rate, alpha_baseline, scale=1 / beta_baseline
        )

    prob, _ = integrate.quad(integrand, 0, np.inf, limit=256, epsabs=1e-9, epsrel=1e-7)
    return float(np.clip(prob, 0.0, 1.0))


def _rate_standard_error(events: pd.Series | np.ndarray, exposure_miles: pd.Series | np.ndarray) -> np.ndarray:
    events = np.asarray(events, dtype=float)
    exposure = np.asarray(exposure_miles, dtype=float)
    exposure = np.where(exposure > 0, exposure, np.nan)
    return np.sqrt(events + PRIOR_ALPHA) / exposure


def _interpretation(prob: float, delta: float, ci_low: float, ci_high: float) -> str:
    if np.isnan(prob):
        return "Insufficient data"
    if prob >= 0.95 and delta < 0:
        return "Candidate likely safer (>=95% posterior)"
    if prob <= 0.05 and delta > 0:
        return "Candidate likely regresses (>=95% posterior)"
    if ci_low <= 0 <= ci_high:
        return "No significant difference"
    return "Mixed evidence; gather more miles"


def _format_ab_rows(merged: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    work = merged.copy()
    work["delta_rate_per_1k_miles"] = (
        work["rate_per_1k_miles_candidate"] - work["rate_per_1k_miles_baseline"]
    )
    denom = work["rate_per_1k_miles_baseline"].replace(0, np.nan)
    work["pct_change"] = (work["delta_rate_per_1k_miles"] / denom) * 100.0
    work.loc[denom.isna() & (work["rate_per_1k_miles_candidate"] > 0), "pct_change"] = np.inf
    work.loc[
        denom.isna() & (work["rate_per_1k_miles_candidate"] == 0), "pct_change"
    ] = 0.0

    work["alpha_baseline"] = PRIOR_ALPHA + work["event_count_baseline"]
    work["beta_baseline"] = PRIOR_BETA + work["exposure_miles_baseline"]
    work["alpha_candidate"] = PRIOR_ALPHA + work["event_count_candidate"]
    work["beta_candidate"] = PRIOR_BETA + work["exposure_miles_candidate"]

    work["prob_candidate_better"] = work.apply(
        lambda row: probability_candidate_better(
            row["alpha_baseline"],
            row["beta_baseline"],
            row["alpha_candidate"],
            row["beta_candidate"],
        ),
        axis=1,
    )

    se_baseline = _rate_standard_error(work["event_count_baseline"], work["exposure_miles_baseline"]) * 1000
    se_candidate = _rate_standard_error(work["event_count_candidate"], work["exposure_miles_candidate"]) * 1000
    work["delta_se"] = np.sqrt(se_baseline**2 + se_candidate**2)
    work["z_score"] = work["delta_rate_per_1k_miles"] / work["delta_se"]
    work.loc[~np.isfinite(work["z_score"]), "z_score"] = np.nan
    work["p_value"] = 2 * stats.norm.sf(np.abs(work["z_score"]))
    work.loc[work["z_score"].isna(), "p_value"] = np.nan

    work["delta_ci_low"] = (
        work["poisson_ci_low_candidate"] - work["poisson_ci_high_baseline"]
    )
    work["delta_ci_high"] = (
        work["poisson_ci_high_candidate"] - work["poisson_ci_low_baseline"]
    )
    work["interpretation_note"] = work.apply(
        lambda row: _interpretation(
            row["prob_candidate_better"],
            row["delta_rate_per_1k_miles"],
            row["delta_ci_low"],
            row["delta_ci_high"],
        ),
        axis=1,
    )

    columns = list(key_cols) + AB_VALUE_COLUMNS
    work = work.assign(
        baseline_event_count=work["event_count_baseline"],
        baseline_exposure_miles=work["exposure_miles_baseline"],
        baseline_rate_per_1k_miles=work["rate_per_1k_miles_baseline"],
        candidate_event_count=work["event_count_candidate"],
        candidate_exposure_miles=work["exposure_miles_candidate"],
        candidate_rate_per_1k_miles=work["rate_per_1k_miles_candidate"],
    )
    return work[columns]


def compute_ab_overall(driver_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_name",
        "driver_version",
        "event_count",
        "exposure_miles",
        "rate_per_1k_miles",
        "poisson_ci_low",
        "poisson_ci_high",
    }
    if driver_metrics.empty or not required.issubset(driver_metrics.columns):
        return _empty_ab_df(AB_OVERALL_COLUMNS)

    baseline = driver_metrics[driver_metrics["driver_version"] == "baseline"].copy()
    candidate = driver_metrics[driver_metrics["driver_version"] == "candidate"].copy()
    if baseline.empty or candidate.empty:
        return _empty_ab_df(AB_OVERALL_COLUMNS)

    merged = baseline.merge(
        candidate,
        on="event_name",
        suffixes=("_baseline", "_candidate"),
    )
    if merged.empty:
        return _empty_ab_df(AB_OVERALL_COLUMNS)

    merged["alpha_baseline"] = PRIOR_ALPHA + merged["event_count_baseline"]
    merged["beta_baseline"] = PRIOR_BETA + merged["exposure_miles_baseline"]
    merged["alpha_candidate"] = PRIOR_ALPHA + merged["event_count_candidate"]
    merged["beta_candidate"] = PRIOR_BETA + merged["exposure_miles_candidate"]

    return _format_ab_rows(merged, ["event_name"]).sort_values("event_name").reset_index(drop=True)


def _filter_event_tables_for_trips(
    event_tables: Dict[str, EventTable], trip_ids: Iterable[str]
) -> Dict[str, EventTable]:
    trip_set = set(trip_ids)
    return {
        name: table[table["trip_id"].isin(trip_set)].copy() if not table.empty else table
        for name, table in event_tables.items()
    }


def _per_driver_slice_metrics(
    frames: pd.DataFrame,
    event_tables: Dict[str, EventTable],
    config: EvaluationConfig,
) -> pd.DataFrame:
    if "driver_version" not in frames.columns:
        return pd.DataFrame()

    per_driver: List[pd.DataFrame] = []
    for driver_version, driver_frames in frames.groupby("driver_version"):
        driver_tables = _filter_event_tables_for_trips(
            event_tables, driver_frames["trip_id"].unique()
        )
        metrics = build_slice_metrics(
            driver_frames,
            driver_tables,
            config.slice_columns,
            config.slice_min_exposure_miles,
        )
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["driver_version"] = driver_version
        per_driver.append(metrics)

    if not per_driver:
        return pd.DataFrame()
    return pd.concat(per_driver, ignore_index=True)


def compute_ab_slice_metrics(
    frames: pd.DataFrame,
    event_tables: Dict[str, EventTable],
    config: EvaluationConfig,
) -> pd.DataFrame:
    per_driver = _per_driver_slice_metrics(frames, event_tables, config)
    if per_driver.empty:
        return _empty_ab_df(AB_SLICE_COLUMNS)

    baseline = per_driver[per_driver["driver_version"] == "baseline"].drop(columns=["driver_version"])
    candidate = per_driver[per_driver["driver_version"] == "candidate"].drop(columns=["driver_version"])
    if baseline.empty or candidate.empty:
        return _empty_ab_df(AB_SLICE_COLUMNS)

    merged = baseline.merge(
        candidate,
        on=["event_name", "slice_name", "slice_value"],
        suffixes=("_baseline", "_candidate"),
    )
    if merged.empty:
        return _empty_ab_df(AB_SLICE_COLUMNS)

    merged["alpha_baseline"] = PRIOR_ALPHA + merged["event_count_baseline"]
    merged["beta_baseline"] = PRIOR_BETA + merged["exposure_miles_baseline"]
    merged["alpha_candidate"] = PRIOR_ALPHA + merged["event_count_candidate"]
    merged["beta_candidate"] = PRIOR_BETA + merged["exposure_miles_candidate"]

    return _format_ab_rows(merged, ["event_name", "slice_name", "slice_value"]).sort_values(
        ["event_name", "slice_name", "slice_value"]
    ).reset_index(drop=True)
