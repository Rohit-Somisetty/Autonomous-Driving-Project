"""Metric aggregation helpers for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..config import EvaluationConfig
from .ab import compute_ab_overall, compute_ab_slice_metrics
from .events import EventTable
from .rates import estimate_rate_bayes, estimate_rate_poisson
from .slicing import build_slice_metrics

METERS_PER_MILE = 1609.34


@dataclass
class EvaluationOutputs:
    overall: pd.DataFrame
    slice_metrics: pd.DataFrame
    trend: pd.DataFrame
    anomalies: pd.DataFrame
    driver_metrics: pd.DataFrame
    ab_overall: pd.DataFrame
    ab_slices: pd.DataFrame
    driver_comparison: pd.DataFrame
    events: Dict[str, EventTable]


def _total_miles(frames: pd.DataFrame) -> float:
    return float((frames["ego_speed_mps"] * frames["dt_s"]).sum() / METERS_PER_MILE)


def compute_overall_metrics(
    frames: pd.DataFrame, event_tables: Dict[str, EventTable]
) -> pd.DataFrame:
    """Return overall counts and rates for each event type."""

    exposure_miles = _total_miles(frames)
    rows = []
    for name, table in event_tables.items():
        count = len(table)
        rate, (lo, hi) = estimate_rate_poisson(count, exposure_miles)
        bayes_rate, (bayes_lo, bayes_hi) = estimate_rate_bayes(count, exposure_miles)
        rows.append(
            {
                "event_name": name,
                "event_count": count,
                "exposure_miles": exposure_miles,
                "rate_per_mile": rate,
                "rate_per_1k_miles": rate * 1000,
                "poisson_ci_low": lo * 1000,
                "poisson_ci_high": hi * 1000,
                "bayes_rate": bayes_rate,
                "bayes_ci_low": bayes_lo * 1000,
                "bayes_ci_high": bayes_hi * 1000,
            }
        )
    return pd.DataFrame(rows)


def compute_driver_version_metrics(
    frames: pd.DataFrame, event_tables: Dict[str, EventTable]
) -> pd.DataFrame:
    """Calculate event metrics per driver_version."""

    if "driver_version" not in frames.columns:
        return pd.DataFrame(
            columns=[
                "event_name",
                "driver_version",
                "event_count",
                "exposure_miles",
                "rate_per_mile",
                "rate_per_1k_miles",
                "poisson_ci_low",
                "poisson_ci_high",
                "bayes_ci_low",
                "bayes_ci_high",
            ]
        )

    exposure = (
        frames.groupby("driver_version")
        .apply(lambda df: (df["ego_speed_mps"] * df["dt_s"]).sum() / METERS_PER_MILE)
        .reset_index(name="exposure_miles")
    )
    trip_meta = frames.drop_duplicates("trip_id")[["trip_id", "driver_version"]]

    rows = []
    for event_name, table in event_tables.items():
        if table.empty:
            counts = pd.DataFrame(columns=["driver_version", "event_count"])
        else:
            enriched = table.merge(trip_meta, on="trip_id", how="left")
            counts = (
                enriched.groupby("driver_version")
                .size()
                .reset_index(name="event_count")
            )
        merged = exposure.merge(counts, on="driver_version", how="left")
        merged["event_count"] = merged["event_count"].fillna(0).astype(int)
        for row in merged.itertuples():
            exposure_miles = float(row.exposure_miles)
            if exposure_miles <= 0:
                continue
            event_count = int(row.event_count)
            rate, (poisson_lo, poisson_hi) = estimate_rate_poisson(event_count, exposure_miles)
            bayes_rate, (bayes_lo, bayes_hi) = estimate_rate_bayes(event_count, exposure_miles)
            rows.append(
                {
                    "event_name": event_name,
                    "driver_version": row.driver_version,
                    "event_count": event_count,
                    "exposure_miles": exposure_miles,
                    "rate_per_mile": rate,
                    "rate_per_1k_miles": rate * 1000,
                    "poisson_ci_low": poisson_lo * 1000,
                    "poisson_ci_high": poisson_hi * 1000,
                    "bayes_ci_low": bayes_lo * 1000,
                    "bayes_ci_high": bayes_hi * 1000,
                }
            )

    return pd.DataFrame(rows)


def build_driver_comparison(driver_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create baseline vs candidate comparison rows with relative deltas."""

    required_columns = {
        "event_name",
        "driver_version",
        "rate_per_1k_miles",
        "poisson_ci_low",
        "poisson_ci_high",
        "event_count",
        "exposure_miles",
    }
    if driver_metrics.empty or not required_columns.issubset(driver_metrics.columns):
        return pd.DataFrame(
            columns=[
                "event_name",
                "baseline_rate_per_1k_miles",
                "candidate_rate_per_1k_miles",
                "relative_change_pct",
                "baseline_exposure_miles",
                "candidate_exposure_miles",
                "uncertainty_note",
            ]
        )

    baseline = driver_metrics[driver_metrics["driver_version"] == "baseline"].set_index(
        "event_name"
    )
    candidate = driver_metrics[driver_metrics["driver_version"] == "candidate"].set_index(
        "event_name"
    )

    rows = []
    for event_name in sorted(set(baseline.index) & set(candidate.index)):
        base_row = baseline.loc[event_name]
        cand_row = candidate.loc[event_name]
        base_rate = float(base_row["rate_per_1k_miles"])
        cand_rate = float(cand_row["rate_per_1k_miles"])
        if base_rate == 0:
            change_pct = np.inf if cand_rate > 0 else 0.0
        else:
            change_pct = ((cand_rate - base_rate) / base_rate) * 100.0

        overlap = not (
            cand_row["poisson_ci_low"] > base_row["poisson_ci_high"]
            or cand_row["poisson_ci_high"] < base_row["poisson_ci_low"]
        )
        if np.isinf(change_pct) and cand_rate == 0:
            note = "No events for either version"
        elif overlap:
            note = "Rates overlap within 95% CI"
        elif cand_rate < base_rate:
            note = "Candidate lower (non-overlapping CI)"
        else:
            note = "Candidate higher (non-overlapping CI)"

        rows.append(
            {
                "event_name": event_name,
                "baseline_rate_per_1k_miles": base_rate,
                "candidate_rate_per_1k_miles": cand_rate,
                "baseline_exposure_miles": float(base_row["exposure_miles"]),
                "candidate_exposure_miles": float(cand_row["exposure_miles"]),
                "relative_change_pct": change_pct,
                "uncertainty_note": note,
            }
        )

    return pd.DataFrame(rows)


def compute_trend_metrics(
    frames: pd.DataFrame,
    event_tables: Dict[str, EventTable],
    time_column: str,
) -> pd.DataFrame:
    """Aggregate event rates by weekly buckets of the provided time column."""

    work = frames.copy()
    work[time_column] = pd.to_datetime(work[time_column])
    work["trend_bucket"] = work[time_column].dt.to_period("W").dt.start_time
    work["miles_increment"] = work["ego_speed_mps"] * work["dt_s"] / METERS_PER_MILE
    exposure = (
        work.groupby("trend_bucket")["miles_increment"].sum().reset_index(name="exposure_miles")
    )

    trip_meta = frames.drop_duplicates("trip_id")[["trip_id", time_column]]
    trip_meta[time_column] = pd.to_datetime(trip_meta[time_column])
    trip_meta["trend_bucket"] = trip_meta[time_column].dt.to_period("W").dt.start_time

    rows = []
    for event_name, table in event_tables.items():
        enriched = table.merge(trip_meta[["trip_id", "trend_bucket"]], on="trip_id", how="left")
        counts = (
            enriched.groupby("trend_bucket").size().reset_index(name="event_count")
            if not enriched.empty
            else pd.DataFrame(columns=["trend_bucket", "event_count"])
        )
        merged = exposure.merge(counts, on="trend_bucket", how="left")
        merged["event_count"] = merged["event_count"].fillna(0)
        merged["event_name"] = event_name
        merged["rate_per_1k_miles"] = (
            merged["event_count"] / merged["exposure_miles"].replace(0, np.nan) * 1000
        ).replace(np.nan, 0.0)
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


def detect_slice_anomalies(
    slice_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    z_threshold: float,
) -> pd.DataFrame:
    """Compute Poisson-style z-scores to surface anomalous slice rates."""

    anomalies = []
    for event_name, group in slice_df.groupby("event_name"):
        baseline = overall_df[overall_df["event_name"] == event_name]
        if baseline.empty:
            continue
        base_rate = baseline["rate_per_mile"].iloc[0]
        expected = group["exposure_miles"] * base_rate
        variance = np.maximum(expected, 1e-6)
        z_scores = (group["event_count"] - expected) / np.sqrt(variance)
        mask = z_scores.abs() >= z_threshold
        if not mask.any():
            continue
        subset = group.loc[mask].copy()
        subset["z_score"] = z_scores[mask]
        subset["expected_events"] = expected[mask]
        anomalies.append(subset)

    if not anomalies:
        return pd.DataFrame(columns=slice_df.columns.tolist() + ["z_score", "expected_events"])

    return pd.concat(anomalies, ignore_index=True)


def run_evaluation(
    frames: pd.DataFrame,
    event_tables: Dict[str, EventTable],
    config: EvaluationConfig,
) -> EvaluationOutputs:
    """End-to-end metric computation for the CLI and programmatic use."""

    overall = compute_overall_metrics(frames, event_tables)
    slice_df = build_slice_metrics(
        frames,
        event_tables,
        config.slice_columns,
        config.slice_min_exposure_miles,
    )
    trend = compute_trend_metrics(frames, event_tables, config.trend_time_column)
    anomalies = detect_slice_anomalies(slice_df, overall, config.anomaly_z_threshold)
    driver_metrics = compute_driver_version_metrics(frames, event_tables)
    driver_comparison = build_driver_comparison(driver_metrics)
    ab_overall = compute_ab_overall(driver_metrics)
    ab_slices = compute_ab_slice_metrics(frames, event_tables, config)

    return EvaluationOutputs(
        overall=overall,
        slice_metrics=slice_df,
        trend=trend,
        anomalies=anomalies,
        driver_metrics=driver_metrics,
        ab_overall=ab_overall,
        ab_slices=ab_slices,
        driver_comparison=driver_comparison,
        events=event_tables,
    )
