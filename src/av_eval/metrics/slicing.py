"""Slice-based exposure and event aggregation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..metrics.events import EventTable
from .rates import estimate_rate_bayes, estimate_rate_poisson

METERS_PER_MILE = 1609.34


def _exposure_by_slice(frames: pd.DataFrame, slice_column: str) -> pd.DataFrame:
    work = frames[[slice_column, "ego_speed_mps", "dt_s"]].copy()
    work["miles_increment"] = work["ego_speed_mps"] * work["dt_s"] / METERS_PER_MILE
    exposure = (
        work.groupby(slice_column)["miles_increment"].sum().reset_index(name="exposure_miles")
    )
    exposure["slice_column"] = slice_column
    exposure = exposure.rename(columns={slice_column: "slice_value"})
    return exposure[["slice_column", "slice_value", "exposure_miles"]]


def build_slice_metrics(
    frames: pd.DataFrame,
    event_tables: Dict[str, EventTable],
    slice_columns: List[str],
    min_exposure_miles: float,
) -> pd.DataFrame:
    """Return a slice-level rate table for each configured event type."""

    if not slice_columns:
        return pd.DataFrame()

    exposures_by_col = {
        col: _exposure_by_slice(frames, col) for col in slice_columns
    }

    trip_meta = frames.drop_duplicates("trip_id")[["trip_id", *slice_columns]]
    records: List[dict] = []

    for event_name, table in event_tables.items():
        enriched = (
            table.merge(trip_meta, on="trip_id", how="left")
            if not table.empty
            else trip_meta.head(0)
        )
        for col in slice_columns:
            base = exposures_by_col[col].copy()
            if enriched.empty:
                counts = pd.DataFrame(columns=["slice_value", "event_count"])
            else:
                counts = (
                    enriched.groupby(col)
                    .size()
                    .reset_index(name="event_count")
                    .rename(columns={col: "slice_value"})
                )
            merged = base.merge(counts, on="slice_value", how="left")
            merged["event_count"] = merged["event_count"].fillna(0).astype(int)
            merged["slice_name"] = col
            merged["event_name"] = event_name

            for row in merged.itertuples():
                exposure = float(row.exposure_miles)
                if exposure <= 0 or exposure < min_exposure_miles:
                    continue
                event_count = int(row.event_count)
                rate, (poisson_lo, poisson_hi) = estimate_rate_poisson(event_count, exposure)
                bayes_rate, (bayes_lo, bayes_hi) = estimate_rate_bayes(event_count, exposure)
                records.append(
                    {
                        "event_name": event_name,
                        "slice_name": col,
                        "slice_value": row.slice_value,
                        "event_count": event_count,
                        "exposure_miles": exposure,
                        "rate_per_mile": rate,
                        "rate_per_1k_miles": rate * 1000,
                        "poisson_ci_low": poisson_lo * 1000,
                        "poisson_ci_high": poisson_hi * 1000,
                        "bayes_ci_low": bayes_lo * 1000,
                        "bayes_ci_high": bayes_hi * 1000,
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "event_name",
                "slice_name",
                "slice_value",
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

    slice_metrics = pd.DataFrame(records)
    slice_metrics = slice_metrics.sort_values(
        "rate_per_1k_miles", ascending=False
    ).reset_index(drop=True)
    return slice_metrics
