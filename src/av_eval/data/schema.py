"""Schema utilities for validating frame-level data."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import pandas as pd
from pandas.api import types as pdt

REQUIRED_COLUMNS = [
    "trip_id",
    "timestamp_s",
    "dt_s",
    "ego_speed_mps",
    "ego_accel_mps2",
    "lane_offset_m",
    "ttc_s",
    "engaged",
    "weather",
    "time_of_day",
    "traffic_density",
    "distance_m",
    "distance_miles",
    "duration_s",
    "log_date",
]


def _is_numeric(series: pd.Series) -> bool:
    return bool(pdt.is_float_dtype(series) or pdt.is_integer_dtype(series))


def _is_string(series: pd.Series) -> bool:
    return bool(pdt.is_string_dtype(series) or pdt.is_object_dtype(series))


def _is_bool(series: pd.Series) -> bool:
    return bool(pdt.is_bool_dtype(series))


def _is_datetime(series: pd.Series) -> bool:
    return bool(pdt.is_datetime64_any_dtype(series))


COLUMN_CHECKS: Dict[str, Callable[[pd.Series], bool]] = {
    "trip_id": _is_string,
    "timestamp_s": _is_numeric,
    "dt_s": _is_numeric,
    "ego_speed_mps": _is_numeric,
    "ego_accel_mps2": _is_numeric,
    "lane_offset_m": _is_numeric,
    "ttc_s": _is_numeric,
    "engaged": _is_bool,
    "weather": _is_string,
    "time_of_day": _is_string,
    "traffic_density": _is_string,
    "distance_m": _is_numeric,
    "distance_miles": _is_numeric,
    "duration_s": _is_numeric,
    "log_date": _is_datetime,
}


def validate_frame_columns(df: pd.DataFrame) -> None:
    """Ensure required columns are present."""

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_frame_dtypes(df: pd.DataFrame) -> None:
    """Ensure canonical dtypes match expectations."""

    mismatches = {}
    for column, checker in COLUMN_CHECKS.items():
        if column not in df.columns:
            continue
        if not checker(df[column]):
            mismatches[column] = str(df[column].dtype)
    if mismatches:
        formatted = ", ".join(f"{col}→{dtype}" for col, dtype in sorted(mismatches.items()))
        raise TypeError(f"Invalid dtypes detected: {formatted}")


def validate_frame_schema(df: pd.DataFrame) -> None:
    """Ensure both required columns and dtypes are satisfied."""

    validate_frame_columns(df)
    validate_frame_dtypes(df)


def assert_categorical_values(
    df: pd.DataFrame, column: str, allowed: Iterable[str]
) -> None:
    """Validate that a categorical column only contains allowed values."""

    invalid = set(df[column].unique()) - set(allowed)
    if invalid:
        raise ValueError(f"Invalid values for {column}: {sorted(invalid)}")
