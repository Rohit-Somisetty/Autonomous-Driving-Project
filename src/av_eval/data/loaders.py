"""I/O helpers for reading and writing driving logs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from .schema import validate_frame_schema

METERS_PER_MILE = 1609.34


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a dataframe against the canonical schema."""

    validate_frame_schema(df)
    return df


class BaseLoader(ABC):
    """Abstract loader that ensures normalization + schema validation."""

    def load(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():  # pragma: no cover - sanity
            raise FileNotFoundError(path)
        df = self._read(path)
        normalized = self.normalize(df)
        return validate_schema(normalized)

    @abstractmethod
    def _read(self, path: Path) -> pd.DataFrame:
        """Return the raw dataframe for the given path."""

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook for subclasses to coerce into canonical schema."""

        return df


class ParquetFrameLoader(BaseLoader):
    """Loader for already canonical parquet datasets."""

    def _read(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)


class CsvFrameLoader(BaseLoader):
    """CSV adapter that normalizes column names, dtypes, and derived fields."""

    COLUMN_ALIASES = {
        "speed_mps": "ego_speed_mps",
        "ego_speed": "ego_speed_mps",
        "accel_mps2": "ego_accel_mps2",
        "acceleration_mps2": "ego_accel_mps2",
        "lane_offset": "lane_offset_m",
        "ttc": "ttc_s",
        "timestamp": "timestamp_s",
        "time_s": "timestamp_s",
        "delta_t": "dt_s",
        "duration": "dt_s",
        "weather_condition": "weather",
        "timeofday": "time_of_day",
        "timeofday_label": "time_of_day",
        "traffic": "traffic_density",
        "traffic_level": "traffic_density",
        "date": "log_date",
        "logdate": "log_date",
        "driver": "driver_version",
        "driverversion": "driver_version",
        "version": "driver_version",
    }
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
        "log_date",
    ]
    NUMERIC_COLUMNS = [
        "timestamp_s",
        "dt_s",
        "ego_speed_mps",
        "ego_accel_mps2",
        "lane_offset_m",
        "ttc_s",
    ]
    CATEGORY_COLUMNS = [
        "weather",
        "time_of_day",
        "traffic_density",
        "driver_version",
    ]
    TRUE_VALUES = {"1", "true", "t", "yes", "y"}
    FALSE_VALUES = {"0", "false", "f", "no", "n"}

    def __init__(
        self,
        alias_map: Optional[Mapping[str, str]] = None,
        read_csv_kwargs: Optional[dict] = None,
    ) -> None:
        base_aliases = {k.lower(): v for k, v in self.COLUMN_ALIASES.items()}
        if alias_map:
            base_aliases.update({k.lower(): v for k, v in alias_map.items()})
        self._alias_map = base_aliases
        self._read_kwargs = read_csv_kwargs or {}

    def _read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, **self._read_kwargs)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = self._rename_columns(df)
        missing = set(self.REQUIRED_COLUMNS) - set(renamed.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        work = renamed.copy()
        for column in self.NUMERIC_COLUMNS:
            work[column] = pd.to_numeric(work[column], errors="coerce")
            if work[column].isna().any():
                raise ValueError(f"Unable to parse numeric values for column '{column}'")

        work["engaged"] = self._parse_booleans(work["engaged"])

        if "log_date" not in work.columns:
            raise ValueError("log_date column required after normalization")
        work["log_date"] = pd.to_datetime(work["log_date"], errors="coerce")
        if work["log_date"].isna().any():
            raise ValueError("Unable to parse log_date values to datetime")

        if "driver_version" not in work.columns:
            work["driver_version"] = pd.Series(pd.NA, index=work.index, dtype="string[python]")

        for column in self.CATEGORY_COLUMNS:
            if column not in work.columns:
                continue
            work[column] = work[column].astype("string[python]").str.strip()

        work.sort_values(["trip_id", "timestamp_s"], inplace=True)
        work.reset_index(drop=True, inplace=True)

        work = self._add_trip_level_fields(work)
        return work

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for column in df.columns:
            key = column.lower()
            target = self._alias_map.get(key)
            if target and target not in df.columns:
                rename_map[column] = target
        if rename_map:
            return df.rename(columns=rename_map)
        return df

    def _parse_booleans(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):  # type: ignore[attr-defined]
            return series.astype(bool)

        normalized = series.astype(str).str.strip().str.lower()
        truth_mask = normalized.isin(self.TRUE_VALUES)
        false_mask = normalized.isin(self.FALSE_VALUES)
        if not (truth_mask | false_mask).all():
            raise ValueError("Unable to coerce engaged column to boolean")
        return truth_mask.astype(bool)

    def _add_trip_level_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["distance_increment_m"] = work["ego_speed_mps"] * work["dt_s"]
        distance = work.groupby("trip_id")["distance_increment_m"].transform("sum")
        work["distance_m"] = distance
        work["distance_miles"] = distance / METERS_PER_MILE
        work["duration_s"] = work.groupby("trip_id")["dt_s"].transform("sum")
        return work.drop(columns=["distance_increment_m"])


def _select_loader(path: Path) -> BaseLoader:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return ParquetFrameLoader()
    if suffix == ".csv":
        return CsvFrameLoader()
    raise ValueError(f"Unsupported file extension: {suffix}")


def load_log_frames(path: Path, loader: Optional[BaseLoader] = None) -> pd.DataFrame:
    """Load driving logs from parquet or CSV using the appropriate adapter."""

    path = Path(path)
    resolved_loader = loader or _select_loader(path)
    return resolved_loader.load(path)


def save_log_frames(df: pd.DataFrame, path: Path) -> None:
    """Persist logs to parquet, ensuring the parent directory exists."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
