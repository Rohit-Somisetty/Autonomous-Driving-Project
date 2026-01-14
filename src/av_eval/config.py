"""Global configuration objects for the AV evaluation framework."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class EventThresholds(BaseModel):
    """Thresholds controlling event detectors."""

    hard_brake_accel_mps2: float = Field(
        default=-3.0, description="Acceleration threshold for hard braking."
    )
    hard_brake_min_duration_s: float = Field(
        default=0.3, description="Minimum duration for hard braking."
    )
    lane_deviation_m: float = Field(
        default=0.6, description="Lateral offset magnitude for lane deviation."
    )
    lane_deviation_min_duration_s: float = Field(
        default=1.0, description="Minimum duration for sustained deviation."
    )
    near_miss_ttc_s: float = Field(
        default=1.5, description="Time-to-collision limit for near-miss events."
    )
    near_miss_min_duration_s: float = Field(
        default=0.2, description="Minimum duration for near-miss classification."
    )


class SyntheticGeneratorConfig(BaseModel):
    """Configuration for the synthetic log generator."""

    num_trips: int = Field(default=200, ge=1)
    seed: int = Field(default=42)
    dt_s: float = Field(default=0.1, gt=0)
    min_duration_s: float = Field(default=30.0, gt=0)
    max_duration_s: float = Field(default=300.0, gt=0)
    weather_options: List[str] = Field(
        default_factory=lambda: ["clear", "rain", "fog", "wind"]
    )
    time_of_day_options: List[str] = Field(
        default_factory=lambda: ["dawn", "day", "dusk", "night"]
    )
    traffic_density_options: List[str] = Field(
        default_factory=lambda: ["light", "moderate", "heavy"]
    )
    version_tags: List[str] = Field(
        default_factory=lambda: ["v1.0", "v1.1", "v1.2"]
    )
    candidate_share: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability that a generated trip uses the candidate driver.",
    )
    candidate_risk_scale: float = Field(
        default=0.6,
        gt=0.0,
        description="Multiplier applied to risk events (lower = safer candidate).",
    )
    candidate_accel_scale: float = Field(
        default=0.8,
        gt=0.0,
        description="Scale factor applied to acceleration noise for candidate trips.",
    )


class EvaluationConfig(BaseModel):
    """Runtime configuration for the evaluation pipeline."""

    thresholds: EventThresholds = Field(default_factory=EventThresholds)
    slice_columns: List[str] = Field(
        default_factory=lambda: ["weather", "time_of_day", "traffic_density"]
    )
    trend_time_column: str = "log_date"
    anomaly_z_threshold: float = 2.0
    slice_min_exposure_miles: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum exposure miles required for slice reporting.",
    )
    report_top_slices: int = Field(
        default=10,
        ge=1,
        description="Number of highest-rate slices to highlight in the report.",
    )
    top_event_limit: int = Field(
        default=50,
        ge=1,
        description="Number of highest-severity events to export for investigations.",
    )
    power_target_lift: float = Field(
        default=-0.1,
        description="Relative change (candidate vs baseline) to size exposure for (e.g., -0.1 means 10% improvement).",
    )
    power_alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for power analysis.",
    )
    power_target_power: float = Field(
        default=0.8,
        gt=0,
        lt=1,
        description="Desired statistical power (1 - beta).",
    )

    class Config:
        arbitrary_types_allowed = True


def resolve_output_paths(outdir: Path) -> dict[str, Path]:
    """Return standard output paths for metrics, plots, and reports."""

    plots_dir = outdir / "plots"
    return {
        "overall": outdir / "metrics_overall.csv",
        "slices": outdir / "metrics_slices.csv",
        "trends": outdir / "metrics_trends.csv",
        "anomalies": outdir / "anomalies.csv",
        "driver_metrics": outdir / "metrics_driver_versions.csv",
        "driver_comparison": outdir / "metrics_driver_comparison.csv",
        "ab_overall": outdir / "metrics_ab.csv",
        "ab_slices": outdir / "metrics_ab_slices.csv",
        "top_events": outdir / "top_events.csv",
        "report": outdir / "report.md",
        "plots_dir": plots_dir,
        "plot_overall": plots_dir / "overall_event_rates.png",
        "plot_weather": plots_dir / "weather_slice_rates.png",
        "plot_time": plots_dir / "time_of_day_slice_rates.png",
        "plot_driver": plots_dir / "driver_comparison.png",
    }
