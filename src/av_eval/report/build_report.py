"""Markdown report builder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

METRIC_DEFINITIONS = {
    "Disengagement": "Autonomy switches from engaged to manual control until re-engagement.",
    "Hard braking": "Longitudinal acceleration below threshold for >=0.3 s.",
    "Lane deviation": "Absolute lateral offset exceeds 0.6 m for >=1.0 s.",
    "Near miss": "Time-to-collision proxy below 1.5 s for >=0.2 s.",
}

UNITS_SECTION = [
    "Distances derive from meters and are converted to miles using 1 mile = 1,609.34 meters before any exposure math.",
    "Event rates in every table/plot are normalized per 1,000 miles (column `rate_per_1k_miles`).",
    "Speeds are meters per second, accelerations are meters per second squared, lane offsets are meters, and TTC values are seconds.",
    "Trend buckets are ISO calendar weeks based on the UTC `log_date` column.",
]

METRICS_GLOSSARY = [
    {
        "metric": "Disengagement",
        "definition": "`engaged` flips True→False until re-engagement (min duration = one sample).",
        "rationale": "Captures any loss of autonomous control regardless of cause.",
    },
    {
        "metric": "Hard braking",
        "definition": "`ego_accel_mps2 < -3.0` m/s² continuously for ≥0.3 s.",
        "rationale": "Flags aggressive decelerations signaling safety maneuvers.",
    },
    {
        "metric": "Lane deviation",
        "definition": "`abs(lane_offset_m) > 0.6` m continuously for ≥1.0 s.",
        "rationale": "Highlights sustained lateral error beyond typical tolerance.",
    },
    {
        "metric": "Near-miss (TTC)",
        "definition": "`ttc_s < 1.5` s continuously for ≥0.2 s.",
        "rationale": "Surfaces short time-to-collision windows indicative of risk.",
    },
    {
        "metric": "Exposure miles",
        "definition": "Integral of `ego_speed_mps × dt_s` converted via 1 mile = 1,609.34 m.",
        "rationale": "Provides the denominator for rates per 1,000 miles.",
    },
]


def _df_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def _glossary_markdown(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "_No glossary entries._"
    header = "| Metric | Default definition | Rationale |"
    separator = "| --- | --- | --- |"
    rows = [
        f"| {item['metric']} | {item['definition']} | {item['rationale']} |" for item in entries
    ]
    return "\n".join([header, separator, *rows])


def _slice_glance_markdown(slice_df: pd.DataFrame, top_n: int) -> str:
    if slice_df.empty:
        return "_No slice metrics available._"
    subset = slice_df.sort_values("rate_per_1k_miles", ascending=False).head(top_n)
    return _df_markdown(
        subset[[
            "event_name",
            "slice_name",
            "slice_value",
            "event_count",
            "exposure_miles",
            "rate_per_1k_miles",
            "poisson_ci_low",
            "poisson_ci_high",
            "bayes_ci_low",
            "bayes_ci_high",
        ]]
    )


def _ab_overview_markdown(ab_df: pd.DataFrame) -> str:
    if ab_df.empty:
        return "_No driver-version comparison available._"
    return _df_markdown(
        ab_df[[
            "event_name",
            "baseline_rate_per_1k_miles",
            "candidate_rate_per_1k_miles",
            "delta_rate_per_1k_miles",
            "pct_change",
            "prob_candidate_better",
            "interpretation_note",
        ]]
    )


def _power_requirements_markdown(power_df: pd.DataFrame) -> str:
    if power_df.empty:
        return "_Power analysis requires driver_version data._"
    return _df_markdown(
        power_df[[
            "event_name",
            "baseline_rate_per_1k_miles",
            "target_rate_per_1k_miles",
            "required_miles_per_arm",
        ]]
    )


def build_markdown_report(
    report_path: Path,
    overall_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    plot_paths: Iterable[Path],
    *,
    slice_min_exposure_miles: float,
    top_slice_rows: int,
    ab_overall: pd.DataFrame,
    power_requirements: pd.DataFrame,
    power_target_lift_pct: float,
    driver_plot_path: Path,
) -> None:
    """Compile a Markdown report that references metrics tables and plots."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Autonomous Driving Evaluation Report", ""]

    lines.append("## Metric Definitions")
    for name, desc in METRIC_DEFINITIONS.items():
        lines.append(f"- **{name}**: {desc}")
    lines.append("")

    lines.append("## Units & Definitions")
    for bullet in UNITS_SECTION:
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("## Metrics Glossary")
    lines.append(_glossary_markdown(METRICS_GLOSSARY))
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append(_df_markdown(overall_df))
    lines.append("")

    lines.append("## Slice Metrics")
    lines.append(
        f"Top {top_slice_rows} slices by rate per 1k miles (exposure ≥ {slice_min_exposure_miles:g} mi)."
    )
    lines.append(_slice_glance_markdown(slice_df, top_slice_rows))
    lines.append("")

    lines.append("## Baseline vs Candidate")
    lines.append(
        "Posterior probability reflects P(candidate rate < baseline rate) using a Gamma-Poisson model."
    )
    lines.append("")
    lines.append(_ab_overview_markdown(ab_overall))
    if driver_plot_path.exists():
        lines.append("")
        lines.append(f"![Driver comparison]({driver_plot_path.as_posix()})")
    lines.append("")

    lines.append(
        f"## Required exposure for detecting {abs(power_target_lift_pct):g}% change"
    )
    lines.append(
        "Miles shown are per arm assuming two-sided alpha and desired power configured for the evaluation."
    )
    lines.append("")
    lines.append(_power_requirements_markdown(power_requirements))
    lines.append("")

    lines.append("## Trend Metrics")
    lines.append(
        _df_markdown(
            trend_df[[
                "event_name",
                "trend_bucket",
                "event_count",
                "exposure_miles",
                "rate_per_1k_miles",
            ]]
        )
        if not trend_df.empty
        else "_No trend metrics available._"
    )
    lines.append("")

    lines.append("## Slice Anomalies")
    lines.append(
        _df_markdown(
            anomalies_df[[
                "event_name",
                "slice_name",
                "slice_value",
                "event_count",
                "expected_events",
                "rate_per_1k_miles",
                "z_score",
            ]]
        )
        if not anomalies_df.empty
        else "_No anomalies detected._"
    )
    lines.append("")

    lines.append("## Plots")
    for path in plot_paths:
        rel = Path(path)
        lines.append(f"![{rel.name}]({rel.as_posix()})")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
