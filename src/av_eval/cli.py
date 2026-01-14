"""Typer-based CLI for the AV evaluation framework."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import typer

from .config import EvaluationConfig, SyntheticGeneratorConfig, resolve_output_paths
from .data.loaders import CsvFrameLoader, load_log_frames, save_log_frames
from .data.synthetic import generate_synthetic_logs
from .metrics import events as event_detectors
from .metrics.summary import run_evaluation
from .report.build_report import build_markdown_report
from .utils.logging import get_logger
from .viz.plots import (
    plot_overall_event_rates,
    plot_slice_event_rates,
    plot_driver_comparison,
)
from .metrics.power import (
    build_power_requirements_table,
    required_exposure_miles,
)
from .metrics.top_events import rank_top_events

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


@app.command("generate-data")
def generate_data(
    out: Path = typer.Option(..., help="Destination parquet file"),
    trips: int = typer.Option(200, min=1, help="Number of trips to simulate"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Generate synthetic driving logs and persist them to parquet."""

    cfg = SyntheticGeneratorConfig(num_trips=trips, seed=seed)
    df = generate_synthetic_logs(cfg)
    save_log_frames(df, out)
    logger.info("Generated %s trips to %s", trips, out)


@app.command("load-csv")
def load_csv(
    csv: Path = typer.Option(..., help="Path to raw CSV logs"),
    out: Path = typer.Option(Path("outputs/data.parquet"), help="Canonical parquet output"),
) -> None:
    """Normalize a CSV export to the canonical parquet schema."""

    loader = CsvFrameLoader()
    df = loader.load(csv)
    save_log_frames(df, out)
    logger.info("Loaded %s frames from %s into %s", len(df), csv, out)


@app.command("power")
def power_command(
    event: str = typer.Option(..., help="Event name (for display only)."),
    baseline_rate_per_1k: float = typer.Option(..., min=0.0, help="Baseline rate per 1,000 miles."),
    lift: float = typer.Option(-0.1, help="Relative change (candidate vs baseline)."),
    alpha: float = typer.Option(0.05, help="Two-sided significance level."),
    power: float = typer.Option(0.8, help="Desired statistical power."),
) -> None:
    """Estimate miles per arm needed to detect a rate change."""

    exposure = required_exposure_miles(baseline_rate_per_1k / 1000.0, lift, alpha, power)
    target_rate = max(0.0, baseline_rate_per_1k * (1.0 + lift))
    typer.echo(
        f"Event '{event}': need ~{exposure:,.0f} mi/arm to detect change from {baseline_rate_per_1k:.3f}"
        f" to {target_rate:.3f} per 1k miles (lift {lift * 100:.1f}%, alpha={alpha}, power={power})."
    )


@app.command("export-events")
def export_events(
    data: Path = typer.Option(..., help="Path to parquet/csv driving logs"),
    outdir: Path = typer.Option(Path("outputs"), help="Directory for outputs"),
    top: int = typer.Option(50, min=1, help="Number of highest severity events"),
) -> None:
    """Export highest-severity events to a CSV for manual review."""

    eval_cfg = EvaluationConfig()
    frames = load_log_frames(data)
    event_tables = event_detectors.detect_all_events(frames, eval_cfg.thresholds)
    top_events = rank_top_events(frames, event_tables, top)

    outdir.mkdir(parents=True, exist_ok=True)
    paths = resolve_output_paths(outdir)
    top_events.to_csv(paths["top_events"], index=False)
    logger.info("Exported top %s events to %s", len(top_events), paths["top_events"])


@app.command("run-eval")
def run_eval(
    data: Path = typer.Option(..., help="Path to parquet/csv driving logs"),
    outdir: Path = typer.Option(Path("outputs"), help="Directory for outputs"),
) -> None:
    """Run the evaluation pipeline given log data."""

    eval_cfg = EvaluationConfig()
    frames = load_log_frames(data)
    event_tables: Dict[str, pd.DataFrame] = event_detectors.detect_all_events(
        frames, eval_cfg.thresholds
    )
    outputs = run_evaluation(frames, event_tables, eval_cfg)

    outdir.mkdir(parents=True, exist_ok=True)
    paths = resolve_output_paths(outdir)
    paths["plots_dir"].mkdir(parents=True, exist_ok=True)

    outputs.overall.to_csv(paths["overall"], index=False)
    outputs.slice_metrics.to_csv(paths["slices"], index=False)
    outputs.trend.to_csv(paths["trends"], index=False)
    outputs.anomalies.to_csv(paths["anomalies"], index=False)
    outputs.driver_metrics.to_csv(paths["driver_metrics"], index=False)
    outputs.driver_comparison.to_csv(paths["driver_comparison"], index=False)
    outputs.ab_overall.to_csv(paths["ab_overall"], index=False)
    outputs.ab_slices.to_csv(paths["ab_slices"], index=False)

    plot_overall_event_rates(outputs.overall, paths["plot_overall"])
    plot_slice_event_rates(
        outputs.slice_metrics,
        slice_name="weather",
        out_path=paths["plot_weather"],
    )
    plot_slice_event_rates(
        outputs.slice_metrics,
        slice_name="time_of_day",
        out_path=paths["plot_time"],
    )
    plot_driver_comparison(
        outputs.driver_metrics,
        out_path=paths["plot_driver"],
    )

    power_requirements = build_power_requirements_table(
        outputs.driver_metrics,
        relative_lift=eval_cfg.power_target_lift,
        alpha=eval_cfg.power_alpha,
        power=eval_cfg.power_target_power,
    )
    top_events = rank_top_events(frames, outputs.events, eval_cfg.top_event_limit)
    top_events.to_csv(paths["top_events"], index=False)

    build_markdown_report(
        report_path=paths["report"],
        overall_df=outputs.overall,
        slice_df=outputs.slice_metrics,
        trend_df=outputs.trend,
        anomalies_df=outputs.anomalies,
        plot_paths=[
            paths["plot_overall"],
            paths["plot_weather"],
            paths["plot_time"],
        ],
        slice_min_exposure_miles=eval_cfg.slice_min_exposure_miles,
        top_slice_rows=eval_cfg.report_top_slices,
        ab_overall=outputs.ab_overall,
        power_requirements=power_requirements,
        power_target_lift_pct=eval_cfg.power_target_lift * 100.0,
        driver_plot_path=paths["plot_driver"],
    )

    logger.info("Evaluation artifacts saved to %s", outdir)


def main() -> None:
    """CLI entrypoint for PyPA console_scripts."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
