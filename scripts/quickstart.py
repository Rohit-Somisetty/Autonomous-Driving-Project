"""Minimal example of the AV evaluation pipeline without the CLI."""

from __future__ import annotations

from pathlib import Path

from av_eval.config import EvaluationConfig, SyntheticGeneratorConfig, resolve_output_paths
from av_eval.data.synthetic import generate_synthetic_logs
from av_eval.metrics import events as event_detectors
from av_eval.metrics.summary import run_evaluation
from av_eval.metrics.top_events import rank_top_events
from av_eval.report.build_report import build_markdown_report
from av_eval.viz.plots import plot_overall_event_rates, plot_driver_comparison
from av_eval.metrics.power import build_power_requirements_table


def main() -> None:
    cfg = SyntheticGeneratorConfig(num_trips=25, seed=7)
    frames = generate_synthetic_logs(cfg)
    eval_cfg = EvaluationConfig()
    event_tables = event_detectors.detect_all_events(frames, eval_cfg.thresholds)
    outputs = run_evaluation(frames, event_tables, eval_cfg)

    outdir = Path("outputs_quickstart")
    outdir.mkdir(exist_ok=True)
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
    top_events = rank_top_events(frames, outputs.events, eval_cfg.top_event_limit)
    top_events.to_csv(paths["top_events"], index=False)

    plot_overall_event_rates(outputs.overall, paths["plot_overall"])
    plot_driver_comparison(outputs.driver_metrics, paths["plot_driver"])

    power_requirements = build_power_requirements_table(
        outputs.driver_metrics,
        relative_lift=eval_cfg.power_target_lift,
        alpha=eval_cfg.power_alpha,
        power=eval_cfg.power_target_power,
    )
    build_markdown_report(
        report_path=paths["report"],
        overall_df=outputs.overall,
        slice_df=outputs.slice_metrics,
        trend_df=outputs.trend,
        anomalies_df=outputs.anomalies,
        plot_paths=[paths["plot_overall"]],
        slice_min_exposure_miles=eval_cfg.slice_min_exposure_miles,
        top_slice_rows=eval_cfg.report_top_slices,
        ab_overall=outputs.ab_overall,
        power_requirements=power_requirements,
        power_target_lift_pct=eval_cfg.power_target_lift * 100.0,
        driver_plot_path=paths["plot_driver"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
