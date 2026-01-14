"""Integration test covering the synthetic generator and evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from av_eval.config import EvaluationConfig, SyntheticGeneratorConfig
from av_eval.data.synthetic import generate_synthetic_logs
from av_eval.metrics import events as event_detectors
from av_eval.metrics.summary import run_evaluation


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    synth_cfg = SyntheticGeneratorConfig(num_trips=6, seed=11, candidate_share=0.5)
    frames = generate_synthetic_logs(synth_cfg)
    eval_cfg = EvaluationConfig()
    event_tables = event_detectors.detect_all_events(frames, eval_cfg.thresholds)

    outputs = run_evaluation(frames, event_tables, eval_cfg)
    assert not outputs.overall.empty
    if not outputs.slice_metrics.empty:
        assert set(outputs.slice_metrics["slice_name"].unique()).issubset(
            set(eval_cfg.slice_columns)
        )
    assert not outputs.trend.empty
    assert not outputs.driver_metrics.empty
    assert {
        "baseline",
        "candidate",
    }.issubset(set(outputs.driver_metrics["driver_version"].unique()))
    if not outputs.driver_comparison.empty:
        assert "relative_change_pct" in outputs.driver_comparison.columns
    if not outputs.ab_overall.empty:
        assert "prob_candidate_better" in outputs.ab_overall.columns
    assert set(outputs.ab_slices.columns).issuperset({"prob_candidate_better"})

    overall_path = tmp_path / "overall.csv"
    outputs.overall.to_csv(overall_path, index=False)
    assert overall_path.exists()
