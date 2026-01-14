"""Tests for rare-event rate estimators."""

from __future__ import annotations

import pytest

from av_eval.metrics.rates import estimate_rate_bayes, estimate_rate_poisson


def test_estimate_rate_poisson_basic() -> None:
    rate, (lo, hi) = estimate_rate_poisson(5, 1200.0)
    assert rate == pytest.approx(5 / 1200.0)
    assert lo < rate < hi


def test_estimate_rate_bayes_reduces_to_prior_when_zero_events() -> None:
    mean_rate, (lo, hi) = estimate_rate_bayes(0, 500.0, prior_alpha=0.5, prior_beta=10.0)
    assert mean_rate == pytest.approx(0.5 / 510.0)
    assert 0 <= lo <= mean_rate <= hi


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        estimate_rate_poisson(1, 0)
    with pytest.raises(ValueError):
        estimate_rate_bayes(1, 100, prior_alpha=-1, prior_beta=1)
