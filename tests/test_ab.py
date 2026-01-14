import numpy as np
import pytest
from scipy import stats

from av_eval.metrics.ab import probability_candidate_better


def test_probability_matches_beta_when_scales_equal():
    alpha_base, beta_base = 4.0, 2.5
    alpha_cand, beta_cand = 3.0, 2.5
    expected = stats.beta.cdf(0.5, alpha_cand, alpha_base)
    prob = probability_candidate_better(alpha_base, beta_base, alpha_cand, beta_cand)
    assert prob == pytest.approx(expected, rel=1e-6)


def test_probability_matches_monte_carlo_for_general_case():
    alpha_base, beta_base = 10.5, 120.0
    alpha_cand, beta_cand = 8.5, 95.0
    prob = probability_candidate_better(alpha_base, beta_base, alpha_cand, beta_cand)

    rng = np.random.default_rng(0)
    cand_samples = rng.gamma(shape=alpha_cand, scale=1 / beta_cand, size=200_000)
    base_samples = rng.gamma(shape=alpha_base, scale=1 / beta_base, size=200_000)
    empirical = float((cand_samples < base_samples).mean())
    assert prob == pytest.approx(empirical, rel=8e-2)


def test_probability_requires_positive_parameters():
    with pytest.raises(ValueError):
        probability_candidate_better(-1.0, 1.0, 1.0, 1.0)
