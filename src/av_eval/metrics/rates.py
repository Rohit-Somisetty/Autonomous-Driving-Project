"""Rare event rate estimators."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def estimate_rate_poisson(
    k_events: int, exposure_miles: float, alpha: float = 0.05
) -> Tuple[float, Tuple[float, float]]:
    """Estimate a Poisson event rate per mile with a two-sided CI."""

    if exposure_miles <= 0:
        raise ValueError("Exposure must be positive")
    if k_events < 0:
        raise ValueError("Event count must be non-negative")

    rate = k_events / exposure_miles
    if k_events == 0:
        lower = 0.0
    else:
        lower = 0.5 * stats.chi2.ppf(alpha / 2, 2 * k_events) / exposure_miles
    upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (k_events + 1)) / exposure_miles
    return rate, (lower, upper)


def estimate_rate_bayes(
    k_events: int,
    exposure_miles: float,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    cred: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """Gamma-Poisson posterior mean and equal-tailed credible interval."""

    if exposure_miles <= 0:
        raise ValueError("Exposure must be positive")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive")

    posterior_alpha = prior_alpha + k_events
    posterior_beta = prior_beta + exposure_miles
    mean_rate = posterior_alpha / posterior_beta
    lower = stats.gamma.ppf((1 - cred) / 2, posterior_alpha, scale=1 / posterior_beta)
    upper = stats.gamma.ppf(1 - (1 - cred) / 2, posterior_alpha, scale=1 / posterior_beta)
    return mean_rate, (lower, upper)
