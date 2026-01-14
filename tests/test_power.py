import pytest

from av_eval.metrics.power import required_exposure_miles


def test_more_improvement_requires_fewer_miles():
    base_rate = 2.0 / 1000  # per mile
    mild = required_exposure_miles(base_rate, -0.05, alpha=0.05, power=0.8)
    strong = required_exposure_miles(base_rate, -0.20, alpha=0.05, power=0.8)
    assert strong < mild


def test_rarer_events_need_more_miles():
    common = required_exposure_miles(3.0 / 1000, -0.1, 0.05, 0.8)
    rare = required_exposure_miles(0.5 / 1000, -0.1, 0.05, 0.8)
    assert rare > common


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        required_exposure_miles(0.0, -0.1, 0.05, 0.8)
    with pytest.raises(ValueError):
        required_exposure_miles(1.0 / 1000, 0.0, 0.05, 0.8)
    with pytest.raises(ValueError):
        required_exposure_miles(1.0 / 1000, -0.1, 1.5, 0.8)
    with pytest.raises(ValueError):
        required_exposure_miles(1.0 / 1000, -0.1, 0.05, 0.0)
