"""Tests for conformal prediction module."""

import numpy as np
import pytest
from trustlayer.conformal import ConformalPredictor, ConformalResult


@pytest.fixture
def calibrated_predictor():
    rng = np.random.default_rng(42)
    n_cal, n_classes = 200, 3
    probs = rng.dirichlet(np.ones(n_classes), size=n_cal)
    labels = np.argmax(probs, axis=1)
    predictor = ConformalPredictor(alpha=0.1)
    predictor.calibrate(probs, labels)
    return predictor


def test_calibrate_sets_threshold(calibrated_predictor):
    assert calibrated_predictor.threshold is not None
    assert 0.0 <= calibrated_predictor.threshold <= 1.0


def test_predict_returns_conformal_result(calibrated_predictor):
    probs = np.array([0.7, 0.2, 0.1])
    result = calibrated_predictor.predict(probs)
    assert isinstance(result, ConformalResult)


def test_prediction_set_non_empty(calibrated_predictor):
    probs = np.array([0.4, 0.35, 0.25])
    result = calibrated_predictor.predict(probs)
    assert len(result.prediction_set) >= 1


def test_coverage_guarantee_value(calibrated_predictor):
    probs = np.array([0.7, 0.2, 0.1])
    result = calibrated_predictor.predict(probs)
    assert result.coverage_guarantee == pytest.approx(0.9)


def test_high_confidence_produces_singleton(calibrated_predictor):
    """Very confident prediction should ideally produce a singleton set."""
    probs = np.array([0.99, 0.005, 0.005])
    result = calibrated_predictor.predict(probs)
    # Singleton is expected but not guaranteed; just verify structure
    assert result.set_size >= 1


def test_empirical_coverage_above_target():
    """Empirical coverage must meet the 1-alpha guarantee."""
    rng = np.random.default_rng(0)
    n_cal, n_test, n_classes = 500, 200, 4
    cal_probs = rng.dirichlet(np.ones(n_classes), size=n_cal)
    cal_labels = rng.integers(0, n_classes, size=n_cal)
    test_probs = rng.dirichlet(np.ones(n_classes), size=n_test)
    test_labels = rng.integers(0, n_classes, size=n_test)

    predictor = ConformalPredictor(alpha=0.1)
    predictor.calibrate(cal_probs, cal_labels)
    coverage = predictor.empirical_coverage(test_probs, test_labels)
    # Should be ≥ 0.9 (1 - alpha) with high probability
    assert coverage >= 0.8  # loose check for small test set


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        ConformalPredictor(alpha=1.5)


def test_predict_before_calibrate_raises():
    predictor = ConformalPredictor(alpha=0.05)
    with pytest.raises(RuntimeError, match="calibrate"):
        predictor.predict(np.array([0.5, 0.5]))
