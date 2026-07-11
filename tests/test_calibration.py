"""Tests for calibration monitoring."""

import pytest
from trustlayer.calibration import CalibrationMonitor, CalibrationReport


@pytest.fixture
def monitor():
    return CalibrationMonitor(n_buckets=5, drift_threshold=0.15)


def test_report_requires_minimum_samples(monitor):
    monitor.log(0.8, True)
    with pytest.raises(RuntimeError):
        monitor.report()


def test_overconfident_model_detected(monitor):
    """Model that says 90% but is only 50% accurate should be flagged."""
    for _ in range(100):
        monitor.log(0.9, True)
    for _ in range(100):
        monitor.log(0.9, False)
    report = monitor.report()
    # mean_confidence = 0.9, mean_accuracy = 0.5 → gap = 0.4 > threshold
    assert report.is_drifting
    assert "overconfident" in report.drift_alert.lower()


def test_well_calibrated_model_not_flagged(monitor):
    """Model where confidence ≈ accuracy should not trigger drift."""
    import random
    random.seed(42)
    for _ in range(200):
        conf = random.uniform(0.6, 0.9)
        correct = random.random() < conf  # accuracy matches confidence
        monitor.log(conf, correct)
    report = monitor.report()
    # ECE should be low for a well-calibrated model
    assert report.expected_calibration_error < 0.2


def test_ece_is_non_negative(monitor):
    for i in range(50):
        monitor.log(0.7, i % 2 == 0)
    report = monitor.report()
    assert report.expected_calibration_error >= 0


def test_reset_clears_data(monitor):
    monitor.log(0.8, True)
    monitor.reset()
    assert monitor.n_samples == 0


def test_log_batch(monitor):
    monitor.log_batch([0.8, 0.6, 0.9], [True, False, True])
    assert monitor.n_samples == 3
