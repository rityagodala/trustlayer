"""
Calibration monitoring for deployed ML models.

A well-calibrated model should have:
    P(correct | confidence = 0.9) ≈ 0.9

In practice, models are often overconfident:
    P(correct | confidence = 0.9) ≈ 0.6   ← "quiet failure"

This module tracks calibration over time (Expected Calibration Error),
detects confidence drift, and produces reliability diagrams.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationReport:
    """Snapshot of calibration quality at a point in time."""

    expected_calibration_error: float   # ECE — lower is better (0 = perfect)
    max_calibration_error: float        # MCE — worst single bucket
    mean_confidence: float
    mean_accuracy: float
    confidence_gap: float               # mean_confidence - mean_accuracy (+ = overconfident)
    bucket_stats: list[dict[str, float]]
    is_drifting: bool
    drift_alert: str


class CalibrationMonitor:
    """
    Online calibration tracker using equal-width confidence buckets.

    Usage:
        monitor = CalibrationMonitor(n_buckets=10, drift_threshold=0.1)

        # Log predictions as they come in
        monitor.log(confidence=0.87, correct=True)
        monitor.log(confidence=0.92, correct=False)

        # Get report
        report = monitor.report()
        print(f"ECE: {report.expected_calibration_error:.3f}")
        print(f"Drifting: {report.is_drifting}")
    """

    def __init__(
        self,
        n_buckets: int = 10,
        drift_threshold: float = 0.1,
        window_size: int = 1000,
    ) -> None:
        self.n_buckets = n_buckets
        self.drift_threshold = drift_threshold
        self.window_size = window_size

        self._confidences: list[float] = []
        self._correctness: list[bool] = []

    def log(self, confidence: float, correct: bool) -> None:
        """Record a single prediction."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        self._confidences.append(confidence)
        self._correctness.append(correct)
        # Sliding window
        if len(self._confidences) > self.window_size:
            self._confidences.pop(0)
            self._correctness.pop(0)

    def log_batch(
        self,
        confidences: list[float] | np.ndarray,
        correctness: list[bool] | np.ndarray,
    ) -> None:
        for c, ok in zip(confidences, correctness):
            self.log(float(c), bool(ok))

    def report(self) -> CalibrationReport:
        """Compute calibration statistics over all logged predictions."""
        if len(self._confidences) < 2:
            raise RuntimeError("Need at least 2 logged predictions to compute calibration")

        confs = np.array(self._confidences)
        correct = np.array(self._correctness, dtype=float)

        bucket_stats = []
        ece = 0.0
        mce = 0.0
        n = len(confs)

        for b in range(self.n_buckets):
            lo = b / self.n_buckets
            hi = (b + 1) / self.n_buckets
            mask = (confs > lo) & (confs <= hi) if b > 0 else (confs >= lo) & (confs <= hi)
            bucket_n = int(mask.sum())
            if bucket_n == 0:
                continue
            bucket_conf = float(confs[mask].mean())
            bucket_acc = float(correct[mask].mean())
            gap = abs(bucket_conf - bucket_acc)
            ece += (bucket_n / n) * gap
            mce = max(mce, gap)
            bucket_stats.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "n": bucket_n,
                "mean_confidence": round(bucket_conf, 3),
                "mean_accuracy": round(bucket_acc, 3),
                "gap": round(gap, 3),
            })

        mean_conf = float(confs.mean())
        mean_acc = float(correct.mean())
        gap = mean_conf - mean_acc
        is_drifting = abs(gap) > self.drift_threshold or ece > self.drift_threshold

        alert = ""
        if is_drifting:
            direction = "overconfident" if gap > 0 else "underconfident"
            alert = (
                f"DRIFT DETECTED: model is {direction} by {abs(gap):.1%}. "
                f"ECE={ece:.3f} exceeds threshold={self.drift_threshold}"
            )

        return CalibrationReport(
            expected_calibration_error=round(ece, 4),
            max_calibration_error=round(mce, 4),
            mean_confidence=round(mean_conf, 3),
            mean_accuracy=round(mean_acc, 3),
            confidence_gap=round(gap, 3),
            bucket_stats=bucket_stats,
            is_drifting=is_drifting,
            drift_alert=alert,
        )

    @property
    def n_samples(self) -> int:
        return len(self._confidences)

    def reset(self) -> None:
        self._confidences.clear()
        self._correctness.clear()
