"""
Conformal Prediction module.

Instead of returning a single point prediction with a scalar confidence,
conformal prediction returns a *set* of plausible labels with a statistical
coverage guarantee:

    P(true_label ∈ prediction_set) ≥ 1 - alpha

This is mathematically guaranteed under exchangeability — no distributional
assumptions required.

Inspired by:
  "A Quiet Failure in Calibrated Virtual Screening" (arXiv 2025)
  "Robust Human-AI Complementarity Under Uncertainty" (arXiv 2025)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConformalResult:
    """Prediction set with coverage guarantee."""

    prediction_set: list[Any]           # set of plausible labels
    point_prediction: Any               # argmax prediction
    point_confidence: float             # raw softmax confidence
    nonconformity_score: float          # how unusual this prediction is
    coverage_guarantee: float           # 1 - alpha (e.g. 0.95)
    set_size: int                       # len(prediction_set) — 1 = confident
    is_uncertain: bool                  # set_size > 1


class ConformalPredictor:
    """
    Split conformal predictor for classification.

    Calibration phase:
        predictor.calibrate(probs, labels)   # on held-out calibration set

    Inference phase:
        result = predictor.predict(probs, labels)

    The nonconformity score used here is 1 - p_true (probability assigned
    to the correct class). The quantile of these scores over the calibration
    set forms the threshold tau.

    Example:
        predictor = ConformalPredictor(alpha=0.05)  # 95% coverage
        predictor.calibrate(cal_probs, cal_labels)
        result = predictor.predict(test_probs, test_labels)
        print(result.prediction_set)  # {YES} or {YES, MAYBE} etc.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.coverage_target = 1.0 - alpha
        self._tau: float | None = None
        self._calibration_scores: np.ndarray | None = None

    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Compute nonconformity scores on a held-out calibration set and
        set the threshold tau to achieve target coverage.

        Args:
            probs:  (n_cal, n_classes) predicted probabilities
            labels: (n_cal,) integer true class indices
        """
        if probs.ndim != 2:
            raise ValueError("probs must be 2D: (n_samples, n_classes)")
        n = len(labels)
        # Nonconformity score: 1 minus the probability assigned to the true class
        scores = 1.0 - probs[np.arange(n), labels]
        self._calibration_scores = scores

        # Adjusted quantile to ensure finite-sample coverage guarantee
        level = np.ceil((n + 1) * self.coverage_target) / n
        level = min(level, 1.0)
        self._tau = float(np.quantile(scores, level))

    def predict(
        self,
        probs: np.ndarray,
        labels: list[Any] | None = None,
    ) -> ConformalResult:
        """
        Build prediction set for a single test example.

        Args:
            probs:  (n_classes,) predicted probabilities for one example
            labels: optional list of class label names (strings)
        Returns:
            ConformalResult with prediction set and coverage guarantee
        """
        if self._tau is None:
            raise RuntimeError("Call calibrate() before predict()")
        if probs.ndim != 1:
            raise ValueError("probs must be 1D for a single example")

        n_classes = len(probs)
        label_names = labels or list(range(n_classes))

        # Include class i in prediction set if its nonconformity score ≤ tau
        prediction_set = [
            label_names[i]
            for i in range(n_classes)
            if (1.0 - probs[i]) <= self._tau
        ]
        if not prediction_set:
            # Edge case: always include the argmax to avoid empty sets
            prediction_set = [label_names[int(np.argmax(probs))]]

        point_pred_idx = int(np.argmax(probs))
        score = float(1.0 - probs[point_pred_idx])

        return ConformalResult(
            prediction_set=prediction_set,
            point_prediction=label_names[point_pred_idx],
            point_confidence=float(probs[point_pred_idx]),
            nonconformity_score=score,
            coverage_guarantee=self.coverage_target,
            set_size=len(prediction_set),
            is_uncertain=len(prediction_set) > 1,
        )

    @property
    def threshold(self) -> float | None:
        """Calibrated nonconformity threshold tau."""
        return self._tau

    def empirical_coverage(
        self, probs: np.ndarray, labels: np.ndarray, label_names: list[Any] | None = None
    ) -> float:
        """Measure actual coverage on a test set (should be ≥ 1 - alpha)."""
        n = len(labels)
        hits = 0
        names = label_names or list(range(probs.shape[1]))
        for i in range(n):
            result = self.predict(probs[i], names)
            true_label = names[labels[i]]
            if true_label in result.prediction_set:
                hits += 1
        return hits / n
