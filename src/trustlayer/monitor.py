"""
TrustLayerMonitor: unified reliability monitor.

Combines conformal prediction, calibration tracking, and human
escalation into a single SDK that wraps any ML model.

This is the main entry point for integrating TrustLayer into
an existing ML pipeline:

    model_fn = lambda x: my_model.predict_proba(x)
    monitor = TrustLayerMonitor.from_model(model_fn, label_names=["benign", "malignant"])
    monitor.calibrate(cal_X, cal_y)

    result = monitor.predict(test_X[0])
    print(result["prediction_set"])   # {"benign"} or {"benign", "malignant"}
    print(result["escalation"])       # EscalationDecision.APPROVE / ESCALATE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from trustlayer.conformal import ConformalPredictor, ConformalResult
from trustlayer.calibration import CalibrationMonitor, CalibrationReport
from trustlayer.escalation import EscalationRouter, EscalationTicket, EscalationDecision


@dataclass
class MonitoredPrediction:
    """Full TrustLayer output for one prediction."""

    # Conformal output
    prediction_set: list[Any]
    point_prediction: Any
    confidence: float
    is_uncertain: bool
    set_size: int
    coverage_guarantee: float

    # Escalation output
    escalation: EscalationDecision
    ticket_id: Optional[str]
    escalation_reason: str

    # Calibration snapshot (updated in background)
    calibration_ece: Optional[float] = None
    calibration_drifting: Optional[bool] = None


class TrustLayerMonitor:
    """
    End-to-end reliability wrapper for any classification model.

    Provides:
      1. Conformal prediction sets (uncertainty-aware outputs)
      2. Online calibration monitoring (detect confidence drift)
      3. Human escalation routing (mandatory review for uncertain cases)

    Usage:
        monitor = TrustLayerMonitor(
            label_names=["benign", "malignant"],
            alpha=0.05,                   # 95% coverage guaranteed
            auto_threshold=0.9,
            escalate_threshold=0.5,
        )
        monitor.calibrate(cal_probs, cal_labels)

        result = monitor.predict(test_probs)
        print(result.prediction_set)   # e.g. ["benign"]
        print(result.escalation)       # EscalationDecision.APPROVE
    """

    def __init__(
        self,
        label_names: list[Any] | None = None,
        alpha: float = 0.05,
        auto_threshold: float = 0.9,
        escalate_threshold: float = 0.5,
        drift_threshold: float = 0.1,
    ) -> None:
        self.label_names = label_names
        self.conformal = ConformalPredictor(alpha=alpha)
        self.calibration = CalibrationMonitor(drift_threshold=drift_threshold)
        self.escalation = EscalationRouter(
            auto_threshold=auto_threshold,
            escalate_threshold=escalate_threshold,
        )

    @classmethod
    def from_model(
        cls,
        model_fn: Callable[[np.ndarray], np.ndarray],
        label_names: list[Any] | None = None,
        **kwargs: Any,
    ) -> "TrustLayerMonitor":
        """
        Construct a monitor wrapping an arbitrary predict_proba function.
        The model_fn should return (n_samples, n_classes) probabilities.
        """
        monitor = cls(label_names=label_names, **kwargs)
        monitor._model_fn = model_fn
        return monitor

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None:
        """Calibrate conformal predictor on a held-out set."""
        self.conformal.calibrate(cal_probs, cal_labels)

    def predict(
        self,
        probs: np.ndarray,
        ground_truth: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> MonitoredPrediction:
        """
        Run a single prediction through the full TrustLayer pipeline.

        Args:
            probs:        (n_classes,) predicted probabilities
            ground_truth: true label if known (for online calibration logging)
            metadata:     optional context dict attached to escalation ticket
        Returns:
            MonitoredPrediction with conformal set + escalation decision
        """
        # Step 1: Conformal prediction
        cp_result: ConformalResult = self.conformal.predict(probs, self.label_names)

        # Step 2: Route through escalation
        ticket: EscalationTicket = self.escalation.route(
            prediction=cp_result.point_prediction,
            confidence=cp_result.point_confidence,
            metadata=metadata,
        )

        # Step 3: Log to calibration monitor if ground truth is known
        cal_ece = None
        cal_drifting = None
        if ground_truth is not None:
            correct = cp_result.point_prediction == ground_truth
            self.calibration.log(confidence=cp_result.point_confidence, correct=correct)
            if self.calibration.n_samples >= 50:
                report = self.calibration.report()
                cal_ece = report.expected_calibration_error
                cal_drifting = report.is_drifting

        return MonitoredPrediction(
            prediction_set=cp_result.prediction_set,
            point_prediction=cp_result.point_prediction,
            confidence=cp_result.point_confidence,
            is_uncertain=cp_result.is_uncertain,
            set_size=cp_result.set_size,
            coverage_guarantee=cp_result.coverage_guarantee,
            escalation=ticket.decision,
            ticket_id=ticket.ticket_id if ticket.decision != EscalationDecision.APPROVE else None,
            escalation_reason=ticket.reason,
            calibration_ece=cal_ece,
            calibration_drifting=cal_drifting,
        )

    def calibration_report(self) -> CalibrationReport:
        return self.calibration.report()

    def dashboard_summary(self) -> dict[str, Any]:
        """JSON-serialisable summary for dashboard / API."""
        esc = self.escalation.stats
        return {
            "conformal_threshold": self.conformal.threshold,
            "coverage_target": self.conformal.coverage_target,
            "escalation": {
                "total_routed": esc.total_routed,
                "auto_approved": esc.auto_approved,
                "reviewed": esc.reviewed,
                "escalated": esc.escalated,
                "escalation_rate": round(esc.escalation_rate, 3),
                "auto_approval_rate": round(esc.auto_approval_rate, 3),
                "human_agreement_rate": round(self.escalation.agreement_rate(), 3),
            },
            "calibration_n_samples": self.calibration.n_samples,
        }
