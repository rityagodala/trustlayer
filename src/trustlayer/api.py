"""
FastAPI server for TrustLayer.

Exposes:
  POST /predict           — run prediction through full TrustLayer pipeline
  POST /calibrate         — load calibration data
  GET  /dashboard         — summary metrics
  POST /escalation/resolve — human resolves a flagged ticket
  GET  /calibration/report — current calibration report
  GET  /health            — liveness probe

Run with:
    uvicorn trustlayer.api:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trustlayer.monitor import TrustLayerMonitor, MonitoredPrediction
from trustlayer.escalation import EscalationDecision

app = FastAPI(
    title="TrustLayer",
    description="Enterprise AI Reliability Platform — Datadog for AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton monitor (in production: one per deployed model, managed via registry)
_monitor = TrustLayerMonitor(
    label_names=["negative", "positive"],
    alpha=0.05,
    auto_threshold=0.9,
    escalate_threshold=0.5,
)
_calibrated = False


# --- Request / Response models ---

class PredictRequest(BaseModel):
    probs: list[float] = Field(..., description="Predicted probabilities per class")
    ground_truth: str | None = Field(None, description="True label if known (for calibration logging)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    prediction_set: list[str]
    point_prediction: str
    confidence: float
    is_uncertain: bool
    set_size: int
    coverage_guarantee: float
    escalation: str  # approve / review / escalate
    ticket_id: str | None
    escalation_reason: str
    calibration_ece: float | None
    calibration_drifting: bool | None


class CalibrateRequest(BaseModel):
    probs: list[list[float]]  # (n_cal, n_classes)
    labels: list[int]         # (n_cal,)
    label_names: list[str] | None = None


class ResolveRequest(BaseModel):
    ticket_id: str
    human_label: str


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "calibrated": str(_calibrated)}


@app.post("/calibrate")
def calibrate(req: CalibrateRequest) -> dict[str, Any]:
    global _calibrated
    probs = np.array(req.probs)
    labels = np.array(req.labels)
    if req.label_names:
        _monitor.label_names = req.label_names
    _monitor.calibrate(probs, labels)
    _calibrated = True
    return {
        "message": "Calibration complete",
        "n_samples": len(labels),
        "threshold": _monitor.conformal.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not _calibrated:
        raise HTTPException(
            status_code=400,
            detail="Model not calibrated. POST to /calibrate first.",
        )
    probs = np.array(req.probs)
    result: MonitoredPrediction = _monitor.predict(
        probs=probs,
        ground_truth=req.ground_truth,
        metadata=req.metadata,
    )
    return PredictResponse(
        prediction_set=[str(p) for p in result.prediction_set],
        point_prediction=str(result.point_prediction),
        confidence=round(result.confidence, 4),
        is_uncertain=result.is_uncertain,
        set_size=result.set_size,
        coverage_guarantee=result.coverage_guarantee,
        escalation=result.escalation.value,
        ticket_id=result.ticket_id,
        escalation_reason=result.escalation_reason,
        calibration_ece=result.calibration_ece,
        calibration_drifting=result.calibration_drifting,
    )


@app.post("/escalation/resolve")
def resolve_ticket(req: ResolveRequest) -> dict[str, Any]:
    try:
        ticket = _monitor.escalation.resolve(req.ticket_id, req.human_label)
        return {
            "ticket_id": ticket.ticket_id,
            "ai_prediction": ticket.prediction,
            "human_label": ticket.human_label,
            "agreement": ticket.prediction == ticket.human_label,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/calibration/report")
def calibration_report() -> dict[str, Any]:
    if _monitor.calibration.n_samples < 2:
        raise HTTPException(status_code=400, detail="Not enough samples for calibration report")
    report = _monitor.calibration_report()
    return {
        "ece": report.expected_calibration_error,
        "mce": report.max_calibration_error,
        "mean_confidence": report.mean_confidence,
        "mean_accuracy": report.mean_accuracy,
        "confidence_gap": report.confidence_gap,
        "is_drifting": report.is_drifting,
        "drift_alert": report.drift_alert,
        "buckets": report.bucket_stats,
    }


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    return _monitor.dashboard_summary()
