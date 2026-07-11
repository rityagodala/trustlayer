# TrustLayer Architecture

## System Overview

```
ML Model (any predict_proba)
         │
   ┌─────▼──────────────────────────────┐
   │      TrustLayerMonitor             │
   │                                    │
   │  ┌─────────────┐                  │
   │  │ Conformal   │ → prediction set  │
   │  │ Predictor   │   + coverage %    │
   │  └─────────────┘                  │
   │                                    │
   │  ┌─────────────┐                  │
   │  │ Calibration │ → ECE + drift    │
   │  │ Monitor     │   alerts          │
   │  └─────────────┘                  │
   │                                    │
   │  ┌─────────────┐                  │
   │  │ Escalation  │ → APPROVE /      │
   │  │ Router      │   REVIEW /        │
   │  └─────────────┘   ESCALATE       │
   └────────────────────────────────────┘
         │
   FastAPI Server (port 8001)
         │
   React Dashboard + PostgreSQL
```

## Conformal Prediction (split method)

1. Reserve ~20% of training data as calibration set
2. Run model on calibration set → collect nonconformity scores (1 - p_true)
3. Set τ = (1-α)-quantile of scores
4. At test time: include class i in prediction set if (1 - p_i) <= Τ

## Calibration (ECE)

Split confidence [0,1] into B equal-width buckets.
For each bucket b:  gap_b = |mean_confidence_b - mean_accuracy_b|
ECE = Σ_b (n_b / n) × gap_b

Drift alert fires when ECE > drift_threshold (default 0.1).

## Escalation Tiers

| Tier | Condition | Action |
|------|-----------|--------|
| APPROVE | conf ≥ 0.90 | Automatic —"no human needed |
| REVIEW | 0.50 <= conf < 0.90 | Soft flag → human may review |
| ESCALATE | conf < 0.50 | Mandatory human review |
