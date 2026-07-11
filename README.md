# TrustLayer

**Datadog for AI Reliability** — detects when models are confidently wrong before it reaches production.

Companies deploying AI models need to know: *is my model actually reliable, or just confident?* TrustLayer answers that question with statistical guarantees.

## Research Basis

Inspired by:
> **A Quiet Failure in Calibrated Virtual Screening** (arXiv 2025)  
> **Robust Human-AI Complementarity Under Uncertainty** (arXiv 2025)

The first paper demonstrates that models can maintain high reported calibration while silently failing on specific sub-populations. The second shows that optimal human-AI collaboration requires selective deferral based on quantified uncertainty — not blanket human review.

## The Problem

Standard model outputs:

```
Prediction:  CANCER = YES
Confidence:  94%
```

The problem: that 94% might only be 60% accurate in practice. The model doesn't know what it doesn't know.

## How TrustLayer Fixes This

```
TrustLayer output:

Prediction set:  {YES, MAYBE}       ← conformal set with 95% coverage guarantee
Point prediction: YES
Escalation:       ESCALATE          ← mandatory human review
Calibration ECE:  0.031             ← model is well-calibrated overall
Drift alert:      None
```

## Install

```bash
pip install trustlayer
```

## Quick Start

### Python SDK

```python
import numpy as np
from trustlayer.monitor import TrustLayerMonitor

monitor = TrustLayerMonitor(
    label_names=["benign", "malignant"],
    alpha=0.05,               # 95% coverage guarantee
    auto_threshold=0.9,       # confidence ≥ 0.9 → auto-approve
    escalate_threshold=0.5,   # confidence < 0.5 → mandatory human review
)

# Calibrate on a held-out set
monitor.calibrate(cal_probs, cal_labels)

# Predict with reliability guarantees
result = monitor.predict(probs=np.array([0.43, 0.57]))

print(result.prediction_set)     # ["malignant"] or ["benign", "malignant"]
print(result.escalation)         # EscalationDecision.ESCALATE
print(result.coverage_guarantee) # 0.95
```

### FastAPI Server

```bash
uvicorn trustlayer.api:app --host 0.0.0.0 --port 8001
```

```bash
# Calibrate
curl -X POST http://localhost:8001/calibrate \
  -d '{"probs": [[0.8,0.2],[0.3,0.7]], "labels": [0,1]}'

# Predict
curl -X POST http://localhost:8001/predict \
  -d '{"probs": [0.43, 0.57]}'

# Monitor calibration drift
curl http://localhost:8001/calibration/report

# Dashboard summary
curl http://localhost:8001/dashboard
```

## Architecture

```
Customer Model
      │
TrustLayer SDK (trustlayer.monitor.TrustLayerMonitor)
      ├── ConformalPredictor  → prediction sets + coverage guarantee
      ├── CalibrationMonitor  → ECE tracking + drift detection
      └── EscalationRouter    → auto-approve / review / escalate
      │
TrustLayer API (FastAPI)
      │
Dashboard (React + Recharts)
      │
PostgreSQL (ticket history + calibration logs)
```

## Core Concepts

### Conformal Prediction

Returns a *set* of labels with a statistical guarantee — not a single number:

```
Standard:  P(Cancer) = 87%          ← no guarantee
Conformal: {Cancer, Unknown} covers true label with P ≥ 95%  ← guaranteed
```

### Calibration Monitoring (ECE)

Tracks Expected Calibration Error over time. A model saying "90% confident" should be right 90% of the time:

```
ECE = 0.02  → well calibrated ✓
ECE = 0.18  → model is overconfident → DRIFT ALERT
```

### Human Escalation

```
confidence ≥ 0.90  → APPROVE   (AI decides)
confidence 0.50–0.89 → REVIEW  (optional human check)
confidence < 0.50  → ESCALATE  (mandatory human review)
```

## Development

```bash
git clone https://github.com/yourusername/trustlayer
cd trustlayer
uv sync --all-extras
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## Resume Bullets

- Developed an AI reliability monitoring platform using conformal prediction and uncertainty estimation to detect model failures before deployment.
- Created automated calibration dashboards tracking confidence drift across production ML models with ECE-based alerting.
- Designed human-in-the-loop escalation pipelines improving prediction reliability for safety-critical AI systems via selective deferral.

## License

MIT
