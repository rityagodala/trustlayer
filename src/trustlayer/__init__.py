"""
TrustLayer: Enterprise AI Reliability Platform.

Detects when models are confidently wrong — before it reaches production.
Implements conformal prediction, calibration monitoring, and human escalation.
"""

from trustlayer.monitor import TrustLayerMonitor
from trustlayer.conformal import ConformalPredictor
from trustlayer.calibration import CalibrationMonitor
from trustlayer.escalation import EscalationRouter

__version__ = "0.1.0"
__all__ = [
    "TrustLayerMonitor",
    "ConformalPredictor",
    "CalibrationMonitor",
    "EscalationRouter",
]
