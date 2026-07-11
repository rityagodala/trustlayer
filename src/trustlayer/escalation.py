"""
Human escalation router.

Automatically routes low-confidence or high-uncertainty predictions
to human reviewers. High-confidence predictions are auto-approved.

Inspired by:
  "Robust Human-AI Complementarity Under Uncertainty" (arXiv 2025)

The core insight: optimal human-AI collaboration is not "human reviews everything"
nor "AI decides everything" — it's selective deferral based on uncertainty.

Decision logic:
  confidence >= auto_threshold    → APPROVE (automatic)
  confidence < escalate_threshold → ESCALATE (human review)
  middle zone                     → REVIEW (optional human check)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EscalationDecision(Enum):
    APPROVE = "approve"       # AI decides automatically
    REVIEW = "review"         # soft flag — human may check
    ESCALATE = "escalate"     # mandatory human review


@dataclass
class EscalationTicket:
    """A flagged prediction awaiting human review."""

    ticket_id: str
    prediction: Any
    confidence: float
    decision: EscalationDecision
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False
    human_label: Any = None

    @property
    def is_pending(self) -> bool:
        return not self.resolved


@dataclass
class EscalationStats:
    total_routed: int = 0
    auto_approved: int = 0
    reviewed: int = 0
    escalated: int = 0
    human_resolved: int = 0

    @property
    def escalation_rate(self) -> float:
        return self.escalated / max(self.total_routed, 1)

    @property
    def auto_approval_rate(self) -> float:
        return self.auto_approved / max(self.total_routed, 1)


class EscalationRouter:
    """
    Routes predictions to humans based on confidence thresholds.

    Example:
        router = EscalationRouter(auto_threshold=0.9, escalate_threshold=0.5)

        ticket = router.route(
            prediction="BENIGN",
            confidence=0.43,
            metadata={"patient_id": "P001", "scan_id": "S42"},
        )
        # → EscalationDecision.ESCALATE

        # Human resolves
        router.resolve(ticket.ticket_id, human_label="MALIGNANT")
    """

    def __init__(
        self,
        auto_threshold: float = 0.9,
        escalate_threshold: float = 0.5,
    ) -> None:
        if not auto_threshold > escalate_threshold:
            raise ValueError("auto_threshold must be greater than escalate_threshold")
        self.auto_threshold = auto_threshold
        self.escalate_threshold = escalate_threshold
        self._queue: dict[str, EscalationTicket] = {}
        self._stats = EscalationStats()

    def route(
        self,
        prediction: Any,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> EscalationTicket:
        """
        Make a routing decision for a single prediction.

        Args:
            prediction:  The model's predicted label / output
            confidence:  Scalar confidence in [0, 1]
            metadata:    Optional dict (request ID, patient ID, etc.)
        Returns:
            EscalationTicket with the routing decision
        """
        if confidence >= self.auto_threshold:
            decision = EscalationDecision.APPROVE
            reason = f"confidence {confidence:.2f} ≥ auto_threshold {self.auto_threshold}"
            self._stats.auto_approved += 1
        elif confidence < self.escalate_threshold:
            decision = EscalationDecision.ESCALATE
            reason = (
                f"confidence {confidence:.2f} < escalate_threshold {self.escalate_threshold} "
                f"— mandatory human review required"
            )
            self._stats.escalated += 1
        else:
            decision = EscalationDecision.REVIEW
            reason = (
                f"confidence {confidence:.2f} in review zone "
                f"[{self.escalate_threshold}, {self.auto_threshold})"
            )
            self._stats.reviewed += 1

        self._stats.total_routed += 1

        ticket = EscalationTicket(
            ticket_id=str(uuid.uuid4())[:8],
            prediction=prediction,
            confidence=confidence,
            decision=decision,
            reason=reason,
            metadata=metadata or {},
        )
        if decision != EscalationDecision.APPROVE:
            self._queue[ticket.ticket_id] = ticket

        return ticket

    def resolve(self, ticket_id: str, human_label: Any) -> EscalationTicket:
        """Mark a ticket as resolved with the human's ground-truth label."""
        if ticket_id not in self._queue:
            raise KeyError(f"Ticket {ticket_id} not found in escalation queue")
        ticket = self._queue[ticket_id]
        ticket.resolved = True
        ticket.human_label = human_label
        self._stats.human_resolved += 1
        return ticket

    def pending_tickets(self) -> list[EscalationTicket]:
        return [t for t in self._queue.values() if t.is_pending]

    @property
    def stats(self) -> EscalationStats:
        return self._stats

    def agreement_rate(self) -> float:
        """Fraction of resolved tickets where AI prediction matched human label."""
        resolved = [t for t in self._queue.values() if t.resolved]
        if not resolved:
            return 0.0
        matches = sum(1 for t in resolved if t.prediction == t.human_label)
        return matches / len(resolved)
