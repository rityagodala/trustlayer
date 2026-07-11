"""Tests for escalation router."""

import pytest
from trustlayer.escalation import EscalationRouter, EscalationDecision


@pytest.fixture
def router():
    return EscalationRouter(auto_threshold=0.9, escalate_threshold=0.5)


def test_high_confidence_auto_approved(router):
    ticket = router.route("POSITIVE", confidence=0.95)
    assert ticket.decision == EscalationDecision.APPROVE


def test_low_confidence_escalated(router):
    ticket = router.route("POSITIVE", confidence=0.3)
    assert ticket.decision == EscalationDecision.ESCALATE


def test_middle_confidence_review(router):
    ticket = router.route("POSITIVE", confidence=0.7)
    assert ticket.decision == EscalationDecision.REVIEW


def test_escalated_ticket_in_queue(router):
    ticket = router.route("POSITIVE", confidence=0.1)
    assert any(t.ticket_id == ticket.ticket_id for t in router.pending_tickets())


def test_approved_ticket_not_in_queue(router):
    ticket = router.route("POSITIVE", confidence=0.99)
    pending_ids = [t.ticket_id for t in router.pending_tickets()]
    assert ticket.ticket_id not in pending_ids


def test_resolve_ticket(router):
    ticket = router.route("POSITIVE", confidence=0.1)
    resolved = router.resolve(ticket.ticket_id, human_label="NEGATIVE")
    assert resolved.resolved
    assert resolved.human_label == "NEGATIVE"


def test_resolve_nonexistent_raises(router):
    with pytest.raises(KeyError):
        router.resolve("nonexistent", human_label="X")


def test_stats_tracking(router):
    router.route("A", 0.95)  # approve
    router.route("B", 0.7)   # review
    router.route("C", 0.2)   # escalate
    assert router.stats.total_routed == 3
    assert router.stats.auto_approved == 1
    assert router.stats.escalated == 1


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        EscalationRouter(auto_threshold=0.4, escalate_threshold=0.6)
