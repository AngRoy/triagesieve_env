"""Unit tests for HintEngine (§16.2).

RED phase: Tests define the contract for deterministic hint generation.
Each test covers one of the 9 ordered predicates, plus mode gating.
"""

from __future__ import annotations

import pytest

from ..models import (
    ActionType,
    CloseReason,
    CustomerTier,
    Impact,
    IssueFamily,
    IssueSubtype,
    NonActionableSubtype,
    Priority,
    QueueId,
    SourceChannel,
    TriageSieveAction,
    TicketStatus,
    Urgency,
    HiddenTicketTruth,
)
from ..server.hint_engine import HintContext, HintEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_truth(**overrides) -> HiddenTicketTruth:
    """Create a HiddenTicketTruth with sensible defaults, overridable."""
    defaults = dict(
        ticket_id="T001",
        customer_tier=CustomerTier.PRO,
        source_channel=SourceChannel.CUSTOMER_EMAIL,
        issue_family=IssueFamily.BILLING,
        issue_subtype=IssueSubtype.REFUND,
        product_area="payments",
        impact=Impact.SINGLE_USER,
        urgency=Urgency.MEDIUM,
        priority=Priority.LOW,
        required_queue=QueueId.REFUND_TEAM,
        required_missing_fields=["order_id"],
        escalation_required=False,
        escalation_target=None,
        is_duplicate=False,
        duplicate_of=None,
        sla_response_deadline=60,
        sla_resolution_deadline=240,
        policy_graph_id="refund_missing_order_id",
        correct_template_ids=["refund_ack"],
        gold_terminal_status=TicketStatus.CLOSED,
        non_actionable_subtype=None,
    )
    defaults.update(overrides)
    return HiddenTicketTruth(**defaults)


def _make_ctx(**overrides) -> HintContext:
    """Create a HintContext with sensible defaults, overridable."""
    defaults = dict(
        last_action=None,
        last_action_result="ok",
        ticket_status=TicketStatus.CLASSIFIED,
        hidden_truth=_make_truth(),
        classification_set=None,
        impact_urgency_set=None,
        info_requested=False,
        info_received=False,
        routed_to=None,
        is_duplicate_truth=False,
        non_actionable_subtype=None,
    )
    defaults.update(overrides)
    return HintContext(**defaults)


@pytest.fixture
def engine() -> HintEngine:
    return HintEngine()


# ---------------------------------------------------------------------------
# Predicate 1: Pushback detected
# ---------------------------------------------------------------------------


class TestPushbackHint:
    def test_pushback_in_last_action_result(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action_result="Pushback: tech_support_l2 requires prerequisites: classification_set",
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.TECH_SUPPORT_L2,
                metadata={},
            ),
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "prerequisites" in hint.lower()

    def test_non_pushback_result_does_not_trigger(self, engine: HintEngine):
        ctx = _make_ctx(last_action_result="ok")
        hint = engine.generate_hint(ctx)
        # With defaults (no wrong action), no hint should fire
        # (other predicates may fire, but pushback specifically should not)
        if hint is not None:
            assert "prerequisites" not in hint.lower() or "routing" not in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 2: Wrong classification family
# ---------------------------------------------------------------------------


class TestWrongFamilyHint:
    def test_wrong_family_triggers_hint(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id="T001",
                issue_family=IssueFamily.TECHNICAL,
                issue_subtype=IssueSubtype.BUG_REPORT,
                metadata={},
            ),
            classification_set=(IssueFamily.TECHNICAL, IssueSubtype.BUG_REPORT),
            hidden_truth=_make_truth(
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
            ),
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "sender domain" in hint.lower() or "classif" in hint.lower()

    def test_correct_family_does_not_trigger_family_hint(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id="T001",
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
                metadata={},
            ),
            classification_set=(IssueFamily.BILLING, IssueSubtype.REFUND),
            hidden_truth=_make_truth(
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
            ),
        )
        hint = engine.generate_hint(ctx)
        # Should NOT get "sender domain" hint
        if hint is not None:
            assert "sender domain" not in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 3: Wrong classification subtype (correct family)
# ---------------------------------------------------------------------------


class TestWrongSubtypeHint:
    def test_wrong_subtype_correct_family(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id="T001",
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.INVOICE_ERROR,
                metadata={},
            ),
            classification_set=(IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR),
            hidden_truth=_make_truth(
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
            ),
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "specific nature" in hint.lower() or "problem" in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 4: Route/close without requesting required missing info
# ---------------------------------------------------------------------------


class TestMissingInfoHint:
    def test_route_without_info_request(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.REFUND_TEAM,
                metadata={},
            ),
            hidden_truth=_make_truth(required_missing_fields=["order_id"]),
            info_requested=False,
            info_received=False,
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "order_id" in hint.lower() or "thread history" in hint.lower()

    def test_close_without_info_request(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.CLOSE_TICKET,
                ticket_id="T001",
                close_reason=CloseReason.RESOLVED,
                metadata={},
            ),
            hidden_truth=_make_truth(required_missing_fields=["invoice_pdf"]),
            info_requested=False,
            info_received=False,
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "invoice_pdf" in hint.lower() or "thread history" in hint.lower()

    def test_no_missing_fields_no_hint(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.REFUND_TEAM,
                metadata={},
            ),
            hidden_truth=_make_truth(required_missing_fields=[]),
            info_received=True,
        )
        hint = engine.generate_hint(ctx)
        # Should not get a missing-info hint
        if hint is not None:
            assert "thread history" not in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 5: Escalation without prior info request
# ---------------------------------------------------------------------------


class TestEscalationHint:
    def test_escalate_without_info_request(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ESCALATE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.SECURITY_TEAM,
                metadata={},
            ),
            hidden_truth=_make_truth(
                required_missing_fields=["login_logs"],
                escalation_required=True,
            ),
            info_requested=False,
            info_received=False,
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "escalation" in hint.lower() and "information" in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 6: Routed to wrong queue
# ---------------------------------------------------------------------------


class TestWrongQueueHint:
    def test_wrong_queue(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.TECH_SUPPORT_L1,
                metadata={},
            ),
            routed_to=QueueId.TECH_SUPPORT_L1,
            hidden_truth=_make_truth(
                required_queue=QueueId.REFUND_TEAM,
                required_missing_fields=[],
            ),
            info_received=True,
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "routing policy" in hint.lower()

    def test_correct_queue_no_hint(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.REFUND_TEAM,
                metadata={},
            ),
            routed_to=QueueId.REFUND_TEAM,
            hidden_truth=_make_truth(
                required_queue=QueueId.REFUND_TEAM,
                required_missing_fields=[],
            ),
            info_received=True,
        )
        hint = engine.generate_hint(ctx)
        if hint is not None:
            assert "routing policy" not in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 7: Wrong impact/urgency
# ---------------------------------------------------------------------------


class TestWrongImpactUrgencyHint:
    def test_wrong_impact(self, engine: HintEngine):
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.SET_IMPACT_URGENCY,
                ticket_id="T001",
                impact=Impact.SINGLE_USER,
                urgency=Urgency.LOW,
                metadata={},
            ),
            impact_urgency_set=(Impact.SINGLE_USER, Urgency.LOW),
            hidden_truth=_make_truth(impact=Impact.ORG_WIDE, urgency=Urgency.HIGH),
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "impact" in hint.lower() or "re-assess" in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 8: Non-actionable mishandled
# ---------------------------------------------------------------------------


class TestNonActionableHint:
    def test_non_actionable_not_identified(self, engine: HintEngine):
        """Ticket is non-actionable but agent is trying to route it normally."""
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.BILLING_TEAM,
                metadata={},
            ),
            hidden_truth=_make_truth(
                non_actionable_subtype=NonActionableSubtype.SPAM_MARKETING,
                required_missing_fields=[],
            ),
            non_actionable_subtype=NonActionableSubtype.SPAM_MARKETING,
            info_received=True,
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "not require standard" in hint.lower() or "non-actionable" in hint.lower()


# ---------------------------------------------------------------------------
# Predicate 9: Duplicate missed
# ---------------------------------------------------------------------------


class TestDuplicateHint:
    def test_duplicate_not_merged(self, engine: HintEngine):
        """Ticket is a duplicate but agent is routing instead of merging."""
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id="T001",
                queue_id=QueueId.BILLING_TEAM,
                metadata={},
            ),
            hidden_truth=_make_truth(
                is_duplicate=True,
                duplicate_of="T000",
                required_missing_fields=[],
                non_actionable_subtype=None,
            ),
            is_duplicate_truth=True,
            non_actionable_subtype=None,
            info_received=True,
        )
        hint = engine.generate_hint(ctx)
        assert hint is not None
        assert "similar" in hint.lower() or "duplicate" in hint.lower()


# ---------------------------------------------------------------------------
# No hint when nothing is wrong
# ---------------------------------------------------------------------------


class TestNoHint:
    def test_all_correct_no_hint(self, engine: HintEngine):
        """When the agent does everything right, no hint fires."""
        ctx = _make_ctx(
            last_action=TriageSieveAction(
                action_type=ActionType.OPEN_TICKET,
                ticket_id="T001",
                metadata={},
            ),
            last_action_result="ok",
            hidden_truth=_make_truth(
                required_missing_fields=[],
                non_actionable_subtype=None,
                is_duplicate=False,
            ),
            non_actionable_subtype=None,
            is_duplicate_truth=False,
        )
        hint = engine.generate_hint(ctx)
        assert hint is None

    def test_none_hidden_truth_no_hint(self, engine: HintEngine):
        """No hidden truth available (e.g., skip/finish) → no hint."""
        ctx = _make_ctx(hidden_truth=None)
        hint = engine.generate_hint(ctx)
        assert hint is None
