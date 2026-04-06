"""Reward scoring regression tests (section 23.2).

Covers:
- PerTicketScore, EpisodePenalties, ScoreBreakdown data structures
- Per-ticket component scoring (8 components with exact weights)
- Priority-weighted aggregation (terminal business score, max 0.85)
- UJCS integration via priority-weighted average
- Episode penalties (invalid, reassignment, escalation, SLA)
- Priority-order score (correctly ordered pairs / total pairs)
- Final score formula: TBS + 0.15 * UJCS - penalties, clamped [0, 1]
- Per-ticket breakdown for traces
- Edge cases: empty episodes, single ticket, all-wrong, all-correct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ..models import (
    ActionType,
    CloseReason,
    CustomerTier,
    HiddenTicketTruth,
    Impact,
    IssueFamily,
    IssueSubtype,
    NonActionableSubtype,
    Priority,
    PRIORITY_WEIGHTS,
    QueueId,
    SourceChannel,
    TriageSieveAction,
    TaskDifficulty,
    TicketStatus,
    Urgency,
)
from ..baseline.scripted_expert import ScriptedExpert
from ..server.episode_engine import RenderedTicket
from ..server.triagesieve_env_environment import TriageSieveEnvironment
from ..server.policy_graph import (
    SOPGraph,
    SOPScoringData,
    SOPTracker,
    TicketGuardContext,
    compute_ujcs,
)
from ..server.scorer import (
    COMPONENT_WEIGHTS,
    EpisodePenalties,
    EpisodeScoringContext,
    PENALTY_AVOIDABLE_REASSIGNMENT,
    PENALTY_INVALID_ACTION,
    PENALTY_SLA_CRITICAL,
    PENALTY_SLA_HIGH,
    PENALTY_UNNECESSARY_ESCALATION,
    PerTicketScore,
    ScoreBreakdown,
    TERMINAL_BUSINESS_SCORE_MAX,
    UJCS_WEIGHT,
    compute_episode_score,
    score_ticket,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal test fixtures
# ---------------------------------------------------------------------------

# Minimal SOP graph data for testing (linear: new -> open -> classify -> route -> close)
_SIMPLE_SOP: dict[str, Any] = {
    "graph_id": "test_simple",
    "nodes": [
        {"id": "new", "checkpoint": False},
        {"id": "open", "checkpoint": False},
        {"id": "classify_billing_refund", "checkpoint": True},
        {"id": "route_billing_team", "checkpoint": True},
        {"id": "close_resolved", "checkpoint": True},
    ],
    "edges": [
        {"from": "new", "to": "open"},
        {"from": "open", "to": "classify_billing_refund"},
        {"from": "classify_billing_refund", "to": "route_billing_team"},
        {"from": "route_billing_team", "to": "close_resolved"},
    ],
    "entry_node": "new",
    "terminal_nodes": ["close_resolved"],
}

# SOP graph with escalation path
_ESCALATION_SOP: dict[str, Any] = {
    "graph_id": "test_escalation",
    "nodes": [
        {"id": "new", "checkpoint": False},
        {"id": "open", "checkpoint": False},
        {"id": "classify_security_suspicious_login", "checkpoint": True},
        {"id": "set_impact_urgency", "checkpoint": True},
        {"id": "route_security_team", "checkpoint": True},
        {"id": "close_resolved", "checkpoint": True},
    ],
    "edges": [
        {"from": "new", "to": "open"},
        {"from": "open", "to": "classify_security_suspicious_login"},
        {"from": "classify_security_suspicious_login", "to": "set_impact_urgency"},
        {"from": "set_impact_urgency", "to": "route_security_team"},
        {"from": "route_security_team", "to": "close_resolved"},
    ],
    "entry_node": "new",
    "terminal_nodes": ["close_resolved"],
}

# SOP graph for duplicate ticket
_DUPLICATE_SOP: dict[str, Any] = {
    "graph_id": "test_duplicate",
    "nodes": [
        {"id": "new", "checkpoint": False},
        {"id": "open", "checkpoint": False},
        {"id": "merge_duplicate", "checkpoint": True},
    ],
    "edges": [
        {"from": "new", "to": "open"},
        {"from": "open", "to": "merge_duplicate"},
    ],
    "entry_node": "new",
    "terminal_nodes": ["merge_duplicate"],
}

# SOP with info request
_INFO_REQUEST_SOP: dict[str, Any] = {
    "graph_id": "test_info_request",
    "nodes": [
        {"id": "new", "checkpoint": False},
        {"id": "open", "checkpoint": False},
        {"id": "classify_billing_refund", "checkpoint": True},
        {"id": "request_order_id", "checkpoint": True},
        {"id": "route_refund_team", "checkpoint": True},
        {"id": "close_resolved", "checkpoint": True},
    ],
    "edges": [
        {"from": "new", "to": "open"},
        {"from": "open", "to": "classify_billing_refund"},
        {"from": "classify_billing_refund", "to": "request_order_id"},
        {"from": "request_order_id", "to": "route_refund_team"},
        {"from": "route_refund_team", "to": "close_resolved"},
    ],
    "entry_node": "new",
    "terminal_nodes": ["close_resolved"],
}


def _make_hidden_truth(
    ticket_id: str = "T001",
    *,
    issue_family: IssueFamily = IssueFamily.BILLING,
    issue_subtype: IssueSubtype = IssueSubtype.REFUND,
    impact: Impact = Impact.SINGLE_USER,
    urgency: Urgency = Urgency.MEDIUM,
    priority: Priority = Priority.LOW,
    required_queue: QueueId = QueueId.BILLING_TEAM,
    required_missing_fields: list[str] | None = None,
    escalation_required: bool = False,
    escalation_target: QueueId | None = None,
    is_duplicate: bool = False,
    duplicate_of: str | None = None,
    correct_template_ids: list[str] | None = None,
    gold_terminal_status: TicketStatus = TicketStatus.CLOSED,
    non_actionable_subtype: NonActionableSubtype | None = None,
    customer_tier: CustomerTier = CustomerTier.FREE,
) -> HiddenTicketTruth:
    return HiddenTicketTruth(
        ticket_id=ticket_id,
        customer_tier=customer_tier,
        source_channel=SourceChannel.CUSTOMER_EMAIL,
        issue_family=issue_family,
        issue_subtype=issue_subtype,
        product_area="test",
        impact=impact,
        urgency=urgency,
        priority=priority,
        required_queue=required_queue,
        required_missing_fields=required_missing_fields or [],
        escalation_required=escalation_required,
        escalation_target=escalation_target,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        sla_response_deadline=1440,
        sla_resolution_deadline=4320,
        policy_graph_id="test_simple",
        correct_template_ids=correct_template_ids or [],
        gold_terminal_status=gold_terminal_status,
        non_actionable_subtype=non_actionable_subtype,
    )


def _make_rendered_ticket(
    ticket_id: str = "T001",
    hidden_truth: HiddenTicketTruth | None = None,
    sop_graph: dict[str, Any] | None = None,
) -> RenderedTicket:
    ht = hidden_truth or _make_hidden_truth(ticket_id=ticket_id)
    return RenderedTicket(
        ticket_id=ticket_id,
        subject="Test ticket",
        body="Test body content for the ticket",
        sender_email="user@test.com",
        received_at="2026-04-05T10:00:00Z",
        customer_tier=ht.customer_tier,
        source_channel=ht.source_channel,
        has_attachment=False,
        attachments=[],
        thread_history=[],
        internal_notes=[],
        hidden_truth=ht,
        sop_graph=sop_graph or _SIMPLE_SOP,
    )


def _make_tracker_at_terminal(sop_data: dict[str, Any]) -> SOPTracker:
    """Build a SOPTracker that has reached the terminal node (perfect path)."""
    graph = SOPGraph.from_archetype_data(sop_data)
    tracker = SOPTracker(graph)
    # Walk through the entire gold path
    ctx_all_true = _all_true_context()
    for node_id in graph.gold_path[1:]:  # skip entry node
        tracker.try_advance(node_id, ctx_all_true)
    return tracker


def _make_tracker_at_entry(sop_data: dict[str, Any]) -> SOPTracker:
    """Build a SOPTracker still at the entry node (no progress)."""
    graph = SOPGraph.from_archetype_data(sop_data)
    return SOPTracker(graph)


def _all_true_context() -> TicketGuardContext:
    """Return a TicketGuardContext with every guard condition satisfied."""
    return TicketGuardContext(
        classification_set=True,
        impact_urgency_set=True,
        missing_fields_requested=True,
        info_received=True,
        escalation_required=True,
        duplicate_confirmed=True,
    )


def _make_context(
    tickets: list[RenderedTicket],
    *,
    ticket_states: dict[str, TicketStatus] | None = None,
    ticket_classifications: dict[str, tuple[IssueFamily, IssueSubtype]] | None = None,
    ticket_impact_urgency: dict[str, tuple[Impact, Urgency]] | None = None,
    ticket_routed_to: dict[str, QueueId] | None = None,
    ticket_escalated_to: dict[str, QueueId] | None = None,
    ticket_close_reasons: dict[str, CloseReason] | None = None,
    ticket_info_requested: dict[str, list[str]] | None = None,
    ticket_info_received: dict[str, bool] | None = None,
    ticket_merged_to: dict[str, str] | None = None,
    ticket_templates_used: dict[str, list[str]] | None = None,
    sop_trackers: dict[str, SOPTracker] | None = None,
    invalid_action_count: int = 0,
    ticket_route_count: dict[str, int] | None = None,
    ticket_first_substantive_step: dict[str, int] | None = None,
) -> EpisodeScoringContext:
    return EpisodeScoringContext(
        tickets=tickets,
        ticket_states=ticket_states or {t.ticket_id: TicketStatus.NEW for t in tickets},
        ticket_classifications=ticket_classifications or {},
        ticket_impact_urgency=ticket_impact_urgency or {},
        ticket_routed_to=ticket_routed_to or {},
        ticket_escalated_to=ticket_escalated_to or {},
        ticket_close_reasons=ticket_close_reasons or {},
        ticket_info_requested=ticket_info_requested or {},
        ticket_info_received=ticket_info_received or {t.ticket_id: False for t in tickets},
        ticket_merged_to=ticket_merged_to or {},
        ticket_templates_used=ticket_templates_used or {},
        sop_trackers=sop_trackers or {
            t.ticket_id: _make_tracker_at_entry(t.sop_graph) for t in tickets
        },
        invalid_action_count=invalid_action_count,
        ticket_route_count=ticket_route_count or {},
        ticket_first_substantive_step=ticket_first_substantive_step or {},
    )


# ---------------------------------------------------------------------------
# Test: Constants match CLAUDE.md exactly
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify all constants match CLAUDE.md section 17 exactly."""

    def test_terminal_business_score_max(self):
        assert TERMINAL_BUSINESS_SCORE_MAX == 0.85

    def test_ujcs_weight(self):
        assert UJCS_WEIGHT == 0.15

    def test_component_weights_sum_to_085(self):
        assert sum(COMPONENT_WEIGHTS.values()) == pytest.approx(0.85)

    def test_component_weights_exact(self):
        assert COMPONENT_WEIGHTS["classification"] == 0.15
        assert COMPONENT_WEIGHTS["impact_urgency"] == 0.15
        assert COMPONENT_WEIGHTS["queue"] == 0.20
        assert COMPONENT_WEIGHTS["missing_info"] == 0.10
        assert COMPONENT_WEIGHTS["escalation"] == 0.10
        assert COMPONENT_WEIGHTS["duplicate_non_actionable"] == 0.05
        assert COMPONENT_WEIGHTS["template"] == 0.05
        assert COMPONENT_WEIGHTS["terminal_status"] == 0.05

    def test_penalty_constants(self):
        assert PENALTY_INVALID_ACTION == 0.03
        assert PENALTY_AVOIDABLE_REASSIGNMENT == 0.05
        assert PENALTY_UNNECESSARY_ESCALATION == 0.05
        assert PENALTY_SLA_HIGH == 0.05
        assert PENALTY_SLA_CRITICAL == 0.10


# ---------------------------------------------------------------------------
# Test: Per-ticket scoring components
# ---------------------------------------------------------------------------


class TestPerTicketScoring:
    """Test individual component scoring logic for a single ticket."""

    def test_perfect_ticket_scores_085(self):
        """A perfectly handled ticket should score 0.85 (sum of all weights)."""
        ht = _make_hidden_truth(priority=Priority.LOW)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            ticket_info_received={"T001": False},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.raw_score == pytest.approx(0.85)

    def test_classification_correct_full_marks(self):
        """Correct family + subtype = full classification score."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.classification == pytest.approx(COMPONENT_WEIGHTS["classification"])

    def test_classification_family_only_half(self):
        """Correct family but wrong subtype = half classification score."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.classification == pytest.approx(COMPONENT_WEIGHTS["classification"] * 0.5)

    def test_classification_wrong_family_zero(self):
        """Wrong family = zero classification score."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            ticket_classifications={"T001": (IssueFamily.TECHNICAL, IssueSubtype.BUG_REPORT)},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.classification == pytest.approx(0.0)

    def test_classification_not_set_zero(self):
        """No classification at all = zero."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.OPENED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.classification == pytest.approx(0.0)

    def test_impact_urgency_both_correct(self):
        """Both impact and urgency correct = full score."""
        ht = _make_hidden_truth(impact=Impact.TEAM, urgency=Urgency.HIGH)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            ticket_impact_urgency={"T001": (Impact.TEAM, Urgency.HIGH)},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.impact_urgency == pytest.approx(COMPONENT_WEIGHTS["impact_urgency"])

    def test_impact_urgency_one_correct_half(self):
        """One of impact/urgency correct = half score."""
        ht = _make_hidden_truth(impact=Impact.TEAM, urgency=Urgency.HIGH)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            ticket_impact_urgency={"T001": (Impact.TEAM, Urgency.LOW)},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.impact_urgency == pytest.approx(COMPONENT_WEIGHTS["impact_urgency"] * 0.5)

    def test_impact_urgency_not_set_zero(self):
        """Impact/urgency not set = zero."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.impact_urgency == pytest.approx(0.0)

    def test_queue_correct(self):
        """Correct queue = full queue score."""
        ht = _make_hidden_truth(required_queue=QueueId.BILLING_TEAM)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.queue == pytest.approx(COMPONENT_WEIGHTS["queue"])

    def test_queue_wrong(self):
        """Wrong queue = zero queue score."""
        ht = _make_hidden_truth(required_queue=QueueId.BILLING_TEAM)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            ticket_routed_to={"T001": QueueId.TECH_SUPPORT_L1},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.queue == pytest.approx(0.0)

    def test_queue_not_routed_zero(self):
        """Ticket never routed = zero queue score."""
        ht = _make_hidden_truth(required_queue=QueueId.BILLING_TEAM)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.queue == pytest.approx(0.0)

    def test_queue_via_escalation_correct(self):
        """Ticket escalated to correct escalation_target counts for queue."""
        ht = _make_hidden_truth(
            required_queue=QueueId.SECURITY_TEAM,
            escalation_required=True,
            escalation_target=QueueId.SECURITY_TEAM,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ESCALATED},
            ticket_escalated_to={"T001": QueueId.SECURITY_TEAM},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        # Queue score: escalation target matches required queue
        assert pts.queue == pytest.approx(COMPONENT_WEIGHTS["queue"])

    def test_missing_info_not_required_full(self):
        """No missing fields required = full missing-info score."""
        ht = _make_hidden_truth(required_missing_fields=[])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.missing_info == pytest.approx(COMPONENT_WEIGHTS["missing_info"])

    def test_missing_info_requested_and_received(self):
        """Required fields requested and response received = full score."""
        ht = _make_hidden_truth(required_missing_fields=["order_id"])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_info_requested={"T001": ["order_id"]},
            ticket_info_received={"T001": True},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.missing_info == pytest.approx(COMPONENT_WEIGHTS["missing_info"])

    def test_missing_info_not_requested_zero(self):
        """Required fields exist but agent never requested = zero."""
        ht = _make_hidden_truth(required_missing_fields=["order_id"])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_info_received={"T001": False},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.missing_info == pytest.approx(0.0)

    def test_missing_info_requested_wrong_fields_partial(self):
        """Requested info but wrong fields (no response received) = partial."""
        ht = _make_hidden_truth(required_missing_fields=["order_id"])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.WAITING_FOR_INFO},
            ticket_info_requested={"T001": ["wrong_field"]},
            ticket_info_received={"T001": False},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.missing_info == pytest.approx(COMPONENT_WEIGHTS["missing_info"] * 0.25)

    def test_escalation_required_and_done_correctly(self):
        """Escalation required and done to correct target = full score."""
        ht = _make_hidden_truth(
            escalation_required=True,
            escalation_target=QueueId.SECURITY_TEAM,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ESCALATED},
            ticket_escalated_to={"T001": QueueId.SECURITY_TEAM},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.escalation == pytest.approx(COMPONENT_WEIGHTS["escalation"])

    def test_escalation_required_but_not_done(self):
        """Escalation required but agent didn't escalate = zero."""
        ht = _make_hidden_truth(escalation_required=True, escalation_target=QueueId.SECURITY_TEAM)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.escalation == pytest.approx(0.0)

    def test_escalation_not_required_not_done_full(self):
        """No escalation required and none done = full score."""
        ht = _make_hidden_truth(escalation_required=False)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.escalation == pytest.approx(COMPONENT_WEIGHTS["escalation"])

    def test_escalation_not_required_but_done_zero(self):
        """Agent escalated when not required = zero."""
        ht = _make_hidden_truth(escalation_required=False)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ESCALATED},
            ticket_escalated_to={"T001": QueueId.SECURITY_TEAM},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.escalation == pytest.approx(0.0)

    def test_duplicate_correctly_merged(self):
        """Duplicate ticket merged to correct target = full dup/na score."""
        ht = _make_hidden_truth(
            is_duplicate=True,
            duplicate_of="T000",
            gold_terminal_status=TicketStatus.MERGED,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht, sop_graph=_DUPLICATE_SOP)
        tracker = _make_tracker_at_entry(_DUPLICATE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.MERGED},
            ticket_merged_to={"T001": "T000"},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.duplicate_non_actionable == pytest.approx(
            COMPONENT_WEIGHTS["duplicate_non_actionable"]
        )

    def test_duplicate_not_merged_zero(self):
        """Duplicate ticket not merged = zero dup/na score."""
        ht = _make_hidden_truth(is_duplicate=True, duplicate_of="T000")
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.duplicate_non_actionable == pytest.approx(0.0)

    def test_duplicate_merged_to_wrong_target_zero(self):
        """Duplicate merged to wrong target = zero dup/na score."""
        ht = _make_hidden_truth(
            is_duplicate=True,
            duplicate_of="T000",
            gold_terminal_status=TicketStatus.MERGED,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht, sop_graph=_DUPLICATE_SOP)
        tracker = _make_tracker_at_entry(_DUPLICATE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.MERGED},
            ticket_merged_to={"T001": "T099"},  # wrong target
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.duplicate_non_actionable == pytest.approx(0.0)

    def test_non_actionable_closed_correctly(self):
        """Non-actionable ticket closed as non_actionable = full score."""
        ht = _make_hidden_truth(
            non_actionable_subtype=NonActionableSubtype.SPAM_MARKETING,
            gold_terminal_status=TicketStatus.CLOSED,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_close_reasons={"T001": CloseReason.NON_ACTIONABLE},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.duplicate_non_actionable == pytest.approx(
            COMPONENT_WEIGHTS["duplicate_non_actionable"]
        )

    def test_neither_duplicate_nor_non_actionable_full(self):
        """Normal ticket (not dup, not non-actionable) = full dup/na score."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.duplicate_non_actionable == pytest.approx(
            COMPONENT_WEIGHTS["duplicate_non_actionable"]
        )

    def test_template_correct(self):
        """Agent used correct template = full template score."""
        ht = _make_hidden_truth(correct_template_ids=["tmpl_refund_request"])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_templates_used={"T001": ["tmpl_refund_request"]},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.template == pytest.approx(COMPONENT_WEIGHTS["template"])

    def test_template_not_needed_full(self):
        """No templates expected = full score."""
        ht = _make_hidden_truth(correct_template_ids=[])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.template == pytest.approx(COMPONENT_WEIGHTS["template"])

    def test_template_wrong_zero(self):
        """Template expected but wrong one used = zero."""
        ht = _make_hidden_truth(correct_template_ids=["tmpl_refund_request"])
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_templates_used={"T001": ["tmpl_wrong"]},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.template == pytest.approx(0.0)

    def test_terminal_status_correct(self):
        """Terminal status matches gold = full score."""
        ht = _make_hidden_truth(gold_terminal_status=TicketStatus.CLOSED)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.terminal_status == pytest.approx(COMPONENT_WEIGHTS["terminal_status"])

    def test_terminal_status_wrong(self):
        """Terminal status doesn't match gold = zero."""
        ht = _make_hidden_truth(gold_terminal_status=TicketStatus.CLOSED)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.terminal_status == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: Priority-weighted aggregation
# ---------------------------------------------------------------------------


class TestPriorityWeightedAggregation:
    """Test priority-weighted terminal business score and UJCS aggregation."""

    def test_single_ticket_tbs_max_085(self):
        """Perfect single ticket TBS = 0.85."""
        ht = _make_hidden_truth(priority=Priority.MEDIUM)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            ticket_info_received={"T001": False},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.terminal_business_score == pytest.approx(0.85)

    def test_critical_ticket_weighted_higher(self):
        """Critical ticket has 2x weight vs medium ticket's 1x."""
        ht_critical = _make_hidden_truth(
            ticket_id="T001",
            priority=Priority.CRITICAL,
            impact=Impact.ORG_WIDE,
            urgency=Urgency.CRITICAL,
        )
        ht_medium = _make_hidden_truth(
            ticket_id="T002",
            priority=Priority.MEDIUM,
            impact=Impact.SINGLE_USER,
            urgency=Urgency.HIGH,
        )
        t1 = _make_rendered_ticket("T001", hidden_truth=ht_critical)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht_medium)

        tracker1 = _make_tracker_at_terminal(_SIMPLE_SOP)
        tracker2 = _make_tracker_at_entry(_SIMPLE_SOP)

        # T001 perfect, T002 all wrong
        ctx = _make_context(
            [t1, t2],
            ticket_states={"T001": TicketStatus.CLOSED, "T002": TicketStatus.NEW},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.ORG_WIDE, Urgency.CRITICAL)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            ticket_info_received={"T001": False, "T002": False},
            sop_trackers={"T001": tracker1, "T002": tracker2},
        )

        result = compute_episode_score(ctx)
        # T001 (critical, weight=2.0) scores 0.85, T002 (medium, weight=1.0) scores ~0
        # TBS = (0.85*2.0 + ~0*1.0) / (2.0+1.0) = ~0.567
        # Critical ticket weight dominates
        assert result.terminal_business_score > 0.5

    def test_priority_weights_match_models(self):
        """Priority weights used in scorer match PRIORITY_WEIGHTS from models."""
        assert PRIORITY_WEIGHTS[Priority.LOW] == 0.5
        assert PRIORITY_WEIGHTS[Priority.MEDIUM] == 1.0
        assert PRIORITY_WEIGHTS[Priority.HIGH] == 1.5
        assert PRIORITY_WEIGHTS[Priority.CRITICAL] == 2.0


# ---------------------------------------------------------------------------
# Test: UJCS integration
# ---------------------------------------------------------------------------


class TestUJCSIntegration:
    """Test UJCS contribution to final score."""

    def test_perfect_ujcs_contributes_015(self):
        """Perfect UJCS (1.0) contributes 0.15 to final score."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.ujcs_openenv == pytest.approx(1.0, abs=0.01)
        assert result.ujcs_contribution == pytest.approx(0.15, abs=0.01)

    def test_zero_ujcs_contributes_zero(self):
        """Zero UJCS contributes 0.0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        # Tracker at entry = 0 checkpoints visited
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.NEW},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.ujcs_contribution == pytest.approx(0.0, abs=0.05)

    def test_ujcs_priority_weighted(self):
        """UJCS is priority-weighted across tickets."""
        ht1 = _make_hidden_truth(ticket_id="T001", priority=Priority.CRITICAL)
        ht2 = _make_hidden_truth(ticket_id="T002", priority=Priority.LOW)
        t1 = _make_rendered_ticket("T001", hidden_truth=ht1)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht2)

        # T001 has perfect UJCS, T002 has zero
        tracker1 = _make_tracker_at_terminal(_SIMPLE_SOP)
        tracker2 = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [t1, t2],
            ticket_states={"T001": TicketStatus.CLOSED, "T002": TicketStatus.NEW},
            sop_trackers={"T001": tracker1, "T002": tracker2},
        )

        result = compute_episode_score(ctx)
        # Weighted avg: (1.0*2.0 + low_ujcs*0.5)/(2.0+0.5)
        # Critical ticket dominates, so UJCS should be > 0.5
        assert result.ujcs_openenv > 0.5


# ---------------------------------------------------------------------------
# Test: Episode penalties
# ---------------------------------------------------------------------------


class TestEpisodePenalties:
    """Test episode-level penalty calculation."""

    def test_no_penalties_zero(self):
        """Clean episode = zero penalties."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.episode_penalties.total_penalty == pytest.approx(0.0)

    def test_invalid_action_penalty(self):
        """Invalid actions incur -0.03 each."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.NEW},
            sop_trackers={"T001": tracker},
            invalid_action_count=3,
        )

        result = compute_episode_score(ctx)
        assert result.episode_penalties.invalid_action_count == 3
        assert result.episode_penalties.total_penalty >= 3 * PENALTY_INVALID_ACTION

    def test_avoidable_reassignment_penalty(self):
        """Routing a ticket multiple times incurs -0.05 per extra route."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ROUTED},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            sop_trackers={"T001": tracker},
            ticket_route_count={"T001": 2},  # routed twice = 1 reassignment
        )

        result = compute_episode_score(ctx)
        assert result.episode_penalties.avoidable_reassignment_count == 1
        assert result.reassignment_count == 1

    def test_unnecessary_escalation_penalty(self):
        """Escalating when not required incurs -0.05."""
        ht = _make_hidden_truth(escalation_required=False)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ESCALATED},
            ticket_escalated_to={"T001": QueueId.SECURITY_TEAM},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.episode_penalties.unnecessary_escalation_count == 1

    def test_sla_mishandling_critical(self):
        """Critical ticket not resolved = -0.10 SLA penalty."""
        ht = _make_hidden_truth(
            priority=Priority.CRITICAL,
            gold_terminal_status=TicketStatus.CLOSED,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.OPENED},  # not resolved
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.episode_penalties.sla_mishandling_penalties >= PENALTY_SLA_CRITICAL

    def test_sla_mishandling_high(self):
        """High priority ticket not resolved = -0.05 SLA penalty."""
        ht = _make_hidden_truth(
            priority=Priority.HIGH,
            gold_terminal_status=TicketStatus.CLOSED,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.OPENED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.episode_penalties.sla_mishandling_penalties >= PENALTY_SLA_HIGH


# ---------------------------------------------------------------------------
# Test: Priority-order score
# ---------------------------------------------------------------------------


class TestPriorityOrderScore:
    """Test the priority-order metric (section 19)."""

    def test_single_ticket_perfect_order(self):
        """Single ticket = perfect order score (1.0) or no pairs."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.priority_order_score == pytest.approx(1.0)

    def test_correct_priority_ordering(self):
        """Higher priority handled first = perfect priority-order score."""
        ht_high = _make_hidden_truth(ticket_id="T001", priority=Priority.HIGH)
        ht_low = _make_hidden_truth(ticket_id="T002", priority=Priority.LOW)
        t1 = _make_rendered_ticket("T001", hidden_truth=ht_high)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht_low)

        ctx = _make_context(
            [t1, t2],
            ticket_states={"T001": TicketStatus.CLOSED, "T002": TicketStatus.CLOSED},
            sop_trackers={
                "T001": _make_tracker_at_entry(_SIMPLE_SOP),
                "T002": _make_tracker_at_entry(_SIMPLE_SOP),
            },
            ticket_first_substantive_step={"T001": 1, "T002": 3},
        )

        result = compute_episode_score(ctx)
        assert result.priority_order_score == pytest.approx(1.0)

    def test_wrong_priority_ordering(self):
        """Lower priority handled first = zero priority-order score."""
        ht_high = _make_hidden_truth(ticket_id="T001", priority=Priority.HIGH)
        ht_low = _make_hidden_truth(ticket_id="T002", priority=Priority.LOW)
        t1 = _make_rendered_ticket("T001", hidden_truth=ht_high)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht_low)

        ctx = _make_context(
            [t1, t2],
            ticket_states={"T001": TicketStatus.CLOSED, "T002": TicketStatus.CLOSED},
            sop_trackers={
                "T001": _make_tracker_at_entry(_SIMPLE_SOP),
                "T002": _make_tracker_at_entry(_SIMPLE_SOP),
            },
            ticket_first_substantive_step={"T001": 5, "T002": 1},  # low handled first
        )

        result = compute_episode_score(ctx)
        assert result.priority_order_score == pytest.approx(0.0)

    def test_same_priority_always_correct(self):
        """Same priority tickets = always correctly ordered."""
        ht1 = _make_hidden_truth(ticket_id="T001", priority=Priority.MEDIUM)
        ht2 = _make_hidden_truth(ticket_id="T002", priority=Priority.MEDIUM)
        t1 = _make_rendered_ticket("T001", hidden_truth=ht1)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht2)

        ctx = _make_context(
            [t1, t2],
            ticket_states={"T001": TicketStatus.CLOSED, "T002": TicketStatus.CLOSED},
            sop_trackers={
                "T001": _make_tracker_at_entry(_SIMPLE_SOP),
                "T002": _make_tracker_at_entry(_SIMPLE_SOP),
            },
            ticket_first_substantive_step={"T001": 3, "T002": 1},
        )

        result = compute_episode_score(ctx)
        # Same priority = not comparable, so perfect score
        assert result.priority_order_score == pytest.approx(1.0)

    def test_unacted_ticket_penalizes_priority_order(self):
        """A critical ticket never acted on should hurt priority-order score."""
        ht_crit = _make_hidden_truth(ticket_id="T001", priority=Priority.CRITICAL)
        ht_low = _make_hidden_truth(ticket_id="T002", priority=Priority.LOW)
        t1 = _make_rendered_ticket("T001", hidden_truth=ht_crit)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht_low)

        ctx = _make_context(
            [t1, t2],
            ticket_states={"T001": TicketStatus.NEW, "T002": TicketStatus.CLOSED},
            sop_trackers={
                "T001": _make_tracker_at_entry(_SIMPLE_SOP),
                "T002": _make_tracker_at_entry(_SIMPLE_SOP),
            },
            # T001 (critical) never acted on, T002 (low) handled at step 2
            ticket_first_substantive_step={"T002": 2},
        )

        result = compute_episode_score(ctx)
        # Critical ticket gets sentinel step > 2, so it's "after" the low ticket = wrong
        assert result.priority_order_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: Final score formula and clamping
# ---------------------------------------------------------------------------


class TestFinalScoreFormula:
    """Test: FinalScore = TBS + 0.15 * UJCS - penalties, clamped [0, 1]."""

    def test_perfect_episode_scores_1(self):
        """Perfectly handled episode = final score 1.0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            ticket_info_received={"T001": False},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.final_score == pytest.approx(1.0, abs=0.01)

    def test_all_wrong_scores_near_zero(self):
        """Everything wrong = score near 0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.NEW},
            sop_trackers={"T001": tracker},
            invalid_action_count=5,
        )

        result = compute_episode_score(ctx)
        assert result.final_score <= 0.20

    def test_score_clamped_at_zero(self):
        """Heavy penalties cannot push score below 0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.NEW},
            sop_trackers={"T001": tracker},
            invalid_action_count=100,  # massive penalties
        )

        result = compute_episode_score(ctx)
        assert result.final_score >= 0.0

    def test_score_clamped_at_one(self):
        """Score cannot exceed 1.0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        assert result.final_score <= 1.0

    def test_formula_tbs_plus_ujcs_minus_penalties(self):
        """Verify final_score = TBS + 0.15 * UJCS - penalties (before clamping)."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_terminal(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_classifications={"T001": (IssueFamily.BILLING, IssueSubtype.REFUND)},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.MEDIUM)},
            ticket_routed_to={"T001": QueueId.BILLING_TEAM},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            sop_trackers={"T001": tracker},
            invalid_action_count=1,
        )

        result = compute_episode_score(ctx)
        expected_raw = (
            result.terminal_business_score
            + result.ujcs_contribution
            - result.episode_penalties.total_penalty
        )
        expected_clamped = max(0.0, min(1.0, expected_raw))
        assert result.final_score == pytest.approx(expected_clamped, abs=0.001)


# ---------------------------------------------------------------------------
# Test: Trace breakdown structure (section 24)
# ---------------------------------------------------------------------------


class TestTraceBreakdown:
    """Test ScoreBreakdown has all fields required by section 24 trace format."""

    def test_breakdown_has_all_trace_fields(self):
        """ScoreBreakdown contains all fields for structured trace output."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)

        # Top-level fields
        assert hasattr(result, "terminal_business_score")
        assert hasattr(result, "ujcs_openenv")
        assert hasattr(result, "ujcs_contribution")
        assert hasattr(result, "episode_penalties")
        assert hasattr(result, "final_score")
        assert hasattr(result, "priority_order_score")
        assert hasattr(result, "reassignment_count")
        assert hasattr(result, "invalid_action_count")
        assert hasattr(result, "per_ticket_scores")

    def test_per_ticket_score_has_all_components(self):
        """PerTicketScore contains all 8 component scores."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]

        assert hasattr(pts, "ticket_id")
        assert hasattr(pts, "classification")
        assert hasattr(pts, "impact_urgency")
        assert hasattr(pts, "queue")
        assert hasattr(pts, "missing_info")
        assert hasattr(pts, "escalation")
        assert hasattr(pts, "duplicate_non_actionable")
        assert hasattr(pts, "template")
        assert hasattr(pts, "terminal_status")
        assert hasattr(pts, "raw_score")
        assert hasattr(pts, "priority_weight")
        assert hasattr(pts, "ujcs")
        assert hasattr(pts, "wrong_parameterizations")

    def test_episode_penalties_structure(self):
        """EpisodePenalties has all required fields."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        ep = result.episode_penalties

        assert hasattr(ep, "invalid_action_count")
        assert hasattr(ep, "avoidable_reassignment_count")
        assert hasattr(ep, "unnecessary_escalation_count")
        assert hasattr(ep, "sla_mishandling_penalties")
        assert hasattr(ep, "total_penalty")


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case coverage."""

    def test_empty_episode_no_tickets(self):
        """Episode with no tickets should return sensible defaults."""
        ctx = _make_context(
            [],
            ticket_states={},
            sop_trackers={},
        )

        result = compute_episode_score(ctx)
        assert result.final_score >= 0.0
        assert result.final_score <= 1.0
        assert result.priority_order_score == pytest.approx(1.0)

    def test_score_ticket_function_exists(self):
        """score_ticket is exported and callable."""
        assert callable(score_ticket)

    def test_per_ticket_raw_score_bounded(self):
        """Per-ticket raw score never exceeds 0.85 or goes below 0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert 0.0 <= pts.raw_score <= 0.85 + 1e-9

    def test_multi_ticket_episode(self):
        """Three tickets with different priorities produce valid breakdown."""
        tickets = []
        states = {}
        trackers = {}
        classifications = {}
        for i, (tid, pri) in enumerate([
            ("T001", Priority.CRITICAL),
            ("T002", Priority.MEDIUM),
            ("T003", Priority.LOW),
        ]):
            ht = _make_hidden_truth(ticket_id=tid, priority=pri)
            t = _make_rendered_ticket(tid, hidden_truth=ht)
            tickets.append(t)
            states[tid] = TicketStatus.CLASSIFIED
            trackers[tid] = _make_tracker_at_entry(_SIMPLE_SOP)
            classifications[tid] = (IssueFamily.BILLING, IssueSubtype.REFUND)

        ctx = _make_context(
            tickets,
            ticket_states=states,
            ticket_classifications=classifications,
            sop_trackers=trackers,
        )

        result = compute_episode_score(ctx)
        assert len(result.per_ticket_scores) == 3
        assert 0.0 <= result.final_score <= 1.0
        assert result.terminal_business_score >= 0.0
        assert result.terminal_business_score <= 0.85 + 1e-9

    def test_impact_urgency_both_wrong(self):
        """Both impact and urgency wrong = zero."""
        ht = _make_hidden_truth(impact=Impact.ORG_WIDE, urgency=Urgency.CRITICAL)
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLASSIFIED},
            ticket_impact_urgency={"T001": (Impact.SINGLE_USER, Urgency.LOW)},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.impact_urgency == pytest.approx(0.0)

    def test_escalation_required_wrong_target(self):
        """Escalation required but to wrong target = partial (0.25)."""
        ht = _make_hidden_truth(
            escalation_required=True,
            escalation_target=QueueId.SECURITY_TEAM,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.ESCALATED},
            ticket_escalated_to={"T001": QueueId.TECH_SUPPORT_L2},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.escalation == pytest.approx(COMPONENT_WEIGHTS["escalation"] * 0.25)

    def test_non_actionable_not_closed_as_non_actionable(self):
        """Non-actionable ticket closed with wrong reason = zero dup/na."""
        ht = _make_hidden_truth(
            non_actionable_subtype=NonActionableSubtype.SPAM_MARKETING,
        )
        ticket = _make_rendered_ticket(hidden_truth=ht)
        tracker = _make_tracker_at_entry(_SIMPLE_SOP)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            ticket_close_reasons={"T001": CloseReason.RESOLVED},
            sop_trackers={"T001": tracker},
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.duplicate_non_actionable == pytest.approx(0.0)

    def test_no_sop_tracker_ujcs_zero(self):
        """Ticket with no SOP tracker gets UJCS = 0."""
        ht = _make_hidden_truth()
        ticket = _make_rendered_ticket(hidden_truth=ht)

        ctx = _make_context(
            [ticket],
            ticket_states={"T001": TicketStatus.CLOSED},
            sop_trackers={},  # no tracker
        )

        result = compute_episode_score(ctx)
        pts = result.per_ticket_scores["T001"]
        assert pts.ujcs == pytest.approx(0.0)

    def test_priority_order_three_tickets_partial(self):
        """Three tickets, one pair wrong = 2/3 priority-order score."""
        ht_crit = _make_hidden_truth(ticket_id="T001", priority=Priority.CRITICAL)
        ht_high = _make_hidden_truth(ticket_id="T002", priority=Priority.HIGH)
        ht_low = _make_hidden_truth(ticket_id="T003", priority=Priority.LOW)
        t1 = _make_rendered_ticket("T001", hidden_truth=ht_crit)
        t2 = _make_rendered_ticket("T002", hidden_truth=ht_high)
        t3 = _make_rendered_ticket("T003", hidden_truth=ht_low)

        ctx = _make_context(
            [t1, t2, t3],
            ticket_states={
                "T001": TicketStatus.CLOSED,
                "T002": TicketStatus.CLOSED,
                "T003": TicketStatus.CLOSED,
            },
            sop_trackers={
                "T001": _make_tracker_at_entry(_SIMPLE_SOP),
                "T002": _make_tracker_at_entry(_SIMPLE_SOP),
                "T003": _make_tracker_at_entry(_SIMPLE_SOP),
            },
            # T001 (crit) first, T003 (low) second, T002 (high) last
            # Pairs: (crit,high) crit@1 < high@5 = correct
            # (crit,low) crit@1 < low@3 = correct
            # (high,low) high@5 > low@3 = WRONG
            ticket_first_substantive_step={"T001": 1, "T002": 5, "T003": 3},
        )

        result = compute_episode_score(ctx)
        # 2 correct out of 3 pairs
        assert result.priority_order_score == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# §23.2 Integration-Level Reward Regression Tests
# ---------------------------------------------------------------------------


class TestRewardRegressions:
    """Integration tests from CLAUDE.md §23.2.

    These exercise the full scoring pipeline (environment + scorer) or
    targeted synthetic setups to verify the 4 declared reward invariants:

    1. Scripted expert on fixed seed → score ≥ 0.90.
    2. Intentionally bad trajectory → score ≤ 0.20.
    3. UJCS drops when mandatory checkpoint is skipped.
    4. Priority weighting changes final score when critical ticket is mishandled.
    """

    # ---- Test 1: Scripted expert ≥ 0.90 ----

    def test_scripted_expert_fixed_seed_ge_090(self):
        """§23.2: Scripted expert on fixed seed → score ≥ 0.90.

        Runs the oracle policy on seed=7, easy difficulty (failed_invoice_charge_dispute).
        With easy budget=6, the expert can complete all SOP checkpoints within budget.
        """
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)

        assert trace["done"] is True
        assert trace["final_score"] >= 0.90

    # ---- Test 2: Intentionally bad trajectory ≤ 0.20 ----

    def test_intentionally_bad_trajectory_le_020(self):
        """§23.2: Intentionally bad trajectory → score ≤ 0.20.

        Plays only SKIP_TURN actions until the budget is exhausted, then
        FINISH_EPISODE. No classification, no routing, no useful work.
        Computes the terminal episode score via the scorer and asserts it
        is very low.
        """
        env = TriageSieveEnvironment()
        obs = env.reset(seed=42, mode="eval_strict", difficulty=TaskDifficulty.EASY.value)

        # Burn budget with skip_turn
        while not obs.done and obs.action_budget_remaining > 1:
            action = TriageSieveAction(
                action_type=ActionType.SKIP_TURN,
                metadata={},
            )
            obs = env.step(action)

        # Finish episode with remaining budget
        if not obs.done:
            finish = TriageSieveAction(
                action_type=ActionType.FINISH_EPISODE,
                metadata={},
            )
            obs = env.step(finish)

        assert obs.done is True

        # Build scoring context from environment state and compute final score.
        # A skip-only trajectory leaves all tickets untouched (NEW status, no
        # classifications, no routing, no info requests, SOP trackers at entry).
        ctx = EpisodeScoringContext(
            tickets=list(env._ticket_index.values()),
            ticket_states=dict(env._ticket_states),
            ticket_classifications=dict(env._ticket_classifications),
            ticket_impact_urgency=dict(env._ticket_impact_urgency),
            ticket_routed_to=dict(env._ticket_routed_to),
            ticket_escalated_to=dict(env._ticket_escalated_to),
            ticket_close_reasons=dict(env._ticket_close_reasons),
            ticket_info_requested=dict(env._ticket_info_requested),
            ticket_info_received=dict(env._ticket_info_received),
            ticket_merged_to=dict(env._ticket_merged_to),
            ticket_templates_used={},  # no close actions taken → no templates used
            sop_trackers=dict(env._sop_trackers),
            invalid_action_count=0,
            ticket_route_count={},
            ticket_first_substantive_step={},
        )
        result = compute_episode_score(ctx)
        assert result.final_score <= 0.20

    # ---- Test 3: UJCS drops when mandatory checkpoint skipped ----

    def test_ujcs_drops_when_checkpoint_skipped(self):
        """§23.2: UJCS drops when mandatory checkpoint is skipped.

        Uses _SIMPLE_SOP (3 checkpoints: classify, route, close).
        - UJCS_full: all 3 checkpoints visited, terminal reached → 1.0
        - UJCS_skip: skip 'classify' checkpoint → tracker gets stuck at 'open'
          because there is no edge open→route. All 3 checkpoints are missed
          and the terminal is never reached, so UJCS drops to 0.0.
        """
        graph = SOPGraph.from_archetype_data(_SIMPLE_SOP)
        assert len(graph.checkpoints) == 3  # classify, route, close

        # Path A: visit all checkpoints (perfect path)
        tracker_full = SOPTracker(graph)
        ctx = _all_true_context()
        for node_id in graph.gold_path[1:]:  # skip entry
            advanced = tracker_full.try_advance(node_id, ctx)
            assert advanced, f"Full path: failed to advance to {node_id!r}"
        data_full = tracker_full.get_scoring_data()
        ujcs_full = compute_ujcs(data_full, wrong_parameterizations=0)

        # Path B: skip the first checkpoint (classify_billing_refund).
        # The tracker is at 'open'. Skipping classify means trying to advance
        # to 'route_billing_team', but there is no edge open→route, so
        # try_advance returns False. The tracker stays stuck at 'open' —
        # all 3 checkpoints are missed and the terminal is never reached.
        tracker_skip = SOPTracker(graph)
        for node_id in graph.gold_path[1:]:
            if node_id == "classify_billing_refund":
                continue  # skip this mandatory checkpoint
            tracker_skip.try_advance(node_id, ctx)
        data_skip = tracker_skip.get_scoring_data()
        ujcs_skip = compute_ujcs(data_skip, wrong_parameterizations=0)

        # Full path should be perfect
        assert ujcs_full == pytest.approx(1.0)
        # Skipping a checkpoint must strictly reduce UJCS
        assert ujcs_skip < ujcs_full
        # Tracker stuck at 'open': 3 checkpoints missed + terminal penalty.
        # raw = 0 - 3 - 3 = -6; min = -6, max = 3; normalized = 0/9 = 0.0
        # (arithmetic tied to compute_ujcs normalization in policy_graph.py)
        assert ujcs_skip == pytest.approx(0.0)

    # ---- Test 4: Priority weighting changes score on critical mishandle ----

    def test_priority_weighting_critical_mishandling_changes_score(self):
        """§23.2: Priority weighting changes final score when critical ticket mishandled.

        Two-ticket episode:
        - T_crit: critical priority (weight=2.0)
        - T_low:  low priority (weight=0.5)

        Scenario A: T_crit perfectly handled, T_low fully botched.
        Scenario B: T_crit fully botched, T_low perfectly handled.

        Score_A must be strictly greater than Score_B because the critical
        ticket's weight (2.0) magnifies the perfect/botched difference.
        """
        ht_crit = _make_hidden_truth(
            ticket_id="T_crit",
            priority=Priority.CRITICAL,
            impact=Impact.REVENUE_AFFECTING,
            urgency=Urgency.CRITICAL,
            required_queue=QueueId.BILLING_TEAM,
            correct_template_ids=["tmpl_1"],
        )
        ht_low = _make_hidden_truth(
            ticket_id="T_low",
            priority=Priority.LOW,
            impact=Impact.SINGLE_USER,
            urgency=Urgency.LOW,
            required_queue=QueueId.BILLING_TEAM,
            correct_template_ids=["tmpl_2"],
        )
        t_crit = _make_rendered_ticket("T_crit", hidden_truth=ht_crit)
        t_low = _make_rendered_ticket("T_low", hidden_truth=ht_low)

        # --- Scenario A: critical perfect, low botched ---
        ctx_a = _make_context(
            [t_crit, t_low],
            ticket_states={
                "T_crit": TicketStatus.CLOSED,
                "T_low": TicketStatus.NEW,  # never touched
            },
            ticket_classifications={
                "T_crit": (IssueFamily.BILLING, IssueSubtype.REFUND),
                # T_low: no classification
            },
            ticket_impact_urgency={
                "T_crit": (Impact.REVENUE_AFFECTING, Urgency.CRITICAL),
            },
            ticket_routed_to={
                "T_crit": QueueId.BILLING_TEAM,
            },
            ticket_close_reasons={
                "T_crit": CloseReason.RESOLVED,
            },
            ticket_templates_used={
                "T_crit": ["tmpl_1"],
            },
            sop_trackers={
                "T_crit": _make_tracker_at_terminal(_SIMPLE_SOP),
                "T_low": _make_tracker_at_entry(_SIMPLE_SOP),
            },
            ticket_first_substantive_step={"T_crit": 1},
        )
        score_a = compute_episode_score(ctx_a)

        # --- Scenario B: critical botched, low perfect ---
        ctx_b = _make_context(
            [t_crit, t_low],
            ticket_states={
                "T_crit": TicketStatus.NEW,  # never touched
                "T_low": TicketStatus.CLOSED,
            },
            ticket_classifications={
                "T_low": (IssueFamily.BILLING, IssueSubtype.REFUND),
            },
            ticket_impact_urgency={
                "T_low": (Impact.SINGLE_USER, Urgency.LOW),
            },
            ticket_routed_to={
                "T_low": QueueId.BILLING_TEAM,
            },
            ticket_close_reasons={
                "T_low": CloseReason.RESOLVED,
            },
            ticket_templates_used={
                "T_low": ["tmpl_2"],
            },
            sop_trackers={
                "T_crit": _make_tracker_at_entry(_SIMPLE_SOP),
                "T_low": _make_tracker_at_terminal(_SIMPLE_SOP),
            },
            ticket_first_substantive_step={"T_low": 1},
        )
        score_b = compute_episode_score(ctx_b)

        # Critical ticket has weight 2.0, low has 0.5.
        # Handling the critical ticket well should yield a much higher score.
        assert score_a.final_score > score_b.final_score
        # The gap should be significant (not just rounding)
        assert score_a.final_score - score_b.final_score > 0.10
        # Verify the weighting mechanism is the cause
        assert score_a.terminal_business_score > score_b.terminal_business_score
