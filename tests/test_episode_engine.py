"""Tests for server/episode_engine.py — deterministic episode rendering.

Covers:
- Data loading (archetypes, templates, routing_rules, sla_rules)
- Single ticket rendering (render_ticket)
- Episode rendering (render_episode) with task ladder
- Deterministic follow-up message generation
- Priority derivation from impact × urgency
- Seed determinism (same inputs → same outputs)
- Accessor helpers
"""

from __future__ import annotations

import pytest

from ..models import (
    CustomerTier,
    HiddenTicketTruth,
    Impact,
    IssueFamily,
    IssueSubtype,
    Priority,
    QueueId,
    SourceChannel,
    TaskDifficulty,
    TicketStatus,
    Urgency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create an EpisodeEngine with default data directory."""
    from ..server.episode_engine import EpisodeEngine

    return EpisodeEngine()


# ---------------------------------------------------------------------------
# §1 Data Loading
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify static data loads correctly from JSON files."""

    def test_loads_all_18_archetypes(self, engine):
        assert len(engine.archetypes) == 18

    def test_archetype_ids_are_unique(self, engine):
        ids = [a["archetype_id"] for a in engine.archetypes]
        assert len(ids) == len(set(ids))

    def test_archetype_lookup_by_id(self, engine):
        arch = engine.get_archetype("refund_missing_order_id")
        assert arch is not None
        assert arch["difficulty"] == "easy"

    def test_unknown_archetype_returns_none(self, engine):
        assert engine.get_archetype("nonexistent") is None

    def test_loads_templates(self, engine):
        assert len(engine.templates) > 0

    def test_template_lookup_by_id(self, engine):
        tmpl = engine.get_template("req_order_id")
        assert tmpl is not None
        assert tmpl["template_id"] == "req_order_id"

    def test_loads_routing_rules(self, engine):
        assert len(engine.routing_rules) == 9  # 9 queues

    def test_routing_rule_lookup(self, engine):
        rule = engine.get_routing_rule("billing_team")
        assert rule is not None
        assert "billing" in rule["handles_families"]

    def test_loads_sla_rules(self, engine):
        assert len(engine.sla_rules) == 4  # 4 tiers

    def test_sla_lookup_by_tier(self, engine):
        sla = engine.get_sla_for_tier(CustomerTier.ENTERPRISE)
        assert sla is not None
        assert sla["response_deadline_minutes"] == 60
        assert sla["resolution_deadline_minutes"] == 480


# ---------------------------------------------------------------------------
# §2 Single Ticket Rendering
# ---------------------------------------------------------------------------


class TestRenderTicket:
    """Verify render_ticket produces correct visible payload + hidden truth."""

    def test_returns_rendered_ticket(self, engine):
        from ..server.episode_engine import RenderedTicket

        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert isinstance(ticket, RenderedTicket)

    def test_ticket_id_format(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert ticket.ticket_id.startswith("T")
        assert "42" in ticket.ticket_id

    def test_visible_fields_populated(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert len(ticket.subject) > 0
        assert len(ticket.body) > 0
        assert len(ticket.sender_email) > 0
        assert len(ticket.received_at) > 0

    def test_hidden_truth_is_hidden_ticket_truth(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert isinstance(ticket.hidden_truth, HiddenTicketTruth)

    def test_hidden_truth_fields_match_archetype(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        ht = ticket.hidden_truth
        assert ht.issue_family == IssueFamily.BILLING
        assert ht.issue_subtype == IssueSubtype.REFUND
        assert ht.required_queue == QueueId.REFUND_TEAM
        assert ht.customer_tier == CustomerTier.PRO
        assert ht.source_channel == SourceChannel.CUSTOMER_EMAIL

    def test_priority_derived_from_impact_urgency(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        ht = ticket.hidden_truth
        # refund: impact=single_user, urgency=medium → priority=low
        assert ht.impact == Impact.SINGLE_USER
        assert ht.urgency == Urgency.MEDIUM
        assert ht.priority == Priority.LOW

    def test_sla_deadlines_from_tier(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        ht = ticket.hidden_truth
        # pro tier: response=480, resolution=2880
        assert ht.sla_response_deadline == 480
        assert ht.sla_resolution_deadline == 2880

    def test_required_missing_fields(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert ticket.hidden_truth.required_missing_fields == ["order_id"]

    def test_sop_graph_present(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert "graph_id" in ticket.sop_graph
        assert ticket.sop_graph["graph_id"] == "refund_missing_order_id"

    def test_thread_history_rendered(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert len(ticket.thread_history) >= 1
        first = ticket.thread_history[0]
        assert "role" in first
        assert "content" in first
        assert "timestamp" in first

    def test_attachments_rendered(self, engine):
        # failed_invoice has attachments with placeholders
        ticket = engine.render_ticket("failed_invoice_charge_dispute", seed=42)
        assert len(ticket.attachments) >= 1
        # Should not contain raw {placeholder}
        for att in ticket.attachments:
            assert "{" not in att

    def test_internal_notes_rendered(self, engine):
        ticket = engine.render_ticket("failed_invoice_charge_dispute", seed=42)
        assert len(ticket.internal_notes) >= 1
        for note in ticket.internal_notes:
            assert "{" not in note

    def test_enterprise_sla_deadlines(self, engine):
        # integration_outage is enterprise tier
        ticket = engine.render_ticket("integration_outage", seed=42)
        ht = ticket.hidden_truth
        assert ht.customer_tier == CustomerTier.ENTERPRISE
        assert ht.sla_response_deadline == 60
        assert ht.sla_resolution_deadline == 480

    def test_critical_priority_derivation(self, engine):
        # integration_outage: impact=org_wide, urgency=critical → priority=critical
        ticket = engine.render_ticket("integration_outage", seed=42)
        assert ticket.hidden_truth.priority == Priority.CRITICAL

    def test_escalation_fields(self, engine):
        ticket = engine.render_ticket("integration_outage", seed=42)
        ht = ticket.hidden_truth
        assert ht.escalation_required is True
        assert ht.escalation_target == QueueId.TECH_SUPPORT_L2

    def test_non_actionable_subtype(self, engine):
        ticket = engine.render_ticket("spam_marketing_email", seed=42)
        ht = ticket.hidden_truth
        assert ht.non_actionable_subtype is not None
        assert ht.non_actionable_subtype.value == "spam_marketing"

    def test_duplicate_ticket_hidden_truth(self, engine):
        ticket = engine.render_ticket("duplicate_complaint", seed=42)
        ht = ticket.hidden_truth
        assert ht.is_duplicate is True
        assert ht.duplicate_of is not None
        # duplicate_of should be resolved (not a raw placeholder)
        assert "{" not in ht.duplicate_of

    def test_gold_terminal_status(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert ticket.hidden_truth.gold_terminal_status == TicketStatus.CLOSED

        ticket2 = engine.render_ticket("integration_outage", seed=42)
        assert ticket2.hidden_truth.gold_terminal_status == TicketStatus.ESCALATED

    def test_policy_graph_id_set(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert ticket.hidden_truth.policy_graph_id == "refund_missing_order_id"

    def test_correct_template_ids(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        assert "req_order_id" in ticket.hidden_truth.correct_template_ids
        assert "close_billing_resolved" in ticket.hidden_truth.correct_template_ids

    def test_unknown_archetype_raises(self, engine):
        with pytest.raises(ValueError, match="Unknown archetype"):
            engine.render_ticket("nonexistent_archetype", seed=42)


# ---------------------------------------------------------------------------
# §3 Seed Determinism
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    """Same seed + same archetype → identical rendered ticket."""

    def test_same_seed_same_ticket(self, engine):
        t1 = engine.render_ticket("refund_missing_order_id", seed=42)
        t2 = engine.render_ticket("refund_missing_order_id", seed=42)
        assert t1.ticket_id == t2.ticket_id
        assert t1.subject == t2.subject
        assert t1.body == t2.body
        assert t1.sender_email == t2.sender_email
        assert t1.hidden_truth.priority == t2.hidden_truth.priority

    def test_different_seed_different_ticket(self, engine):
        t1 = engine.render_ticket("refund_missing_order_id", seed=42)
        t2 = engine.render_ticket("refund_missing_order_id", seed=99)
        # At minimum, variation params differ (different index picks)
        assert t1.ticket_id != t2.ticket_id

    def test_episode_determinism(self, engine):
        ep1 = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        ep2 = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        assert ep1.episode_id == ep2.episode_id
        assert len(ep1.tickets) == len(ep2.tickets)
        for t1, t2 in zip(ep1.tickets, ep2.tickets):
            assert t1.ticket_id == t2.ticket_id
            assert t1.subject == t2.subject


# ---------------------------------------------------------------------------
# §4 Episode Rendering (Task Ladder)
# ---------------------------------------------------------------------------


class TestRenderEpisode:
    """Verify render_episode respects task ladder constraints."""

    def test_easy_episode_has_1_ticket(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        assert len(ep.tickets) == 1

    def test_easy_episode_budget_6(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        assert ep.action_budget == 6

    def test_medium_episode_has_2_or_3_tickets(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.MEDIUM)
        assert 2 <= len(ep.tickets) <= 3

    def test_medium_episode_budget_12(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.MEDIUM)
        assert ep.action_budget == 12

    def test_hard_episode_has_3_or_4_tickets(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.HARD)
        assert 3 <= len(ep.tickets) <= 4

    def test_hard_episode_budget_14(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.HARD)
        assert ep.action_budget == 14

    def test_episode_id_contains_seed(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        assert "42" in ep.episode_id

    def test_episode_difficulty_matches(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.HARD)
        assert ep.task_difficulty == TaskDifficulty.HARD

    def test_episode_seed_stored(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        assert ep.seed == 42

    def test_episode_base_time_is_iso8601(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.EASY)
        assert "T" in ep.base_time
        assert ep.base_time.endswith("Z") or "+" in ep.base_time

    def test_ticket_ids_unique_within_episode(self, engine):
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.HARD)
        ids = [t.ticket_id for t in ep.tickets]
        assert len(ids) == len(set(ids))

    def test_default_difficulty_selection(self, engine):
        """When difficulty is None, engine picks based on seed."""
        ep = engine.render_episode(seed=42)
        assert ep.task_difficulty in (TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD)

    def test_sub_seeds_are_deterministic(self, engine):
        """Sub-seeds follow seed * 1000 + i pattern — verified via ticket ID format."""
        ep = engine.render_episode(seed=42, difficulty=TaskDifficulty.MEDIUM)
        for i, ticket in enumerate(ep.tickets):
            expected_sub_seed = 42 * 1000 + i
            # Ticket ID format: T{sub_seed:04d}-{archetype[:8]}
            assert ticket.ticket_id.startswith(f"T{expected_sub_seed:04d}"), (
                f"Ticket {i} ID {ticket.ticket_id!r} does not start with T{expected_sub_seed:04d}"
            )


# ---------------------------------------------------------------------------
# §5 Follow-up Message Generation
# ---------------------------------------------------------------------------


class TestFollowUpGeneration:
    """Verify deterministic follow-up message generation."""

    def test_correct_fields_returns_message(self, engine):
        # refund_missing_order_id requires ["order_id"]
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        msg = engine.generate_follow_up_message(ticket, requested_fields=["order_id"])
        assert msg is not None
        assert len(msg) > 0

    def test_wrong_fields_returns_none(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        msg = engine.generate_follow_up_message(ticket, requested_fields=["tracking_number"])
        assert msg is None

    def test_superset_fields_returns_message(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        msg = engine.generate_follow_up_message(
            ticket, requested_fields=["order_id", "extra_field"]
        )
        assert msg is not None

    def test_no_required_fields_returns_none(self, engine):
        ticket = engine.render_ticket("failed_invoice_charge_dispute", seed=42)
        assert ticket.hidden_truth.required_missing_fields == [], (
            "Precondition: archetype must have no required fields"
        )
        msg = engine.generate_follow_up_message(ticket, requested_fields=["anything"])
        assert msg is None

    def test_follow_up_is_deterministic(self, engine):
        ticket = engine.render_ticket("refund_missing_order_id", seed=42)
        msg1 = engine.generate_follow_up_message(ticket, requested_fields=["order_id"])
        msg2 = engine.generate_follow_up_message(ticket, requested_fields=["order_id"])
        assert msg1 == msg2


# ---------------------------------------------------------------------------
# §6 Archetype Accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    """Verify convenience accessors."""

    def test_get_archetypes_by_difficulty_easy(self, engine):
        easy = engine.get_archetypes_by_difficulty(TaskDifficulty.EASY)
        assert all(a["difficulty"] == "easy" for a in easy)
        assert len(easy) >= 1

    def test_get_archetypes_by_difficulty_medium(self, engine):
        medium = engine.get_archetypes_by_difficulty(TaskDifficulty.MEDIUM)
        assert all(a["difficulty"] == "medium" for a in medium)

    def test_get_archetypes_by_difficulty_hard(self, engine):
        hard = engine.get_archetypes_by_difficulty(TaskDifficulty.HARD)
        assert all(a["difficulty"] == "hard" for a in hard)

    def test_all_difficulties_covered(self, engine):
        for diff in TaskDifficulty:
            archs = engine.get_archetypes_by_difficulty(diff)
            assert len(archs) >= 1, f"No archetypes for difficulty {diff}"
