"""Tests for triagesieve_env_environment.py Part 2 — action handler transition logic.

RED phase: These tests define the contract for:
- classify_ticket: opened/waiting_for_info → classified, family-subtype validation
- set_impact_urgency: classified → classified (stays), stores impact/urgency
- route_ticket: classified/waiting_for_info → routed, gated queue pushback
- escalate_ticket: classified/waiting_for_info/routed → escalated, gated queue pushback
- request_information: classified → waiting_for_info, deterministic follow-up
- merge_duplicate: opened/classified → merged, duplicate validation
- close_ticket: multiple sources → closed, close-reason constraints
- SOP tracker advancement for each action
- Invalid actions: never crash, consume budget, precise last_action_result
"""

from __future__ import annotations

import pytest

from ..models import (
    ActionType,
    CloseReason,
    Impact,
    IssueFamily,
    IssueSubtype,
    QueueId,
    TriageSieveAction,
    TaskDifficulty,
    TicketStatus,
    Urgency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Create a fresh TriageSieveEnvironment instance."""
    from ..server.triagesieve_env_environment import TriageSieveEnvironment

    return TriageSieveEnvironment()


@pytest.fixture
def easy_env(env):
    """Environment after reset with seed=42, easy difficulty (1 ticket, budget=6).

    Returns (env, obs, ticket_id).
    """
    obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
    ticket_id = obs.inbox_summaries[0].ticket_id
    return env, obs, ticket_id


@pytest.fixture
def medium_env(env):
    """Environment with seed=42 medium difficulty (budget=12).

    Uses the second ticket (entitlement_mismatch: billing/invoice_error, no missing fields)
    for tests needing 5+ steps.

    Returns (env, obs, ticket_id).
    """
    obs = env.reset(seed=42, mode="eval_strict", difficulty="medium")
    # Second ticket has no required_missing_fields → easier to close
    ticket_id = obs.inbox_summaries[1].ticket_id
    return env, obs, ticket_id


def _open_ticket(env, ticket_id: str):
    """Helper: open a ticket and return the observation."""
    return env.step(TriageSieveAction(
        action_type=ActionType.OPEN_TICKET,
        ticket_id=ticket_id,
        metadata={},
    ))


def _classify_ticket(env, ticket_id: str, family: IssueFamily, subtype: IssueSubtype):
    """Helper: classify a ticket."""
    return env.step(TriageSieveAction(
        action_type=ActionType.CLASSIFY_TICKET,
        ticket_id=ticket_id,
        issue_family=family,
        issue_subtype=subtype,
        metadata={},
    ))


def _set_impact_urgency(env, ticket_id: str, impact: Impact, urgency: Urgency):
    """Helper: set impact and urgency."""
    return env.step(TriageSieveAction(
        action_type=ActionType.SET_IMPACT_URGENCY,
        ticket_id=ticket_id,
        impact=impact,
        urgency=urgency,
        metadata={},
    ))


def _route_ticket(env, ticket_id: str, queue_id: QueueId):
    """Helper: route a ticket."""
    return env.step(TriageSieveAction(
        action_type=ActionType.ROUTE_TICKET,
        ticket_id=ticket_id,
        queue_id=queue_id,
        metadata={},
    ))


def _request_info(env, ticket_id: str, fields: list[str]):
    """Helper: request information."""
    return env.step(TriageSieveAction(
        action_type=ActionType.REQUEST_INFORMATION,
        ticket_id=ticket_id,
        requested_fields=fields,
        metadata={},
    ))


def _escalate_ticket(env, ticket_id: str, queue_id: QueueId):
    """Helper: escalate a ticket."""
    return env.step(TriageSieveAction(
        action_type=ActionType.ESCALATE_TICKET,
        ticket_id=ticket_id,
        queue_id=queue_id,
        metadata={},
    ))


def _close_ticket(env, ticket_id: str, reason: CloseReason):
    """Helper: close a ticket."""
    return env.step(TriageSieveAction(
        action_type=ActionType.CLOSE_TICKET,
        ticket_id=ticket_id,
        close_reason=reason,
        metadata={},
    ))


def _merge_ticket(env, ticket_id: str, target_ticket_id: str):
    """Helper: merge a duplicate ticket."""
    return env.step(TriageSieveAction(
        action_type=ActionType.MERGE_DUPLICATE,
        ticket_id=ticket_id,
        target_ticket_id=target_ticket_id,
        metadata={},
    ))


# ---------------------------------------------------------------------------
# §1 classify_ticket
# ---------------------------------------------------------------------------


class TestClassifyTicket:
    """Verify classify_ticket transitions and validation."""

    def test_classify_from_opened(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        # Ticket should now be classified
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLASSIFIED

    def test_classify_sets_last_action_result_ok(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        assert obs.last_action_result == "ok"

    def test_classify_correct_gives_positive_reward(self, easy_env):
        """Correct classification (matching hidden truth) → +0.02."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        # The easy seed=42 ticket is refund_missing_order_id: billing/refund
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        assert obs.reward == 0.02

    def test_classify_wrong_gives_base_reward(self, easy_env):
        """Wrong classification → +0.01 (valid action, not correct)."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.BUG_REPORT)
        assert obs.reward == 0.01

    def test_classify_logs_action(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        focused = obs.focused_ticket
        assert focused is not None
        assert any("classif" in a.lower() for a in focused.prior_actions_taken)

    def test_classify_from_waiting_for_info(self, env):
        """Re-classify after info received (waiting_for_info → classified)."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        # Request info to get to waiting_for_info
        _request_info(env, tid, ["order_id"])
        # Re-classify should work from waiting_for_info
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLASSIFIED

    def test_classify_from_new_is_illegal(self, easy_env):
        """Cannot classify a ticket that hasn't been opened yet."""
        env, obs, tid = easy_env
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        assert "Illegal action" in obs.last_action_result
        assert obs.reward == -0.02

    def test_classify_stores_classification(self, easy_env):
        """Internal state records the classification for later scoring."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        # Verify via internal state (white-box check)
        assert tid in env._ticket_classifications
        assert env._ticket_classifications[tid] == (IssueFamily.BILLING, IssueSubtype.REFUND)

    def test_classify_advances_sop_tracker(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        tracker = env._sop_trackers[tid]
        # Should have advanced past the open and classify nodes
        assert len(tracker.visited_nodes) > 2


# ---------------------------------------------------------------------------
# §2 set_impact_urgency
# ---------------------------------------------------------------------------


class TestSetImpactUrgency:
    """Verify set_impact_urgency behavior."""

    def test_set_from_classified(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _set_impact_urgency(env, tid, Impact.SINGLE_USER, Urgency.MEDIUM)
        assert obs.last_action_result == "ok"

    def test_status_stays_classified(self, easy_env):
        """set_impact_urgency does NOT transition status — ticket stays classified."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _set_impact_urgency(env, tid, Impact.SINGLE_USER, Urgency.MEDIUM)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLASSIFIED

    def test_stores_impact_urgency(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _set_impact_urgency(env, tid, Impact.SINGLE_USER, Urgency.MEDIUM)
        assert env._ticket_impact_urgency[tid] == (Impact.SINGLE_USER, Urgency.MEDIUM)

    def test_reward_is_base(self, easy_env):
        """set_impact_urgency always gives +0.01 (valid action reward)."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _set_impact_urgency(env, tid, Impact.SINGLE_USER, Urgency.MEDIUM)
        assert obs.reward == 0.01

    def test_set_from_opened_is_illegal(self, easy_env):
        """Cannot set impact/urgency before classification."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _set_impact_urgency(env, tid, Impact.SINGLE_USER, Urgency.MEDIUM)
        assert "Illegal action" in obs.last_action_result

    def test_logs_action(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _set_impact_urgency(env, tid, Impact.SINGLE_USER, Urgency.MEDIUM)
        assert any("impact" in a.lower() for a in obs.focused_ticket.prior_actions_taken)


# ---------------------------------------------------------------------------
# §3 route_ticket
# ---------------------------------------------------------------------------


class TestRouteTicket:
    """Verify route_ticket transitions and gated queue pushback."""

    def test_route_from_classified(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _route_ticket(env, tid, QueueId.REFUND_TEAM)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.ROUTED

    def test_route_result_ok(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _route_ticket(env, tid, QueueId.REFUND_TEAM)
        assert obs.last_action_result == "ok"

    def test_route_reward(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _route_ticket(env, tid, QueueId.REFUND_TEAM)
        assert obs.reward == 0.01

    def test_route_before_classify_is_illegal(self, easy_env):
        """§12: cannot route before classification."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _route_ticket(env, tid, QueueId.REFUND_TEAM)
        assert "Illegal action" in obs.last_action_result

    def test_route_stores_queue(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _route_ticket(env, tid, QueueId.REFUND_TEAM)
        assert env._ticket_routed_to[tid] == QueueId.REFUND_TEAM

    def test_gated_queue_pushback_without_prerequisites(self, env):
        """§15: routing to tech_support_l2 without impact/urgency → pushback."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        # No impact/urgency set → pushback
        obs = _route_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        assert "Pushback" in obs.last_action_result
        # Ticket stays classified (not routed)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLASSIFIED

    def test_gated_queue_pushback_penalty(self, env):
        """Pushback costs -0.03."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        obs = _route_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        assert obs.reward == -0.03

    def test_gated_queue_succeeds_with_prerequisites(self, env):
        """Routing to gated queue works when prerequisites are met."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        _set_impact_urgency(env, tid, Impact.ORG_WIDE, Urgency.CRITICAL)
        obs = _route_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        assert obs.last_action_result == "ok"
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.ROUTED

    def test_gated_security_team_pushback(self, env):
        """§15: routing to security_team without prerequisites → pushback."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.SECURITY, IssueSubtype.SUSPICIOUS_LOGIN)
        obs = _route_ticket(env, tid, QueueId.SECURITY_TEAM)
        assert "Pushback" in obs.last_action_result

    def test_route_from_waiting_for_info(self, env):
        """Can route from waiting_for_info status."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _request_info(env, tid, ["order_id"])
        obs = _route_ticket(env, tid, QueueId.REFUND_TEAM)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.ROUTED

    def test_route_logs_action(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _route_ticket(env, tid, QueueId.REFUND_TEAM)
        assert any("route" in a.lower() for a in obs.focused_ticket.prior_actions_taken)


# ---------------------------------------------------------------------------
# §4 escalate_ticket
# ---------------------------------------------------------------------------


class TestEscalateTicket:
    """Verify escalate_ticket transitions and gated queue pushback."""

    def test_escalate_from_classified(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        _set_impact_urgency(env, tid, Impact.ORG_WIDE, Urgency.CRITICAL)
        obs = _escalate_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.ESCALATED
        assert obs.last_action_result == "ok"

    def test_escalate_from_routed(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs = _escalate_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        # Gated queue but routed status means classification_set=True,
        # but impact_urgency might not be set → pushback
        # Actually gated queue requires both classification_set AND impact_urgency_set
        assert "Pushback" in obs.last_action_result

    def test_escalate_from_routed_with_prereqs(self, medium_env):
        """Escalate from routed with prerequisites met (needs budget > 4)."""
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        _set_impact_urgency(env, tid, Impact.ORG_WIDE, Urgency.CRITICAL)
        _route_ticket(env, tid, QueueId.TECH_SUPPORT_L1)
        obs = _escalate_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.ESCALATED

    def test_escalate_gated_pushback(self, env):
        """Escalate to gated queue without prerequisites → pushback."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        # No impact/urgency → pushback
        obs = _escalate_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        assert "Pushback" in obs.last_action_result
        assert obs.reward == -0.03

    def test_escalate_non_gated_queue(self, env):
        """Escalate to non-gated queue works without special prerequisites."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _escalate_ticket(env, tid, QueueId.BILLING_TEAM)
        assert obs.last_action_result == "ok"
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.ESCALATED

    def test_escalate_reward(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _escalate_ticket(env, tid, QueueId.BILLING_TEAM)
        assert obs.reward == 0.01

    def test_escalate_from_opened_is_illegal(self, easy_env):
        """Cannot escalate before classification."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _escalate_ticket(env, tid, QueueId.BILLING_TEAM)
        assert "Illegal action" in obs.last_action_result


# ---------------------------------------------------------------------------
# §5 request_information
# ---------------------------------------------------------------------------


class TestRequestInformation:
    """Verify request_information and deterministic follow-up generation."""

    def test_request_info_transitions_to_waiting(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["order_id"])
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.WAITING_FOR_INFO

    def test_correct_fields_reward(self, easy_env):
        """Correct info request (matching required_missing_fields) → +0.03."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["order_id"])
        assert obs.reward == 0.03

    def test_wrong_fields_base_reward(self, easy_env):
        """Wrong fields requested → +0.01 (valid action, not correct)."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["wrong_field"])
        assert obs.reward == 0.01

    def test_correct_fields_generates_follow_up(self, easy_env):
        """When correct fields requested, follow-up message is appended to thread."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["order_id"])
        # Thread history should have grown
        assert obs.focused_ticket is not None
        thread = obs.focused_ticket.thread_history
        # At least one message from customer follow-up
        customer_msgs = [m for m in thread if m.get("role") == "customer"]
        assert len(customer_msgs) >= 1

    def test_correct_fields_sets_info_received(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _request_info(env, tid, ["order_id"])
        assert env._ticket_info_received[tid] is True

    def test_wrong_fields_does_not_set_info_received(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _request_info(env, tid, ["wrong_field"])
        assert env._ticket_info_received[tid] is False

    def test_superset_fields_also_correct(self, easy_env):
        """Superset of required fields → also counts as correct."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["order_id", "extra_field"])
        assert obs.reward == 0.03
        assert env._ticket_info_received[tid] is True

    def test_request_info_stores_fields(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        _request_info(env, tid, ["order_id"])
        assert env._ticket_info_requested[tid] == ["order_id"]

    def test_request_info_from_opened_is_illegal(self, easy_env):
        """Cannot request info before classification."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _request_info(env, tid, ["order_id"])
        assert "Illegal action" in obs.last_action_result

    def test_request_info_result_ok(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["order_id"])
        assert obs.last_action_result == "ok"

    def test_request_info_logs_action(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _request_info(env, tid, ["order_id"])
        assert any("request" in a.lower() for a in obs.focused_ticket.prior_actions_taken)


# ---------------------------------------------------------------------------
# §6 merge_duplicate
# ---------------------------------------------------------------------------


class TestMergeDuplicate:
    """Verify merge_duplicate validation and transitions."""

    @pytest.fixture
    def dup_env(self, env):
        """Medium episode with seed=1 which contains a duplicate ticket.

        Returns (env, obs, dup_ticket_id, dup_target_id).
        """
        obs = env.reset(seed=1, mode="eval_strict", difficulty="medium")
        # seed=1 medium → first ticket is duplicate_complaint
        dup_tid = obs.inbox_summaries[0].ticket_id
        dup_target = env._ticket_index[dup_tid].hidden_truth.duplicate_of
        return env, obs, dup_tid, dup_target

    def test_merge_valid_duplicate_from_opened(self, dup_env):
        """Merge a ticket confirmed as duplicate with correct target."""
        env, obs, dup_tid, target_tid = dup_env
        _open_ticket(env, dup_tid)
        obs = _merge_ticket(env, dup_tid, target_tid)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == dup_tid)
        assert summary.status == TicketStatus.MERGED

    def test_merge_non_duplicate_fails(self, easy_env):
        """Merging a ticket that is NOT a duplicate → error."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _merge_ticket(env, tid, "T-fake-target")
        assert obs.last_action_result != "ok"
        assert obs.reward == -0.02

    def test_merge_wrong_target_fails(self, dup_env):
        """Merge with wrong target_ticket_id → error."""
        env, obs, dup_tid, _target = dup_env
        _open_ticket(env, dup_tid)
        obs = _merge_ticket(env, dup_tid, "T-wrong-target")
        assert obs.last_action_result != "ok"
        assert obs.reward == -0.02

    def test_merge_from_new_is_illegal(self, easy_env):
        """Cannot merge a ticket in NEW status."""
        env, obs, tid = easy_env
        obs = _merge_ticket(env, tid, "T-fake")
        assert "Illegal action" in obs.last_action_result

    def test_merge_is_terminal(self, dup_env):
        """Merged ticket cannot have further actions."""
        env, obs, dup_tid, target_tid = dup_env
        _open_ticket(env, dup_tid)
        _merge_ticket(env, dup_tid, target_tid)
        obs = _classify_ticket(env, dup_tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        assert "Illegal action" in obs.last_action_result

    def test_merge_stores_target(self, dup_env):
        env, obs, dup_tid, target_tid = dup_env
        _open_ticket(env, dup_tid)
        _merge_ticket(env, dup_tid, target_tid)
        assert env._ticket_merged_to[dup_tid] == target_tid


# ---------------------------------------------------------------------------
# §7 close_ticket
# ---------------------------------------------------------------------------


class TestCloseTicket:
    """Verify close_ticket constraints and transitions."""

    def test_close_from_routed(self, medium_env):
        """Standard close path: routed → closed (no missing fields)."""
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLOSED

    def test_close_result_ok(self, medium_env):
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        assert obs.last_action_result == "ok"

    def test_close_from_escalated(self, medium_env):
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        _set_impact_urgency(env, tid, Impact.ORG_WIDE, Urgency.CRITICAL)
        _escalate_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLOSED

    def test_close_non_actionable_from_opened(self, env):
        """§12: can close from opened ONLY if non-actionable."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        obs = _close_ticket(env, tid, CloseReason.NON_ACTIONABLE)
        assert obs.last_action_result == "ok"
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLOSED

    def test_close_resolved_from_opened_fails(self, easy_env):
        """§12: closing from opened with reason=resolved is not allowed (only non-actionable)."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        assert obs.last_action_result != "ok"
        assert obs.reward == -0.02

    def test_close_with_missing_fields_unfulfilled(self, easy_env):
        """§12 hard rule: cannot close while required_missing_fields unfulfilled."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        # Don't request info → missing fields still unfulfilled
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        assert obs.last_action_result != "ok"
        assert obs.reward == -0.02

    def test_close_non_actionable_exempt_from_missing_fields(self, easy_env):
        """Non-actionable close bypasses missing fields check."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _close_ticket(env, tid, CloseReason.NON_ACTIONABLE)
        assert obs.last_action_result == "ok"

    def test_close_duplicate_exempt_from_missing_fields(self, easy_env):
        """Duplicate close bypasses missing fields check."""
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.REFUND)
        obs = _close_ticket(env, tid, CloseReason.DUPLICATE)
        assert obs.last_action_result == "ok"

    def test_close_stores_reason(self, medium_env):
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        _close_ticket(env, tid, CloseReason.RESOLVED)
        assert env._ticket_close_reasons[tid] == CloseReason.RESOLVED

    def test_close_is_terminal(self, medium_env):
        """Closed ticket allows no further actions."""
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        _close_ticket(env, tid, CloseReason.RESOLVED)
        obs = _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        assert "Illegal action" in obs.last_action_result

    def test_close_from_new_is_illegal(self, easy_env):
        env, obs, tid = easy_env
        obs = _close_ticket(env, tid, CloseReason.NON_ACTIONABLE)
        assert "Illegal action" in obs.last_action_result

    def test_close_reward(self, medium_env):
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        assert obs.reward == 0.01

    def test_close_logs_action(self, medium_env):
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        assert any("close" in a.lower() for a in obs.focused_ticket.prior_actions_taken)


# ---------------------------------------------------------------------------
# §8 Budget and Invalid Action Behavior
# ---------------------------------------------------------------------------


class TestInvalidActionBehavior:
    """Verify that invalid actions never crash, always consume budget."""

    def test_invalid_action_does_not_crash(self, easy_env):
        """Invalid transition does not raise an exception."""
        env, obs, tid = easy_env
        # Try to route a NEW ticket (illegal)
        obs = _route_ticket(env, tid, QueueId.BILLING_TEAM)
        assert obs is not None
        assert obs.reward == -0.02

    def test_invalid_action_consumes_budget(self, easy_env):
        env, obs, tid = easy_env
        initial_budget = obs.action_budget_remaining
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs2 = env.step(TriageSieveAction(
            action_type=ActionType.SKIP_TURN, metadata={}
        ))
        # Budget should have decreased by 2 (one invalid + one skip)
        assert obs2.action_budget_remaining == initial_budget - 2

    def test_pushback_does_not_crash(self, env):
        """Pushback from gated queue does not crash."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        obs = _route_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        assert obs is not None
        assert "Pushback" in obs.last_action_result

    def test_merge_invalid_does_not_crash(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _merge_ticket(env, tid, "nonexistent")
        assert obs is not None
        assert obs.last_action_result != "ok"

    def test_close_invalid_does_not_crash(self, easy_env):
        env, obs, tid = easy_env
        _open_ticket(env, tid)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        assert obs is not None
        assert obs.last_action_result != "ok"


# ---------------------------------------------------------------------------
# §9 Full Workflow Integration
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """End-to-end workflow tests combining multiple actions."""

    def test_happy_path_no_missing_fields(self, medium_env):
        """Complete happy path: open → classify → route → close (no missing fields)."""
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.BILLING, IssueSubtype.INVOICE_ERROR)
        _route_ticket(env, tid, QueueId.BILLING_TEAM)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLOSED
        assert obs.last_action_result == "ok"

    def test_escalation_path(self, medium_env):
        """open → classify → set impact/urgency → escalate → close."""
        env, obs, tid = medium_env
        _open_ticket(env, tid)
        _classify_ticket(env, tid, IssueFamily.TECHNICAL, IssueSubtype.INTEGRATION_FAILURE)
        _set_impact_urgency(env, tid, Impact.ORG_WIDE, Urgency.CRITICAL)
        _escalate_ticket(env, tid, QueueId.TECH_SUPPORT_L2)
        obs = _close_ticket(env, tid, CloseReason.RESOLVED)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLOSED

    def test_non_actionable_shortcut(self, env):
        """open → close(non_actionable) — shortest valid path for spam."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _open_ticket(env, tid)
        obs = _close_ticket(env, tid, CloseReason.NON_ACTIONABLE)
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.CLOSED

    def test_budget_exhaustion_ends_episode(self, env):
        """Episode ends when action budget reaches 0."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        # Easy budget = 6
        for _ in range(6):
            obs = env.step(TriageSieveAction(
                action_type=ActionType.SKIP_TURN, metadata={}
            ))
        assert obs.done is True

    def test_actions_after_done_raises(self, env):
        """Cannot step after episode is done."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        for _ in range(6):
            obs = env.step(TriageSieveAction(
                action_type=ActionType.SKIP_TURN, metadata={}
            ))
        with pytest.raises(RuntimeError, match="done"):
            env.step(TriageSieveAction(
                action_type=ActionType.SKIP_TURN, metadata={}
            ))
