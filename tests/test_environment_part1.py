"""Tests for triagesieve_env_environment.py Part 1 — skeleton, reset, state, helpers.

RED phase: These tests define the contract for:
- Environment subclass construction and SUPPORTS_CONCURRENT_SESSIONS
- reset() returning valid TriageSieveObservation
- state property returning valid TriageSieveState
- Action format gate (_validate_action_format)
- Legal action computation (_compute_legal_actions)
- Observation assembly helpers
- Minimal step() stub behavior
"""

from __future__ import annotations

import pytest

from ..models import (
    ActionType,
    CustomerTier,
    Impact,
    IssueFamily,
    IssueSubtype,
    QueueId,
    TriageSieveAction,
    TriageSieveObservation,
    TriageSieveState,
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
def env_after_reset(env):
    """Environment after a deterministic reset with seed=42, easy difficulty."""
    obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
    return env, obs


@pytest.fixture
def env_medium(env):
    """Environment after reset with seed=100, medium difficulty."""
    obs = env.reset(seed=100, mode="eval_strict", difficulty="medium")
    return env, obs


# ---------------------------------------------------------------------------
# §1 Class Structure
# ---------------------------------------------------------------------------


class TestClassStructure:
    def test_is_environment_subclass(self):
        from openenv.core.env_server.interfaces import Environment

        from ..server.triagesieve_env_environment import TriageSieveEnvironment

        assert issubclass(TriageSieveEnvironment, Environment)

    def test_supports_concurrent_sessions(self):
        from ..server.triagesieve_env_environment import TriageSieveEnvironment

        assert TriageSieveEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True

    def test_construction_no_args(self, env):
        """Environment can be constructed with no arguments."""
        assert env is not None


# ---------------------------------------------------------------------------
# §2 reset() Contract
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_observation(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert isinstance(obs, TriageSieveObservation)

    def test_observation_not_done(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert obs.done is False

    def test_observation_reward_none_or_zero(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert obs.reward is None or obs.reward == 0.0

    def test_observation_step_count_zero(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert obs.step_count == 0

    def test_observation_last_action_result_ok(self, env):
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert obs.last_action_result == "ok"

    def test_observation_has_inbox_summaries(self, env_after_reset):
        _, obs = env_after_reset
        assert len(obs.inbox_summaries) >= 1

    def test_easy_has_one_ticket(self, env_after_reset):
        _, obs = env_after_reset
        assert len(obs.inbox_summaries) == 1

    def test_medium_has_multiple_tickets(self, env_medium):
        _, obs = env_medium
        assert 2 <= len(obs.inbox_summaries) <= 3

    def test_inbox_summary_fields(self, env_after_reset):
        _, obs = env_after_reset
        item = obs.inbox_summaries[0]
        assert item.ticket_id
        assert item.subject
        assert item.sender_email
        assert item.received_at
        assert item.status == TicketStatus.NEW
        assert isinstance(item.customer_tier, CustomerTier)
        assert isinstance(item.has_attachment, bool)
        assert item.short_preview

    def test_all_tickets_start_new(self, env_after_reset):
        _, obs = env_after_reset
        for item in obs.inbox_summaries:
            assert item.status == TicketStatus.NEW

    def test_focused_ticket_is_none(self, env_after_reset):
        _, obs = env_after_reset
        assert obs.focused_ticket is None

    def test_allowed_queues_populated(self, env_after_reset):
        _, obs = env_after_reset
        assert len(obs.allowed_queues) == 9  # 9 queues in taxonomy

    def test_routing_policy_cards_populated(self, env_after_reset):
        _, obs = env_after_reset
        assert len(obs.routing_policy_cards) == 9

    def test_sla_policy_cards_populated(self, env_after_reset):
        _, obs = env_after_reset
        assert len(obs.sla_policy_cards) == 4  # 4 customer tiers

    def test_available_templates_populated(self, env_after_reset):
        _, obs = env_after_reset
        assert len(obs.available_templates) > 0

    def test_action_budget_matches_difficulty(self, env_after_reset):
        _, obs = env_after_reset
        assert obs.action_budget_remaining == 6  # easy budget

    def test_legal_actions_initial(self, env_after_reset):
        _, obs = env_after_reset
        # Initially: can open tickets, skip, finish
        assert ActionType.OPEN_TICKET in obs.legal_actions
        assert ActionType.SKIP_TURN in obs.legal_actions
        assert ActionType.FINISH_EPISODE in obs.legal_actions
        # Cannot classify/route before opening
        assert ActionType.CLASSIFY_TICKET not in obs.legal_actions
        assert ActionType.ROUTE_TICKET not in obs.legal_actions

    def test_task_difficulty_set(self, env_after_reset):
        _, obs = env_after_reset
        assert obs.task_difficulty == TaskDifficulty.EASY

    def test_current_time_is_iso(self, env_after_reset):
        _, obs = env_after_reset
        assert "T" in obs.current_time  # basic ISO 8601 check

    def test_hint_none_in_strict_mode(self, env_after_reset):
        _, obs = env_after_reset
        assert obs.hint is None

    def test_deterministic_reset(self, env):
        """Same seed + difficulty → identical observation."""
        obs1 = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        obs2 = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert obs1.inbox_summaries[0].ticket_id == obs2.inbox_summaries[0].ticket_id
        assert obs1.inbox_summaries[0].subject == obs2.inbox_summaries[0].subject

    def test_reset_clears_previous_state(self, env):
        """Calling reset twice clears the previous episode state."""
        env.reset(seed=42, mode="eval_strict", difficulty="easy")
        obs2 = env.reset(seed=99, mode="eval_strict", difficulty="easy")
        assert obs2.step_count == 0
        assert obs2.last_action_result == "ok"

    def test_default_mode_is_eval_strict(self, env):
        """If mode not provided, defaults to eval_strict."""
        env.reset(seed=42, difficulty="easy")
        assert env.state.mode == "eval_strict"

    def test_train_guided_mode(self, env):
        env.reset(seed=42, mode="train_guided", difficulty="easy")
        assert env.state.mode == "train_guided"

    def test_invalid_mode_raises(self, env):
        with pytest.raises(ValueError, match="Invalid mode"):
            env.reset(seed=42, mode="bad_mode", difficulty="easy")


# ---------------------------------------------------------------------------
# §3 state Property
# ---------------------------------------------------------------------------


class TestState:
    def test_returns_state(self, env_after_reset):
        e, _ = env_after_reset
        state = e.state
        assert isinstance(state, TriageSieveState)

    def test_episode_id(self, env_after_reset):
        e, _ = env_after_reset
        state = e.state
        assert state.episode_id is not None
        assert isinstance(state.episode_id, str)

    def test_step_count_zero(self, env_after_reset):
        e, _ = env_after_reset
        assert e.state.step_count == 0

    def test_task_difficulty(self, env_after_reset):
        e, _ = env_after_reset
        assert e.state.task_difficulty == TaskDifficulty.EASY

    def test_seed(self, env_after_reset):
        e, _ = env_after_reset
        assert e.state.seed == 42

    def test_total_tickets(self, env_after_reset):
        e, _ = env_after_reset
        assert e.state.total_tickets == 1

    def test_action_budget(self, env_after_reset):
        e, _ = env_after_reset
        assert e.state.action_budget == 6
        assert e.state.action_budget_remaining == 6

    def test_mode(self, env_after_reset):
        e, _ = env_after_reset
        assert e.state.mode == "eval_strict"

    def test_tickets_summary(self, env_after_reset):
        e, _ = env_after_reset
        ts = e.state.tickets_summary
        assert len(ts) == 1
        assert "ticket_id" in ts[0]
        assert "status" in ts[0]
        # gold_priority intentionally removed — hidden truth must not leak into state
        assert "gold_priority" not in ts[0]

    def test_state_raises_before_reset(self, env):
        """Accessing state before reset raises RuntimeError."""
        with pytest.raises(RuntimeError):
            _ = env.state


# ---------------------------------------------------------------------------
# §4 Action Format Gate (_validate_action_format)
# ---------------------------------------------------------------------------


class TestFormatGate:
    """Tests for the format gate per §17.1.

    The format gate checks:
    1. Action parses as valid TriageSieveAction (guaranteed by Pydantic)
    2. All enum values are valid (guaranteed by Pydantic)
    3. Required arguments for action_type are present
    4. ticket_id exists in the inbox
    5. Action is legal in the ticket's current state
    """

    def test_open_ticket_valid(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=tid,
        )
        result = e._validate_action_format(action)
        assert result is None  # None means valid

    def test_open_ticket_missing_ticket_id(self, env_after_reset):
        e, _ = env_after_reset
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=None,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "ticket_id" in result.lower()

    def test_nonexistent_ticket_id(self, env_after_reset):
        e, _ = env_after_reset
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id="FAKE_ID",
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "not found" in result.lower() or "does not exist" in result.lower()

    def test_classify_missing_family(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id=tid,
            issue_subtype=IssueSubtype.REFUND,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "issue_family" in result.lower()

    def test_classify_missing_subtype(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id=tid,
            issue_family=IssueFamily.BILLING,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "issue_subtype" in result.lower()

    def test_classify_invalid_family_subtype_pair(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id=tid,
            issue_family=IssueFamily.BILLING,
            issue_subtype=IssueSubtype.BUG_REPORT,  # wrong family
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "subtype" in result.lower()

    def test_set_impact_urgency_missing_impact(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.SET_IMPACT_URGENCY,
            ticket_id=tid,
            urgency=Urgency.HIGH,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "impact" in result.lower()

    def test_set_impact_urgency_missing_urgency(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.SET_IMPACT_URGENCY,
            ticket_id=tid,
            impact=Impact.SINGLE_USER,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "urgency" in result.lower()

    def test_route_missing_queue_id(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.ROUTE_TICKET,
            ticket_id=tid,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "queue_id" in result.lower()

    def test_merge_missing_target(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.MERGE_DUPLICATE,
            ticket_id=tid,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "target_ticket_id" in result.lower()

    def test_close_missing_reason(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.CLOSE_TICKET,
            ticket_id=tid,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "close_reason" in result.lower()

    def test_skip_turn_no_ticket_needed(self, env_after_reset):
        e, _ = env_after_reset
        action = TriageSieveAction(action_type=ActionType.SKIP_TURN)
        result = e._validate_action_format(action)
        assert result is None

    def test_finish_episode_no_ticket_needed(self, env_after_reset):
        e, _ = env_after_reset
        action = TriageSieveAction(action_type=ActionType.FINISH_EPISODE)
        result = e._validate_action_format(action)
        assert result is None

    def test_escalate_missing_queue_id(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.ESCALATE_TICKET,
            ticket_id=tid,
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "queue_id" in result.lower()

    def test_request_info_missing_fields(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.REQUEST_INFORMATION,
            ticket_id=tid,
        )
        result = e._validate_action_format(action)
        assert result is not None

    def test_request_info_empty_fields_rejected(self, env_after_reset):
        """Empty requested_fields=[] should also fail the format gate."""
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.REQUEST_INFORMATION,
            ticket_id=tid,
            requested_fields=[],
        )
        result = e._validate_action_format(action)
        assert result is not None
        assert "requested_fields" in result.lower()


# ---------------------------------------------------------------------------
# §5 Legal Actions (_compute_legal_actions)
# ---------------------------------------------------------------------------


class TestLegalActions:
    """Tests for _compute_legal_actions per §12 transition rules."""

    def test_new_ticket_allows_open(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        legal = e._compute_legal_actions(tid)
        assert ActionType.OPEN_TICKET in legal

    def test_new_ticket_disallows_classify(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        legal = e._compute_legal_actions(tid)
        assert ActionType.CLASSIFY_TICKET not in legal

    def test_new_ticket_disallows_route(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        legal = e._compute_legal_actions(tid)
        assert ActionType.ROUTE_TICKET not in legal

    def test_opened_ticket_allows_classify(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        # Manually set ticket to opened status
        e._ticket_states[tid] = TicketStatus.OPENED
        legal = e._compute_legal_actions(tid)
        assert ActionType.CLASSIFY_TICKET in legal

    def test_opened_ticket_allows_merge(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.OPENED
        legal = e._compute_legal_actions(tid)
        assert ActionType.MERGE_DUPLICATE in legal

    def test_opened_ticket_allows_close(self, env_after_reset):
        """Can close from opened if non-actionable."""
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.OPENED
        legal = e._compute_legal_actions(tid)
        assert ActionType.CLOSE_TICKET in legal

    def test_classified_ticket_allows_route(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.CLASSIFIED
        legal = e._compute_legal_actions(tid)
        assert ActionType.ROUTE_TICKET in legal

    def test_classified_ticket_allows_set_impact_urgency(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.CLASSIFIED
        legal = e._compute_legal_actions(tid)
        assert ActionType.SET_IMPACT_URGENCY in legal

    def test_classified_ticket_allows_escalate(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.CLASSIFIED
        legal = e._compute_legal_actions(tid)
        assert ActionType.ESCALATE_TICKET in legal

    def test_classified_ticket_allows_request_info(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.CLASSIFIED
        legal = e._compute_legal_actions(tid)
        assert ActionType.REQUEST_INFORMATION in legal

    def test_routed_ticket_allows_escalate(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.ROUTED
        legal = e._compute_legal_actions(tid)
        assert ActionType.ESCALATE_TICKET in legal

    def test_routed_ticket_allows_close(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.ROUTED
        legal = e._compute_legal_actions(tid)
        assert ActionType.CLOSE_TICKET in legal

    def test_closed_ticket_no_actions(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.CLOSED
        legal = e._compute_legal_actions(tid)
        assert len(legal) == 0

    def test_merged_ticket_no_actions(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.MERGED
        legal = e._compute_legal_actions(tid)
        assert len(legal) == 0

    def test_waiting_for_info_allows_classify(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.WAITING_FOR_INFO
        legal = e._compute_legal_actions(tid)
        assert ActionType.CLASSIFY_TICKET in legal

    def test_waiting_for_info_allows_route(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.WAITING_FOR_INFO
        legal = e._compute_legal_actions(tid)
        assert ActionType.ROUTE_TICKET in legal

    def test_escalated_ticket_allows_close(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.ESCALATED
        legal = e._compute_legal_actions(tid)
        assert ActionType.CLOSE_TICKET in legal

    def test_escalated_ticket_disallows_route(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        e._ticket_states[tid] = TicketStatus.ESCALATED
        legal = e._compute_legal_actions(tid)
        assert ActionType.ROUTE_TICKET not in legal


# ---------------------------------------------------------------------------
# §6 Observation Assembly
# ---------------------------------------------------------------------------


class TestObservationAssembly:
    def test_routing_policy_cards_have_queue_ids(self, env_after_reset):
        _, obs = env_after_reset
        queue_ids = {card.queue_id for card in obs.routing_policy_cards}
        assert QueueId.BILLING_TEAM in queue_ids
        assert QueueId.SECURITY_TEAM in queue_ids
        assert QueueId.TECH_SUPPORT_L2 in queue_ids

    def test_sla_cards_have_all_tiers(self, env_after_reset):
        _, obs = env_after_reset
        tiers = {card.tier for card in obs.sla_policy_cards}
        assert tiers == {CustomerTier.FREE, CustomerTier.PRO, CustomerTier.ENTERPRISE, CustomerTier.INTERNAL}

    def test_templates_have_required_keys(self, env_after_reset):
        _, obs = env_after_reset
        for t in obs.available_templates:
            assert "template_id" in t
            assert "name" in t
            assert "description" in t
            assert "applies_to" in t

    def test_global_legal_actions(self, env_after_reset):
        """Global legal actions combine per-ticket legals + SKIP/FINISH."""
        _, obs = env_after_reset
        assert ActionType.SKIP_TURN in obs.legal_actions
        assert ActionType.FINISH_EPISODE in obs.legal_actions


# ---------------------------------------------------------------------------
# §7 Minimal step() Stub
# ---------------------------------------------------------------------------


class TestStepStub:
    """step() in Part 1 should at minimum handle the format gate and budget."""

    def test_step_returns_observation(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=tid,
        )
        result = e.step(action)
        assert isinstance(result, TriageSieveObservation)

    def test_step_increments_step_count(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=tid,
        )
        result = e.step(action)
        assert result.step_count == 1

    def test_step_decrements_budget(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=tid,
        )
        result = e.step(action)
        assert result.action_budget_remaining == 5  # 6 - 1

    def test_step_format_gate_rejects_bad_action(self, env_after_reset):
        e, obs = env_after_reset
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id="NONEXISTENT",
        )
        result = e.step(action)
        assert "not found" in result.last_action_result.lower() or "does not exist" in result.last_action_result.lower()
        assert result.reward is not None and result.reward < 0

    def test_step_budget_exhaustion_ends_episode(self, env_after_reset):
        e, obs = env_after_reset
        # Exhaust budget with skip_turns (6 steps for easy)
        for _ in range(6):
            action = TriageSieveAction(action_type=ActionType.SKIP_TURN)
            result = e.step(action)
        assert result.done is True
        assert result.action_budget_remaining == 0

    def test_finish_episode_sets_done(self, env_after_reset):
        e, _ = env_after_reset
        action = TriageSieveAction(action_type=ActionType.FINISH_EPISODE)
        result = e.step(action)
        assert result.done is True

    def test_open_ticket_sets_focused_ticket(self, env_after_reset):
        e, obs = env_after_reset
        tid = obs.inbox_summaries[0].ticket_id
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=tid,
        )
        result = e.step(action)
        assert result.focused_ticket is not None
        assert result.focused_ticket.ticket_id == tid

    def test_step_after_done_raises(self, env_after_reset):
        e, _ = env_after_reset
        action = TriageSieveAction(action_type=ActionType.FINISH_EPISODE)
        e.step(action)
        with pytest.raises(RuntimeError):
            e.step(TriageSieveAction(action_type=ActionType.SKIP_TURN))
