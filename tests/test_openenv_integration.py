"""OpenEnv integration contract tests (§23.4).

Covers:
- reset() returns valid TriageSieveObservation
- step() returns valid TriageSieveObservation with done and reward
- state returns valid TriageSieveState
- Observation fields validate against Pydantic schemas
"""

from __future__ import annotations

import pytest

from ..models import (
    ActionType,
    InboxSummaryItem,
    QueueId,
    RoutingPolicyCard,
    SlaPolicyCard,
    TriageSieveAction,
    TriageSieveObservation,
    TriageSieveState,
    TaskDifficulty,
)
from ..server.triagesieve_env_environment import TriageSieveEnvironment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action(action_type: ActionType, **kwargs: object) -> TriageSieveAction:
    return TriageSieveAction(action_type=action_type, metadata={}, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> TriageSieveEnvironment:
    return TriageSieveEnvironment()


@pytest.fixture
def reset_obs(env: TriageSieveEnvironment) -> TriageSieveObservation:
    """Easy episode, seed=42 — known to produce refund_missing_order_id."""
    return env.reset(seed=42, difficulty="easy")


# ---------------------------------------------------------------------------
# 1–3: reset() contract
# ---------------------------------------------------------------------------


class TestResetContract:
    """Verify reset() returns a valid TriageSieveObservation per §5.1 / §9."""

    def test_reset_returns_observation_type(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs, TriageSieveObservation)

    def test_reset_observation_done_false(self, reset_obs: TriageSieveObservation) -> None:
        assert reset_obs.done is False

    def test_reset_observation_reward_none(self, reset_obs: TriageSieveObservation) -> None:
        assert reset_obs.reward is None


# ---------------------------------------------------------------------------
# 4: reset() observation required fields
# ---------------------------------------------------------------------------


class TestResetObservationFields:
    """All §9 fields present with correct types after reset()."""

    def test_inbox_summaries_non_empty(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs.inbox_summaries, list)
        assert len(reset_obs.inbox_summaries) >= 1
        for item in reset_obs.inbox_summaries:
            assert isinstance(item, InboxSummaryItem)

    def test_focused_ticket_none_after_reset(self, reset_obs: TriageSieveObservation) -> None:
        assert reset_obs.focused_ticket is None

    def test_allowed_queues_are_queue_ids(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs.allowed_queues, list)
        for q in reset_obs.allowed_queues:
            assert isinstance(q, QueueId)

    def test_routing_policy_cards(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs.routing_policy_cards, list)
        for card in reset_obs.routing_policy_cards:
            assert isinstance(card, RoutingPolicyCard)

    def test_sla_policy_cards(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs.sla_policy_cards, list)
        for card in reset_obs.sla_policy_cards:
            assert isinstance(card, SlaPolicyCard)

    def test_legal_actions_are_action_types(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs.legal_actions, list)
        for a in reset_obs.legal_actions:
            assert isinstance(a, ActionType)

    def test_scalar_fields(self, reset_obs: TriageSieveObservation) -> None:
        assert isinstance(reset_obs.action_budget_remaining, int)
        assert reset_obs.action_budget_remaining > 0
        assert isinstance(reset_obs.step_count, int)
        assert reset_obs.step_count == 0
        assert isinstance(reset_obs.current_time, str)
        assert isinstance(reset_obs.last_action_result, str)
        assert isinstance(reset_obs.task_difficulty, TaskDifficulty)
        assert reset_obs.task_difficulty == TaskDifficulty.EASY

    def test_hint_none_in_strict_mode(self, reset_obs: TriageSieveObservation) -> None:
        assert reset_obs.hint is None


# ---------------------------------------------------------------------------
# 5: Pydantic round-trip for observation
# ---------------------------------------------------------------------------


class TestObservationPydanticRoundtrip:
    """model_dump → model_validate round-trip preserves all fields."""

    def test_reset_observation_roundtrip(self, reset_obs: TriageSieveObservation) -> None:
        dumped = reset_obs.model_dump()
        restored = TriageSieveObservation.model_validate(dumped)
        assert restored.model_dump() == dumped


# ---------------------------------------------------------------------------
# 6–8: step() contract
# ---------------------------------------------------------------------------


class TestStepContract:
    """Verify step() returns a valid TriageSieveObservation per §5.1."""

    def test_step_returns_observation_type(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        ticket_id = reset_obs.inbox_summaries[0].ticket_id
        action = _action(ActionType.OPEN_TICKET, ticket_id=ticket_id)
        result = env.step(action)
        assert isinstance(result, TriageSieveObservation)

    def test_step_observation_has_reward(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        ticket_id = reset_obs.inbox_summaries[0].ticket_id
        action = _action(ActionType.OPEN_TICKET, ticket_id=ticket_id)
        result = env.step(action)
        assert isinstance(result.reward, float)

    def test_step_count_increments(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        ticket_id = reset_obs.inbox_summaries[0].ticket_id
        action = _action(ActionType.OPEN_TICKET, ticket_id=ticket_id)
        result = env.step(action)
        assert result.step_count == 1

    def test_step_observation_pydantic_roundtrip(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        ticket_id = reset_obs.inbox_summaries[0].ticket_id
        action = _action(ActionType.OPEN_TICKET, ticket_id=ticket_id)
        result = env.step(action)
        dumped = result.model_dump()
        restored = TriageSieveObservation.model_validate(dumped)
        assert restored.model_dump() == dumped


# ---------------------------------------------------------------------------
# 9: finish_episode → done=True
# ---------------------------------------------------------------------------


class TestFinishEpisode:
    """FINISH_EPISODE action sets done=True and reward is not None."""

    def test_finish_episode_done_true(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        action = _action(ActionType.FINISH_EPISODE)
        result = env.step(action)
        assert result.done is True
        assert result.reward is not None
        assert isinstance(result.reward, float)


# ---------------------------------------------------------------------------
# 10–11: state property contract
# ---------------------------------------------------------------------------


class TestStateContract:
    """Verify state property returns a valid TriageSieveState per §5.1 / §10."""

    def test_state_returns_state_type(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        state = env.state
        assert isinstance(state, TriageSieveState)

    def test_state_required_fields(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        state = env.state
        assert state.episode_id is not None
        assert isinstance(state.episode_id, str)
        assert isinstance(state.step_count, int)
        assert state.step_count == 0
        assert isinstance(state.task_difficulty, TaskDifficulty)
        assert isinstance(state.seed, int)
        assert isinstance(state.total_tickets, int)
        assert state.total_tickets >= 1
        assert isinstance(state.action_budget, int)
        assert isinstance(state.action_budget_remaining, int)
        assert state.mode in ("eval_strict", "train_guided")
        assert isinstance(state.tickets_summary, list)

    def test_state_pydantic_roundtrip(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        state = env.state
        dumped = state.model_dump()
        restored = TriageSieveState.model_validate(dumped)
        assert restored.model_dump() == dumped

    def test_state_step_count_tracks_steps(
        self, env: TriageSieveEnvironment, reset_obs: TriageSieveObservation
    ) -> None:
        assert env.state.step_count == 0
        ticket_id = reset_obs.inbox_summaries[0].ticket_id
        env.step(_action(ActionType.OPEN_TICKET, ticket_id=ticket_id))
        assert env.state.step_count == 1
