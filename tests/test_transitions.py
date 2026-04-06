"""State machine transition tests (§23.1) and guided-mode hint integration tests (§16).

Covers:
- route_before_classify → invalid action error
- merge_non_duplicate → invalid action error
- correct_info_request → follow-up generated, state updated
- escalate_without_prerequisites → pushback
- close_with_missing_fields → invalid action error
- close_non_actionable_without_classification → allowed
- Guided-mode hints appear in train_guided, absent in eval_strict
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
    TriageSieveObservation,
    TicketStatus,
    Urgency,
)
from ..server.triagesieve_env_environment import TriageSieveEnvironment

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> TriageSieveEnvironment:
    return TriageSieveEnvironment()


def _action(action_type: ActionType, **kwargs: object) -> TriageSieveAction:
    return TriageSieveAction(action_type=action_type, metadata={}, **kwargs)


def _step_ok(env: TriageSieveEnvironment, action: TriageSieveAction) -> TriageSieveObservation:
    """Execute a setup step and assert it succeeded."""
    obs = env.step(action)
    assert obs.last_action_result == "ok", f"Setup step failed: {obs.last_action_result}"
    return obs


# ---------------------------------------------------------------------------
# §23.1 Transition Tests
# ---------------------------------------------------------------------------

# Seed contracts:
#   seed=42,  difficulty="easy" → refund_missing_order_id archetype (billing/refund, needs order_id)
#   seed=100, difficulty="easy" → benign_expected archetype (non-actionable)


class TestRouteBeforeClassify:
    def test_route_new_ticket_fails(self, env: TriageSieveEnvironment) -> None:
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        obs = env.step(
            _action(ActionType.ROUTE_TICKET, ticket_id=tid, queue_id=QueueId.BILLING_TEAM)
        )
        result = obs.last_action_result.lower()
        assert "illegal" in result or "not allowed" in result


class TestMergeNonDuplicate:
    def test_merge_non_duplicate_fails(self, env: TriageSieveEnvironment) -> None:
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        obs = env.step(_action(ActionType.MERGE_DUPLICATE, ticket_id=tid, target_ticket_id="T999"))
        result = obs.last_action_result.lower()
        assert "not a duplicate" in result or "merge failed" in result


class TestCorrectInfoRequest:
    def test_follow_up_generated(self, env: TriageSieveEnvironment) -> None:
        """Correct info request → follow-up in thread history, +0.03 reward."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        _step_ok(
            env,
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
            ),
        )
        obs = env.step(
            _action(
                ActionType.REQUEST_INFORMATION,
                ticket_id=tid,
                requested_fields=["order_id"],
            )
        )
        assert obs.reward == pytest.approx(0.03)
        assert obs.last_action_result == "ok"
        summary = next(s for s in obs.inbox_summaries if s.ticket_id == tid)
        assert summary.status == TicketStatus.WAITING_FOR_INFO


class TestEscalateWithoutPrerequisites:
    def test_escalate_to_gated_queue_pushback(self, env: TriageSieveEnvironment) -> None:
        """Classify as TECHNICAL so queue family matches; omit impact/urgency → pushback."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        _step_ok(
            env,
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.TECHNICAL,
                issue_subtype=IssueSubtype.API_ERROR,
            ),
        )
        obs = env.step(
            _action(
                ActionType.ESCALATE_TICKET,
                ticket_id=tid,
                queue_id=QueueId.TECH_SUPPORT_L2,
            )
        )
        assert obs.last_action_result.startswith("Pushback:")
        assert obs.reward == pytest.approx(-0.03)


class TestCloseWithMissingFields:
    def test_close_resolved_with_unfulfilled_fields(self, env: TriageSieveEnvironment) -> None:
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        _step_ok(
            env,
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
            ),
        )
        obs = env.step(
            _action(
                ActionType.CLOSE_TICKET,
                ticket_id=tid,
                close_reason=CloseReason.RESOLVED,
            )
        )
        result = obs.last_action_result.lower()
        assert "close failed" in result or "missing" in result


class TestCloseNonActionable:
    def test_close_non_actionable_from_opened(self, env: TriageSieveEnvironment) -> None:
        """Non-actionable close from opened status is allowed.

        seed=100 produces benign_expected archetype (non_actionable_subtype=BENIGN_EXPECTED).
        """
        obs = env.reset(seed=100, mode="eval_strict", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        obs = env.step(
            _action(
                ActionType.CLOSE_TICKET,
                ticket_id=tid,
                close_reason=CloseReason.NON_ACTIONABLE,
            )
        )
        assert obs.last_action_result == "ok"


# ---------------------------------------------------------------------------
# §16 Guided-Mode Hint Integration Tests
# ---------------------------------------------------------------------------


class TestGuidedModeHints:
    """Integration tests: hints flow through step() → observation.hint."""

    def test_no_hints_in_eval_strict(self, env: TriageSieveEnvironment) -> None:
        """eval_strict mode never produces hints."""
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        assert obs.hint is None
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        obs = env.step(
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.TECHNICAL,
                issue_subtype=IssueSubtype.BUG_REPORT,
            )
        )
        assert obs.hint is None

    def test_hints_in_train_guided_wrong_family(self, env: TriageSieveEnvironment) -> None:
        """train_guided mode produces hint for wrong classification family."""
        obs = env.reset(seed=42, mode="train_guided", difficulty="easy")
        assert obs.hint is None
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        obs = env.step(
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.TECHNICAL,
                issue_subtype=IssueSubtype.BUG_REPORT,
            )
        )
        assert obs.hint is not None
        assert len(obs.hint) > 0

    def test_hints_in_train_guided_pushback(self, env: TriageSieveEnvironment) -> None:
        """train_guided mode produces hint on queue pushback."""
        obs = env.reset(seed=42, mode="train_guided", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        _step_ok(
            env,
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.TECHNICAL,
                issue_subtype=IssueSubtype.API_ERROR,
            ),
        )
        obs = env.step(
            _action(
                ActionType.ESCALATE_TICKET,
                ticket_id=tid,
                queue_id=QueueId.TECH_SUPPORT_L2,
            )
        )
        assert obs.hint is not None
        assert len(obs.hint) > 0

    def test_no_hint_on_correct_action(self, env: TriageSieveEnvironment) -> None:
        """No hint when action is correct in train_guided mode."""
        obs = env.reset(seed=42, mode="train_guided", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        obs = _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        assert obs.hint is None

    def test_hint_for_route_without_info(self, env: TriageSieveEnvironment) -> None:
        """Hint when routing without requesting required missing fields."""
        obs = env.reset(seed=42, mode="train_guided", difficulty="easy")
        tid = obs.inbox_summaries[0].ticket_id
        _step_ok(env, _action(ActionType.OPEN_TICKET, ticket_id=tid))
        _step_ok(
            env,
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family=IssueFamily.BILLING,
                issue_subtype=IssueSubtype.REFUND,
            ),
        )
        _step_ok(
            env,
            _action(
                ActionType.SET_IMPACT_URGENCY,
                ticket_id=tid,
                impact=Impact.SINGLE_USER,
                urgency=Urgency.MEDIUM,
            ),
        )
        obs = env.step(
            _action(
                ActionType.ROUTE_TICKET,
                ticket_id=tid,
                queue_id=QueueId.REFUND_TEAM,
            )
        )
        assert obs.hint is not None
        assert len(obs.hint) > 0
