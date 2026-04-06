"""End-to-end tests for inference.py — the hackathon inference script.

Layers:
1. Unit tests for parse_action, serialize_observation, action_to_str, log functions
2. Integration tests with mocked LLM (no Docker, no network)
3. Real LLM smoke tests via HuggingFace router (skipped without HF_TOKEN)

Run:
    pytest tests/test_e2e_inference.py -v
    pytest tests/test_e2e_inference.py -v -m "not slow"   # skip real LLM tests
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import inference helpers directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference import (
    action_to_str,
    log_end,
    log_start,
    log_step,
    parse_action,
    run_task,
    serialize_observation,
)

from ..models import (
    ActionType,
    CloseReason,
    CustomerTier,
    FocusedTicket,
    Impact,
    InboxSummaryItem,
    IssueFamily,
    IssueSubtype,
    QueueId,
    RoutingPolicyCard,
    SlaPolicyCard,
    TriageSieveAction,
    TriageSieveObservation,
    TaskDifficulty,
    TicketStatus,
    Urgency,
)
from ..server.triagesieve_env_environment import TriageSieveEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observation(**overrides: Any) -> TriageSieveObservation:
    """Build a minimal valid TriageSieveObservation for testing."""
    defaults = dict(
        done=False,
        reward=0.01,
        metadata={},
        inbox_summaries=[
            InboxSummaryItem(
                ticket_id="T001",
                subject="Test ticket",
                sender_email="test@example.com",
                received_at="2026-04-05T08:00:00Z",
                status=TicketStatus.NEW,
                customer_tier=CustomerTier.PRO,
                has_attachment=False,
                sla_remaining_minutes=480,
                short_preview="Test preview text",
            )
        ],
        focused_ticket=None,
        available_templates=[],
        allowed_queues=[QueueId.BILLING_TEAM],
        routing_policy_cards=[],
        sla_policy_cards=[],
        legal_actions=[ActionType.OPEN_TICKET, ActionType.SKIP_TURN],
        action_budget_remaining=6,
        step_count=0,
        current_time="2026-04-05T08:00:00Z",
        last_action_result="ok",
        task_difficulty=TaskDifficulty.EASY,
        hint=None,
    )
    defaults.update(overrides)
    return TriageSieveObservation(**defaults)


@dataclass
class FakeStepResult:
    """Mimics openenv StepResult for the async adapter."""

    observation: TriageSieveObservation
    reward: float | None
    done: bool


class LocalAsyncEnvAdapter:
    """Async wrapper around TriageSieveEnvironment for testing run_task()."""

    def __init__(self) -> None:
        self._env = TriageSieveEnvironment()

    async def reset(self, **kwargs: Any) -> FakeStepResult:
        obs = self._env.reset(**kwargs)
        return FakeStepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def step(self, action: TriageSieveAction) -> FakeStepResult:
        obs = self._env.step(action)
        return FakeStepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def close(self) -> None:
        pass


def _mock_openai_client(responses: list[str]) -> MagicMock:
    """Create a mock OpenAI client that returns canned responses in order."""
    client = MagicMock()
    call_idx = [0]

    def _create(**kwargs: Any) -> MagicMock:
        idx = min(call_idx[0], len(responses) - 1)
        call_idx[0] += 1
        msg = MagicMock()
        msg.content = responses[idx]
        choice = MagicMock()
        choice.message = msg
        result = MagicMock()
        result.choices = [choice]
        return result

    client.chat.completions.create = _create
    return client


# ===========================================================================
# Layer 1: Unit Tests for inference.py helpers
# ===========================================================================


class TestParseAction:
    """Tests for inference.parse_action()."""

    def test_valid_open_ticket(self) -> None:
        action = parse_action('{"action_type": "open_ticket", "ticket_id": "T001"}')
        assert action is not None
        assert action.action_type == ActionType.OPEN_TICKET
        assert action.ticket_id == "T001"

    def test_valid_classify(self) -> None:
        action = parse_action(
            '{"action_type": "classify_ticket", "ticket_id": "T001", '
            '"issue_family": "billing", "issue_subtype": "refund"}'
        )
        assert action is not None
        assert action.issue_family == IssueFamily.BILLING
        assert action.issue_subtype == IssueSubtype.REFUND

    def test_valid_set_impact_urgency(self) -> None:
        action = parse_action(
            '{"action_type": "set_impact_urgency", "ticket_id": "T001", '
            '"impact": "single_user", "urgency": "medium"}'
        )
        assert action is not None
        assert action.impact == Impact.SINGLE_USER
        assert action.urgency == Urgency.MEDIUM

    def test_valid_route(self) -> None:
        action = parse_action(
            '{"action_type": "route_ticket", "ticket_id": "T001", "queue_id": "billing_team"}'
        )
        assert action is not None
        assert action.queue_id == QueueId.BILLING_TEAM

    def test_valid_close(self) -> None:
        action = parse_action(
            '{"action_type": "close_ticket", "ticket_id": "T001", "close_reason": "resolved"}'
        )
        assert action is not None
        assert action.close_reason == CloseReason.RESOLVED

    def test_valid_skip(self) -> None:
        action = parse_action('{"action_type": "skip_turn"}')
        assert action is not None
        assert action.action_type == ActionType.SKIP_TURN

    def test_valid_finish(self) -> None:
        action = parse_action('{"action_type": "finish_episode"}')
        assert action is not None
        assert action.action_type == ActionType.FINISH_EPISODE

    def test_valid_merge(self) -> None:
        action = parse_action(
            '{"action_type": "merge_duplicate", "ticket_id": "T001", "target_ticket_id": "T002"}'
        )
        assert action is not None
        assert action.target_ticket_id == "T002"

    def test_markdown_fence_stripped(self) -> None:
        action = parse_action(
            '```json\n{"action_type": "open_ticket", "ticket_id": "T001"}\n```'
        )
        assert action is not None
        assert action.action_type == ActionType.OPEN_TICKET

    def test_json_in_prose(self) -> None:
        action = parse_action(
            'I think the best action is: {"action_type": "skip_turn"} because ...'
        )
        assert action is not None
        assert action.action_type == ActionType.SKIP_TURN

    def test_uppercase_enums_normalized(self) -> None:
        action = parse_action(
            '{"action_type": "CLASSIFY_TICKET", "ticket_id": "T001", '
            '"issue_family": "BILLING", "issue_subtype": "REFUND"}'
        )
        assert action is not None
        assert action.action_type == ActionType.CLASSIFY_TICKET

    def test_empty_string_returns_none(self) -> None:
        assert parse_action("") is None

    def test_none_returns_none(self) -> None:
        assert parse_action(None) is None

    def test_garbage_text_returns_none(self) -> None:
        assert parse_action("just some random text") is None

    def test_missing_action_type_returns_none(self) -> None:
        assert parse_action('{"ticket_id": "T001"}') is None

    def test_invalid_enum_value_returns_none(self) -> None:
        assert parse_action(
            '{"action_type": "classify_ticket", "ticket_id": "T001", '
            '"issue_family": "nonexistent_family", "issue_subtype": "refund"}'
        ) is None

    def test_unbalanced_braces_returns_none(self) -> None:
        assert parse_action('{"action_type": "open_ticket"') is None


class TestActionToStr:
    """Tests for inference.action_to_str()."""

    def test_open_ticket(self) -> None:
        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET, ticket_id="T001"
        )
        assert action_to_str(action) == "open_ticket:T001"

    def test_classify(self) -> None:
        action = TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id="T001",
            issue_family=IssueFamily.BILLING,
            issue_subtype=IssueSubtype.REFUND,
        )
        result = action_to_str(action)
        assert "classify_ticket" in result
        assert "T001" in result
        assert "billing" in result

    def test_skip_turn(self) -> None:
        action = TriageSieveAction(action_type=ActionType.SKIP_TURN)
        assert action_to_str(action) == "skip_turn"


class TestSerializeObservation:
    """Tests for inference.serialize_observation()."""

    def test_contains_inbox(self) -> None:
        obs = _make_observation()
        text = serialize_observation(obs)
        assert "T001" in text
        assert "Test ticket" in text
        assert "test@example.com" in text

    def test_contains_legal_actions(self) -> None:
        obs = _make_observation()
        text = serialize_observation(obs)
        assert "open_ticket" in text

    def test_contains_budget(self) -> None:
        obs = _make_observation(action_budget_remaining=5)
        text = serialize_observation(obs)
        assert "5" in text

    def test_contains_focused_ticket(self) -> None:
        ft = FocusedTicket(
            ticket_id="T001",
            subject="Test",
            latest_message="Help me please",
            thread_history=[],
            attachments=[],
            visible_internal_notes=[],
            prior_actions_taken=[],
        )
        obs = _make_observation(focused_ticket=ft)
        text = serialize_observation(obs)
        assert "Help me please" in text

    def test_contains_hint_when_present(self) -> None:
        obs = _make_observation(hint="Check the sender domain")
        text = serialize_observation(obs)
        assert "Check the sender domain" in text


class TestLogFunctions:
    """Verify [START]/[STEP]/[END] stdout format."""

    def test_log_start(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_start("easy", "triagesieve_env", "test-model")
        captured = capsys.readouterr().out
        assert captured.strip() == "[START] task=easy env=triagesieve_env model=test-model"

    def test_log_step(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_step(step=1, action="open_ticket:T001", reward=0.01, done=False, error=None)
        captured = capsys.readouterr().out
        assert "[STEP] step=1" in captured
        assert "reward=0.01" in captured
        assert "done=false" in captured
        assert "error=null" in captured

    def test_log_step_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_step(step=2, action="close:T001", reward=-0.02, done=False, error="Illegal action")
        captured = capsys.readouterr().out
        assert "error=Illegal action" in captured

    def test_log_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_end(success=True, steps=5, score=0.95, rewards=[0.01, 0.02, 0.01, 0.01, 0.95])
        captured = capsys.readouterr().out
        assert "[END]" in captured
        assert "success=true" in captured
        assert "score=0.950" in captured
        assert "rewards=0.01,0.02,0.01,0.01,0.95" in captured

    def test_log_end_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_end(success=False, steps=2, score=0.0, rewards=[0.01, -0.02])
        captured = capsys.readouterr().out
        assert "success=false" in captured


# ===========================================================================
# Layer 2: Integration Tests with Mocked LLM (no Docker, no network)
# ===========================================================================


class TestRunTaskMocked:
    """Full run_task() loop with mocked LLM and local environment."""

    def _run(
        self,
        responses: list[str],
        seed: int = 42,
        difficulty: str = "easy",
        max_steps: int = 8,
    ) -> dict[str, Any]:
        """Helper to run a task with canned LLM responses."""
        client = _mock_openai_client(responses)
        env = LocalAsyncEnvAdapter()
        return asyncio.run(
            run_task(
                client=client,
                env=env,
                task_name=difficulty,
                seed=seed,
                difficulty=difficulty,
                max_steps=max_steps,
            )
        )

    def test_all_garbage_completes(self) -> None:
        """LLM returns nonsense — episode should still complete without crash."""
        result = self._run(["not json at all"] * 10)
        assert result["score"] >= 0.0
        assert result["score"] <= 1.0
        assert result["steps"] > 0

    def test_empty_responses_completes(self) -> None:
        """LLM returns empty strings — should fall back to skip_turn."""
        result = self._run([""] * 10)
        assert result["score"] >= 0.0
        assert result["steps"] > 0

    def test_skip_turn_budget_exhaustion(self) -> None:
        """All skip_turns exhaust budget — episode terminates with a score."""
        result = self._run(['{"action_type": "skip_turn"}'] * 20)
        assert result["score"] >= 0.0
        assert result["score"] <= 1.0

    def test_easy_happy_path(self) -> None:
        """Oracle-like sequence for seed=42 easy (refund_missing_order_id)."""
        import json as _json

        # Discover correct ticket ID and hidden truth
        env = TriageSieveEnvironment()
        obs = env.reset(seed=42, difficulty="easy", mode="eval_strict")
        tid = obs.inbox_summaries[0].ticket_id
        ht = env._ticket_index[tid].hidden_truth

        # Build valid JSON strings (use json.dumps for lists)
        responses = [
            _json.dumps({"action_type": "open_ticket", "ticket_id": tid}),
            _json.dumps({
                "action_type": "classify_ticket", "ticket_id": tid,
                "issue_family": ht.issue_family.value,
                "issue_subtype": ht.issue_subtype.value,
            }),
            _json.dumps({
                "action_type": "set_impact_urgency", "ticket_id": tid,
                "impact": ht.impact.value, "urgency": ht.urgency.value,
            }),
            _json.dumps({
                "action_type": "request_information", "ticket_id": tid,
                "requested_fields": list(ht.required_missing_fields),
                "template_id": ht.correct_template_ids[0],
            }),
            _json.dumps({
                "action_type": "route_ticket", "ticket_id": tid,
                "queue_id": ht.required_queue.value,
            }),
            _json.dumps({
                "action_type": "close_ticket", "ticket_id": tid,
                "close_reason": "resolved",
                "template_id": ht.correct_template_ids[-1],
            }),
        ]
        result = self._run(responses, seed=42, difficulty="easy")
        assert result["score"] >= 0.90, f"Expected >=0.90 but got {result['score']}"
        assert result["success"] is True

    def test_invalid_ticket_id_handled(self) -> None:
        """Action with nonexistent ticket_id gets penalty, no crash."""
        responses = [
            '{"action_type": "open_ticket", "ticket_id": "T-NONEXISTENT"}',
        ] * 10
        result = self._run(responses)
        assert result["score"] >= 0.0
        assert result["steps"] > 0

    def test_all_three_difficulties(self) -> None:
        """All difficulty tiers complete without exception."""
        for diff, seed in [("easy", 42), ("medium", 42), ("hard", 42)]:
            result = self._run(
                ['{"action_type": "skip_turn"}'] * 20,
                seed=seed,
                difficulty=diff,
                max_steps=20,
            )
            assert 0.0 <= result["score"] <= 1.0, f"{diff} score out of range"
            assert result["steps"] > 0

    def test_terminal_score_in_0_1_range(self) -> None:
        """Terminal score is always in [0, 1] regardless of actions taken."""
        for _ in range(5):
            result = self._run(
                ['{"action_type": "skip_turn"}'] * 10,
                seed=42,
                difficulty="easy",
            )
            assert 0.0 <= result["score"] <= 1.0

    def test_result_has_required_keys(self) -> None:
        """run_task() return dict has all required keys."""
        result = self._run(['{"action_type": "skip_turn"}'] * 10)
        assert "task" in result
        assert "score" in result
        assert "success" in result
        assert "steps" in result


# ===========================================================================
# Layer 3: Real LLM Smoke Tests (skip if no HF_TOKEN)
# ===========================================================================

_HF_TOKEN = os.getenv("HF_TOKEN")
_SKIP_REAL_LLM = not _HF_TOKEN


@pytest.mark.slow
@pytest.mark.skipif(_SKIP_REAL_LLM, reason="HF_TOKEN not set — skipping real LLM tests")
class TestRealLLMSmoke:
    """Smoke tests with a real sub-1B model via HuggingFace router."""

    # Use the same model configured in inference.py env vars, or a sensible default
    MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

    def _get_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(base_url=self.BASE_URL, api_key=_HF_TOKEN)

    def test_openai_client_connects(self) -> None:
        """Verify OpenAI client can reach HF router and get a response."""
        client = self._get_client()
        completion = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            temperature=0.0,
        )
        assert completion.choices[0].message.content is not None
        assert len(completion.choices[0].message.content) > 0

    def test_easy_episode_completes(self) -> None:
        """Run easy episode with real LLM — must complete without crash."""
        client = self._get_client()
        env = LocalAsyncEnvAdapter()
        result = asyncio.run(
            run_task(
                client=client,
                env=env,
                task_name="easy",
                seed=42,
                difficulty="easy",
                max_steps=8,
            )
        )
        assert 0.0 <= result["score"] <= 1.0
        assert result["steps"] > 0

    def test_all_difficulties_complete(self) -> None:
        """All 3 difficulty tiers complete with real LLM — no crashes."""
        client = self._get_client()
        for diff, seed, max_steps in [("easy", 42, 8), ("medium", 42, 14), ("hard", 42, 20)]:
            env = LocalAsyncEnvAdapter()
            result = asyncio.run(
                run_task(
                    client=client,
                    env=env,
                    task_name=diff,
                    seed=seed,
                    difficulty=diff,
                    max_steps=max_steps,
                )
            )
            assert 0.0 <= result["score"] <= 1.0, f"{diff} failed with score {result['score']}"
