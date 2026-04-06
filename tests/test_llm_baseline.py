"""Tests for the LLM baseline agent (§22.2).

Tests pure helpers (observation serialization, action parsing, system prompt)
without requiring litellm installed. Integration test mocks litellm.completion.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ..models import (
    ActionType,
    CloseReason,
    CustomerTier,
    FocusedTicket,
    InboxSummaryItem,
    Impact,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_observation(**overrides: Any) -> TriageSieveObservation:
    """Build a minimal valid observation for testing."""
    defaults: dict[str, Any] = {
        "done": False,
        "reward": 0.01,
        "metadata": {},
        "inbox_summaries": [
            InboxSummaryItem(
                ticket_id="T001",
                subject="Refund request",
                sender_email="alice@example.com",
                received_at="2026-04-05T10:00:00Z",
                status=TicketStatus.NEW,
                customer_tier=CustomerTier.PRO,
                has_attachment=False,
                sla_remaining_minutes=120,
                short_preview="I need a refund for order #12345",
            ),
        ],
        "focused_ticket": None,
        "available_templates": [
            {"template_id": "tpl_refund_ack", "name": "Refund Acknowledgement", "description": "Ack refund", "applies_to": "billing"},
        ],
        "allowed_queues": [QueueId.BILLING_TEAM, QueueId.REFUND_TEAM],
        "routing_policy_cards": [
            RoutingPolicyCard(
                queue_id=QueueId.BILLING_TEAM,
                description="General billing",
                prerequisites=[],
                handles_families=[IssueFamily.BILLING],
            ),
        ],
        "sla_policy_cards": [
            SlaPolicyCard(tier=CustomerTier.PRO, response_deadline_minutes=60, resolution_deadline_minutes=240),
        ],
        "legal_actions": [ActionType.OPEN_TICKET, ActionType.SKIP_TURN, ActionType.FINISH_EPISODE],
        "action_budget_remaining": 4,
        "step_count": 0,
        "current_time": "2026-04-05T10:00:00Z",
        "last_action_result": "ok",
        "task_difficulty": TaskDifficulty.EASY,
        "hint": None,
    }
    defaults.update(overrides)
    return TriageSieveObservation(**defaults)


def _make_focused_observation() -> TriageSieveObservation:
    """Observation with a focused ticket visible."""
    return _make_observation(
        focused_ticket=FocusedTicket(
            ticket_id="T001",
            subject="Refund request",
            latest_message="I need a refund for order #12345. Invoice attached.",
            thread_history=[
                {"role": "customer", "content": "I need a refund for order #12345.", "timestamp": "2026-04-05T09:50:00Z"},
            ],
            attachments=["invoice.pdf"],
            visible_internal_notes=["Verified customer since 2024."],
            prior_actions_taken=["open_ticket(T001)"],
        ),
        legal_actions=[
            ActionType.CLASSIFY_TICKET,
            ActionType.MERGE_DUPLICATE,
            ActionType.CLOSE_TICKET,
            ActionType.SKIP_TURN,
            ActionType.FINISH_EPISODE,
        ],
        step_count=1,
        action_budget_remaining=3,
    )


# ---------------------------------------------------------------------------
# Lazy import helper — avoids ImportError if litellm not installed
# ---------------------------------------------------------------------------


def _import_baseline():
    """Import LLMBaseline, skip test if litellm missing."""
    try:
        from ..baseline.llm_baseline import LLMBaseline
        return LLMBaseline
    except ImportError:
        pytest.skip("litellm not installed")


# ---------------------------------------------------------------------------
# Tests: _serialize_observation
# ---------------------------------------------------------------------------


class TestSerializeObservation:
    """Tests for observation → prompt text serialization."""

    def test_includes_inbox_summary(self) -> None:
        LLMBaseline = _import_baseline()
        obs = _make_observation()
        baseline = LLMBaseline.__new__(LLMBaseline)
        text = baseline._serialize_observation(obs)
        assert "T001" in text
        assert "Refund request" in text
        assert "alice@example.com" in text

    def test_includes_focused_ticket_when_present(self) -> None:
        LLMBaseline = _import_baseline()
        obs = _make_focused_observation()
        baseline = LLMBaseline.__new__(LLMBaseline)
        text = baseline._serialize_observation(obs)
        assert "FOCUSED TICKET" in text.upper() or "Focused Ticket" in text or "focused_ticket" in text.lower()
        assert "invoice.pdf" in text
        assert "order #12345" in text

    def test_no_focused_section_when_none(self) -> None:
        LLMBaseline = _import_baseline()
        obs = _make_observation(focused_ticket=None)
        baseline = LLMBaseline.__new__(LLMBaseline)
        text = baseline._serialize_observation(obs)
        # Should not have focused ticket details
        assert "invoice.pdf" not in text

    def test_includes_legal_actions(self) -> None:
        LLMBaseline = _import_baseline()
        obs = _make_observation()
        baseline = LLMBaseline.__new__(LLMBaseline)
        text = baseline._serialize_observation(obs)
        assert "open_ticket" in text
        assert "skip_turn" in text

    def test_includes_budget_and_step(self) -> None:
        LLMBaseline = _import_baseline()
        obs = _make_observation(action_budget_remaining=3, step_count=2)
        baseline = LLMBaseline.__new__(LLMBaseline)
        text = baseline._serialize_observation(obs)
        assert "3" in text  # budget
        assert "2" in text  # step

    def test_includes_hint_when_present(self) -> None:
        LLMBaseline = _import_baseline()
        obs = _make_observation(hint="Check thread history for order identifier")
        baseline = LLMBaseline.__new__(LLMBaseline)
        text = baseline._serialize_observation(obs)
        assert "Check thread history" in text


# ---------------------------------------------------------------------------
# Tests: _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for the static system prompt."""

    def test_contains_all_action_types(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        prompt = baseline._build_system_prompt()
        for at in ActionType:
            assert at.value in prompt, f"Missing action type {at.value} in system prompt"

    def test_contains_json_format_instruction(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        prompt = baseline._build_system_prompt()
        assert "json" in prompt.lower() or "JSON" in prompt

    def test_contains_enum_values(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        prompt = baseline._build_system_prompt()
        # Spot-check key enum values
        assert "billing" in prompt
        assert "refund" in prompt
        assert "tech_support_l2" in prompt
        assert "single_user" in prompt

    def test_contains_priority_matrix_info(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        prompt = baseline._build_system_prompt()
        assert "impact" in prompt.lower()
        assert "urgency" in prompt.lower()


# ---------------------------------------------------------------------------
# Tests: _parse_action
# ---------------------------------------------------------------------------


class TestParseAction:
    """Tests for raw LLM text → TriageSieveAction parsing."""

    def test_parses_valid_json(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "open_ticket", "ticket_id": "T001"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.action_type == ActionType.OPEN_TICKET
        assert action.ticket_id == "T001"

    def test_strips_markdown_fences(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '```json\n{"action_type": "skip_turn"}\n```'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.action_type == ActionType.SKIP_TURN

    def test_extracts_json_from_surrounding_text(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = 'I think we should do this: {"action_type": "open_ticket", "ticket_id": "T001"} and that should work.'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.action_type == ActionType.OPEN_TICKET

    def test_handles_uppercase_enum(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "CLASSIFY_TICKET", "ticket_id": "T001", "issue_family": "BILLING", "issue_subtype": "REFUND"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.action_type == ActionType.CLASSIFY_TICKET
        assert action.issue_family == IssueFamily.BILLING

    def test_returns_none_on_garbage(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        assert baseline._parse_action("not json at all") is None
        assert baseline._parse_action("") is None
        assert baseline._parse_action("{}") is None  # missing action_type

    def test_returns_none_on_invalid_enum(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "nonexistent_action", "ticket_id": "T001"}'
        assert baseline._parse_action(raw) is None

    def test_parses_classify_with_all_fields(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "classify_ticket", "ticket_id": "T001", "issue_family": "billing", "issue_subtype": "refund"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.issue_family == IssueFamily.BILLING
        assert action.issue_subtype == IssueSubtype.REFUND

    def test_parses_close_ticket(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "close_ticket", "ticket_id": "T001", "close_reason": "resolved", "template_id": "tpl_refund_ack"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.close_reason == CloseReason.RESOLVED
        assert action.template_id == "tpl_refund_ack"

    def test_parses_route_ticket(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "route_ticket", "ticket_id": "T001", "queue_id": "billing_team"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.queue_id == QueueId.BILLING_TEAM

    def test_parses_request_information(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "request_information", "ticket_id": "T001", "requested_fields": ["order_id", "invoice_pdf"]}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.action_type == ActionType.REQUEST_INFORMATION
        assert action.requested_fields == ["order_id", "invoice_pdf"]

    def test_parses_request_information_with_template(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "request_information", "ticket_id": "T001", "requested_fields": ["order_id"], "template_id": "tpl_missing_info"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.template_id == "tpl_missing_info"

    def test_parses_set_impact_urgency(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        raw = '{"action_type": "set_impact_urgency", "ticket_id": "T001", "impact": "single_user", "urgency": "medium"}'
        action = baseline._parse_action(raw)
        assert action is not None
        assert action.impact == Impact.SINGLE_USER
        assert action.urgency == Urgency.MEDIUM


# ---------------------------------------------------------------------------
# Tests: _fallback_action
# ---------------------------------------------------------------------------


class TestFallbackAction:
    """Tests for the fallback action on parse failure."""

    def test_returns_skip_turn(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        action = baseline._fallback_action()
        assert action.action_type == ActionType.SKIP_TURN

    def test_fallback_is_valid_action(self) -> None:
        LLMBaseline = _import_baseline()
        baseline = LLMBaseline.__new__(LLMBaseline)
        action = baseline._fallback_action()
        # Should be a valid TriageSieveAction
        assert isinstance(action, TriageSieveAction)


# ---------------------------------------------------------------------------
# Tests: run_episode (integration, mocked LLM)
# ---------------------------------------------------------------------------


class TestRunEpisode:
    """Integration test: LLM calls are mocked, environment is real."""

    def _make_llm_response(self, content: str) -> MagicMock:
        """Build a mock litellm completion response."""
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_run_episode_returns_trace(self) -> None:
        """Full episode with mocked LLM producing valid actions."""
        LLMBaseline = _import_baseline()

        from ..server.triagesieve_env_environment import TriageSieveEnvironment

        env = TriageSieveEnvironment()

        # We'll feed the LLM mock a sequence of valid actions for an easy episode.
        # Reset first to see what tickets exist.
        obs = env.reset(seed=42, mode="eval_strict", difficulty="easy")
        ticket_id = obs.inbox_summaries[0].ticket_id

        # Re-create env for the baseline to use fresh
        env2 = TriageSieveEnvironment()

        responses = [
            self._make_llm_response(f'{{"action_type": "open_ticket", "ticket_id": "{ticket_id}"}}'),
            self._make_llm_response(f'{{"action_type": "classify_ticket", "ticket_id": "{ticket_id}", "issue_family": "billing", "issue_subtype": "refund"}}'),
            self._make_llm_response(f'{{"action_type": "set_impact_urgency", "ticket_id": "{ticket_id}", "impact": "single_user", "urgency": "medium"}}'),
            self._make_llm_response(f'{{"action_type": "finish_episode"}}'),
        ]

        baseline = LLMBaseline(env=env2, model="gpt-4o-mini", temperature=0.0)

        with patch("triagesieve_env.baseline.llm_baseline.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = responses
            trace = baseline.run_episode(seed=42, difficulty=TaskDifficulty.EASY)

        assert "episode_id" in trace
        assert "seed" in trace
        assert trace["seed"] == 42
        assert "action_sequence" in trace
        assert "final_score" in trace
        assert isinstance(trace["action_sequence"], list)
        assert len(trace["action_sequence"]) > 0
        assert trace["done"] is True

    def test_run_episode_handles_parse_failure(self) -> None:
        """When LLM returns garbage, fallback to SKIP_TURN, episode still completes."""
        LLMBaseline = _import_baseline()

        from ..server.triagesieve_env_environment import TriageSieveEnvironment

        env = TriageSieveEnvironment()

        # All garbage → skip_turn until budget exhausted or done
        garbage_responses = [
            self._make_llm_response("I don't know what to do"),
        ] * 10  # enough for any budget

        baseline = LLMBaseline(env=env, model="gpt-4o-mini", temperature=0.0)

        with patch("triagesieve_env.baseline.llm_baseline.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = garbage_responses
            trace = baseline.run_episode(seed=42, difficulty=TaskDifficulty.EASY)

        assert trace["done"] is True
        assert trace["final_score"] is not None
