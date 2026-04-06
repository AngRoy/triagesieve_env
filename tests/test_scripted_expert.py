"""Tests for the scripted expert oracle baseline (§22.1).

Verifies:
- Expert follows gold SOP for each ticket within budget constraints.
- Priority ordering: higher-priority tickets get first substantive actions.
- Returns a trace dict with score breakdown.
- No invalid actions are ever produced.
- Deterministic: same seed → identical trace.
- Score breakdown includes proper terminal business + UJCS scoring.
"""

from __future__ import annotations

import pytest

from ..baseline.scripted_expert import ScriptedExpert
from ..models import (
    ActionType,
    TaskDifficulty,
)
from ..server.triagesieve_env_environment import TriageSieveEnvironment


@pytest.fixture
def env() -> TriageSieveEnvironment:
    """Fresh environment instance."""
    return TriageSieveEnvironment()


@pytest.fixture
def expert(env: TriageSieveEnvironment) -> ScriptedExpert:
    """Expert policy wired to the environment."""
    return ScriptedExpert(env)


class TestScriptedExpertAPI:
    """Expert class public interface."""

    def test_constructor_accepts_environment(self, env: TriageSieveEnvironment) -> None:
        expert = ScriptedExpert(env)
        assert expert.env is env

    def test_run_episode_returns_trace_dict(self, expert: ScriptedExpert) -> None:
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert isinstance(trace, dict)
        assert "episode_id" in trace
        assert "seed" in trace
        assert "action_sequence" in trace
        assert "final_score" in trace
        assert "score_breakdown" in trace

    def test_run_episode_returns_done(self, expert: ScriptedExpert) -> None:
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert trace["done"] is True

    def test_trace_score_breakdown_keys(self, expert: ScriptedExpert) -> None:
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        breakdown = trace["score_breakdown"]
        assert "terminal_business_score" in breakdown
        assert "ujcs_openenv" in breakdown
        assert "episode_penalties" in breakdown
        assert "priority_order_score" in breakdown
        assert "invalid_action_count" in breakdown
        assert "reassignment_count" in breakdown


class TestScriptedExpertCorrectness:
    """No invalid actions, correct scores for known archetypes."""

    @pytest.mark.parametrize("seed", range(20))
    def test_no_invalid_actions_easy(self, seed: int) -> None:
        """Expert should never produce an invalid action on any easy seed."""
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=seed, difficulty=TaskDifficulty.EASY)
        for entry in trace["action_sequence"]:
            assert entry["result"] == "ok", (
                f"seed={seed}, step {entry['step']}: {entry['result']}"
            )

    @pytest.mark.parametrize("seed", range(10))
    def test_no_invalid_actions_medium(self, seed: int) -> None:
        """Expert should never produce an invalid action on medium seeds."""
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=seed, difficulty=TaskDifficulty.MEDIUM)
        for entry in trace["action_sequence"]:
            assert entry["result"] == "ok", (
                f"seed={seed}, step {entry['step']}: {entry['result']}"
            )

    @pytest.mark.parametrize("seed", range(10))
    def test_no_invalid_actions_hard(self, seed: int) -> None:
        """Expert should never produce an invalid action on hard seeds."""
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=seed, difficulty=TaskDifficulty.HARD)
        for entry in trace["action_sequence"]:
            assert entry["result"] == "ok", (
                f"seed={seed}, step {entry['step']}: {entry['result']}"
            )

    def test_zero_invalid_actions_in_breakdown(self, expert: ScriptedExpert) -> None:
        """Score breakdown must show 0 invalid actions."""
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert trace["score_breakdown"]["invalid_action_count"] == 0

    def test_zero_reassignments(self, expert: ScriptedExpert) -> None:
        """Expert never routes a ticket twice."""
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert trace["score_breakdown"]["reassignment_count"] == 0

    def test_zero_episode_penalties(self, expert: ScriptedExpert) -> None:
        """Expert produces no penalties."""
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert trace["score_breakdown"]["episode_penalties"] == 0.0

    def test_positive_terminal_business_score(self, expert: ScriptedExpert) -> None:
        """Expert achieves a positive terminal business score."""
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert trace["score_breakdown"]["terminal_business_score"] > 0.0

    def test_normal_archetype_high_score(self) -> None:
        """Seed=7 (failed_invoice_charge_dispute, no missing fields) scores >= 0.80."""
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        assert trace["final_score"] >= 0.80, (
            f"Expert scored {trace['final_score']:.3f} on seed=7 easy"
        )


class TestScriptedExpertDeterminism:
    """Same seed + same difficulty → identical trace."""

    def test_same_seed_same_trace(self) -> None:
        expert_a = ScriptedExpert(TriageSieveEnvironment())
        trace_a = expert_a.run_episode(seed=7, difficulty=TaskDifficulty.EASY)

        expert_b = ScriptedExpert(TriageSieveEnvironment())
        trace_b = expert_b.run_episode(seed=7, difficulty=TaskDifficulty.EASY)

        assert trace_a["final_score"] == trace_b["final_score"]
        assert trace_a["action_sequence"] == trace_b["action_sequence"]

    def test_different_seed_different_trace(self) -> None:
        expert_a = ScriptedExpert(TriageSieveEnvironment())
        trace_a = expert_a.run_episode(seed=7, difficulty=TaskDifficulty.EASY)

        expert_b = ScriptedExpert(TriageSieveEnvironment())
        trace_b = expert_b.run_episode(seed=0, difficulty=TaskDifficulty.EASY)

        assert trace_a["episode_id"] != trace_b["episode_id"]


class TestScriptedExpertActionSequence:
    """Verify action types follow expected SOP patterns."""

    def test_easy_begins_with_open(self, expert: ScriptedExpert) -> None:
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        first_action = trace["action_sequence"][0]
        assert first_action["action"]["action_type"] == ActionType.OPEN_TICKET.value

    def test_ends_with_finish_episode(self, expert: ScriptedExpert) -> None:
        trace = expert.run_episode(seed=1, difficulty=TaskDifficulty.EASY)
        last_action = trace["action_sequence"][-1]
        assert last_action["action"]["action_type"] == ActionType.FINISH_EPISODE.value

    def test_classify_follows_open(self) -> None:
        """For normal (non-special) archetypes, classify follows open."""
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        # seed=7 = failed_invoice_charge_dispute (normal flow)
        trace = expert.run_episode(seed=7, difficulty=TaskDifficulty.EASY)
        types = [e["action"]["action_type"] for e in trace["action_sequence"]]
        assert types[0] == ActionType.OPEN_TICKET.value
        assert types[1] == ActionType.CLASSIFY_TICKET.value


class TestScriptedExpertMultiTicket:
    """Multi-ticket priority ordering (§19)."""

    def test_hard_processes_multiple_tickets(self) -> None:
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=0, difficulty=TaskDifficulty.HARD)
        ticket_ids_opened = [
            entry["action"].get("ticket_id")
            for entry in trace["action_sequence"]
            if entry["action"]["action_type"] == ActionType.OPEN_TICKET.value
        ]
        assert len(ticket_ids_opened) >= 2, "Hard episode should open multiple tickets"

    def test_medium_priority_order_score(self) -> None:
        """Expert achieves perfect or near-perfect priority ordering on medium."""
        env = TriageSieveEnvironment()
        expert = ScriptedExpert(env)
        trace = expert.run_episode(seed=0, difficulty=TaskDifficulty.MEDIUM)
        assert trace["score_breakdown"]["priority_order_score"] >= 0.5
