"""Determinism tests (§23.3).

Covers:
- Same seed + episode_id → identical initial observation
- Same action sequence → identical final state and score
- No random branch without a fixed seed

Invariants tested:
1. reset(seed=S, difficulty=D) called twice → identical observation and state.
2. Same action sequence replayed → identical intermediate observations, final state, score.
3. N repeated runs with the same seed are all identical; different seeds diverge.
"""

from __future__ import annotations

from typing import Any

import pytest

from ..models import (
    ActionType,
    TriageSieveAction,
    TriageSieveObservation,
    TriageSieveState,
    TaskDifficulty,
)
from ..baseline.scripted_expert import ScriptedExpert
from ..server.triagesieve_env_environment import TriageSieveEnvironment

# ---------------------------------------------------------------------------
# Seed contracts (must match episode_engine archetype selection):
#   seed=42,  difficulty="easy"   → refund_missing_order_id (billing/refund, needs order_id)
#   seed=100, difficulty="easy"   → benign_expected (non-actionable)
#   seed=99,  difficulty="easy"   → different archetype from seed=42
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs_dump(obs: TriageSieveObservation) -> dict[str, Any]:
    """Serialize an observation to a comparable dict."""
    return obs.model_dump()


def _state_dump(state: TriageSieveState) -> dict[str, Any]:
    """Serialize a state to a comparable dict."""
    return state.model_dump()


def _action(action_type: ActionType, **kwargs: object) -> TriageSieveAction:
    return TriageSieveAction(action_type=action_type, metadata={}, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> TriageSieveEnvironment:
    return TriageSieveEnvironment()


@pytest.fixture
def env_pair() -> tuple[TriageSieveEnvironment, TriageSieveEnvironment]:
    """Two independent environment instances for parallel replay comparison."""
    return TriageSieveEnvironment(), TriageSieveEnvironment()


# ---------------------------------------------------------------------------
# Invariant 1: Same seed + difficulty → identical reset observation & state
# ---------------------------------------------------------------------------


class TestResetDeterminism:
    """Two resets with the same seed and difficulty must produce identical outputs."""

    @pytest.mark.parametrize("seed", [42, 100, 7])
    def test_same_seed_identical_observation(
        self, env_pair: tuple[TriageSieveEnvironment, TriageSieveEnvironment], seed: int
    ) -> None:
        env_a, env_b = env_pair
        obs_a = env_a.reset(seed=seed, mode="eval_strict", difficulty="easy")
        obs_b = env_b.reset(seed=seed, mode="eval_strict", difficulty="easy")
        assert _obs_dump(obs_a) == _obs_dump(obs_b)

    @pytest.mark.parametrize("seed", [42, 100, 7])
    def test_same_seed_identical_state(
        self, env_pair: tuple[TriageSieveEnvironment, TriageSieveEnvironment], seed: int
    ) -> None:
        env_a, env_b = env_pair
        env_a.reset(seed=seed, mode="eval_strict", difficulty="easy")
        env_b.reset(seed=seed, mode="eval_strict", difficulty="easy")
        assert _state_dump(env_a.state) == _state_dump(env_b.state)

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_across_difficulties(
        self, env_pair: tuple[TriageSieveEnvironment, TriageSieveEnvironment], difficulty: str
    ) -> None:
        """Determinism holds regardless of difficulty tier."""
        seed = 42
        env_a, env_b = env_pair
        obs_a = env_a.reset(seed=seed, mode="eval_strict", difficulty=difficulty)
        obs_b = env_b.reset(seed=seed, mode="eval_strict", difficulty=difficulty)
        assert _obs_dump(obs_a) == _obs_dump(obs_b)
        assert _state_dump(env_a.state) == _state_dump(env_b.state)


# ---------------------------------------------------------------------------
# Invariant 2: Same action sequence → identical final state and score
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    """Replaying the same action sequence must produce identical trajectories."""

    @pytest.mark.parametrize(
        "seed,difficulty",
        [(42, TaskDifficulty.EASY), (100, TaskDifficulty.EASY), (42, TaskDifficulty.MEDIUM)],
    )
    def test_expert_replay_identical_trace(self, seed: int, difficulty: TaskDifficulty) -> None:
        """ScriptedExpert run twice with same seed → identical traces."""
        expert_a = ScriptedExpert(TriageSieveEnvironment())
        expert_b = ScriptedExpert(TriageSieveEnvironment())
        trace_a = expert_a.run_episode(seed=seed, difficulty=difficulty)
        trace_b = expert_b.run_episode(seed=seed, difficulty=difficulty)

        assert trace_a["episode_id"] == trace_b["episode_id"]
        assert trace_a["final_score"] == pytest.approx(trace_b["final_score"], abs=1e-9)
        assert trace_a["score_breakdown"] == trace_b["score_breakdown"]
        assert trace_a["action_sequence"] == trace_b["action_sequence"]

    def test_manual_action_replay_identical_observations(
        self, env_pair: tuple[TriageSieveEnvironment, TriageSieveEnvironment]
    ) -> None:
        """Manually replay a short action sequence on two envs and compare every step."""
        env_a, env_b = env_pair
        seed = 42

        obs_a = env_a.reset(seed=seed, mode="eval_strict", difficulty="easy")
        obs_b = env_b.reset(seed=seed, mode="eval_strict", difficulty="easy")
        assert _obs_dump(obs_a) == _obs_dump(obs_b)

        tid = obs_a.inbox_summaries[0].ticket_id

        actions = [
            _action(ActionType.OPEN_TICKET, ticket_id=tid),
            _action(
                ActionType.CLASSIFY_TICKET,
                ticket_id=tid,
                issue_family="billing",
                issue_subtype="refund",
            ),
            _action(ActionType.FINISH_EPISODE),
        ]

        for action in actions:
            obs_a = env_a.step(action)
            obs_b = env_b.step(action)
            assert _obs_dump(obs_a) == _obs_dump(
                obs_b
            ), f"Divergence after {action.action_type.value}"

        assert _state_dump(env_a.state) == _state_dump(env_b.state)


# ---------------------------------------------------------------------------
# Invariant 3: No unfixed randomness
# ---------------------------------------------------------------------------


class TestNoUnfixedRandomness:
    """Multiple runs with the same seed are identical; different seeds diverge."""

    def test_five_repeated_runs_identical(self) -> None:
        """5 runs with seed=42/easy must all produce the same trace."""
        traces: list[dict[str, Any]] = []
        for _ in range(5):
            expert = ScriptedExpert(TriageSieveEnvironment())
            traces.append(expert.run_episode(seed=42, difficulty=TaskDifficulty.EASY))

        reference = traces[0]
        for i, trace in enumerate(traces[1:], start=2):
            assert trace["episode_id"] == reference["episode_id"], f"Run {i} episode_id diverged"
            assert trace["final_score"] == pytest.approx(
                reference["final_score"], abs=1e-9
            ), f"Run {i} score diverged"
            assert (
                trace["action_sequence"] == reference["action_sequence"]
            ), f"Run {i} action sequence diverged"
            assert (
                trace["score_breakdown"] == reference["score_breakdown"]
            ), f"Run {i} score breakdown diverged"

    def test_different_seeds_differ(self) -> None:
        """Two different seeds must produce different initial observations."""
        env_a = TriageSieveEnvironment()
        env_b = TriageSieveEnvironment()
        obs_42 = env_a.reset(seed=42, mode="eval_strict", difficulty="easy")
        dump_42 = _obs_dump(obs_42)

        obs_99 = env_b.reset(seed=99, mode="eval_strict", difficulty="easy")
        dump_99 = _obs_dump(obs_99)

        # At minimum, ticket_ids or subjects must differ
        ids_42 = {s["ticket_id"] for s in dump_42["inbox_summaries"]}
        ids_99 = {s["ticket_id"] for s in dump_99["inbox_summaries"]}
        subjects_42 = {s["subject"] for s in dump_42["inbox_summaries"]}
        subjects_99 = {s["subject"] for s in dump_99["inbox_summaries"]}

        assert (
            ids_42 != ids_99 or subjects_42 != subjects_99
        ), "Different seeds produced identical observations"
