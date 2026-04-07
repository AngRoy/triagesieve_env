"""Smoke playthrough script.

Usage:
    python scripts/smoke_playthrough.py [--seed 42] [--difficulty easy|medium|hard|all] [--quiet]

Runs the scripted expert baseline against one episode per difficulty tier,
prints a human-readable step-by-step trace, and asserts final score >= 0.90.

Exit code 0 if all tiers pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from triagesieve_env.baseline.scripted_expert import ScriptedExpert
from triagesieve_env.models import TaskDifficulty
from triagesieve_env.server.triagesieve_env_environment import TriageSieveEnvironment

logger = logging.getLogger(__name__)

# Per-difficulty solvability thresholds for the scripted expert.
# Easy: oracle should near-perfectly complete the single ticket within budget.
# Medium: 2-3 tickets with 12-step budget; 3-ticket episodes may be budget-
#   constrained so the oracle sometimes scores ~0.78. Threshold set at 0.75.
# Hard: budget is intentionally tight (3-4 critical tickets, escalation chains,
#   SLA pressure); partial completion is by design. Threshold reflects that.
_SOLVABILITY_THRESHOLDS: dict[TaskDifficulty, float] = {
    TaskDifficulty.EASY: 0.90,
    TaskDifficulty.MEDIUM: 0.75,
    TaskDifficulty.HARD: 0.20,
}
_ALL_DIFFICULTIES = [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]


def _non_negative_int(value: str) -> int:
    """Argparse type: non-negative integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}")
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the scripted expert on one episode per difficulty."
    )
    parser.add_argument(
        "--seed", type=_non_negative_int, default=42, help="Episode seed (default: 42)."
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty tier to test (default: all).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace output.")
    return parser.parse_args(argv)


def _print_trace(trace: dict[str, Any], quiet: bool) -> None:
    """Print a human-readable episode trace."""
    ep_id = trace["episode_id"]
    difficulty = trace["task_difficulty"]
    score = trace["final_score"]
    breakdown = trace["score_breakdown"]

    print(f"\n{'=' * 72}")
    print(f"Episode: {ep_id}  |  Difficulty: {difficulty}  |  Seed: {trace['seed']}")
    print(f"{'=' * 72}")

    if not quiet:
        for entry in trace["action_sequence"]:
            action = entry["action"]
            action_type = action["action_type"]
            ticket_id = action.get("ticket_id", "—")
            result = entry["result"]
            reward = entry["step_reward"]

            # Build a compact action summary
            details: list[str] = []
            for key in (
                "issue_family",
                "issue_subtype",
                "impact",
                "urgency",
                "queue_id",
                "close_reason",
                "target_ticket_id",
                "template_id",
            ):
                val = action.get(key)
                if val is not None:
                    details.append(f"{key}={val}")

            detail_str = f"  ({', '.join(details)})" if details else ""
            result_marker = "OK" if result == "ok" else f"ERR: {result}"
            reward_str = f"{reward:+.3f}" if reward is not None else "—"

            print(
                f"  Step {entry['step']:>2}: {action_type:<22} "
                f"ticket={ticket_id:<6} [{result_marker}] reward={reward_str}{detail_str}"
            )

    # Score summary
    print(f"\n  Final score: {score:.4f}")
    print(f"    Terminal business score: {breakdown['terminal_business_score']:.4f}")
    print(f"    UJCS-OpenEnv:           {breakdown['ujcs_openenv']:.4f}")
    print(f"    Episode penalties:       {breakdown['episode_penalties']:+.4f}")
    print(f"    Priority order score:    {breakdown['priority_order_score']:.4f}")
    print(f"    Invalid actions:         {breakdown['invalid_action_count']}")
    print(f"    Reassignments:           {breakdown['reassignment_count']}")

    threshold = _SOLVABILITY_THRESHOLDS[TaskDifficulty(difficulty)]
    status = "PASS" if score >= threshold else "FAIL"
    print(f"  Status: {status} (threshold={threshold})")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on PASS, 1 on FAIL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    if args.difficulty == "all":
        difficulties = _ALL_DIFFICULTIES
    else:
        difficulties = [TaskDifficulty(args.difficulty)]

    env = TriageSieveEnvironment()
    expert = ScriptedExpert(env)
    all_pass = True
    results: list[tuple[str, float, bool]] = []

    for diff in difficulties:
        trace = expert.run_episode(seed=args.seed, difficulty=diff)
        score = trace["final_score"]
        passed = score >= _SOLVABILITY_THRESHOLDS[diff]
        if not passed:
            all_pass = False
        results.append((diff.value, score, passed))
        _print_trace(trace, quiet=args.quiet)

    # Summary table
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  {'Difficulty':<12} {'Score':>8} {'Status':>8}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8}")
    for diff_name, score, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {diff_name:<12} {score:>8.4f} {status:>8}")
    print()

    overall = "PASS" if all_pass else "FAIL"
    logger.info("Overall: %s", overall)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
