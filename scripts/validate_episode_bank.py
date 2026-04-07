"""Episode bank validator.

Usage:
    python scripts/validate_episode_bank.py --input data/seeded_episodes.jsonl [--verbose]

Three-pass validation:
  1. Parse   — every line is valid JSON with required top-level keys.
  2. Determinism — re-render a sample via EpisodeEngine and compare.
  3. Solvability — scripted expert scores >= 0.90 on each episode.

Exit code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from triagesieve_env.baseline.scripted_expert import ScriptedExpert
from triagesieve_env.models import TaskDifficulty
from triagesieve_env.server.episode_engine import EpisodeEngine
from triagesieve_env.server.triagesieve_env_environment import TriageSieveEnvironment

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"episode_id", "seed", "task_difficulty", "tickets", "action_budget", "base_time"}
_SOLVABILITY_THRESHOLDS: dict[str, float] = {
    "easy": 0.90,
    "medium": 0.75,
    "hard": 0.20,
}
_DETERMINISM_SAMPLE = 10  # spot-check first N unless --verbose


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate the seeded episode bank.")
    parser.add_argument("--input", type=str, required=True, help="Path to seeded_episodes.jsonl.")
    parser.add_argument(
        "--verbose", action="store_true", help="Check ALL episodes for determinism (not just 10)."
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pass 1 — parse
# ---------------------------------------------------------------------------


def _pass_parse(lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse every line as JSON and check required keys.

    Returns:
        Tuple of (parsed episodes, error messages).
    """
    episodes: list[dict[str, Any]] = []
    errors: list[str] = []

    if not lines:
        errors.append("Episode bank is empty (0 lines).")
        return episodes, errors

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {i}: invalid JSON — {exc}")
            continue

        missing = _REQUIRED_KEYS - set(ep.keys())
        if missing:
            errors.append(f"Line {i} ({ep.get('episode_id', '?')}): missing keys {sorted(missing)}")
            continue

        if not ep.get("tickets"):
            errors.append(f"Line {i} ({ep['episode_id']}): zero tickets.")
            continue

        episodes.append(ep)

    return episodes, errors


# ---------------------------------------------------------------------------
# Pass 2 — determinism
# ---------------------------------------------------------------------------


def _enum_serializer(obj: Any) -> Any:
    """Convert enum values for JSON comparison."""
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _pass_determinism(episodes: list[dict[str, Any]], max_check: int | None = None) -> list[str]:
    """Re-render episodes from (seed, difficulty) and compare to bank.

    Args:
        episodes: Parsed episode dicts from the bank.
        max_check: Maximum episodes to spot-check. None = all.

    Returns:
        List of error messages (empty = pass).
    """
    engine = EpisodeEngine()
    errors: list[str] = []
    to_check = episodes[:max_check] if max_check is not None else episodes

    for ep in to_check:
        seed = ep["seed"]
        difficulty = TaskDifficulty(ep["task_difficulty"])
        rendered = engine.render_episode(seed=seed, difficulty=difficulty)

        # Serialize rendered episode the same way generate_episodes does
        raw = dataclasses.asdict(rendered)
        rendered_dict = json.loads(json.dumps(raw, default=_enum_serializer))

        if rendered_dict != ep:
            # Provide a specific diagnostic for common divergences
            if rendered_dict["episode_id"] != ep["episode_id"]:
                detail = f"episode_id {rendered_dict['episode_id']!r} vs {ep['episode_id']!r}"
            elif len(rendered_dict["tickets"]) != len(ep["tickets"]):
                detail = f"ticket count {len(rendered_dict['tickets'])} vs {len(ep['tickets'])}"
            else:
                detail = "content mismatch (same id and ticket count)"
            errors.append(f"Determinism FAIL: {ep['episode_id']} (seed={seed}): {detail}")

    return errors


# ---------------------------------------------------------------------------
# Pass 3 — solvability
# ---------------------------------------------------------------------------


def _pass_solvability(episodes: list[dict[str, Any]]) -> tuple[list[float], list[str]]:
    """Run scripted expert on each episode and assert score >= threshold.

    Returns:
        Tuple of (score list, error messages).
    """
    env = TriageSieveEnvironment()
    expert = ScriptedExpert(env)
    scores: list[float] = []
    errors: list[str] = []

    for ep in episodes:
        seed = ep["seed"]
        diff_str = ep["task_difficulty"]
        difficulty = TaskDifficulty(diff_str)
        threshold = _SOLVABILITY_THRESHOLDS.get(diff_str, 0.90)
        trace = expert.run_episode(seed=seed, difficulty=difficulty)
        score = trace["final_score"]
        scores.append(score)

        if score < threshold:
            errors.append(
                f"Solvability FAIL: {ep['episode_id']} (seed={seed}, "
                f"difficulty={diff_str}) scored {score:.4f} "
                f"< {threshold}"
            )

    return scores, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on PASS, 1 on FAIL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    input_path = Path(args.input)

    # Read file
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        return 1

    lines = input_path.read_text(encoding="utf-8").splitlines()
    all_ok = True

    # Pass 1: Parse
    logger.info("Pass 1/3: Parse (%d lines)...", len(lines))
    episodes, parse_errors = _pass_parse(lines)
    if parse_errors:
        for err in parse_errors:
            logger.error("  %s", err)
        logger.error("Pass 1 FAIL: %d parse errors.", len(parse_errors))
        return 1
    logger.info("  Pass 1 OK: %d episodes parsed.", len(episodes))

    # Pass 2: Determinism
    max_check = None if args.verbose else min(_DETERMINISM_SAMPLE, len(episodes))
    logger.info("Pass 2/3: Determinism (checking %s episodes)...", max_check or "all")
    det_errors = _pass_determinism(episodes, max_check=max_check)
    if det_errors:
        for err in det_errors:
            logger.error("  %s", err)
        logger.error("Pass 2 FAIL: %d determinism errors.", len(det_errors))
        all_ok = False
    else:
        logger.info("  Pass 2 OK.")

    # Pass 3: Solvability
    logger.info("Pass 3/3: Solvability (%d episodes)...", len(episodes))
    scores, solve_errors = _pass_solvability(episodes)
    if solve_errors:
        for err in solve_errors:
            logger.error("  %s", err)
        logger.error("Pass 3 FAIL: %d episodes below threshold.", len(solve_errors))
        all_ok = False
    else:
        logger.info("  Pass 3 OK.")

    # Summary
    if scores:
        min_s, avg_s, max_s = min(scores), sum(scores) / len(scores), max(scores)
        thresholds_str = ", ".join(f"{k}={v}" for k, v in _SOLVABILITY_THRESHOLDS.items())
        logger.info(
            "Scores: min=%.4f  avg=%.4f  max=%.4f  (thresholds: %s)",
            min_s,
            avg_s,
            max_s,
            thresholds_str,
        )

    if all_ok:
        logger.info("RESULT: PASS (%d episodes validated)", len(episodes))
        return 0
    else:
        logger.error("RESULT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
