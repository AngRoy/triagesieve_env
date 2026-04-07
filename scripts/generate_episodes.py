"""CLI episode generator.

Usage:
    python scripts/generate_episodes.py --seed 42 --count 100 --difficulty all --output data/seeded_episodes.jsonl

Generates deterministic episode instances from archetypes. Same (archetype_id, seed) always
produces an identical episode. Delegates to EpisodeEngine.render_episode — no rendering logic
is duplicated here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

from triagesieve_env.models import TaskDifficulty
from triagesieve_env.server.episode_engine import EpisodeEngine, RenderedEpisode

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Argparse type: positive integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}") from None
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return n


def _non_negative_int(value: str) -> int:
    """Argparse type: non-negative integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a non-negative integer, got {value!r}"
        ) from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}")
    return n


def _enum_serializer(obj: Any) -> Any:
    """Convert enum values to their string representation for JSON."""
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _episode_to_dict(episode: RenderedEpisode) -> dict[str, Any]:
    """Serialize a RenderedEpisode (frozen dataclass with enum fields) to a plain dict.

    Uses a json round-trip to coerce all nested Enum fields via _enum_serializer,
    avoiding a manual recursive walk.
    """
    raw = dataclasses.asdict(episode)
    return json.loads(json.dumps(raw, default=_enum_serializer))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic episode instances from archetypes."
    )
    parser.add_argument(
        "--seed", type=_non_negative_int, required=True, help="Master seed for generation."
    )
    parser.add_argument(
        "--count", type=_positive_int, required=True, help="Number of episodes to generate."
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        required=True,
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty filter. 'all' cycles through easy/medium/hard evenly.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the episode generator CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    fixed_difficulty: TaskDifficulty | None
    if args.difficulty == "all":
        fixed_difficulty = None
    else:
        fixed_difficulty = TaskDifficulty(args.difficulty)

    # When --difficulty all, rotate through difficulties explicitly rather than
    # passing None to render_episode. This ensures the stored task_difficulty
    # matches what render_episode(seed, explicit_difficulty) would produce, so
    # the determinism check in validate_episode_bank passes.
    _difficulty_cycle = [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]

    engine = EpisodeEngine()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.warning("Output file %s already exists and will be overwritten.", output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(args.count):
            episode_seed = args.seed + i
            if fixed_difficulty is None:
                episode_difficulty = _difficulty_cycle[i % len(_difficulty_cycle)]
            else:
                episode_difficulty = fixed_difficulty
            episode = engine.render_episode(seed=episode_seed, difficulty=episode_difficulty)
            line = json.dumps(_episode_to_dict(episode), separators=(",", ":"))  # compact JSONL
            f.write(line + "\n")

    logger.info("Wrote %d episodes to %s", args.count, output_path)


if __name__ == "__main__":
    main()
