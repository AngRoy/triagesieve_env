"""Tests for scripts/validate_episode_bank.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..scripts.validate_episode_bank import (
    _parse_args,
    _pass_determinism,
    _pass_parse,
    _pass_solvability,
    main,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self) -> None:
        args = _parse_args(["--input", "data/seeded_episodes.jsonl"])
        assert args.input == "data/seeded_episodes.jsonl"
        assert args.verbose is False

    def test_verbose_flag(self) -> None:
        args = _parse_args(["--input", "x.jsonl", "--verbose"])
        assert args.verbose is True

    def test_missing_input_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([])


# ---------------------------------------------------------------------------
# Pass 1 — parse (unit tests)
# ---------------------------------------------------------------------------


class TestPassParse:
    def test_valid_episode(self) -> None:
        episodes, errors = _pass_parse([_make_minimal_episode_line()])
        assert len(episodes) == 1
        assert errors == []

    def test_empty_bank(self) -> None:
        episodes, errors = _pass_parse([])
        assert len(episodes) == 0
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_blank_lines_skipped(self) -> None:
        episodes, errors = _pass_parse(["", "  ", _make_minimal_episode_line()])
        assert len(episodes) == 1
        assert errors == []

    def test_invalid_json(self) -> None:
        episodes, errors = _pass_parse(["not json"])
        assert len(episodes) == 0
        assert len(errors) == 1
        assert "invalid JSON" in errors[0]

    def test_missing_keys(self) -> None:
        episodes, errors = _pass_parse([json.dumps({"episode_id": "x"})])
        assert len(episodes) == 0
        assert len(errors) == 1
        assert "missing keys" in errors[0]

    def test_zero_tickets(self) -> None:
        ep = json.loads(_make_minimal_episode_line())
        ep["tickets"] = []
        episodes, errors = _pass_parse([json.dumps(ep)])
        assert len(episodes) == 0
        assert len(errors) == 1
        assert "zero tickets" in errors[0]


# ---------------------------------------------------------------------------
# Pass 2 — determinism (unit test with real engine)
# ---------------------------------------------------------------------------


class TestPassDeterminism:
    def test_real_episode_deterministic(self, generated_bank_episodes: list[dict]) -> None:
        """First episode from real bank should pass determinism."""
        errors = _pass_determinism(generated_bank_episodes[:1], max_check=1)
        assert errors == [], f"Determinism errors: {errors}"


# ---------------------------------------------------------------------------
# Pass 3 — solvability
# ---------------------------------------------------------------------------


class TestPassSolvability:
    def test_solvability_returns_scores(self, generated_bank_episodes: list[dict]) -> None:
        """Solvability pass returns a score per episode and error details."""
        scores, errors = _pass_solvability(generated_bank_episodes[:1])
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert 0.0 <= scores[0] <= 1.0
        # Errors are strings describing failures (may be empty if episode passes)
        assert all(isinstance(e, str) for e in errors)


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_missing_file(self) -> None:
        exit_code = main(["--input", "/tmp/does_not_exist_12345.jsonl"])
        assert exit_code == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        bank = tmp_path / "empty.jsonl"
        bank.write_text("")
        exit_code = main(["--input", str(bank)])
        assert exit_code == 1

    def test_invalid_json_file(self, tmp_path: Path) -> None:
        bank = tmp_path / "bad.jsonl"
        bank.write_text("not json\n")
        exit_code = main(["--input", str(bank)])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generated_bank_episodes() -> list[dict]:
    """Load parsed episodes from the real bank."""
    bank = Path(__file__).resolve().parent.parent / "data" / "seeded_episodes.jsonl"
    if not bank.exists() or bank.stat().st_size == 0:
        pytest.skip("seeded_episodes.jsonl is empty; run generate_episodes.py first")
    episodes = []
    for line in bank.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            episodes.append(json.loads(line))
    if not episodes:
        pytest.skip("seeded_episodes.jsonl has no valid episodes")
    return episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_episode_line() -> str:
    """Minimal valid episode JSON (passes parse, NOT determinism/solvability)."""
    return json.dumps(
        {
            "episode_id": "ep_test_001",
            "seed": 42,
            "task_difficulty": "easy",
            "tickets": [{"ticket_id": "T001", "subject": "Test"}],
            "action_budget": 4,
            "base_time": "2026-04-05T08:00:00+00:00",
        }
    )
