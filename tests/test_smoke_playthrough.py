"""Tests for scripts/smoke_playthrough.py."""

from __future__ import annotations

import pytest

from ..scripts.smoke_playthrough import main, _parse_args

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self) -> None:
        args = _parse_args([])
        assert args.seed == 42
        assert args.difficulty == "all"
        assert args.quiet is False

    def test_custom_seed(self) -> None:
        args = _parse_args(["--seed", "100"])
        assert args.seed == 100

    def test_single_difficulty(self) -> None:
        args = _parse_args(["--difficulty", "easy"])
        assert args.difficulty == "easy"

    def test_quiet_flag(self) -> None:
        args = _parse_args(["--quiet"])
        assert args.quiet is True

    def test_invalid_difficulty_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--difficulty", "impossible"])

    def test_negative_seed_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--seed", "-1"])

    def test_non_integer_seed_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--seed", "abc"])


# ---------------------------------------------------------------------------
# Full run — these test that the script executes without crashing and
# produces a deterministic exit code on a fixed seed.
# ---------------------------------------------------------------------------


class TestSmokeRun:
    def test_single_easy_completes(self) -> None:
        """Script completes without exception on a single easy episode."""
        exit_code = main(["--seed", "42", "--difficulty", "easy"])
        assert isinstance(exit_code, int)

    def test_quiet_mode_completes(self) -> None:
        """Quiet mode runs without exception."""
        exit_code = main(["--seed", "42", "--difficulty", "easy", "--quiet"])
        assert isinstance(exit_code, int)

    def test_all_difficulties_complete(self) -> None:
        """All difficulties complete without exception."""
        exit_code = main(["--seed", "42", "--difficulty", "all"])
        assert isinstance(exit_code, int)

    def test_deterministic_exit_code(self) -> None:
        """Same seed + difficulty always produces the same exit code."""
        code1 = main(["--seed", "99", "--difficulty", "easy", "--quiet"])
        code2 = main(["--seed", "99", "--difficulty", "easy", "--quiet"])
        assert code1 == code2
