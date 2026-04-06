"""Tests for scripts/generate_episodes.py -- CLI episode generator.

Covers:
- CLI argument parsing (--seed, --count, --difficulty, --output)
- JSONL output format and structure
- Determinism (same seed -> same output)
- Difficulty filtering ("all" -> None, specific -> TaskDifficulty enum)
- Integration with EpisodeEngine.render_episode
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "generate_episodes.py")


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke generate_episodes.py as a subprocess to test the CLI contract end-to-end."""
    return subprocess.run(  # noqa: S603 S607
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def run_cli_ok(*args: str) -> subprocess.CompletedProcess[str]:
    """Run CLI and assert success. Returns the CompletedProcess."""
    result = run_cli(*args)
    assert result.returncode == 0, f"CLI failed (rc={result.returncode}): {result.stderr}"
    return result


class TestCLIContract:
    """CLI accepts exactly --seed, --count, --difficulty, --output."""

    def test_missing_args_exits_nonzero(self) -> None:
        result = run_cli()
        assert result.returncode != 0

    def test_help_flag(self) -> None:
        result = run_cli("--help")
        assert result.returncode == 0
        for flag in ("--seed", "--count", "--difficulty", "--output"):
            assert flag in result.stdout

    def test_negative_count_exits_nonzero(self, tmp_path: Path) -> None:
        result = run_cli(
            "--seed", "1", "--count", "-5",
            "--difficulty", "easy", "--output", str(tmp_path / "out.jsonl"),
        )
        assert result.returncode != 0

    def test_negative_seed_exits_nonzero(self, tmp_path: Path) -> None:
        result = run_cli(
            "--seed", "-1", "--count", "1",
            "--difficulty", "easy", "--output", str(tmp_path / "out.jsonl"),
        )
        assert result.returncode != 0

    def test_invalid_difficulty_exits_nonzero(self, tmp_path: Path) -> None:
        result = run_cli(
            "--seed", "1", "--count", "1",
            "--difficulty", "impossible", "--output", str(tmp_path / "out.jsonl"),
        )
        assert result.returncode != 0


class TestJSONLOutput:
    """Output is valid JSONL with expected fields."""

    def test_generates_correct_count(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "42", "--count", "3",
            "--difficulty", "easy", "--output", str(out),
        )
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "10", "--count", "2",
            "--difficulty", "medium", "--output", str(out),
        )
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            episode = json.loads(line)
            assert isinstance(episode, dict)

    def test_episode_has_required_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "7", "--count", "1",
            "--difficulty", "easy", "--output", str(out),
        )
        episode = json.loads(out.read_text(encoding="utf-8").strip())
        required = {
            "episode_id", "seed", "task_difficulty",
            "tickets", "action_budget", "base_time",
        }
        assert required.issubset(episode.keys())

    def test_ticket_has_required_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "7", "--count", "1",
            "--difficulty", "easy", "--output", str(out),
        )
        episode = json.loads(out.read_text(encoding="utf-8").strip())
        ticket = episode["tickets"][0]
        required = {
            "ticket_id", "subject", "body", "sender_email", "received_at",
            "customer_tier", "source_channel", "has_attachment", "attachments",
            "thread_history", "internal_notes", "hidden_truth", "sop_graph",
            "follow_up_replies",
        }
        assert required.issubset(ticket.keys())

    def test_hidden_truth_has_required_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "7", "--count", "1",
            "--difficulty", "easy", "--output", str(out),
        )
        episode = json.loads(out.read_text(encoding="utf-8").strip())
        ht = episode["tickets"][0]["hidden_truth"]
        required = {
            "ticket_id", "customer_tier", "source_channel", "issue_family",
            "issue_subtype", "product_area", "impact", "urgency", "priority",
            "required_queue", "required_missing_fields", "escalation_required",
            "is_duplicate", "sla_response_deadline", "sla_resolution_deadline",
            "policy_graph_id", "correct_template_ids", "gold_terminal_status",
        }
        assert required.issubset(ht.keys())

    def test_enum_values_are_strings(self, tmp_path: Path) -> None:
        """Enum fields serialize as plain strings, not EnumClass.VALUE."""
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "7", "--count", "1",
            "--difficulty", "easy", "--output", str(out),
        )
        episode = json.loads(out.read_text(encoding="utf-8").strip())
        assert isinstance(episode["task_difficulty"], str)
        assert "." not in episode["task_difficulty"]
        ticket = episode["tickets"][0]
        assert isinstance(ticket["customer_tier"], str)
        assert "." not in ticket["customer_tier"]


class TestDeterminism:
    """Same seed + count + difficulty -> identical JSONL output."""

    def test_same_seed_same_output(self, tmp_path: Path) -> None:
        out1 = tmp_path / "ep1.jsonl"
        out2 = tmp_path / "ep2.jsonl"
        args = ["--seed", "42", "--count", "5", "--difficulty", "all"]
        run_cli_ok(*args, "--output", str(out1))
        run_cli_ok(*args, "--output", str(out2))
        assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    def test_different_seed_different_output(self, tmp_path: Path) -> None:
        out1 = tmp_path / "ep1.jsonl"
        out2 = tmp_path / "ep2.jsonl"
        run_cli_ok(
            "--seed", "1", "--count", "3",
            "--difficulty", "all", "--output", str(out1),
        )
        run_cli_ok(
            "--seed", "2", "--count", "3",
            "--difficulty", "all", "--output", str(out2),
        )
        ep1 = json.loads(out1.read_text(encoding="utf-8").splitlines()[0])
        ep2 = json.loads(out2.read_text(encoding="utf-8").splitlines()[0])
        assert ep1["seed"] != ep2["seed"]


class TestDifficultyFilter:
    """--difficulty flag correctly filters or passes None."""

    def test_easy_produces_easy_episodes(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "42", "--count", "3",
            "--difficulty", "easy", "--output", str(out),
        )
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            ep = json.loads(line)
            assert ep["task_difficulty"] == "easy"

    def test_medium_produces_medium_episodes(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "42", "--count", "3",
            "--difficulty", "medium", "--output", str(out),
        )
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            ep = json.loads(line)
            assert ep["task_difficulty"] == "medium"

    def test_hard_produces_hard_episodes(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "42", "--count", "3",
            "--difficulty", "hard", "--output", str(out),
        )
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            ep = json.loads(line)
            assert ep["task_difficulty"] == "hard"

    def test_all_can_produce_mixed_difficulties(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "42", "--count", "50",
            "--difficulty", "all", "--output", str(out),
        )
        difficulties = set()
        for line in out.read_text(encoding="utf-8").strip().splitlines():
            ep = json.loads(line)
            difficulties.add(ep["task_difficulty"])
        assert len(difficulties) >= 2


class TestSeedStrategy:
    """Episode i uses seed = master_seed + i."""

    def test_seeds_are_sequential(self, tmp_path: Path) -> None:
        out = tmp_path / "episodes.jsonl"
        run_cli_ok(
            "--seed", "100", "--count", "3",
            "--difficulty", "easy", "--output", str(out),
        )
        episodes = [json.loads(line) for line in out.read_text(encoding="utf-8").strip().splitlines()]
        assert episodes[0]["seed"] == 100
        assert episodes[1]["seed"] == 101
        assert episodes[2]["seed"] == 102
