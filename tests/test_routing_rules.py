"""Tests for data/routing_rules.json — validates schema, queue coverage, and gated prerequisites."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROUTING_RULES_PATH = DATA_DIR / "routing_rules.json"

QUEUE_IDS = {
    "billing_team",
    "tech_support_l1",
    "tech_support_l2",
    "account_team",
    "security_team",
    "shipping_team",
    "refund_team",
    "spam_filter",
    "sales_or_feature_requests",
}

GATED_QUEUE_IDS = {"tech_support_l2", "security_team"}

ISSUE_FAMILIES = {"billing", "technical", "account", "security", "shipping"}

REQUIRED_ENTRY_FIELDS = {"description", "gated", "prerequisites", "handles_families"}


@pytest.fixture(scope="module")
def routing_rules() -> dict:
    """Load routing rules from JSON."""
    raw = json.loads(ROUTING_RULES_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "Root must be an object"
    assert "queues" in raw, "Missing 'queues' key"
    return raw["queues"]


def test_file_exists():
    assert ROUTING_RULES_PATH.exists(), "routing_rules.json must exist"


def test_all_queues_present(routing_rules):
    assert set(routing_rules.keys()) == QUEUE_IDS, (
        f"Queue mismatch. Missing: {QUEUE_IDS - set(routing_rules.keys())}. "
        f"Extra: {set(routing_rules.keys()) - QUEUE_IDS}"
    )


def test_required_fields(routing_rules):
    for qid, entry in routing_rules.items():
        missing = REQUIRED_ENTRY_FIELDS - set(entry.keys())
        assert not missing, f"Queue {qid} missing fields: {missing}"


def test_gated_flag_matches(routing_rules):
    for qid, entry in routing_rules.items():
        if qid in GATED_QUEUE_IDS:
            assert entry["gated"] is True, f"{qid} must be gated"
            assert len(entry["prerequisites"]) > 0, f"Gated queue {qid} must have prerequisites"
        else:
            assert entry["gated"] is False, f"{qid} must not be gated"
            assert entry["prerequisites"] == [], f"Non-gated queue {qid} must have empty prerequisites"


def test_handles_families_valid(routing_rules):
    for qid, entry in routing_rules.items():
        for fam in entry["handles_families"]:
            assert fam in ISSUE_FAMILIES, f"{qid}: invalid family '{fam}'"


def test_every_family_handled(routing_rules):
    """Every issue family must be handled by at least one queue."""
    covered = set()
    for entry in routing_rules.values():
        covered.update(entry["handles_families"])
    assert ISSUE_FAMILIES <= covered, f"Unhandled families: {ISSUE_FAMILIES - covered}"


def test_description_non_empty(routing_rules):
    for qid, entry in routing_rules.items():
        assert isinstance(entry["description"], str) and len(entry["description"]) > 0, (
            f"{qid}: description must be a non-empty string"
        )


def test_prerequisites_are_strings(routing_rules):
    for qid, entry in routing_rules.items():
        assert isinstance(entry["prerequisites"], list), f"{qid}: prerequisites must be a list"
        for p in entry["prerequisites"]:
            assert isinstance(p, str), f"{qid}: prerequisite must be a string, got {type(p)}"


def test_no_extra_top_level_keys():
    """Root object should only have 'queues' key."""
    raw = json.loads(ROUTING_RULES_PATH.read_text(encoding="utf-8"))
    assert set(raw.keys()) == {"queues"}, f"Unexpected top-level keys: {set(raw.keys()) - {'queues'}}"
