"""Tests for data/templates.json — validates schema, coverage, and deterministic usability."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEMPLATES_PATH = DATA_DIR / "templates.json"

VALID_CATEGORIES = {
    "request_info",
    "close_resolved",
    "close_non_actionable",
    "close_duplicate",
    "close_feature_request",
    "close_no_response",
}

ISSUE_FAMILIES = {"billing", "technical", "account", "security", "shipping"}

ISSUE_SUBTYPES = {
    "refund",
    "invoice_error",
    "failed_charge",
    "bug_report",
    "api_error",
    "integration_failure",
    "password_reset",
    "sso_issue",
    "account_lockout",
    "suspicious_login",
    "exposure_risk",
    "abuse_report",
    "delay",
    "tracking_problem",
    "lost_package",
}

CLOSE_REASONS = {"resolved", "duplicate", "non_actionable", "feature_request", "no_response"}


@pytest.fixture(scope="module")
def templates() -> list[dict]:
    """Load templates from JSON."""
    raw = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "Root must be an object with a 'templates' key"
    assert "templates" in raw, "Missing 'templates' key"
    return raw["templates"]


def test_file_exists():
    assert TEMPLATES_PATH.exists(), "templates.json must exist"


def test_non_empty(templates):
    assert len(templates) > 0, "Template list must not be empty"


def test_unique_ids(templates):
    ids = [t["template_id"] for t in templates]
    assert len(ids) == len(set(ids)), f"Duplicate template_ids: {[x for x in ids if ids.count(x) > 1]}"


def test_required_fields(templates):
    required = {"template_id", "name", "description", "category", "applies_to"}
    for t in templates:
        missing = required - set(t.keys())
        assert not missing, f"Template {t.get('template_id', '?')} missing fields: {missing}"


def test_valid_categories(templates):
    for t in templates:
        assert t["category"] in VALID_CATEGORIES, (
            f"Template {t['template_id']} has invalid category: {t['category']}"
        )


def test_applies_to_structure(templates):
    for t in templates:
        at = t["applies_to"]
        assert "issue_families" in at, f"{t['template_id']}: applies_to missing issue_families"
        assert "issue_subtypes" in at, f"{t['template_id']}: applies_to missing issue_subtypes"
        assert "close_reasons" in at or at.get("close_reasons") is None, (
            f"{t['template_id']}: applies_to missing close_reasons"
        )
        for fam in at["issue_families"]:
            assert fam in ISSUE_FAMILIES or fam == "*", (
                f"{t['template_id']}: invalid family {fam}"
            )
        for sub in at["issue_subtypes"]:
            assert sub in ISSUE_SUBTYPES or sub == "*", (
                f"{t['template_id']}: invalid subtype {sub}"
            )
        if at.get("close_reasons"):
            for cr in at["close_reasons"]:
                assert cr in CLOSE_REASONS, f"{t['template_id']}: invalid close_reason {cr}"


def test_request_info_templates_have_fields(templates):
    req_templates = [t for t in templates if t["category"] == "request_info"]
    assert len(req_templates) >= 1, "Must have at least one request_info template"
    for t in req_templates:
        assert t.get("requested_fields") and len(t["requested_fields"]) > 0, (
            f"request_info template {t['template_id']} must have non-empty requested_fields"
        )
        assert t.get("reply_body_template"), (
            f"request_info template {t['template_id']} must have reply_body_template"
        )


def test_close_templates_have_body(templates):
    close_templates = [t for t in templates if t["category"].startswith("close_")]
    assert len(close_templates) >= 1, "Must have at least one close template"
    for t in close_templates:
        assert t.get("close_body_template"), (
            f"close template {t['template_id']} must have close_body_template"
        )


def test_all_close_reasons_covered(templates):
    """Every CloseReason must have at least one template."""
    covered = set()
    for t in templates:
        if t["category"].startswith("close_") and t["applies_to"].get("close_reasons"):
            covered.update(t["applies_to"]["close_reasons"])
    assert CLOSE_REASONS <= covered, f"Uncovered close reasons: {CLOSE_REASONS - covered}"


def test_all_issue_families_have_close_resolved(templates):
    """Each issue family should have a close_resolved template."""
    resolved = [t for t in templates if t["category"] == "close_resolved"]
    covered_families = set()
    for t in resolved:
        for fam in t["applies_to"]["issue_families"]:
            covered_families.add(fam)
    assert ISSUE_FAMILIES <= covered_families, (
        f"Families without close_resolved template: {ISSUE_FAMILIES - covered_families}"
    )


def test_template_count_range(templates):
    """Expect 15-25 templates — compact but complete."""
    assert 15 <= len(templates) <= 25, f"Expected 15-25 templates, got {len(templates)}"
