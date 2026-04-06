"""Tests for data/archetypes.json schema and first 6 archetypes.

Validates:
- JSON schema structure (schema_version, archetypes list)
- Each archetype has all required keys
- Enum values match models.py definitions
- SOP graph structure (nodes, edges, entry/terminal)
- Variation parameters are non-empty lists
- Hidden truth fields are valid
- No priority field in hidden_truth (derived at runtime)
- No SLA fields in archetypes (derived from customer_tier + sla_rules.json)
- Deterministic rendering: same seed -> same output
- Graph connectivity: entry_node reachable, terminal_nodes exist
- All pattern placeholders resolvable from variation_parameters
"""

import json
import random
import string
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from ..models import (
    CustomerTier,
    IssueFamily,
    IssueSubtype,
    NonActionableSubtype,
    QueueId,
    Impact,
    SourceChannel,
    TaskDifficulty,
    TicketStatus,
    Urgency,
    VALID_FAMILY_SUBTYPES,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHETYPES_PATH = DATA_DIR / "archetypes.json"
TEMPLATES_PATH = DATA_DIR / "templates.json"

# All 18 archetypes per CLAUDE.md §20.
EXPECTED_ALL_18 = [
    "refund_missing_order_id",
    "failed_invoice_charge_dispute",
    "password_reset_account_lockout",
    "sso_login_issue",
    "api_key_failure",
    "integration_outage",
    "suspicious_login_security_review",
    "shipment_tracking_problem",
    "duplicate_complaint",
    "spam_marketing_email",
    "benign_expected_behavior",
    "automation_false_positive",
    "data_error_generated_ticket",
    "feature_request_misfiled_as_support",
    "org_wide_outage_report",
    "internal_escalation_from_sales",
    "entitlement_mismatch",
    "thread_conflicting_urgency",
]

REQUIRED_TOP_KEYS = {"schema_version", "archetypes"}

REQUIRED_ARCHETYPE_KEYS = {
    "archetype_id",
    "display_name",
    "difficulty",
    "visible_template",
    "hidden_truth",
    "sop_graph",
    "variation_parameters",
}

REQUIRED_VISIBLE_TEMPLATE_KEYS = {
    "subject_pattern",
    "body_pattern",
    "sender_pattern",
    "attachments",
    "thread_history_pattern",
    "internal_notes_pattern",
}

REQUIRED_HIDDEN_TRUTH_KEYS = {
    "customer_tier",
    "source_channel",
    "issue_family",
    "issue_subtype",
    "product_area",
    "impact",
    "urgency",
    "required_queue",
    "required_missing_fields",
    "escalation_required",
    "escalation_target",
    "is_duplicate",
    "duplicate_of",
    "correct_template_ids",
    "gold_terminal_status",
    "non_actionable_subtype",
}

REQUIRED_SOP_GRAPH_KEYS = {
    "graph_id",
    "nodes",
    "edges",
    "entry_node",
    "terminal_nodes",
}

REQUIRED_NODE_KEYS = {"id", "checkpoint"}
REQUIRED_EDGE_KEYS = {"from", "to", "guard"}

# Valid terminal statuses per state machine (CLAUDE.md §12)
TERMINAL_STATUSES = frozenset({"routed", "escalated", "merged", "closed"})

ALL_SPEC_IDS = {
    "refund_missing_order_id",
    "failed_invoice_charge_dispute",
    "password_reset_account_lockout",
    "sso_login_issue",
    "api_key_failure",
    "integration_outage",
    "suspicious_login_security_review",
    "shipment_tracking_problem",
    "duplicate_complaint",
    "spam_marketing_email",
    "benign_expected_behavior",
    "automation_false_positive",
    "data_error_generated_ticket",
    "feature_request_misfiled_as_support",
    "org_wide_outage_report",
    "internal_escalation_from_sales",
    "entitlement_mismatch",
    "thread_conflicting_urgency",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_all_pattern_text(vt: dict[str, Any]) -> str:
    """Concatenate all pattern strings from a visible_template for placeholder analysis."""
    parts = [vt["subject_pattern"], vt["body_pattern"], vt["sender_pattern"]]
    for entry in vt.get("thread_history_pattern", []):
        if isinstance(entry, dict):
            parts.append(entry.get("content_pattern", ""))
    for note in vt.get("internal_notes_pattern", []):
        if isinstance(note, str):
            parts.append(note)
    return " ".join(parts)


def _extract_placeholders(pattern: str) -> set[str]:
    """Extract all {placeholder} names from a format string."""
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(pattern)
        if field_name is not None
    }


def _collect_all_placeholders(vt: dict[str, Any]) -> set[str]:
    """Extract all unique placeholders across all patterns in a visible_template."""
    placeholders: set[str] = set()
    for pattern in [vt["subject_pattern"], vt["body_pattern"], vt["sender_pattern"]]:
        placeholders |= _extract_placeholders(pattern)
    for entry in vt.get("thread_history_pattern", []):
        if isinstance(entry, dict):
            placeholders |= _extract_placeholders(entry.get("content_pattern", ""))
    for note in vt.get("internal_notes_pattern", []):
        if isinstance(note, str):
            placeholders |= _extract_placeholders(note)
    return placeholders


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def archetypes_data() -> dict[str, Any]:
    """Load and return the full archetypes.json data."""
    assert ARCHETYPES_PATH.exists(), f"archetypes.json not found at {ARCHETYPES_PATH}"
    with open(ARCHETYPES_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def archetypes_list(archetypes_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the archetypes list from archetypes.json."""
    return archetypes_data["archetypes"]


@pytest.fixture(scope="module")
def templates_data() -> dict[str, Any]:
    """Load templates.json for cross-referencing correct_template_ids."""
    assert TEMPLATES_PATH.exists(), f"templates.json not found at {TEMPLATES_PATH}"
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def valid_template_ids(templates_data: dict[str, Any]) -> set[str]:
    """Set of all valid template_ids from templates.json."""
    return {t["template_id"] for t in templates_data["templates"]}


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------


class TestTopLevelStructure:
    """Tests for the top-level JSON structure."""

    def test_file_exists(self) -> None:
        assert ARCHETYPES_PATH.exists()

    def test_required_top_keys(self, archetypes_data: dict[str, Any]) -> None:
        missing = REQUIRED_TOP_KEYS - set(archetypes_data.keys())
        assert not missing, f"Top-level JSON missing required keys: {missing}"

    def test_schema_version(self, archetypes_data: dict[str, Any]) -> None:
        assert archetypes_data["schema_version"] == "1.0"

    def test_archetypes_is_list(self, archetypes_data: dict[str, Any]) -> None:
        assert isinstance(archetypes_data["archetypes"], list)

    def test_exactly_18_archetypes(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """All 18 archetypes per CLAUDE.md §20."""
        assert len(archetypes_list) == 18

    def test_expected_archetype_ids(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        ids = [a["archetype_id"] for a in archetypes_list]
        assert ids == EXPECTED_ALL_18

    def test_unique_archetype_ids(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        ids = [a["archetype_id"] for a in archetypes_list]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Per-archetype structure validation
# ---------------------------------------------------------------------------


class TestArchetypeStructure:
    """Tests that every archetype has the correct keys and types."""

    def test_required_keys(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            missing = REQUIRED_ARCHETYPE_KEYS - set(arch.keys())
            assert not missing, f"{arch.get('archetype_id', '?')}: missing keys {missing}"

    def test_difficulty_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in TaskDifficulty}
        for arch in archetypes_list:
            assert arch["difficulty"] in valid, (
                f"{arch['archetype_id']}: invalid difficulty {arch['difficulty']}"
            )

    def test_display_name_non_empty(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            assert isinstance(arch["display_name"], str) and len(arch["display_name"]) > 0


# ---------------------------------------------------------------------------
# Visible template validation
# ---------------------------------------------------------------------------


class TestVisibleTemplate:
    """Tests for the visible_template sub-object."""

    def test_required_keys(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            vt = arch["visible_template"]
            missing = REQUIRED_VISIBLE_TEMPLATE_KEYS - set(vt.keys())
            assert not missing, (
                f"{arch['archetype_id']}: missing visible_template keys {missing}"
            )

    def test_patterns_are_strings(self, archetypes_list: list[dict[str, Any]]) -> None:
        string_fields = {"subject_pattern", "body_pattern", "sender_pattern"}
        for arch in archetypes_list:
            vt = arch["visible_template"]
            for field in string_fields:
                assert isinstance(vt[field], str) and len(vt[field]) > 0, (
                    f"{arch['archetype_id']}: {field} must be a non-empty string"
                )

    def test_attachments_is_list(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            assert isinstance(arch["visible_template"]["attachments"], list)

    def test_thread_history_pattern_is_list(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            assert isinstance(arch["visible_template"]["thread_history_pattern"], list)

    def test_internal_notes_pattern_is_list(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            assert isinstance(arch["visible_template"]["internal_notes_pattern"], list)

    def test_all_placeholders_in_variation_params(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """Every {placeholder} in any pattern must have a matching variation_parameters key."""
        for arch in archetypes_list:
            vp_keys = set(arch["variation_parameters"].keys())
            placeholders = _collect_all_placeholders(arch["visible_template"])
            missing = placeholders - vp_keys
            assert not missing, (
                f"{arch['archetype_id']}: placeholders {missing} not in variation_parameters"
            )


# ---------------------------------------------------------------------------
# Hidden truth validation
# ---------------------------------------------------------------------------


class TestHiddenTruth:
    """Tests for the hidden_truth sub-object."""

    def test_required_keys(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            ht = arch["hidden_truth"]
            missing = REQUIRED_HIDDEN_TRUTH_KEYS - set(ht.keys())
            assert not missing, (
                f"{arch['archetype_id']}: missing hidden_truth keys {missing}"
            )

    def test_no_priority_field(self, archetypes_list: list[dict[str, Any]]) -> None:
        """Priority is derived at runtime from impact x urgency. Must not be stored."""
        for arch in archetypes_list:
            assert "priority" not in arch["hidden_truth"], (
                f"{arch['archetype_id']}: priority must not be in hidden_truth"
            )

    def test_no_sla_fields(self, archetypes_list: list[dict[str, Any]]) -> None:
        """SLA deadlines are derived from customer_tier + sla_rules.json."""
        for arch in archetypes_list:
            ht = arch["hidden_truth"]
            assert "sla_response_deadline" not in ht
            assert "sla_resolution_deadline" not in ht

    def test_customer_tier_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in CustomerTier}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["customer_tier"] in valid

    def test_source_channel_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in SourceChannel}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["source_channel"] in valid

    def test_issue_family_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in IssueFamily}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["issue_family"] in valid

    def test_issue_subtype_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in IssueSubtype}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["issue_subtype"] in valid

    def test_family_subtype_consistency(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """issue_subtype must belong to the declared issue_family."""
        for arch in archetypes_list:
            ht = arch["hidden_truth"]
            family = IssueFamily(ht["issue_family"])
            subtype = IssueSubtype(ht["issue_subtype"])
            assert subtype in VALID_FAMILY_SUBTYPES[family], (
                f"{arch['archetype_id']}: {subtype} not valid for family {family}"
            )

    def test_impact_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in Impact}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["impact"] in valid

    def test_urgency_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in Urgency}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["urgency"] in valid

    def test_required_queue_enum(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in QueueId}
        for arch in archetypes_list:
            assert arch["hidden_truth"]["required_queue"] in valid

    def test_gold_terminal_status_is_terminal(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """gold_terminal_status must be a valid terminal status per state machine."""
        for arch in archetypes_list:
            status = arch["hidden_truth"]["gold_terminal_status"]
            assert status in TERMINAL_STATUSES, (
                f"{arch['archetype_id']}: gold_terminal_status '{status}' "
                f"not in terminal statuses {TERMINAL_STATUSES}"
            )

    def test_required_missing_fields_is_list(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            assert isinstance(arch["hidden_truth"]["required_missing_fields"], list)

    def test_required_missing_fields_in_variation_params(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """Every required_missing_field must exist as a variation_parameters key."""
        for arch in archetypes_list:
            vp_keys = set(arch["variation_parameters"].keys())
            for field_name in arch["hidden_truth"]["required_missing_fields"]:
                assert field_name in vp_keys, (
                    f"{arch['archetype_id']}: required_missing_field '{field_name}' "
                    "not in variation_parameters"
                )

    def test_escalation_fields(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            ht = arch["hidden_truth"]
            assert isinstance(ht["escalation_required"], bool)
            if ht["escalation_required"]:
                valid = {e.value for e in QueueId}
                assert ht["escalation_target"] in valid
                assert ht["escalation_target"] != ht["required_queue"], (
                    f"{arch['archetype_id']}: escalation_target must differ from required_queue"
                )
            else:
                assert ht["escalation_target"] is None

    def test_duplicate_fields(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            ht = arch["hidden_truth"]
            assert isinstance(ht["is_duplicate"], bool)
            if ht["is_duplicate"]:
                assert isinstance(ht["duplicate_of"], str) and len(ht["duplicate_of"]) > 0, (
                    f"{arch['archetype_id']}: is_duplicate=True but duplicate_of is empty/null"
                )
            else:
                assert ht["duplicate_of"] is None

    def test_correct_template_ids_exist(
        self,
        archetypes_list: list[dict[str, Any]],
        valid_template_ids: set[str],
    ) -> None:
        """Every referenced template_id must exist in templates.json."""
        for arch in archetypes_list:
            for tid in arch["hidden_truth"]["correct_template_ids"]:
                assert tid in valid_template_ids, (
                    f"{arch['archetype_id']}: template_id '{tid}' not in templates.json"
                )

    def test_actionable_archetypes_have_templates(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """Actionable archetypes with gold_terminal_status='closed' must have templates."""
        for arch in archetypes_list:
            ht = arch["hidden_truth"]
            if ht["gold_terminal_status"] == "closed" and ht["non_actionable_subtype"] is None:
                assert len(ht["correct_template_ids"]) > 0, (
                    f"{arch['archetype_id']}: actionable closed ticket has empty "
                    "correct_template_ids"
                )

    def test_non_actionable_subtype(self, archetypes_list: list[dict[str, Any]]) -> None:
        valid = {e.value for e in NonActionableSubtype}
        for arch in archetypes_list:
            nas = arch["hidden_truth"]["non_actionable_subtype"]
            if nas is not None:
                assert nas in valid


# ---------------------------------------------------------------------------
# SOP graph validation
# ---------------------------------------------------------------------------


class TestSopGraph:
    """Tests for the sop_graph sub-object."""

    def test_required_keys(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            sg = arch["sop_graph"]
            missing = REQUIRED_SOP_GRAPH_KEYS - set(sg.keys())
            assert not missing, f"{arch['archetype_id']}: missing sop_graph keys {missing}"

    def test_graph_id_matches_archetype_id(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            assert arch["sop_graph"]["graph_id"] == arch["archetype_id"]

    def test_nodes_have_required_keys(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            for node in arch["sop_graph"]["nodes"]:
                missing = REQUIRED_NODE_KEYS - set(node.keys())
                assert not missing, (
                    f"{arch['archetype_id']}: node missing keys {missing}"
                )

    def test_edges_have_required_keys(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            for edge in arch["sop_graph"]["edges"]:
                missing = REQUIRED_EDGE_KEYS - set(edge.keys())
                assert not missing, (
                    f"{arch['archetype_id']}: edge missing keys {missing}"
                )

    def test_checkpoint_is_bool(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            for node in arch["sop_graph"]["nodes"]:
                assert isinstance(node["checkpoint"], bool)

    def test_at_least_one_checkpoint(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            checkpoints = [n for n in arch["sop_graph"]["nodes"] if n["checkpoint"]]
            assert len(checkpoints) >= 1, (
                f"{arch['archetype_id']}: must have at least one checkpoint node"
            )

    def test_entry_node_exists(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            sg = arch["sop_graph"]
            node_ids = {n["id"] for n in sg["nodes"]}
            assert sg["entry_node"] in node_ids

    def test_terminal_nodes_exist(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            sg = arch["sop_graph"]
            node_ids = {n["id"] for n in sg["nodes"]}
            for tn in sg["terminal_nodes"]:
                assert tn in node_ids

    def test_edge_references_valid_nodes(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            sg = arch["sop_graph"]
            node_ids = {n["id"] for n in sg["nodes"]}
            for edge in sg["edges"]:
                assert edge["from"] in node_ids, (
                    f"{arch['archetype_id']}: edge from '{edge['from']}' not in nodes"
                )
                assert edge["to"] in node_ids, (
                    f"{arch['archetype_id']}: edge to '{edge['to']}' not in nodes"
                )

    def test_terminal_nodes_have_no_outgoing_edges(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            sg = arch["sop_graph"]
            terminals = set(sg["terminal_nodes"])
            for edge in sg["edges"]:
                assert edge["from"] not in terminals, (
                    f"{arch['archetype_id']}: terminal node '{edge['from']}' has outgoing edge"
                )

    def test_entry_reachable_to_terminal(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """BFS from entry_node must reach at least one terminal_node."""
        for arch in archetypes_list:
            sg = arch["sop_graph"]
            adj: dict[str, list[str]] = {}
            for edge in sg["edges"]:
                adj.setdefault(edge["from"], []).append(edge["to"])
            visited: set[str] = set()
            queue: deque[str] = deque([sg["entry_node"]])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(adj.get(node, []))
            terminals = set(sg["terminal_nodes"])
            assert visited & terminals, (
                f"{arch['archetype_id']}: no path from entry to any terminal node"
            )


# ---------------------------------------------------------------------------
# Variation parameters
# ---------------------------------------------------------------------------


class TestVariationParameters:
    """Tests for variation_parameters."""

    def test_is_dict(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            assert isinstance(arch["variation_parameters"], dict)

    def test_non_empty(self, archetypes_list: list[dict[str, Any]]) -> None:
        for arch in archetypes_list:
            assert len(arch["variation_parameters"]) > 0, (
                f"{arch['archetype_id']}: variation_parameters must not be empty"
            )

    def test_all_values_are_non_empty_lists(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        for arch in archetypes_list:
            for key, val in arch["variation_parameters"].items():
                assert isinstance(val, list) and len(val) > 0, (
                    f"{arch['archetype_id']}: variation_parameters['{key}'] "
                    "must be a non-empty list"
                )

    def test_patterns_reference_variation_keys(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """At least one pattern in visible_template should reference a variation key."""
        for arch in archetypes_list:
            vp_keys = set(arch["variation_parameters"].keys())
            all_text = _collect_all_pattern_text(arch["visible_template"])
            referenced = {k for k in vp_keys if f"{{{k}}}" in all_text}
            assert len(referenced) > 0, (
                f"{arch['archetype_id']}: no variation parameter referenced in patterns"
            )

    def test_parallel_lower_lists_same_length(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """Lists like customer_name and customer_name_lower must have the same length."""
        for arch in archetypes_list:
            vp = arch["variation_parameters"]
            for key in list(vp.keys()):
                lower_key = f"{key}_lower"
                if lower_key in vp:
                    assert len(vp[key]) == len(vp[lower_key]), (
                        f"{arch['archetype_id']}: '{key}' ({len(vp[key])}) and "
                        f"'{lower_key}' ({len(vp[lower_key])}) have different lengths"
                    )


# ---------------------------------------------------------------------------
# Determinism: seeded rendering produces identical output
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests that seeded variation rendering is deterministic."""

    @staticmethod
    def _render_all_params(variation_params: dict[str, list[str]], seed: int) -> dict[str, str]:
        """Simulate deterministic selection of all variation parameters.

        Uses sorted key ordering for reproducible RNG consumption.
        The actual episode_engine must use the same ordering convention.
        """
        rng = random.Random(seed)
        return {k: rng.choice(v) for k, v in sorted(variation_params.items())}

    def test_same_seed_same_full_render(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """Full parameter dict must be identical for same seed."""
        for arch in archetypes_list:
            vp = arch["variation_parameters"]
            for seed in [42, 123, 999]:
                render1 = self._render_all_params(vp, seed)
                render2 = self._render_all_params(vp, seed)
                assert render1 == render2, (
                    f"{arch['archetype_id']}: non-deterministic render at seed {seed}"
                )

    def test_cross_seed_isolation(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """Rendering at seed A must not affect rendering at seed B."""
        for arch in archetypes_list:
            vp = arch["variation_parameters"]
            # Render seed 42 alone
            baseline = self._render_all_params(vp, 42)
            # Render seed 99 first (to consume RNG state), then 42 again
            self._render_all_params(vp, 99)
            after_interleave = self._render_all_params(vp, 42)
            assert baseline == after_interleave, (
                f"{arch['archetype_id']}: seed isolation broken"
            )

    def test_different_seeds_can_differ(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """With enough variation, different seeds should produce different results."""
        for arch in archetypes_list:
            vp = arch["variation_parameters"]
            total_combos = 1
            for vals in vp.values():
                total_combos *= len(vals)
            if total_combos < 3:
                continue
            results = set()
            for seed in range(20):
                params = self._render_all_params(vp, seed)
                results.add(tuple(sorted(params.items())))
            assert len(results) > 1, (
                f"{arch['archetype_id']}: all 20 seeds produced identical params"
            )


# ---------------------------------------------------------------------------
# Cross-archetype consistency
# ---------------------------------------------------------------------------


class TestCrossArchetypeConsistency:
    """Tests for consistency across all archetypes."""

    def test_all_difficulty_levels_present(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """First 6 should include easy, medium, and hard."""
        difficulties = {a["difficulty"] for a in archetypes_list}
        assert difficulties == {"easy", "medium", "hard"}, (
            f"Expected all three difficulty levels, got: {difficulties}"
        )

    def test_all_archetype_ids_from_spec(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """All IDs must come from the CLAUDE.md section 20 list."""
        for arch in archetypes_list:
            assert arch["archetype_id"] in ALL_SPEC_IDS, (
                f"{arch['archetype_id']} not in CLAUDE.md section 20 list"
            )

    def test_all_issue_families_covered(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """All 5 issue families should appear across the 18 archetypes."""
        families = {a["hidden_truth"]["issue_family"] for a in archetypes_list}
        expected = {e.value for e in IssueFamily}
        assert families == expected, f"Missing families: {expected - families}"

    def test_all_source_channels_covered(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """All 3 source channels should appear across the 18 archetypes."""
        channels = {a["hidden_truth"]["source_channel"] for a in archetypes_list}
        expected = {e.value for e in SourceChannel}
        assert channels == expected, f"Missing channels: {expected - channels}"

    def test_non_actionable_subtypes_covered(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """At least 4 of 5 non-actionable subtypes should appear (no_response_needed excluded for v1)."""
        subtypes = {
            a["hidden_truth"]["non_actionable_subtype"]
            for a in archetypes_list
            if a["hidden_truth"]["non_actionable_subtype"] is not None
        }
        required = {"spam_marketing", "benign_expected", "automation_false_positive", "data_error"}
        assert required <= subtypes, f"Missing non-actionable subtypes: {required - subtypes}"

    def test_duplicate_archetype_exists(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """At least one archetype must have is_duplicate=True."""
        assert any(a["hidden_truth"]["is_duplicate"] for a in archetypes_list)

    def test_escalation_archetype_exists(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """At least two archetypes should require escalation."""
        escalation_count = sum(
            1 for a in archetypes_list if a["hidden_truth"]["escalation_required"]
        )
        assert escalation_count >= 2

    def test_gated_queue_archetypes_exist(
        self, archetypes_list: list[dict[str, Any]]
    ) -> None:
        """At least one archetype should route to each gated queue."""
        gated_queues_used = set()
        for a in archetypes_list:
            ht = a["hidden_truth"]
            q = ht["required_queue"]
            if q in {"tech_support_l2", "security_team"}:
                gated_queues_used.add(q)
            if ht["escalation_target"] in {"tech_support_l2", "security_team"}:
                gated_queues_used.add(ht["escalation_target"])
        assert "tech_support_l2" in gated_queues_used
        assert "security_team" in gated_queues_used
