"""Tests for server/policy_graph.py — SOP DAG representation, tracking, and UJCS scoring.

Covers:
- Data structure construction (SOPNode, SOPEdge, SOPGraph, TicketGuardContext, SOPTracker)
- Graph loading from archetype dicts + validation
- Guard evaluation
- Transition queries
- Action-to-node mapping
- SOP tracker advancement (including auto-advance)
- UJCS computation
- Batch loading from real archetypes.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ..models import ActionType
from ..server.policy_graph import (
    GUARD_EVALUATORS,
    SOPEdge,
    SOPGraph,
    SOPNode,
    SOPScoringData,
    SOPTracker,
    TicketGuardContext,
    compute_ujcs,
    load_sop_graphs,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic graph
# ---------------------------------------------------------------------------

SIMPLE_GRAPH_DATA: dict[str, Any] = {
    "graph_id": "test_simple",
    "nodes": [
        {"id": "new", "checkpoint": False},
        {"id": "open", "checkpoint": False},
        {"id": "classify_billing_refund", "checkpoint": True},
        {"id": "set_impact_urgency", "checkpoint": True},
        {"id": "route_billing_team", "checkpoint": True},
        {"id": "close_resolved", "checkpoint": True},
    ],
    "edges": [
        {"from": "new", "to": "open", "guard": None},
        {"from": "open", "to": "classify_billing_refund", "guard": None},
        {"from": "classify_billing_refund", "to": "set_impact_urgency", "guard": None},
        {"from": "set_impact_urgency", "to": "route_billing_team", "guard": None},
        {"from": "route_billing_team", "to": "close_resolved", "guard": None},
    ],
    "entry_node": "new",
    "terminal_nodes": ["close_resolved"],
}


GUARDED_GRAPH_DATA: dict[str, Any] = {
    "graph_id": "test_guarded",
    "nodes": [
        {"id": "new", "checkpoint": False},
        {"id": "open", "checkpoint": False},
        {"id": "classify_billing_refund", "checkpoint": True},
        {"id": "set_impact_urgency", "checkpoint": True},
        {"id": "request_info_order_id", "checkpoint": True},
        {"id": "receive_reply", "checkpoint": False},
        {"id": "route_refund_team", "checkpoint": True},
        {"id": "close_resolved", "checkpoint": True},
    ],
    "edges": [
        {"from": "new", "to": "open", "guard": None},
        {"from": "open", "to": "classify_billing_refund", "guard": None},
        {"from": "classify_billing_refund", "to": "set_impact_urgency", "guard": None},
        {"from": "set_impact_urgency", "to": "request_info_order_id", "guard": "missing_order_id"},
        {"from": "request_info_order_id", "to": "receive_reply", "guard": None},
        {"from": "receive_reply", "to": "route_refund_team", "guard": "info_received"},
        {"from": "route_refund_team", "to": "close_resolved", "guard": None},
    ],
    "entry_node": "new",
    "terminal_nodes": ["close_resolved"],
}


@pytest.fixture
def simple_graph() -> SOPGraph:
    return SOPGraph.from_archetype_data(SIMPLE_GRAPH_DATA)


@pytest.fixture
def guarded_graph() -> SOPGraph:
    return SOPGraph.from_archetype_data(GUARDED_GRAPH_DATA)


@pytest.fixture
def default_ctx() -> TicketGuardContext:
    """Guard context with nothing set."""
    return TicketGuardContext()


@pytest.fixture
def full_ctx() -> TicketGuardContext:
    """Guard context with everything set."""
    return TicketGuardContext(
        classification_set=True,
        impact_urgency_set=True,
        missing_fields_requested=True,
        info_received=True,
        escalation_required=True,
        duplicate_confirmed=True,
    )


# =========================================================================
# 1. Data Structure Construction
# =========================================================================


class TestSOPNode:
    def test_construction(self) -> None:
        node = SOPNode(node_id="classify_billing", is_checkpoint=True)
        assert node.node_id == "classify_billing"
        assert node.is_checkpoint is True

    def test_frozen(self) -> None:
        node = SOPNode(node_id="x", is_checkpoint=False)
        with pytest.raises(AttributeError):
            node.node_id = "y"  # type: ignore[misc]


class TestSOPEdge:
    def test_construction(self) -> None:
        edge = SOPEdge(from_node="a", to_node="b", guard="info_received")
        assert edge.from_node == "a"
        assert edge.to_node == "b"
        assert edge.guard == "info_received"

    def test_no_guard(self) -> None:
        edge = SOPEdge(from_node="a", to_node="b", guard=None)
        assert edge.guard is None

    def test_frozen(self) -> None:
        edge = SOPEdge(from_node="a", to_node="b", guard=None)
        with pytest.raises(AttributeError):
            edge.guard = "x"  # type: ignore[misc]


class TestTicketGuardContext:
    def test_defaults_all_false(self) -> None:
        ctx = TicketGuardContext()
        assert ctx.classification_set is False
        assert ctx.impact_urgency_set is False
        assert ctx.missing_fields_requested is False
        assert ctx.info_received is False
        assert ctx.escalation_required is False
        assert ctx.duplicate_confirmed is False

    def test_frozen(self) -> None:
        ctx = TicketGuardContext()
        with pytest.raises(AttributeError):
            ctx.classification_set = True  # type: ignore[misc]


# =========================================================================
# 2. SOPGraph Construction + Validation
# =========================================================================


class TestSOPGraphConstruction:
    def test_from_archetype_data_simple(self, simple_graph: SOPGraph) -> None:
        assert simple_graph.graph_id == "test_simple"
        assert len(simple_graph.nodes) == 6
        assert len(simple_graph.edges) == 5
        assert simple_graph.entry_node == "new"
        assert simple_graph.terminal_nodes == frozenset({"close_resolved"})

    def test_node_index(self, simple_graph: SOPGraph) -> None:
        assert "new" in simple_graph.node_index
        assert "close_resolved" in simple_graph.node_index
        assert simple_graph.node_index["classify_billing_refund"].is_checkpoint is True
        assert simple_graph.node_index["new"].is_checkpoint is False

    def test_adjacency(self, simple_graph: SOPGraph) -> None:
        adj = simple_graph.adjacency
        assert len(adj["new"]) == 1
        assert adj["new"][0].to_node == "open"
        # Terminal node has no outgoing edges
        assert len(adj.get("close_resolved", [])) == 0

    def test_checkpoints(self, simple_graph: SOPGraph) -> None:
        expected = frozenset({
            "classify_billing_refund",
            "set_impact_urgency",
            "route_billing_team",
            "close_resolved",
        })
        assert simple_graph.checkpoints == expected

    def test_gold_path(self, simple_graph: SOPGraph) -> None:
        expected = (
            "new",
            "open",
            "classify_billing_refund",
            "set_impact_urgency",
            "route_billing_team",
            "close_resolved",
        )
        assert simple_graph.gold_path == expected

    def test_gold_path_guarded(self, guarded_graph: SOPGraph) -> None:
        expected = (
            "new",
            "open",
            "classify_billing_refund",
            "set_impact_urgency",
            "request_info_order_id",
            "receive_reply",
            "route_refund_team",
            "close_resolved",
        )
        assert guarded_graph.gold_path == expected


class TestSOPGraphValidation:
    def test_missing_entry_node(self) -> None:
        data = {
            "graph_id": "bad",
            "nodes": [{"id": "a", "checkpoint": False}],
            "edges": [],
            "entry_node": "nonexistent",
            "terminal_nodes": ["a"],
        }
        with pytest.raises(ValueError, match="entry_node"):
            SOPGraph.from_archetype_data(data)

    def test_missing_terminal_node(self) -> None:
        data = {
            "graph_id": "bad",
            "nodes": [{"id": "a", "checkpoint": False}],
            "edges": [],
            "entry_node": "a",
            "terminal_nodes": ["nonexistent"],
        }
        with pytest.raises(ValueError, match="terminal"):
            SOPGraph.from_archetype_data(data)

    def test_edge_references_missing_node(self) -> None:
        data = {
            "graph_id": "bad",
            "nodes": [
                {"id": "a", "checkpoint": False},
                {"id": "b", "checkpoint": False},
            ],
            "edges": [{"from": "a", "to": "ghost", "guard": None}],
            "entry_node": "a",
            "terminal_nodes": ["b"],
        }
        with pytest.raises(ValueError, match="ghost"):
            SOPGraph.from_archetype_data(data)

    def test_cycle_detection(self) -> None:
        data = {
            "graph_id": "cycle",
            "nodes": [
                {"id": "a", "checkpoint": False},
                {"id": "b", "checkpoint": False},
                {"id": "c", "checkpoint": True},
            ],
            "edges": [
                {"from": "a", "to": "b", "guard": None},
                {"from": "b", "to": "c", "guard": None},
                {"from": "c", "to": "a", "guard": None},
            ],
            "entry_node": "a",
            "terminal_nodes": ["c"],
        }
        with pytest.raises(ValueError, match="[Cc]ycle"):
            SOPGraph.from_archetype_data(data)

    def test_terminal_has_outgoing_edges(self) -> None:
        data = {
            "graph_id": "bad_terminal",
            "nodes": [
                {"id": "a", "checkpoint": False},
                {"id": "b", "checkpoint": True},
                {"id": "c", "checkpoint": False},
            ],
            "edges": [
                {"from": "a", "to": "b", "guard": None},
                {"from": "b", "to": "c", "guard": None},
            ],
            "entry_node": "a",
            "terminal_nodes": ["b"],
        }
        with pytest.raises(ValueError, match="terminal.*outgoing|outgoing.*terminal"):
            SOPGraph.from_archetype_data(data)

    def test_duplicate_edges(self) -> None:
        data = {
            "graph_id": "dup_edge",
            "nodes": [
                {"id": "a", "checkpoint": False},
                {"id": "b", "checkpoint": True},
            ],
            "edges": [
                {"from": "a", "to": "b", "guard": None},
                {"from": "a", "to": "b", "guard": None},
            ],
            "entry_node": "a",
            "terminal_nodes": ["b"],
        }
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            SOPGraph.from_archetype_data(data)

    def test_terminal_unreachable_from_entry(self) -> None:
        data = {
            "graph_id": "unreachable",
            "nodes": [
                {"id": "a", "checkpoint": False},
                {"id": "b", "checkpoint": True},
                {"id": "c", "checkpoint": True},
            ],
            "edges": [
                {"from": "a", "to": "b", "guard": None},
            ],
            "entry_node": "a",
            "terminal_nodes": ["c"],
        }
        with pytest.raises(ValueError, match="[Uu]nreachable"):
            SOPGraph.from_archetype_data(data)

    def test_branching_graph_rejected(self) -> None:
        """v1 only supports linear graphs; branching must be rejected."""
        data = {
            "graph_id": "branching",
            "nodes": [
                {"id": "a", "checkpoint": False},
                {"id": "b", "checkpoint": True},
                {"id": "c", "checkpoint": True},
            ],
            "edges": [
                {"from": "a", "to": "b", "guard": None},
                {"from": "a", "to": "c", "guard": None},
            ],
            "entry_node": "a",
            "terminal_nodes": ["b", "c"],
        }
        with pytest.raises(ValueError, match="outgoing edges"):
            SOPGraph.from_archetype_data(data)


# =========================================================================
# 3. Guard Evaluation
# =========================================================================


class TestGuardEvaluation:
    def test_none_guard_always_passes(self, simple_graph: SOPGraph, default_ctx: TicketGuardContext) -> None:
        assert simple_graph.evaluate_guard(None, default_ctx) is True

    def test_missing_order_id_guard(self, guarded_graph: SOPGraph) -> None:
        ctx_not_requested = TicketGuardContext(missing_fields_requested=False)
        assert guarded_graph.evaluate_guard("missing_order_id", ctx_not_requested) is True

        ctx_requested = TicketGuardContext(missing_fields_requested=True)
        assert guarded_graph.evaluate_guard("missing_order_id", ctx_requested) is False

    def test_info_received_guard(self, guarded_graph: SOPGraph) -> None:
        ctx_no = TicketGuardContext(info_received=False)
        assert guarded_graph.evaluate_guard("info_received", ctx_no) is False

        ctx_yes = TicketGuardContext(info_received=True)
        assert guarded_graph.evaluate_guard("info_received", ctx_yes) is True

    def test_classification_set_guard(self, simple_graph: SOPGraph) -> None:
        ctx_no = TicketGuardContext(classification_set=False)
        assert simple_graph.evaluate_guard("classification_set", ctx_no) is False

        ctx_yes = TicketGuardContext(classification_set=True)
        assert simple_graph.evaluate_guard("classification_set", ctx_yes) is True

    def test_escalation_required_guard(self, simple_graph: SOPGraph) -> None:
        ctx_no = TicketGuardContext(escalation_required=False)
        assert simple_graph.evaluate_guard("escalation_required", ctx_no) is False

        ctx_yes = TicketGuardContext(escalation_required=True)
        assert simple_graph.evaluate_guard("escalation_required", ctx_yes) is True

    def test_duplicate_confirmed_guard(self, simple_graph: SOPGraph) -> None:
        ctx_no = TicketGuardContext(duplicate_confirmed=False)
        assert simple_graph.evaluate_guard("duplicate_confirmed", ctx_no) is False

        ctx_yes = TicketGuardContext(duplicate_confirmed=True)
        assert simple_graph.evaluate_guard("duplicate_confirmed", ctx_yes) is True

    def test_missing_verification_guard(self, simple_graph: SOPGraph) -> None:
        ctx_not_req = TicketGuardContext(missing_fields_requested=False)
        assert simple_graph.evaluate_guard("missing_verification", ctx_not_req) is True

    def test_missing_api_details_guard(self, simple_graph: SOPGraph) -> None:
        ctx_not_req = TicketGuardContext(missing_fields_requested=False)
        assert simple_graph.evaluate_guard("missing_api_details", ctx_not_req) is True

    def test_missing_security_details_guard(self, simple_graph: SOPGraph) -> None:
        ctx_not_req = TicketGuardContext(missing_fields_requested=False)
        assert simple_graph.evaluate_guard("missing_security_details", ctx_not_req) is True

    def test_missing_shipping_details_guard(self, simple_graph: SOPGraph) -> None:
        ctx_not_req = TicketGuardContext(missing_fields_requested=False)
        assert simple_graph.evaluate_guard("missing_shipping_details", ctx_not_req) is True

    def test_unknown_guard_raises(self, simple_graph: SOPGraph, default_ctx: TicketGuardContext) -> None:
        with pytest.raises(ValueError, match="Unknown guard"):
            simple_graph.evaluate_guard("totally_bogus_guard", default_ctx)

    def test_all_guards_in_evaluator_table(self) -> None:
        """Every guard used in real archetypes must exist in GUARD_EVALUATORS."""
        data_path = Path(__file__).resolve().parent.parent / "data" / "archetypes.json"
        with open(data_path, encoding="utf-8") as f:
            archetypes = json.load(f)["archetypes"]
        guards_in_data: set[str] = set()
        for a in archetypes:
            for e in a["sop_graph"]["edges"]:
                if e["guard"] is not None:
                    guards_in_data.add(e["guard"])
        missing = guards_in_data - set(GUARD_EVALUATORS.keys())
        assert missing == set(), f"Guards missing from GUARD_EVALUATORS: {missing}"


# =========================================================================
# 4. Transition Queries
# =========================================================================


class TestGetAvailableTransitions:
    def test_from_entry(self, simple_graph: SOPGraph, default_ctx: TicketGuardContext) -> None:
        transitions = simple_graph.get_available_transitions("new", default_ctx)
        assert len(transitions) == 1
        assert transitions[0].to_node == "open"

    def test_from_terminal(self, simple_graph: SOPGraph, default_ctx: TicketGuardContext) -> None:
        transitions = simple_graph.get_available_transitions("close_resolved", default_ctx)
        assert transitions == []

    def test_guard_blocks_transition(self, guarded_graph: SOPGraph) -> None:
        # receive_reply → route_refund_team has guard "info_received"
        ctx_no_info = TicketGuardContext(info_received=False)
        transitions = guarded_graph.get_available_transitions("receive_reply", ctx_no_info)
        assert transitions == []

    def test_guard_allows_transition(self, guarded_graph: SOPGraph) -> None:
        ctx_info = TicketGuardContext(info_received=True)
        transitions = guarded_graph.get_available_transitions("receive_reply", ctx_info)
        assert len(transitions) == 1
        assert transitions[0].to_node == "route_refund_team"


# =========================================================================
# 5. Action-to-Node Mapping
# =========================================================================


class TestFindMatchingNode:
    def test_open_ticket(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.OPEN_TICKET)
        assert "open" in matches

    def test_classify_ticket(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.CLASSIFY_TICKET)
        assert "classify_billing_refund" in matches

    def test_set_impact_urgency(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.SET_IMPACT_URGENCY)
        assert "set_impact_urgency" in matches

    def test_route_ticket(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.ROUTE_TICKET)
        assert "route_billing_team" in matches

    def test_close_ticket(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.CLOSE_TICKET)
        assert "close_resolved" in matches

    def test_request_information(self, guarded_graph: SOPGraph) -> None:
        matches = guarded_graph.find_matching_nodes(ActionType.REQUEST_INFORMATION)
        assert "request_info_order_id" in matches

    def test_no_match_returns_empty(self, simple_graph: SOPGraph) -> None:
        # simple_graph has no escalate nodes
        matches = simple_graph.find_matching_nodes(ActionType.ESCALATE_TICKET)
        assert matches == []

    def test_skip_turn_returns_empty(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.SKIP_TURN)
        assert matches == []

    def test_finish_episode_returns_empty(self, simple_graph: SOPGraph) -> None:
        matches = simple_graph.find_matching_nodes(ActionType.FINISH_EPISODE)
        assert matches == []


# =========================================================================
# 6. SOPTracker
# =========================================================================


class TestSOPTracker:
    def test_initial_state(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        assert tracker.current_node == "new"
        assert tracker.visited_nodes == ["new"]
        assert tracker.visited_checkpoints == set()
        assert tracker.completed is False

    def test_try_advance_valid(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        assert tracker.try_advance("open", ctx) is True
        assert tracker.current_node == "open"
        assert tracker.visited_nodes == ["new", "open"]

    def test_try_advance_no_edge(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        # Can't jump from "new" to "close_resolved"
        assert tracker.try_advance("close_resolved", ctx) is False
        assert tracker.current_node == "new"

    def test_try_advance_guard_blocks(self, guarded_graph: SOPGraph) -> None:
        tracker = SOPTracker(guarded_graph)
        ctx = TicketGuardContext()
        # Advance to set_impact_urgency first
        tracker.try_advance("open", ctx)
        tracker.try_advance("classify_billing_refund", ctx)
        tracker.try_advance("set_impact_urgency", ctx)
        # Guard "missing_order_id" requires missing_fields_requested=False (meaning fields not yet requested → guard fires)
        # So with default ctx (missing_fields_requested=False), the guard passes
        assert tracker.try_advance("request_info_order_id", ctx) is True

    def test_try_advance_guard_blocks_info_received(self, guarded_graph: SOPGraph) -> None:
        tracker = SOPTracker(guarded_graph)
        ctx_no_info = TicketGuardContext()
        tracker.try_advance("open", ctx_no_info)
        tracker.try_advance("classify_billing_refund", ctx_no_info)
        tracker.try_advance("set_impact_urgency", ctx_no_info)
        tracker.try_advance("request_info_order_id", ctx_no_info)
        tracker.try_advance("receive_reply", ctx_no_info)
        # Now need info_received=True to advance to route_refund_team
        assert tracker.try_advance("route_refund_team", ctx_no_info) is False

        ctx_info = TicketGuardContext(info_received=True)
        assert tracker.try_advance("route_refund_team", ctx_info) is True

    def test_checkpoint_tracking(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        tracker.try_advance("open", ctx)  # not a checkpoint
        assert tracker.visited_checkpoints == set()
        tracker.try_advance("classify_billing_refund", ctx)  # checkpoint
        assert "classify_billing_refund" in tracker.visited_checkpoints

    def test_completed_on_terminal(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        tracker.try_advance("open", ctx)
        tracker.try_advance("classify_billing_refund", ctx)
        tracker.try_advance("set_impact_urgency", ctx)
        tracker.try_advance("route_billing_team", ctx)
        assert tracker.completed is False
        tracker.try_advance("close_resolved", ctx)
        assert tracker.completed is True

    def test_no_advance_after_completed(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        for node in ["open", "classify_billing_refund", "set_impact_urgency",
                      "route_billing_team", "close_resolved"]:
            tracker.try_advance(node, ctx)
        assert tracker.completed is True
        # Should not advance further
        assert tracker.try_advance("new", ctx) is False

    def test_try_advance_by_action(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        tracker.try_advance("open", ctx)  # move to open first
        # Now try advancing by CLASSIFY_TICKET action
        result = tracker.try_advance_by_action(ActionType.CLASSIFY_TICKET, ctx)
        assert result is True
        assert tracker.current_node == "classify_billing_refund"

    def test_try_advance_by_action_no_match(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        # From "new", CLASSIFY_TICKET doesn't have an edge
        result = tracker.try_advance_by_action(ActionType.CLASSIFY_TICKET, ctx)
        assert result is False

    def test_auto_advance_non_checkpoints(self, guarded_graph: SOPGraph) -> None:
        tracker = SOPTracker(guarded_graph)
        ctx = TicketGuardContext()
        # Start at "new" (non-checkpoint). Auto-advance should go to "open" (also non-checkpoint)
        # but stop before "classify_billing_refund" (checkpoint)
        advanced = tracker.auto_advance_non_checkpoints(ctx)
        # Should advance through "open" (non-checkpoint with unguarded edge)
        assert "open" in advanced
        assert tracker.current_node == "open"

    def test_auto_advance_stops_at_checkpoint(self, guarded_graph: SOPGraph) -> None:
        tracker = SOPTracker(guarded_graph)
        ctx = TicketGuardContext()
        tracker.auto_advance_non_checkpoints(ctx)
        # Now at "open". Next is "classify_billing_refund" which is a checkpoint
        # auto_advance should not go there
        assert tracker.current_node == "open"

    def test_auto_advance_through_receive_reply(self, guarded_graph: SOPGraph) -> None:
        """receive_reply is non-checkpoint; auto-advance should pass through it."""
        tracker = SOPTracker(guarded_graph)
        ctx_full = TicketGuardContext(
            missing_fields_requested=False,  # guard fires (fields not requested yet)
            info_received=True,
        )
        # Manually advance to request_info_order_id
        for node in ["open", "classify_billing_refund", "set_impact_urgency",
                      "request_info_order_id"]:
            tracker.try_advance(node, ctx_full)

        # Now at request_info_order_id. Next is receive_reply (non-checkpoint, no guard).
        # Auto-advance should go through receive_reply.
        # But receive_reply → route_refund_team has guard "info_received", needs info_received=True.
        advanced = tracker.auto_advance_non_checkpoints(ctx_full)
        assert "receive_reply" in advanced
        # Should stop before route_refund_team (checkpoint)
        assert tracker.current_node == "receive_reply"


# =========================================================================
# 7. SOPScoringData + UJCS Computation
# =========================================================================


class TestSOPScoringData:
    def test_from_tracker(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        for node in ["open", "classify_billing_refund", "set_impact_urgency",
                      "route_billing_team", "close_resolved"]:
            tracker.try_advance(node, ctx)

        data = tracker.get_scoring_data()
        assert isinstance(data, SOPScoringData)
        assert data.gold_path == simple_graph.gold_path
        assert data.gold_checkpoints == simple_graph.checkpoints
        assert data.reached_terminal is True
        assert data.terminal_node == "close_resolved"
        assert len(data.agent_path) == 6  # new + 5 advances
        assert data.agent_checkpoints == simple_graph.checkpoints


class TestComputeUJCS:
    def test_perfect_path(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        for node in ["open", "classify_billing_refund", "set_impact_urgency",
                      "route_billing_team", "close_resolved"]:
            tracker.try_advance(node, ctx)
        score = compute_ujcs(tracker.get_scoring_data())
        assert score == 1.0

    def test_no_progress(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        # Never advanced from entry
        score = compute_ujcs(tracker.get_scoring_data())
        assert score == 0.0

    def test_partial_progress(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        tracker.try_advance("open", ctx)
        tracker.try_advance("classify_billing_refund", ctx)
        # Only visited 1 of 4 checkpoints, didn't reach terminal
        score = compute_ujcs(tracker.get_scoring_data())
        assert 0.0 < score < 1.0

    def test_wrong_parameterization_reduces_score(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        for node in ["open", "classify_billing_refund", "set_impact_urgency",
                      "route_billing_team", "close_resolved"]:
            tracker.try_advance(node, ctx)
        perfect = compute_ujcs(tracker.get_scoring_data(), wrong_parameterizations=0)
        with_errors = compute_ujcs(tracker.get_scoring_data(), wrong_parameterizations=2)
        assert with_errors < perfect

    def test_score_clamped_to_zero_one(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        # Many wrong parameterizations on empty path should still give ≥ 0
        score = compute_ujcs(tracker.get_scoring_data(), wrong_parameterizations=100)
        assert 0.0 <= score <= 1.0

    def test_most_checkpoints_visited(self, simple_graph: SOPGraph) -> None:
        """Visit all but one checkpoint, reach terminal. Score should be high but not 1.0."""
        tracker = SOPTracker(simple_graph)
        ctx = TicketGuardContext()
        # Skip set_impact_urgency by directly jumping (will fail, so partial)
        tracker.try_advance("open", ctx)
        tracker.try_advance("classify_billing_refund", ctx)
        tracker.try_advance("set_impact_urgency", ctx)
        # Skip route_billing_team — can't advance to close_resolved from here
        # This means we have 2/4 checkpoints + not at terminal
        score = compute_ujcs(tracker.get_scoring_data())
        assert 0.0 < score < 1.0


# =========================================================================
# 8. Batch Loading from Real Archetypes
# =========================================================================


class TestLoadSOPGraphs:
    @pytest.fixture
    def archetypes_data(self) -> list[dict[str, Any]]:
        data_path = Path(__file__).resolve().parent.parent / "data" / "archetypes.json"
        with open(data_path, encoding="utf-8") as f:
            return json.load(f)["archetypes"]

    def test_loads_all_18(self, archetypes_data: list[dict]) -> None:
        graphs = load_sop_graphs(archetypes_data)
        assert len(graphs) == 18

    def test_returns_dict_keyed_by_graph_id(self, archetypes_data: list[dict]) -> None:
        graphs = load_sop_graphs(archetypes_data)
        assert "refund_missing_order_id" in graphs
        assert "spam_marketing_email" in graphs
        assert "suspicious_login_security_review" in graphs

    def test_each_graph_is_valid(self, archetypes_data: list[dict]) -> None:
        """All 18 real archetypes must produce valid SOPGraphs."""
        graphs = load_sop_graphs(archetypes_data)
        for graph_id, graph in graphs.items():
            assert isinstance(graph, SOPGraph), f"{graph_id} is not SOPGraph"
            assert len(graph.gold_path) >= 2, f"{graph_id} gold_path too short"
            assert len(graph.checkpoints) >= 1, f"{graph_id} has no checkpoints"
            assert graph.entry_node == "new", f"{graph_id} entry_node is not 'new'"

    def test_refund_graph_structure(self, archetypes_data: list[dict]) -> None:
        graphs = load_sop_graphs(archetypes_data)
        g = graphs["refund_missing_order_id"]
        assert g.entry_node == "new"
        assert g.terminal_nodes == frozenset({"close_resolved"})
        assert "classify_billing_refund" in g.checkpoints
        assert "request_info_order_id" in g.checkpoints
        assert "route_refund_team" in g.checkpoints

    def test_suspicious_login_has_escalation(self, archetypes_data: list[dict]) -> None:
        graphs = load_sop_graphs(archetypes_data)
        g = graphs["suspicious_login_security_review"]
        assert g.terminal_nodes == frozenset({"escalate_security_team"})
        assert "escalate_security_team" in g.checkpoints

    def test_spam_graph_is_short(self, archetypes_data: list[dict]) -> None:
        graphs = load_sop_graphs(archetypes_data)
        g = graphs["spam_marketing_email"]
        # spam: new → open → identify_spam → close_non_actionable
        assert len(g.gold_path) == 4
        assert g.terminal_nodes == frozenset({"close_non_actionable"})


# =========================================================================
# 9. Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_single_node_graph(self) -> None:
        """A graph with only entry = terminal should be valid."""
        data = {
            "graph_id": "trivial",
            "nodes": [{"id": "done", "checkpoint": True}],
            "edges": [],
            "entry_node": "done",
            "terminal_nodes": ["done"],
        }
        g = SOPGraph.from_archetype_data(data)
        assert g.gold_path == ("done",)
        tracker = SOPTracker(g)
        assert tracker.completed is True  # starts at terminal
        score = compute_ujcs(tracker.get_scoring_data())
        assert score == 1.0  # already at terminal, checkpoint visited

    def test_tracker_scoring_data_immutable(self, simple_graph: SOPGraph) -> None:
        tracker = SOPTracker(simple_graph)
        data = tracker.get_scoring_data()
        with pytest.raises(AttributeError):
            data.reached_terminal = False  # type: ignore[misc]
