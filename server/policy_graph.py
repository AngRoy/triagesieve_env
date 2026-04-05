"""SOP DAG definitions, tracking, and UJCS scoring.

Represents SOP (Standard Operating Procedure) directed acyclic graphs for each
ticket archetype (§13). Provides:

- Immutable graph data structures (SOPNode, SOPEdge, SOPGraph).
- Graph loading from archetype JSON dicts with validation.
- Guard evaluation against ticket state context.
- Mutable SOPTracker for recording the agent's path through the graph.
- UJCS (User Journey Coverage Score) computation (§17.4).

This module does NOT implement environment step mechanics or scoring logic.
The environment and scorer import from here.

Python 3.11+, frozen dataclasses for immutable structures.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..models import ActionType

__all__ = [
    "SOPNode",
    "SOPEdge",
    "SOPGraph",
    "TicketGuardContext",
    "SOPTracker",
    "SOPScoringData",
    "GUARD_EVALUATORS",
    "load_sop_graphs",
    "compute_ujcs",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SOPNode:
    """A single node in an SOP DAG.

    Args:
        node_id: Unique identifier within the graph (e.g. "classify_billing_refund").
        is_checkpoint: Whether this node is a mandatory checkpoint for UJCS scoring.
    """

    node_id: str
    is_checkpoint: bool


@dataclass(frozen=True)
class SOPEdge:
    """A directed edge between two SOP nodes.

    Args:
        from_node: Source node ID.
        to_node: Destination node ID.
        guard: Optional guard condition name. None means always passable.
    """

    from_node: str
    to_node: str
    guard: str | None


@dataclass(frozen=True)
class TicketGuardContext:
    """Snapshot of ticket state used for guard evaluation.

    The environment builds this from the current ticket state and passes it
    to guard evaluation and tracker advancement methods.
    """

    classification_set: bool = False
    impact_urgency_set: bool = False
    missing_fields_requested: bool = False
    info_received: bool = False
    escalation_required: bool = False
    duplicate_confirmed: bool = False


@dataclass(frozen=True)
class SOPScoringData:
    """Immutable snapshot of tracker state for UJCS computation.

    Produced by ``SOPTracker.get_scoring_data()``, consumed by ``compute_ujcs()``.
    """

    gold_path: tuple[str, ...]
    gold_checkpoints: frozenset[str]
    agent_path: tuple[str, ...]
    agent_checkpoints: frozenset[str]
    reached_terminal: bool
    terminal_node: str | None


# ---------------------------------------------------------------------------
# Guard Evaluator Dispatch Table
# ---------------------------------------------------------------------------

# Each guard is a function: (TicketGuardContext) -> bool
# "missing_*" guards fire (return True) when the field has NOT been requested yet.
# "info_received" fires when info HAS been received.
# "classification_set" fires when classification IS set.
# etc.
#
# KNOWN LIMITATION (v1): All "missing_*" guards collapse to a single boolean
# (missing_fields_requested). This means the system cannot distinguish which
# specific field the agent requested. This is safe in v1 because each archetype's
# SOP graph uses at most one missing-field guard. If a future archetype requires
# two different missing-field guards in the same SOP, TicketGuardContext must be
# extended with a set-valued `requested_fields: frozenset[str]` field and each
# guard must check for its specific field.

GUARD_EVALUATORS: dict[str, Callable[[TicketGuardContext], bool]] = {
    "missing_order_id": lambda ctx: not ctx.missing_fields_requested,
    "missing_verification": lambda ctx: not ctx.missing_fields_requested,
    "missing_api_details": lambda ctx: not ctx.missing_fields_requested,
    "missing_security_details": lambda ctx: not ctx.missing_fields_requested,
    "missing_shipping_details": lambda ctx: not ctx.missing_fields_requested,
    "info_received": lambda ctx: ctx.info_received,
    "classification_set": lambda ctx: ctx.classification_set,
    "escalation_required": lambda ctx: ctx.escalation_required,
    "duplicate_confirmed": lambda ctx: ctx.duplicate_confirmed,
}


# ---------------------------------------------------------------------------
# Action → SOP Node Prefix Mapping
# ---------------------------------------------------------------------------

# Maps prefixes of SOP node IDs to the ActionType they correspond to.
# Order matters: longer/more-specific prefixes checked first.
ACTION_NODE_PREFIX_MAP: list[tuple[str, ActionType]] = [
    ("open", ActionType.OPEN_TICKET),
    ("classify_", ActionType.CLASSIFY_TICKET),
    ("identify_", ActionType.CLASSIFY_TICKET),
    ("set_impact_urgency", ActionType.SET_IMPACT_URGENCY),
    ("route_", ActionType.ROUTE_TICKET),
    ("request_", ActionType.REQUEST_INFORMATION),
    ("escalate_", ActionType.ESCALATE_TICKET),
    ("merge_", ActionType.MERGE_DUPLICATE),
    ("close_", ActionType.CLOSE_TICKET),
]


def _node_id_to_action_type(node_id: str) -> ActionType | None:
    """Map a SOP node ID to an ActionType using prefix matching.

    For prefixes ending in '_' (e.g. "classify_"), a simple startswith is safe.
    For bare-word prefixes (e.g. "open"), requires word boundary (followed by
    '_' or end-of-string) to avoid false positives like "opening" matching "open".

    Returns None if the node doesn't correspond to any agent action
    (e.g. "new", "receive_reply", "check_order_id", "inspect_thread_history").
    """
    for prefix, action_type in ACTION_NODE_PREFIX_MAP:
        if node_id == prefix:
            return action_type
        if prefix.endswith("_"):
            # Prefix already has separator — simple startswith is safe
            if node_id.startswith(prefix):
                return action_type
        else:
            # Bare-word prefix — require underscore boundary
            if node_id.startswith(prefix + "_"):
                return action_type
    return None


# ---------------------------------------------------------------------------
# SOPGraph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SOPGraph:
    """Immutable SOP DAG loaded from an archetype definition.

    Constructed via ``from_archetype_data()`` factory which validates the graph.

    Derived fields (computed at construction time):
        node_index: Fast lookup from node_id to SOPNode.
        adjacency: Outgoing edges per node (tuples, not lists, for immutability).
        checkpoints: Set of mandatory checkpoint node IDs.
        gold_path: Ordered tuple of node IDs from entry to terminal.

    Note:
        ``node_index`` and ``adjacency`` are dicts (not hashable). They are excluded
        from ``__hash__`` and ``__eq__`` to prevent ``TypeError`` if instances are
        placed in sets. Do not mutate these dicts after construction.
    """

    graph_id: str
    nodes: tuple[SOPNode, ...]
    edges: tuple[SOPEdge, ...]
    entry_node: str
    terminal_nodes: frozenset[str]
    node_index: dict[str, SOPNode] = field(default_factory=dict, hash=False, compare=False)
    adjacency: dict[str, tuple[SOPEdge, ...]] = field(
        default_factory=dict, hash=False, compare=False
    )
    checkpoints: frozenset[str] = frozenset()
    gold_path: tuple[str, ...] = ()

    @staticmethod
    def from_archetype_data(sop_dict: dict[str, Any]) -> SOPGraph:
        """Build and validate an SOPGraph from an archetype's sop_graph dict.

        Args:
            sop_dict: Raw dict with keys: graph_id, nodes, edges, entry_node,
                terminal_nodes. Matches the schema in data/archetypes.json.

        Returns:
            Validated, frozen SOPGraph.

        Raises:
            ValueError: If the graph fails any validation check.
        """
        graph_id = sop_dict["graph_id"]

        # Build nodes
        nodes_list: list[SOPNode] = []
        node_index: dict[str, SOPNode] = {}
        for n in sop_dict["nodes"]:
            node = SOPNode(node_id=n["id"], is_checkpoint=n["checkpoint"])
            nodes_list.append(node)
            node_index[n["id"]] = node

        # Build edges (mutable during construction, frozen after)
        edges_list: list[SOPEdge] = []
        adj_build: dict[str, list[SOPEdge]] = {n.node_id: [] for n in nodes_list}
        for e in sop_dict["edges"]:
            edge = SOPEdge(from_node=e["from"], to_node=e["to"], guard=e.get("guard"))
            edges_list.append(edge)
            adj_build[e["from"]].append(edge)

        entry_node = sop_dict["entry_node"]
        terminal_nodes = frozenset(sop_dict["terminal_nodes"])

        # Compute checkpoints
        checkpoints = frozenset(n.node_id for n in nodes_list if n.is_checkpoint)

        # Compute gold path via topological walk from entry
        gold_path = _compute_gold_path(entry_node, adj_build)

        # Validate
        _validate_graph(
            graph_id=graph_id,
            node_index=node_index,
            edges=edges_list,
            adjacency=adj_build,
            entry_node=entry_node,
            terminal_nodes=terminal_nodes,
            gold_path=gold_path,
        )

        # Freeze adjacency lists to tuples for immutability
        adjacency: dict[str, tuple[SOPEdge, ...]] = {
            k: tuple(v) for k, v in adj_build.items()
        }

        return SOPGraph(
            graph_id=graph_id,
            nodes=tuple(nodes_list),
            edges=tuple(edges_list),
            entry_node=entry_node,
            terminal_nodes=terminal_nodes,
            node_index=node_index,
            adjacency=adjacency,
            checkpoints=checkpoints,
            gold_path=gold_path,
        )

    def evaluate_guard(self, guard: str | None, ctx: TicketGuardContext) -> bool:
        """Evaluate a guard condition against a ticket state context.

        Args:
            guard: Guard name (key in GUARD_EVALUATORS), or None for unconditional.
            ctx: Current ticket state snapshot.

        Returns:
            True if the guard passes (transition is allowed).

        Raises:
            ValueError: If the guard name is not recognized.
        """
        if guard is None:
            return True
        evaluator = GUARD_EVALUATORS.get(guard)
        if evaluator is None:
            raise ValueError(f"Unknown guard: {guard!r}")
        return evaluator(ctx)

    def get_available_transitions(
        self, current_node: str, ctx: TicketGuardContext
    ) -> list[SOPEdge]:
        """Return edges from current_node whose guards pass.

        Args:
            current_node: Node ID to query transitions from.
            ctx: Current ticket state for guard evaluation.

        Returns:
            List of passable SOPEdge objects (may be empty).
        """
        edges = self.adjacency.get(current_node, [])
        return [e for e in edges if self.evaluate_guard(e.guard, ctx)]

    def find_matching_nodes(self, action_type: ActionType) -> list[str]:
        """Find SOP node IDs that correspond to a given ActionType.

        Uses prefix matching via ACTION_NODE_PREFIX_MAP.

        Args:
            action_type: The agent action type to match.

        Returns:
            List of matching node IDs in this graph (may be empty).
        """
        result: list[str] = []
        for node in self.nodes:
            mapped = _node_id_to_action_type(node.node_id)
            if mapped == action_type:
                result.append(node.node_id)
        return result


# ---------------------------------------------------------------------------
# SOPTracker (Mutable)
# ---------------------------------------------------------------------------


class SOPTracker:
    """Mutable tracker recording the agent's path through an SOP graph.

    Observational only — does not block actions. The environment's state machine
    (CLAUDE.md §12) determines valid actions; the tracker records what happened
    for UJCS scoring.

    Args:
        graph: The SOPGraph to track against.
    """

    def __init__(self, graph: SOPGraph) -> None:
        self._graph = graph
        self._current_node: str = graph.entry_node
        self._visited_nodes: list[str] = [graph.entry_node]
        self._visited_checkpoints: set[str] = set()
        self._completed: bool = graph.entry_node in graph.terminal_nodes

        # Record entry checkpoint if applicable
        if graph.node_index[graph.entry_node].is_checkpoint:
            self._visited_checkpoints.add(graph.entry_node)

    @property
    def graph(self) -> SOPGraph:
        """The SOPGraph being tracked."""
        return self._graph

    @property
    def current_node(self) -> str:
        """Current position in the SOP graph."""
        return self._current_node

    @property
    def visited_nodes(self) -> list[str]:
        """Ordered list of all visited node IDs."""
        return list(self._visited_nodes)

    @property
    def visited_checkpoints(self) -> set[str]:
        """Set of checkpoint node IDs that have been visited."""
        return set(self._visited_checkpoints)

    @property
    def completed(self) -> bool:
        """Whether the tracker has reached a terminal node."""
        return self._completed

    def try_advance(self, target_node: str, ctx: TicketGuardContext) -> bool:
        """Try to advance the tracker to a specific target node.

        Succeeds only if there is a direct edge from the current node to the
        target and the edge's guard passes.

        Args:
            target_node: The node ID to advance to.
            ctx: Current ticket state for guard evaluation.

        Returns:
            True if advancement succeeded, False otherwise.
        """
        if self._completed:
            return False

        edges = self._graph.adjacency.get(self._current_node, [])
        for edge in edges:
            if edge.to_node == target_node and self._graph.evaluate_guard(edge.guard, ctx):
                self._advance_to(target_node)
                return True
        return False

    def try_advance_by_action(self, action_type: ActionType, ctx: TicketGuardContext) -> bool:
        """Try to advance the tracker based on an ActionType.

        Finds SOP nodes matching the action type, then tries to advance to the
        first reachable one (connected by an edge from current node with passing guard).

        Args:
            action_type: The agent's action type.
            ctx: Current ticket state for guard evaluation.

        Returns:
            True if advancement succeeded, False otherwise.
        """
        if self._completed:
            return False

        matching_nodes = self._graph.find_matching_nodes(action_type)
        edges = self._graph.adjacency.get(self._current_node, [])
        for edge in edges:
            if edge.to_node in matching_nodes and self._graph.evaluate_guard(edge.guard, ctx):
                self._advance_to(edge.to_node)
                return True
        return False

    def auto_advance_non_checkpoints(self, ctx: TicketGuardContext) -> list[str]:
        """Auto-advance through consecutive non-checkpoint nodes.

        Walks forward through unguarded or guard-passing edges as long as the
        next node is NOT a checkpoint. Stops at checkpoints, terminal nodes,
        or when no passable edge exists.

        Args:
            ctx: Current ticket state for guard evaluation.

        Returns:
            List of node IDs that were auto-advanced through.
        """
        advanced: list[str] = []
        while not self._completed:
            transitions = self._graph.get_available_transitions(self._current_node, ctx)
            if not transitions:
                break
            # Take the first available transition (linear graphs have at most one)
            next_edge = transitions[0]
            next_node = self._graph.node_index.get(next_edge.to_node)
            if next_node is None:
                break
            # Stop before advancing INTO a checkpoint
            if next_node.is_checkpoint:
                break
            self._advance_to(next_edge.to_node)
            advanced.append(next_edge.to_node)
        return advanced

    def get_scoring_data(self) -> SOPScoringData:
        """Extract an immutable scoring snapshot from the current tracker state.

        Returns:
            SOPScoringData for consumption by ``compute_ujcs()``.
        """
        return SOPScoringData(
            gold_path=self._graph.gold_path,
            gold_checkpoints=self._graph.checkpoints,
            agent_path=tuple(self._visited_nodes),
            agent_checkpoints=frozenset(self._visited_checkpoints),
            reached_terminal=self._completed,
            terminal_node=self._current_node if self._completed else None,
        )

    def _advance_to(self, node_id: str) -> None:
        """Internal: move to a node, update bookkeeping."""
        self._current_node = node_id
        self._visited_nodes.append(node_id)
        node = self._graph.node_index[node_id]
        if node.is_checkpoint:
            self._visited_checkpoints.add(node_id)
        if node_id in self._graph.terminal_nodes:
            self._completed = True


# ---------------------------------------------------------------------------
# UJCS Computation
# ---------------------------------------------------------------------------


def compute_ujcs(scoring_data: SOPScoringData, wrong_parameterizations: int = 0) -> float:
    """Compute the User Journey Coverage Score (UJCS) for a ticket.

    Compares the agent's path against the gold SOP path (§17.4).

    Scoring formula:
        +1 for each mandatory checkpoint visited
        -1 for each mandatory checkpoint missed
        -1 for each wrong parameterization
        Terminal penalty if terminal not reached or wrong terminal

    Normalized to [0, 1].

    Args:
        scoring_data: Immutable snapshot from SOPTracker.get_scoring_data().
        wrong_parameterizations: Count of steps with correct node but wrong parameters.

    Returns:
        UJCS score in [0.0, 1.0].
    """
    total_checkpoints = len(scoring_data.gold_checkpoints)
    if total_checkpoints == 0:
        # No checkpoints to evaluate. Score based on terminal only.
        return 1.0 if scoring_data.reached_terminal else 0.0

    # Checkpoint scoring
    visited = scoring_data.agent_checkpoints & scoring_data.gold_checkpoints
    missed = scoring_data.gold_checkpoints - scoring_data.agent_checkpoints

    raw_score = len(visited) - len(missed) - wrong_parameterizations

    # Terminal penalty
    if not scoring_data.reached_terminal:
        raw_score -= total_checkpoints  # significant penalty

    # Normalize via min-max scaling to [0, 1].
    # max_score = all checkpoints visited (no penalties).
    # min_score = all checkpoints missed + terminal not reached.
    max_score = total_checkpoints
    min_score = -2 * total_checkpoints  # missed all + terminal penalty
    if max_score == min_score:
        return 1.0 if raw_score >= max_score else 0.0

    normalized = (raw_score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized))


# ---------------------------------------------------------------------------
# Batch Loader
# ---------------------------------------------------------------------------


def load_sop_graphs(archetypes: Sequence[dict[str, Any]]) -> dict[str, SOPGraph]:
    """Load and validate SOPGraphs from a list of archetype dicts.

    Args:
        archetypes: List of archetype dicts, each containing a "sop_graph" key.

    Returns:
        Dict mapping graph_id to validated SOPGraph.

    Raises:
        ValueError: If any graph fails validation.
    """
    result: dict[str, SOPGraph] = {}
    for arch in archetypes:
        sop_data = arch["sop_graph"]
        try:
            graph = SOPGraph.from_archetype_data(sop_data)
        except ValueError as exc:
            archetype_id = arch.get("archetype_id", sop_data.get("graph_id", "?"))
            raise ValueError(f"Archetype {archetype_id!r}: {exc}") from exc
        result[graph.graph_id] = graph
    return result


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------


def _compute_gold_path(
    entry_node: str,
    adjacency: dict[str, list[SOPEdge]] | dict[str, tuple[SOPEdge, ...]],
) -> tuple[str, ...]:
    """Compute the gold (canonical) path from entry to terminal via BFS/DFS.

    For v1 all SOP graphs are linear (single path), so this follows the first
    outgoing edge at each node.

    Args:
        entry_node: Starting node ID.
        adjacency: Adjacency map.

    Returns:
        Tuple of node IDs in path order.
    """
    path: list[str] = [entry_node]
    visited: set[str] = {entry_node}
    current = entry_node
    while True:
        edges = adjacency.get(current, [])
        if not edges:
            break
        next_node = edges[0].to_node
        if next_node in visited:
            break  # safety: stop if we'd loop
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return tuple(path)


def _validate_graph(
    *,
    graph_id: str,
    node_index: dict[str, SOPNode],
    edges: list[SOPEdge],
    adjacency: dict[str, list[SOPEdge]] | dict[str, tuple[SOPEdge, ...]],
    entry_node: str,
    terminal_nodes: frozenset[str],
    gold_path: tuple[str, ...],
) -> None:
    """Validate an SOP graph structure.

    Raises ValueError on any fatal validation failure.
    """
    # Entry node must exist
    if entry_node not in node_index:
        raise ValueError(
            f"Graph {graph_id!r}: entry_node {entry_node!r} not found in nodes"
        )

    # Terminal nodes must exist
    for tn in terminal_nodes:
        if tn not in node_index:
            raise ValueError(
                f"Graph {graph_id!r}: terminal node {tn!r} not found in nodes"
            )

    # Edge endpoints must exist
    for edge in edges:
        if edge.from_node not in node_index:
            raise ValueError(
                f"Graph {graph_id!r}: edge from_node {edge.from_node!r} not in nodes"
            )
        if edge.to_node not in node_index:
            raise ValueError(
                f"Graph {graph_id!r}: edge to_node {edge.to_node!r} not in nodes"
            )

    # No duplicate edges
    edge_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (edge.from_node, edge.to_node)
        if pair in edge_pairs:
            raise ValueError(
                f"Graph {graph_id!r}: duplicate edge {edge.from_node!r} → {edge.to_node!r}"
            )
        edge_pairs.add(pair)

    # Terminal nodes must not have outgoing edges
    for tn in terminal_nodes:
        if adjacency.get(tn, []):
            raise ValueError(
                f"Graph {graph_id!r}: terminal node {tn!r} has outgoing edges"
            )

    # DAG check: no cycles (Kahn's algorithm)
    _check_no_cycles(graph_id, node_index, edges)

    # v1 linear constraint: each non-terminal node must have at most one outgoing edge.
    # _compute_gold_path follows edges[0]; branching would silently produce a wrong path.
    for node_id, out_edges in adjacency.items():
        if node_id not in terminal_nodes and len(out_edges) > 1:
            raise ValueError(
                f"Graph {graph_id!r}: node {node_id!r} has {len(out_edges)} outgoing edges; "
                "only linear (single-path) graphs are supported in v1"
            )

    # Terminal reachability: all terminals must appear in gold path or be reachable from entry
    reachable = _reachable_from(entry_node, adjacency)
    for tn in terminal_nodes:
        if tn not in reachable:
            raise ValueError(
                f"Graph {graph_id!r}: terminal node {tn!r} is unreachable from entry"
            )


def _check_no_cycles(
    graph_id: str,
    node_index: dict[str, SOPNode],
    edges: list[SOPEdge],
) -> None:
    """Check for cycles using Kahn's topological sort algorithm."""
    in_degree: dict[str, int] = {nid: 0 for nid in node_index}
    adj: dict[str, list[str]] = {nid: [] for nid in node_index}
    for edge in edges:
        adj[edge.from_node].append(edge.to_node)
        in_degree[edge.to_node] += 1

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if processed != len(node_index):
        raise ValueError(f"Graph {graph_id!r}: cycle detected (not a DAG)")


def _reachable_from(
    start: str, adjacency: dict[str, list[SOPEdge]] | dict[str, tuple[SOPEdge, ...]]
) -> set[str]:
    """Return all nodes reachable from start via BFS."""
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for edge in adjacency.get(node, []):
            if edge.to_node not in visited:
                queue.append(edge.to_node)
    return visited
