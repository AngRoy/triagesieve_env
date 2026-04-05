"""Deterministic episode + ticket generation engine.

Loads archetypes from data/archetypes.json, generates fully deterministic
episode instances given (archetype_id, seed). No runtime LLM calls.

Public API:
    EpisodeEngine      — main class, loads static data, renders episodes/tickets
    RenderedTicket     — frozen dataclass for a single rendered ticket
    RenderedEpisode    — frozen dataclass for a complete episode
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from ..models import (
    CustomerTier,
    HiddenTicketTruth,
    Impact,
    IssueFamily,
    IssueSubtype,
    NonActionableSubtype,
    QueueId,
    SourceChannel,
    TaskDifficulty,
    TicketStatus,
    Urgency,
    derive_priority,
)

__all__ = ["EpisodeEngine", "RenderedTicket", "RenderedEpisode"]

# ---------------------------------------------------------------------------
# Base time for episode timestamps
# ---------------------------------------------------------------------------
_BASE_DATETIME = datetime(2026, 4, 5, 8, 0, 0, tzinfo=timezone.utc)

# Task ladder: difficulty → (min_tickets, max_tickets, budget)
# Easy budget = 6: covers the longest easy SOP (5 checkpoints: classify + set_iu +
#   request_info + route + close) plus the open action, with 1 step margin.
# Medium budget = 12: covers 2 tickets at ~6 steps each.
# Hard budget = 14: covers 2 full tickets + 1 partial, tests prioritisation under
#   pressure (escalation arcs + SLA urgency discrimination).
_TASK_LADDER: dict[TaskDifficulty, tuple[int, int, int]] = {
    TaskDifficulty.EASY: (1, 1, 6),
    TaskDifficulty.MEDIUM: (2, 3, 12),
    TaskDifficulty.HARD: (3, 4, 14),
}


# ---------------------------------------------------------------------------
# Data classes for rendered output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedTicket:
    """A fully rendered ticket with visible payload and hidden ground truth."""

    ticket_id: str
    subject: str
    body: str
    sender_email: str
    received_at: str  # ISO 8601
    customer_tier: CustomerTier
    source_channel: SourceChannel
    has_attachment: bool
    attachments: list[str]
    thread_history: list[dict[str, Any]]
    internal_notes: list[str]
    hidden_truth: HiddenTicketTruth
    sop_graph: dict[str, Any]
    follow_up_replies: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedEpisode:
    """A complete episode containing one or more rendered tickets."""

    episode_id: str
    seed: int
    task_difficulty: TaskDifficulty
    tickets: list[RenderedTicket]
    action_budget: int
    base_time: str  # ISO 8601


# ---------------------------------------------------------------------------
# Enum coercion helpers
# ---------------------------------------------------------------------------


_E = TypeVar("_E", bound=Enum)


def _to_enum(enum_cls: type[_E], value: str | _E) -> _E:
    """Convert a string value to an enum member, or return as-is if already an enum."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _to_optional_enum(enum_cls: type[_E], value: str | None) -> _E | None:
    """Convert a string value to an enum member, or return None."""
    if value is None:
        return None
    return _to_enum(enum_cls, value)


# ---------------------------------------------------------------------------
# EpisodeEngine
# ---------------------------------------------------------------------------


class EpisodeEngine:
    """Deterministic episode and ticket generation engine.

    Loads archetypes and static data from JSON files. Renders fully deterministic
    episode instances from (archetype_id, seed). All randomness uses a seeded
    ``random.Random`` instance.

    Args:
        data_dir: Path to the data/ directory. Defaults to the package's
            bundled ``data/`` directory.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self._data_dir = data_dir

        # Load all static data files
        self._archetypes_raw = self._load_json("archetypes.json")
        self._templates_raw = self._load_json("templates.json")
        self._routing_rules_raw = self._load_json("routing_rules.json")
        self._sla_rules_raw = self._load_json("sla_rules.json")

        # Build indices
        self._archetypes: list[dict[str, Any]] = self._archetypes_raw["archetypes"]
        self._archetype_index: dict[str, dict[str, Any]] = {
            a["archetype_id"]: a for a in self._archetypes
        }

        self._templates: list[dict[str, Any]] = self._templates_raw["templates"]
        self._template_index: dict[str, dict[str, Any]] = {
            t["template_id"]: t for t in self._templates
        }

        self._routing_rules: dict[str, dict[str, Any]] = self._routing_rules_raw["queues"]

        self._sla_rules: list[dict[str, Any]] = self._sla_rules_raw
        self._sla_index: dict[str, dict[str, Any]] = {s["tier"]: s for s in self._sla_rules}

        # Pre-group archetypes by difficulty
        self._by_difficulty: dict[str, list[dict[str, Any]]] = {}
        for a in self._archetypes:
            self._by_difficulty.setdefault(a["difficulty"], []).append(a)

    # -- Private helpers ----------------------------------------------------

    def _load_json(self, filename: str) -> Any:
        """Load and parse a JSON file from the data directory."""
        path = self._data_dir / filename
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Required data file not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    # -- Public properties --------------------------------------------------

    @property
    def archetypes(self) -> Sequence[dict[str, Any]]:
        """All loaded archetypes (read-only view)."""
        return self._archetypes

    @property
    def templates(self) -> Sequence[dict[str, Any]]:
        """All loaded templates (read-only view)."""
        return self._templates

    @property
    def routing_rules(self) -> Mapping[str, dict[str, Any]]:
        """All routing rules keyed by queue_id string (read-only view)."""
        return self._routing_rules

    @property
    def sla_rules(self) -> Sequence[dict[str, Any]]:
        """All SLA rules (read-only view)."""
        return self._sla_rules

    # -- Accessors ----------------------------------------------------------

    def get_archetype(self, archetype_id: str) -> dict[str, Any] | None:
        """Look up an archetype by its ID. Returns None if not found."""
        return self._archetype_index.get(archetype_id)

    def get_template(self, template_id: str) -> dict[str, Any] | None:
        """Look up a template by its ID. Returns None if not found."""
        return self._template_index.get(template_id)

    def get_routing_rule(self, queue_id: str) -> dict[str, Any] | None:
        """Look up routing rule by queue_id string. Returns None if not found."""
        return self._routing_rules.get(queue_id)

    def get_sla_for_tier(self, tier: CustomerTier) -> dict[str, Any] | None:
        """Look up SLA rule by customer tier. Returns None if not found."""
        return self._sla_index.get(tier.value)

    def get_archetypes_by_difficulty(self, difficulty: TaskDifficulty) -> list[dict[str, Any]]:
        """Return all archetypes matching the given difficulty level."""
        return list(self._by_difficulty.get(difficulty.value, []))

    # -- Ticket Rendering ---------------------------------------------------

    def render_ticket(self, archetype_id: str, seed: int) -> RenderedTicket:
        """Render a single ticket from an archetype and seed.

        Deterministic: same (archetype_id, seed) → identical output.

        Args:
            archetype_id: ID of the archetype to render.
            seed: Integer seed for deterministic variation selection.

        Returns:
            A fully rendered RenderedTicket with visible payload and hidden truth.

        Raises:
            ValueError: If archetype_id is not found.
        """
        arch = self._archetype_index.get(archetype_id)
        if arch is None:
            raise ValueError(f"Unknown archetype: {archetype_id!r}")

        rng = random.Random(seed)  # nosec B311 — deterministic simulation, not security

        # Resolve variation parameters deterministically
        params = self._resolve_variation_params(arch["variation_parameters"], rng)

        # Build ticket ID
        ticket_id = f"T{seed:04d}-{archetype_id[:8]}"

        # Compute base timestamp
        time_offset = timedelta(minutes=seed % 1440)
        base_dt = _BASE_DATETIME + time_offset
        received_at = base_dt.isoformat().replace("+00:00", "Z")

        # Render visible template fields
        vt = arch["visible_template"]
        subject = self._render_template_str(vt["subject_pattern"], params)
        body = self._render_template_str(vt["body_pattern"], params)
        sender_email = self._render_template_str(vt["sender_pattern"], params)

        # Render attachments
        attachments = [self._render_template_str(a, params) for a in vt.get("attachments", [])]
        has_attachment = len(attachments) > 0

        # Render thread history
        thread_history = self._render_thread_history(
            vt.get("thread_history_pattern", []), params, base_dt
        )

        # Render internal notes
        internal_notes = [
            self._render_template_str(n, params) for n in vt.get("internal_notes_pattern", [])
        ]

        # Build hidden truth
        ht_raw = arch["hidden_truth"]
        customer_tier = _to_enum(CustomerTier, ht_raw["customer_tier"])
        source_channel = _to_enum(SourceChannel, ht_raw["source_channel"])
        impact = _to_enum(Impact, ht_raw["impact"])
        urgency = _to_enum(Urgency, ht_raw["urgency"])
        priority = derive_priority(impact, urgency)

        # SLA from tier
        sla = self._sla_index.get(customer_tier.value, {})
        sla_response = sla.get("response_deadline_minutes", 0)
        sla_resolution = sla.get("resolution_deadline_minutes", 0)

        # Resolve duplicate_of placeholder if present
        duplicate_of = ht_raw.get("duplicate_of")
        if duplicate_of is not None and "{" in duplicate_of:
            duplicate_of = self._render_template_str(duplicate_of, params)

        hidden_truth = HiddenTicketTruth(
            ticket_id=ticket_id,
            customer_tier=customer_tier,
            source_channel=source_channel,
            issue_family=_to_enum(IssueFamily, ht_raw["issue_family"]),
            issue_subtype=_to_enum(IssueSubtype, ht_raw["issue_subtype"]),
            product_area=ht_raw["product_area"],
            impact=impact,
            urgency=urgency,
            priority=priority,
            required_queue=_to_enum(QueueId, ht_raw["required_queue"]),
            required_missing_fields=list(ht_raw.get("required_missing_fields", [])),
            escalation_required=ht_raw.get("escalation_required", False),
            escalation_target=_to_optional_enum(QueueId, ht_raw.get("escalation_target")),
            is_duplicate=ht_raw.get("is_duplicate", False),
            duplicate_of=duplicate_of,
            sla_response_deadline=sla_response,
            sla_resolution_deadline=sla_resolution,
            policy_graph_id=arch["sop_graph"]["graph_id"],
            correct_template_ids=list(ht_raw.get("correct_template_ids", [])),
            gold_terminal_status=_to_enum(TicketStatus, ht_raw["gold_terminal_status"]),
            non_actionable_subtype=_to_optional_enum(
                NonActionableSubtype, ht_raw.get("non_actionable_subtype")
            ),
        )

        # Pre-compute follow-up replies for required fields
        follow_up_replies = self._build_follow_up_replies(
            ht_raw.get("required_missing_fields", []), params, seed
        )

        return RenderedTicket(
            ticket_id=ticket_id,
            subject=subject,
            body=body,
            sender_email=sender_email,
            received_at=received_at,
            customer_tier=customer_tier,
            source_channel=source_channel,
            has_attachment=has_attachment,
            attachments=attachments,
            thread_history=thread_history,
            internal_notes=internal_notes,
            hidden_truth=hidden_truth,
            sop_graph=arch["sop_graph"],
            follow_up_replies=follow_up_replies,
        )

    # -- Episode Rendering --------------------------------------------------

    def render_episode(
        self,
        seed: int,
        difficulty: TaskDifficulty | None = None,
    ) -> RenderedEpisode:
        """Render a complete episode with deterministic ticket selection.

        Args:
            seed: Master seed for the episode.
            difficulty: Task difficulty. If None, derived from seed.

        Returns:
            A RenderedEpisode containing rendered tickets and metadata.
        """
        rng = random.Random(seed)  # nosec B311 — deterministic simulation, not security

        # Pick difficulty if not specified
        if difficulty is None:
            difficulty = rng.choice(list(TaskDifficulty))

        min_tickets, max_tickets, budget = _TASK_LADDER[difficulty]
        num_tickets = rng.randint(min_tickets, max_tickets)

        # Select archetypes for this episode
        matching = self.get_archetypes_by_difficulty(difficulty)
        # If not enough matching archetypes, also pull from easier difficulties
        all_archetypes = list(matching)
        if len(all_archetypes) < num_tickets:
            for diff in TaskDifficulty:
                if diff.value != difficulty.value:
                    all_archetypes.extend(self.get_archetypes_by_difficulty(diff))

        # Deterministically select archetypes
        selected: list[dict[str, Any]] = []
        pool = list(all_archetypes)
        for _ in range(num_tickets):
            if not pool:
                pool = list(all_archetypes)
            idx = rng.randrange(len(pool))
            selected.append(pool.pop(idx))

        # Render tickets with sub-seeds
        tickets: list[RenderedTicket] = []
        for i, arch in enumerate(selected):
            sub_seed = seed * 1000 + i
            ticket = self.render_ticket(arch["archetype_id"], sub_seed)
            tickets.append(ticket)

        # Compute base time
        time_offset = timedelta(minutes=seed % 1440)
        base_dt = _BASE_DATETIME + time_offset
        base_time = base_dt.isoformat().replace("+00:00", "Z")

        episode_id = f"ep-{seed}-{difficulty.value}"

        return RenderedEpisode(
            episode_id=episode_id,
            seed=seed,
            task_difficulty=difficulty,
            tickets=tickets,
            action_budget=budget,
            base_time=base_time,
        )

    # -- Follow-up Message Generation ---------------------------------------

    def generate_follow_up_message(
        self,
        ticket: RenderedTicket,
        requested_fields: list[str],
    ) -> str | None:
        """Generate a deterministic follow-up reply if requested fields are sufficient.

        Returns a follow-up message string if ``requested_fields`` is a superset of
        the ticket's ``required_missing_fields``. Returns None otherwise, or if the
        ticket has no required missing fields.

        Args:
            ticket: The rendered ticket to generate follow-up for.
            requested_fields: Fields the agent requested from the customer.

        Returns:
            Deterministic reply string, or None.
        """
        required = set(ticket.hidden_truth.required_missing_fields)
        if not required:
            return None

        if not required.issubset(requested_fields):
            return None

        # Build reply from pre-computed follow-up replies
        if ticket.follow_up_replies:
            parts = []
            for fld in sorted(required):
                if fld in ticket.follow_up_replies:
                    parts.append(f"{fld}: {ticket.follow_up_replies[fld]}")
            if parts:
                return "Thank you for the follow-up. Here are the details: " + "; ".join(parts)

        # Fallback: generic reply
        return "Thank you for providing the requested information."

    # -- Private rendering helpers ------------------------------------------

    def _resolve_variation_params(
        self, variation_params: dict[str, list[str]], rng: random.Random
    ) -> dict[str, str]:
        """Deterministically pick one value per variation parameter.

        Iterates keys in sorted order for reproducibility.
        """
        result: dict[str, str] = {}
        for key in sorted(variation_params.keys()):
            values = variation_params[key]
            result[key] = rng.choice(values)
        return result

    def _render_template_str(self, template: str, params: dict[str, str]) -> str:
        """Render a template string by substituting {key} placeholders.

        Unknown placeholders are preserved as-is (e.g., ``{unknown}`` remains literal).
        """

        class _KeepMissing(dict[str, str]):
            def __missing__(self, key: str) -> str:
                return f"{{{key}}}"

        return template.format_map(_KeepMissing(params))

    def _render_thread_history(
        self,
        history_pattern: list[dict[str, Any]],
        params: dict[str, str],
        base_dt: datetime,
    ) -> list[dict[str, Any]]:
        """Render thread history entries with timestamps."""
        result = []
        for entry in history_pattern:
            offset_minutes = entry.get("timestamp_offset_minutes", 0)
            ts = base_dt + timedelta(minutes=offset_minutes)
            result.append(
                {
                    "role": entry["role"],
                    "content": self._render_template_str(entry["content_pattern"], params),
                    "timestamp": ts.isoformat().replace("+00:00", "Z"),
                }
            )
        return result

    def _build_follow_up_replies(
        self,
        required_fields: list[str],
        params: dict[str, str],
        seed: int,
    ) -> dict[str, str]:
        """Build deterministic follow-up reply values for required fields.

        Uses variation parameters if available, otherwise generates a
        deterministic placeholder value.
        """
        replies: dict[str, str] = {}
        for fld in required_fields:
            if fld in params:
                replies[fld] = params[fld]
            else:
                # Deterministic placeholder from field name + seed
                replies[fld] = f"{fld}-{seed}"
        return replies
