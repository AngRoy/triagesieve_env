"""TriageSieveEnvironment — OpenEnv Environment subclass.

Implements reset(), step(), and state property per §5.1.

Covers:
- Environment subclass skeleton with SUPPORTS_CONCURRENT_SESSIONS
- reset() — episode initialization via EpisodeEngine
- state property — TriageSieveState from internal bookkeeping
- _validate_action_format() — §17.1 format gate
- _compute_legal_actions() — §12 transition rules per ticket
- Observation assembly helpers (_build_inbox_summaries, _build_observation, etc.)
- Full step() with action dispatch:
  - open_ticket, classify_ticket, set_impact_urgency, route_ticket
  - escalate_ticket, request_information, merge_duplicate, close_ticket
  - skip_turn, finish_episode
- §12 state machine transitions with hard rule enforcement
- §14 deterministic follow-up generation on correct info requests
- §15 gated queue pushback for tech_support_l2 and security_team
- §17.2 step shaping rewards (+0.01 valid, +0.02 correct classify, +0.03 correct info)
- SOP tracker advancement after each successful action
"""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.interfaces import Environment

from ..models import (
    ActionType,
    CloseReason,
    CustomerTier,
    FocusedTicket,
    Impact,
    IssueFamily,
    IssueSubtype,
    InboxSummaryItem,
    QueueId,
    RoutingPolicyCard,
    SlaPolicyCard,
    TriageSieveAction,
    TriageSieveObservation,
    TriageSieveState,
    TaskDifficulty,
    TicketStatus,
    Urgency,
    VALID_FAMILY_SUBTYPES,
    GATED_QUEUES,
)
from .episode_engine import EpisodeEngine, RenderedEpisode, RenderedTicket
from .hint_engine import HintContext, HintEngine
from .policy_graph import SOPGraph, SOPTracker, TicketGuardContext
from .scorer import EpisodeScoringContext, compute_episode_score

__all__ = ["TriageSieveEnvironment"]

_VALID_MODES = frozenset({"eval_strict", "train_guided"})

# Required fields per action_type for format gate validation (§17.1 check 3).
_REQUIRED_FIELDS: dict[ActionType, list[str]] = {
    ActionType.OPEN_TICKET: ["ticket_id"],
    ActionType.CLASSIFY_TICKET: ["ticket_id", "issue_family", "issue_subtype"],
    ActionType.SET_IMPACT_URGENCY: ["ticket_id", "impact", "urgency"],
    ActionType.ROUTE_TICKET: ["ticket_id", "queue_id"],
    ActionType.REQUEST_INFORMATION: ["ticket_id", "requested_fields"],
    ActionType.ESCALATE_TICKET: ["ticket_id", "queue_id"],
    ActionType.MERGE_DUPLICATE: ["ticket_id", "target_ticket_id"],
    ActionType.CLOSE_TICKET: ["ticket_id", "close_reason"],
    ActionType.SKIP_TURN: [],
    ActionType.FINISH_EPISODE: [],
}

# §12 state machine: status → set of allowed action types on that ticket.
_STATUS_LEGAL_ACTIONS: dict[TicketStatus, frozenset[ActionType]] = {
    TicketStatus.NEW: frozenset({ActionType.OPEN_TICKET}),
    TicketStatus.OPENED: frozenset({
        ActionType.CLASSIFY_TICKET,
        ActionType.MERGE_DUPLICATE,
        ActionType.CLOSE_TICKET,
    }),
    TicketStatus.CLASSIFIED: frozenset({
        ActionType.SET_IMPACT_URGENCY,
        ActionType.ROUTE_TICKET,
        ActionType.ESCALATE_TICKET,
        ActionType.REQUEST_INFORMATION,
        ActionType.MERGE_DUPLICATE,
        ActionType.CLOSE_TICKET,
    }),
    TicketStatus.WAITING_FOR_INFO: frozenset({
        ActionType.CLASSIFY_TICKET,
        ActionType.ROUTE_TICKET,
        ActionType.ESCALATE_TICKET,
        ActionType.CLOSE_TICKET,
    }),
    TicketStatus.ROUTED: frozenset({
        ActionType.ESCALATE_TICKET,
        ActionType.CLOSE_TICKET,
    }),
    TicketStatus.ESCALATED: frozenset({
        ActionType.CLOSE_TICKET,
    }),
    TicketStatus.MERGED: frozenset(),
    TicketStatus.CLOSED: frozenset(),
}


class TriageSieveEnvironment(
    Environment[TriageSieveAction, TriageSieveObservation, TriageSieveState]
):
    """Deterministic support-ticket triage environment for agentic RL.

    Implements the OpenEnv Environment contract (§5.1). Tickets are generated
    from seeded archetypes via EpisodeEngine. Scoring is programmatic against
    hidden ground truth.

    Class Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: True — each instance is fully isolated.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._engine = EpisodeEngine()
        self._hint_engine = HintEngine()
        self._episode: RenderedEpisode | None = None
        self._ticket_index: dict[str, RenderedTicket] = {}

        # Per-ticket mutable state (populated by reset)
        self._ticket_states: dict[str, TicketStatus] = {}
        self._ticket_classifications: dict[str, tuple[IssueFamily, IssueSubtype]] = {}
        self._ticket_impact_urgency: dict[str, tuple[Impact, Urgency]] = {}
        self._ticket_routed_to: dict[str, QueueId] = {}
        self._ticket_escalated_to: dict[str, QueueId] = {}
        self._ticket_close_reasons: dict[str, CloseReason] = {}
        self._ticket_actions_log: dict[str, list[str]] = {}
        self._ticket_info_requested: dict[str, list[str]] = {}
        self._ticket_info_received: dict[str, bool] = {}
        self._ticket_merged_to: dict[str, str] = {}
        self._ticket_thread_histories: dict[str, list[dict[str, Any]]] = {}
        self._focused_ticket_id: str | None = None
        self._sop_trackers: dict[str, SOPTracker] = {}

        # Episode-level tracking for terminal scoring (§17.3–§17.5, §19)
        self._invalid_action_count: int = 0
        self._ticket_route_count: dict[str, int] = {}
        self._ticket_templates_used: dict[str, list[str]] = {}
        self._ticket_first_substantive_step: dict[str, int] = {}

        # Episode-level state
        self._step_count: int = 0
        self._action_budget_remaining: int = 0
        self._mode: Literal["eval_strict", "train_guided"] = "eval_strict"
        self._last_action_result: str = "ok"
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._seed: int = 0
        self._last_action: TriageSieveAction | None = None

        # Cached static data (built once per reset from engine data)
        self._routing_policy_cards: list[RoutingPolicyCard] = []
        self._sla_policy_cards: list[SlaPolicyCard] = []
        self._available_templates: list[dict[str, Any]] = []
        self._allowed_queues: list[QueueId] = []

    # ------------------------------------------------------------------
    # OpenEnv contract: reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> TriageSieveObservation:
        """Reset the environment and return the initial observation.

        Args:
            seed: Deterministic seed for episode generation. Defaults to 0.
            episode_id: Optional episode ID override (unused; derived from seed).
            **kwargs: Extra keyword arguments:
                mode: "eval_strict" (default) or "train_guided".
                difficulty: "easy", "medium", or "hard". Defaults to seed-derived.

        Returns:
            Initial TriageSieveObservation with all tickets in NEW status.
        """
        self._reset_rubric()

        effective_seed = seed if seed is not None else 0
        self._seed = effective_seed
        mode_raw = kwargs.get("mode", "eval_strict")
        if mode_raw not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode_raw!r}. Must be one of {sorted(_VALID_MODES)}")
        self._mode = mode_raw

        difficulty_raw = kwargs.get("difficulty")
        difficulty = TaskDifficulty(difficulty_raw) if difficulty_raw else None

        # Generate episode
        self._episode = self._engine.render_episode(effective_seed, difficulty)
        self._ticket_index = {t.ticket_id: t for t in self._episode.tickets}

        # Initialize per-ticket mutable state
        self._ticket_states = {t.ticket_id: TicketStatus.NEW for t in self._episode.tickets}
        self._ticket_classifications = {}
        self._ticket_impact_urgency = {}
        self._ticket_routed_to = {}
        self._ticket_escalated_to = {}
        self._ticket_close_reasons = {}
        self._ticket_actions_log = {t.ticket_id: [] for t in self._episode.tickets}
        self._ticket_info_requested = {}
        self._ticket_info_received = {t.ticket_id: False for t in self._episode.tickets}
        self._ticket_merged_to = {}
        self._ticket_thread_histories = {
            t.ticket_id: list(t.thread_history) for t in self._episode.tickets
        }
        self._focused_ticket_id = None

        # Build SOP trackers
        self._sop_trackers = {}
        for t in self._episode.tickets:
            graph = SOPGraph.from_archetype_data(t.sop_graph)
            self._sop_trackers[t.ticket_id] = SOPTracker(graph)

        # Episode-level state
        self._step_count = 0
        self._action_budget_remaining = self._episode.action_budget
        self._last_action_result = "ok"
        self._cumulative_reward = 0.0
        self._done = False
        self._last_action = None

        # Reset terminal scoring trackers
        self._invalid_action_count = 0
        self._ticket_route_count = {}
        self._ticket_templates_used = {}
        self._ticket_first_substantive_step = {}

        # Cache static observation data
        self._routing_policy_cards = self._build_routing_policy_cards()
        self._sla_policy_cards = self._build_sla_policy_cards()
        self._available_templates = self._build_available_templates()
        self._allowed_queues = list(QueueId)

        return self._build_observation(reward=None, done=False)

    # ------------------------------------------------------------------
    # OpenEnv contract: step()
    # ------------------------------------------------------------------

    def step(
        self,
        action: TriageSieveAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TriageSieveObservation:
        """Take a step in the environment.

        Handles format gate validation, legality checks, budget tracking,
        and dispatches to the appropriate action handler. Invalid actions
        never crash — they return a penalty and a descriptive error message.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        self._action_budget_remaining = max(0, self._action_budget_remaining - 1)

        # Always record what the agent tried, even on failure, for hint coherence
        self._last_action = action

        # Format gate (§17.1)
        error = self._validate_action_format(action)
        if error is not None:
            self._last_action_result = error
            self._invalid_action_count += 1
            step_reward = -0.02
            self._cumulative_reward += step_reward
            done = self._action_budget_remaining <= 0
            self._done = done
            if done:
                return self._build_observation(
                    reward=self._compute_terminal_score(), done=True
                )
            return self._build_observation(reward=step_reward, done=False)

        # Check legal action against ticket state (§17.1 check 5)
        if action.action_type not in (ActionType.SKIP_TURN, ActionType.FINISH_EPISODE):
            ticket_id = action.ticket_id
            if ticket_id is None:
                raise RuntimeError("ticket_id is None after format gate passed")
            legal = self._compute_legal_actions(ticket_id)
            if action.action_type not in legal:
                self._last_action_result = (
                    f"Illegal action: {action.action_type.value} not allowed "
                    f"for ticket {ticket_id} in status {self._ticket_states[ticket_id].value}"
                )
                self._invalid_action_count += 1
                step_reward = -0.02
                self._cumulative_reward += step_reward
                done = self._action_budget_remaining <= 0
                self._done = done
                if done:
                    return self._build_observation(
                        reward=self._compute_terminal_score(), done=True
                    )
                return self._build_observation(reward=step_reward, done=False)

        # Dispatch action and track per-ticket metrics for terminal scoring
        step_reward = self._execute_action(action)
        self._cumulative_reward += step_reward
        self._track_action_metrics(action)

        # Check termination
        done = (
            self._done
            or self._action_budget_remaining <= 0
            or action.action_type == ActionType.FINISH_EPISODE
        )
        self._done = done

        # On terminal step, compute the full episode score (§17.6)
        if done:
            final_score = self._compute_terminal_score()
            return self._build_observation(reward=final_score, done=True)

        return self._build_observation(reward=step_reward, done=done)

    # ------------------------------------------------------------------
    # OpenEnv contract: state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> TriageSieveState:
        """Return current episode state for debugging and TRL integration."""
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return TriageSieveState(
            episode_id=self._episode.episode_id,
            step_count=self._step_count,
            task_difficulty=self._episode.task_difficulty,
            seed=self._seed,
            total_tickets=len(self._episode.tickets),
            action_budget=self._episode.action_budget,
            action_budget_remaining=self._action_budget_remaining,
            mode=self._mode,
            tickets_summary=[
                {
                    "ticket_id": t.ticket_id,
                    "status": self._ticket_states[t.ticket_id].value,
                }
                for t in self._episode.tickets
            ],
        )

    # ------------------------------------------------------------------
    # Format Gate (§17.1)
    # ------------------------------------------------------------------

    def _validate_action_format(self, action: TriageSieveAction) -> str | None:
        """Validate action format per §17.1.

        Checks:
        1. Action parses as valid TriageSieveAction (guaranteed by Pydantic).
        2. All enum values are valid (guaranteed by Pydantic).
        3. Required arguments for action_type are present.
        4. ticket_id exists in the inbox (when required).
        5. Family-subtype pairing is valid for classify actions.

        Args:
            action: The action to validate.

        Returns:
            None if valid, or a descriptive error string.
        """
        required = _REQUIRED_FIELDS.get(action.action_type, [])

        # Check 3: required fields present (None or empty list both fail)
        for field_name in required:
            value = getattr(action, field_name, None)
            if value is None or (isinstance(value, list) and len(value) == 0):
                return f"Missing required field: {field_name} for {action.action_type.value}"

        # Check 4: ticket_id exists (when required)
        if "ticket_id" in required and action.ticket_id is not None:
            if action.ticket_id not in self._ticket_index:
                return f"Ticket ID does not exist: {action.ticket_id} not found in inbox"

        # Check 5: family-subtype validation for classify actions
        if action.action_type == ActionType.CLASSIFY_TICKET:
            if action.issue_family is not None and action.issue_subtype is not None:
                valid_subtypes = VALID_FAMILY_SUBTYPES.get(action.issue_family, frozenset())
                if action.issue_subtype not in valid_subtypes:
                    return (
                        f"Invalid family-subtype pair: {action.issue_subtype.value} "
                        f"is not a valid subtype for {action.issue_family.value}"
                    )

        return None

    # ------------------------------------------------------------------
    # Legal Actions (§12)
    # ------------------------------------------------------------------

    def _compute_legal_actions(self, ticket_id: str) -> list[ActionType]:
        """Compute legal action types for a specific ticket based on its status.

        Implements §12 state machine transition rules.

        Args:
            ticket_id: The ticket to compute legal actions for.

        Returns:
            List of ActionType values that are legal for this ticket.
        """
        status = self._ticket_states.get(ticket_id)
        if status is None:
            return []
        return sorted(_STATUS_LEGAL_ACTIONS.get(status, frozenset()), key=lambda a: a.value)

    def _compute_global_legal_actions(self) -> list[ActionType]:
        """Compute the union of all per-ticket legal actions plus SKIP/FINISH.

        Returns:
            Deduplicated list of all currently legal ActionType values.
        """
        legal: set[ActionType] = {ActionType.SKIP_TURN, ActionType.FINISH_EPISODE}
        for ticket_id in self._ticket_states:
            legal.update(self._compute_legal_actions(ticket_id))
        return sorted(legal, key=lambda a: a.value)

    # ------------------------------------------------------------------
    # Per-action metric tracking (for terminal scoring)
    # ------------------------------------------------------------------

    # Actions that count as "substantive" for priority-order scoring (§19).
    _SUBSTANTIVE_ACTIONS: frozenset[ActionType] = frozenset({
        ActionType.CLASSIFY_TICKET,
        ActionType.ROUTE_TICKET,
        ActionType.CLOSE_TICKET,
    })

    def _track_action_metrics(self, action: TriageSieveAction) -> None:
        """Track metrics needed for terminal scoring after each successful action."""
        tid = action.ticket_id
        if tid is None:
            return

        # Track template usage
        if action.template_id is not None:
            self._ticket_templates_used.setdefault(tid, []).append(action.template_id)

        # Track route count for reassignment penalty (§17.5)
        if action.action_type == ActionType.ROUTE_TICKET:
            self._ticket_route_count[tid] = self._ticket_route_count.get(tid, 0) + 1

        # Track first substantive step for priority-order scoring (§19)
        if (
            action.action_type in self._SUBSTANTIVE_ACTIONS
            and tid not in self._ticket_first_substantive_step
        ):
            self._ticket_first_substantive_step[tid] = self._step_count

    def _compute_terminal_score(self) -> float:
        """Compute the full episode score per §17.6.

        Builds EpisodeScoringContext from environment state and delegates
        to compute_episode_score(). Returns the final [0, 1] score.
        """
        ctx = EpisodeScoringContext(
            tickets=list(self._ticket_index.values()),
            ticket_states=dict(self._ticket_states),
            ticket_classifications=dict(self._ticket_classifications),
            ticket_impact_urgency=dict(self._ticket_impact_urgency),
            ticket_routed_to=dict(self._ticket_routed_to),
            ticket_escalated_to=dict(self._ticket_escalated_to),
            ticket_close_reasons=dict(self._ticket_close_reasons),
            ticket_info_requested=dict(self._ticket_info_requested),
            ticket_info_received=dict(self._ticket_info_received),
            ticket_merged_to=dict(self._ticket_merged_to),
            ticket_templates_used=dict(self._ticket_templates_used),
            sop_trackers=dict(self._sop_trackers),
            invalid_action_count=self._invalid_action_count,
            ticket_route_count=dict(self._ticket_route_count),
            ticket_first_substantive_step=dict(self._ticket_first_substantive_step),
        )
        breakdown = compute_episode_score(ctx)
        return breakdown.final_score

    # ------------------------------------------------------------------
    # Action Execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: TriageSieveAction) -> float:
        """Execute a validated, legal action. Returns step reward."""
        if action.action_type == ActionType.SKIP_TURN:
            self._last_action_result = "ok"
            return -0.01  # wasted step

        if action.action_type == ActionType.FINISH_EPISODE:
            self._last_action_result = "ok"
            self._done = True
            return 0.0

        # Narrowing: ticket_id is guaranteed non-None by format gate + legality
        # check in step() for all ticket-based actions.
        ticket_id = action.ticket_id
        if ticket_id is None:
            raise RuntimeError("Unreachable: ticket_id None after step() validation")

        dispatch = {
            ActionType.OPEN_TICKET: self._handle_open_ticket,
            ActionType.CLASSIFY_TICKET: self._handle_classify_ticket,
            ActionType.SET_IMPACT_URGENCY: self._handle_set_impact_urgency,
            ActionType.ROUTE_TICKET: self._handle_route_ticket,
            ActionType.ESCALATE_TICKET: self._handle_escalate_ticket,
            ActionType.REQUEST_INFORMATION: self._handle_request_information,
            ActionType.MERGE_DUPLICATE: self._handle_merge_duplicate,
            ActionType.CLOSE_TICKET: self._handle_close_ticket,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            raise RuntimeError(f"No handler for action_type: {action.action_type}")
        return handler(action, ticket_id)

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------

    def _handle_open_ticket(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle OPEN_TICKET: new → opened."""
        self._ticket_states[ticket_id] = TicketStatus.OPENED
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append("opened")
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.OPEN_TICKET)
        return 0.01

    def _handle_classify_ticket(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle CLASSIFY_TICKET: opened/waiting_for_info → classified."""
        if action.issue_family is None or action.issue_subtype is None:
            raise RuntimeError("issue_family/issue_subtype None after format gate")

        self._ticket_states[ticket_id] = TicketStatus.CLASSIFIED
        self._ticket_classifications[ticket_id] = (action.issue_family, action.issue_subtype)
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(
            f"classified as {action.issue_family.value}/{action.issue_subtype.value}"
        )
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.CLASSIFY_TICKET)

        # Step shaping: +0.02 if correct, +0.01 otherwise
        ticket = self._ticket_index[ticket_id]
        ht = ticket.hidden_truth
        if action.issue_family == ht.issue_family and action.issue_subtype == ht.issue_subtype:
            return 0.02
        return 0.01

    def _handle_set_impact_urgency(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle SET_IMPACT_URGENCY: classified → classified (no transition)."""
        if action.impact is None or action.urgency is None:
            raise RuntimeError("impact/urgency None after format gate")

        self._ticket_impact_urgency[ticket_id] = (action.impact, action.urgency)
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(
            f"set impact={action.impact.value}, urgency={action.urgency.value}"
        )
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.SET_IMPACT_URGENCY)
        return 0.01

    def _check_gated_queue_prerequisites(self, ticket_id: str, queue_id: QueueId) -> str | None:
        """Check if prerequisites are met for a gated queue.

        Returns None if the queue is not gated or prerequisites are met.
        Returns a pushback error string if prerequisites are missing.
        """
        if queue_id not in GATED_QUEUES:
            return None

        missing: list[str] = []
        if ticket_id not in self._ticket_classifications:
            missing.append("classification_set")
        if ticket_id not in self._ticket_impact_urgency:
            missing.append("impact_urgency_set")

        if missing:
            return f"Pushback: {queue_id.value} requires prerequisites: {', '.join(missing)}"
        return None

    def _handle_route_ticket(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle ROUTE_TICKET: classified/waiting_for_info → routed."""
        if action.queue_id is None:
            raise RuntimeError("queue_id None after format gate")

        # Check gated queue prerequisites (§15)
        pushback = self._check_gated_queue_prerequisites(ticket_id, action.queue_id)
        if pushback is not None:
            self._last_action_result = pushback
            self._ticket_actions_log[ticket_id].append(
                f"route to {action.queue_id.value} pushed back"
            )
            return -0.03

        self._ticket_states[ticket_id] = TicketStatus.ROUTED
        self._ticket_routed_to[ticket_id] = action.queue_id
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(f"routed to {action.queue_id.value}")
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.ROUTE_TICKET)
        return 0.01

    def _handle_escalate_ticket(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle ESCALATE_TICKET: classified/waiting_for_info/routed → escalated."""
        if action.queue_id is None:
            raise RuntimeError("queue_id None after format gate")

        # Check gated queue prerequisites (§15)
        pushback = self._check_gated_queue_prerequisites(ticket_id, action.queue_id)
        if pushback is not None:
            self._last_action_result = pushback
            self._ticket_actions_log[ticket_id].append(
                f"escalate to {action.queue_id.value} pushed back"
            )
            return -0.03

        self._ticket_states[ticket_id] = TicketStatus.ESCALATED
        self._ticket_escalated_to[ticket_id] = action.queue_id
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(f"escalated to {action.queue_id.value}")
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.ESCALATE_TICKET)
        return 0.01

    def _handle_request_information(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle REQUEST_INFORMATION: classified → waiting_for_info.

        If correct fields requested, generates deterministic follow-up (§14).
        """
        if action.requested_fields is None:
            raise RuntimeError("requested_fields None after format gate")

        # Advance SOP tracker BEFORE updating info_requested state.
        # SOP edges to request_info nodes are guarded by "missing_*" guards
        # (e.g. "missing_order_id") which fire when NOT yet requested.
        # Advancing here uses the pre-action guard context (fields not yet
        # marked as requested), so the guard correctly evaluates True.
        self._advance_sop_tracker(ticket_id, ActionType.REQUEST_INFORMATION)

        self._ticket_states[ticket_id] = TicketStatus.WAITING_FOR_INFO
        self._ticket_info_requested[ticket_id] = list(action.requested_fields)
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(
            f"requested info: {', '.join(action.requested_fields)}"
        )
        self._last_action_result = "ok"

        # Try to generate deterministic follow-up (§14)
        ticket = self._ticket_index[ticket_id]
        follow_up = self._engine.generate_follow_up_message(ticket, action.requested_fields)

        if follow_up is not None:
            # Correct fields: append follow-up to environment-owned thread history copy.
            # v1: static timestamp; elapsed-time tracking not yet implemented.
            self._ticket_thread_histories[ticket_id].append({
                "role": "customer",
                "content": follow_up,
                "timestamp": self._episode.base_time if self._episode else "",
            })
            self._ticket_info_received[ticket_id] = True
            reward = 0.03  # correct info request
        else:
            reward = 0.01  # valid but wrong fields

        return reward

    def _handle_merge_duplicate(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle MERGE_DUPLICATE: opened/classified → merged.

        Validates that hidden truth confirms is_duplicate and duplicate_of matches.
        The target_ticket_id may be external to the current episode (a historical
        ticket not in the inbox), which is by design in the archetype data.
        """
        if action.target_ticket_id is None:
            raise RuntimeError("target_ticket_id None after format gate")

        ticket = self._ticket_index[ticket_id]
        ht = ticket.hidden_truth

        # Validate merge eligibility
        if not ht.is_duplicate:
            self._last_action_result = (
                f"Merge failed: ticket {ticket_id} is not a duplicate"
            )
            self._ticket_actions_log[ticket_id].append("merge failed: not a duplicate")
            return -0.02

        if ht.duplicate_of != action.target_ticket_id:
            self._last_action_result = (
                f"Merge failed: target {action.target_ticket_id} does not match "
                f"the original ticket for {ticket_id}"
            )
            self._ticket_actions_log[ticket_id].append("merge failed: wrong target")
            return -0.02

        self._ticket_states[ticket_id] = TicketStatus.MERGED
        self._ticket_merged_to[ticket_id] = action.target_ticket_id
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(
            f"merged into {action.target_ticket_id}"
        )
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.MERGE_DUPLICATE)
        return 0.01

    def _handle_close_ticket(self, action: TriageSieveAction, ticket_id: str) -> float:
        """Handle CLOSE_TICKET: multiple sources → closed.

        Enforces §12 hard rules:
        - From opened: only NON_ACTIONABLE close reason allowed.
        - Cannot close with RESOLVED if required_missing_fields unfulfilled
          (unless non-actionable or duplicate close reason).
        """
        if action.close_reason is None:
            raise RuntimeError("close_reason None after format gate")

        status = self._ticket_states[ticket_id]
        ticket = self._ticket_index[ticket_id]
        ht = ticket.hidden_truth

        # From opened: only non-actionable close allowed
        if status == TicketStatus.OPENED:
            if action.close_reason != CloseReason.NON_ACTIONABLE:
                self._last_action_result = (
                    f"Close failed: from opened status, only non_actionable close is allowed, "
                    f"got {action.close_reason.value}"
                )
                self._ticket_actions_log[ticket_id].append("close failed: invalid reason from opened")
                return -0.02

        # Missing fields guard: cannot close RESOLVED while required fields unfulfilled.
        # Non-actionable, duplicate, feature request, and no-response are exempt since
        # these closures don't require the support information to be fulfilled.
        exempt_reasons = {
            CloseReason.NON_ACTIONABLE,
            CloseReason.DUPLICATE,
            CloseReason.FEATURE_REQUEST,
            CloseReason.NO_RESPONSE,
        }
        if action.close_reason not in exempt_reasons:
            if ht.required_missing_fields and not self._ticket_info_received.get(ticket_id, False):
                self._last_action_result = (
                    f"Close failed: required fields {ht.required_missing_fields} not yet fulfilled"
                )
                self._ticket_actions_log[ticket_id].append("close failed: missing fields unfulfilled")
                return -0.02

        self._ticket_states[ticket_id] = TicketStatus.CLOSED
        self._ticket_close_reasons[ticket_id] = action.close_reason
        self._focused_ticket_id = ticket_id
        self._ticket_actions_log[ticket_id].append(f"closed as {action.close_reason.value}")
        self._last_action_result = "ok"
        self._advance_sop_tracker(ticket_id, ActionType.CLOSE_TICKET)
        return 0.01

    # ------------------------------------------------------------------
    # SOP Tracker Helper
    # ------------------------------------------------------------------

    def _advance_sop_tracker(self, ticket_id: str, action_type: ActionType) -> None:
        """Advance the SOP tracker for a ticket after a successful action.

        Builds guard context from current state, attempts to advance by action
        type, then auto-advances through non-checkpoint nodes.

        Args:
            ticket_id: The ticket whose tracker to advance.
            action_type: The action type that was just executed.
        """
        tracker = self._sop_trackers.get(ticket_id)
        if tracker is None:
            return
        ctx = self._build_guard_context(ticket_id)
        tracker.try_advance_by_action(action_type, ctx)
        tracker.auto_advance_non_checkpoints(ctx)

    # ------------------------------------------------------------------
    # Guard Context Builder
    # ------------------------------------------------------------------

    def _build_guard_context(self, ticket_id: str) -> TicketGuardContext:
        """Build a TicketGuardContext snapshot for SOP guard evaluation.

        Args:
            ticket_id: The ticket to build context for.

        Returns:
            TicketGuardContext reflecting current ticket state.
        """
        ticket = self._ticket_index[ticket_id]
        return TicketGuardContext(
            classification_set=ticket_id in self._ticket_classifications,
            impact_urgency_set=ticket_id in self._ticket_impact_urgency,
            missing_fields_requested=ticket_id in self._ticket_info_requested,
            info_received=self._ticket_info_received.get(ticket_id, False),
            escalation_required=ticket.hidden_truth.escalation_required,
            duplicate_confirmed=ticket.hidden_truth.is_duplicate,
        )

    # ------------------------------------------------------------------
    # Observation Assembly Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float | None,
        done: bool,
    ) -> TriageSieveObservation:
        """Assemble a complete TriageSieveObservation.

        Args:
            reward: Step reward (None for initial observation).
            done: Whether the episode is finished.

        Returns:
            Fully populated TriageSieveObservation.
        """
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return TriageSieveObservation(
            done=done,
            reward=reward,
            metadata={},
            inbox_summaries=self._build_inbox_summaries(),
            focused_ticket=self._build_focused_ticket(),
            available_templates=self._available_templates,
            allowed_queues=self._allowed_queues,
            routing_policy_cards=self._routing_policy_cards,
            sla_policy_cards=self._sla_policy_cards,
            legal_actions=self._compute_global_legal_actions(),
            action_budget_remaining=self._action_budget_remaining,
            step_count=self._step_count,
            current_time=self._episode.base_time,
            last_action_result=self._last_action_result,
            task_difficulty=self._episode.task_difficulty,
            hint=self._generate_hint(),
        )

    def _generate_hint(self) -> str | None:
        """Generate a guided-mode hint, or None.

        Returns None immediately if mode is not train_guided, or if there
        is no last action with a ticket_id to evaluate against.
        """
        if self._mode != "train_guided":
            return None
        if self._last_action is None:
            return None

        ticket_id = self._last_action.ticket_id
        if ticket_id is None:
            return None

        ticket = self._ticket_index.get(ticket_id)
        if ticket is None:
            return None

        ht = ticket.hidden_truth
        ctx = HintContext(
            last_action=self._last_action,
            last_action_result=self._last_action_result,
            ticket_status=self._ticket_states.get(ticket_id),
            hidden_truth=ht,
            classification_set=self._ticket_classifications.get(ticket_id),
            impact_urgency_set=self._ticket_impact_urgency.get(ticket_id),
            info_requested=ticket_id in self._ticket_info_requested,
            info_received=self._ticket_info_received.get(ticket_id, False),
            routed_to=self._ticket_routed_to.get(ticket_id),
            is_duplicate_truth=ht.is_duplicate,
            non_actionable_subtype=ht.non_actionable_subtype,
        )
        return self._hint_engine.generate_hint(ctx)

    def _build_inbox_summaries(self) -> list[InboxSummaryItem]:
        """Build inbox summary items from rendered tickets and current state."""
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        items: list[InboxSummaryItem] = []
        for ticket in self._episode.tickets:
            sla = self._engine.get_sla_for_tier(ticket.customer_tier)
            # v1: uses full SLA deadline as proxy for remaining time (no elapsed tracking yet)
            sla_remaining = sla["response_deadline_minutes"] if sla else None
            items.append(
                InboxSummaryItem(
                    ticket_id=ticket.ticket_id,
                    subject=ticket.subject,
                    sender_email=ticket.sender_email,
                    received_at=ticket.received_at,
                    status=self._ticket_states[ticket.ticket_id],
                    customer_tier=ticket.customer_tier,
                    has_attachment=ticket.has_attachment,
                    sla_remaining_minutes=sla_remaining,
                    short_preview=ticket.body[:80],
                )
            )
        return items

    def _build_focused_ticket(self) -> FocusedTicket | None:
        """Build FocusedTicket for the currently focused ticket, or None."""
        if self._focused_ticket_id is None:
            return None

        ticket = self._ticket_index.get(self._focused_ticket_id)
        if ticket is None:
            return None

        thread = self._ticket_thread_histories.get(ticket.ticket_id, [])
        # Show the most recent message: the last thread entry if any, otherwise the original body.
        latest = thread[-1]["content"] if thread else ticket.body

        return FocusedTicket(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            latest_message=latest,
            thread_history=list(thread),
            attachments=list(ticket.attachments),
            visible_internal_notes=list(ticket.internal_notes),
            prior_actions_taken=list(self._ticket_actions_log.get(ticket.ticket_id, [])),
        )

    def _build_routing_policy_cards(self) -> list[RoutingPolicyCard]:
        """Build routing policy cards from engine's routing rules data."""
        cards: list[RoutingPolicyCard] = []
        for queue_id_str, rule in self._engine.routing_rules.items():
            cards.append(
                RoutingPolicyCard(
                    queue_id=QueueId(queue_id_str),
                    description=rule["description"],
                    prerequisites=list(rule.get("prerequisites", [])),
                    handles_families=[
                        IssueFamily(f) for f in rule.get("handles_families", [])
                    ],
                )
            )
        return cards

    def _build_sla_policy_cards(self) -> list[SlaPolicyCard]:
        """Build SLA policy cards from engine's SLA rules data."""
        return [
            SlaPolicyCard(
                tier=CustomerTier(rule["tier"]),
                response_deadline_minutes=rule["response_deadline_minutes"],
                resolution_deadline_minutes=rule["resolution_deadline_minutes"],
            )
            for rule in self._engine.sla_rules
        ]

    def _build_available_templates(self) -> list[dict[str, Any]]:
        """Build template list for observation from engine's template data."""
        return [
            {
                "template_id": t["template_id"],
                "name": t["name"],
                "description": t["description"],
                "applies_to": t["applies_to"],
            }
            for t in self._engine.templates
        ]
