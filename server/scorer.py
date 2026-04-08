"""Terminal business scorer, UJCS-OpenEnv scorer, priority-order scorer, and penalty logic.

Implements the full reward design from section 17:
- Format gate support (section 17.1) — counted via invalid_action_count in context
- Step shaping (section 17.2) — handled in environment, NOT re-computed here
- Terminal business score with priority weighting (section 17.3)
- UJCS-OpenEnv process adherence (section 17.4)
- Episode penalties (section 17.5)
- Final score formula (section 17.6):
    FinalScore = TerminalBusinessScore + 0.15 * UJCS_OpenEnv - EpisodePenalties
    Clamped to [0, 1].
- Priority-order score (section 19) — reported in trace, not in final formula.
- Per-ticket breakdown for structured traces (section 24).

Python 3.11+, frozen dataclasses, no runtime LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    CloseReason,
    HiddenTicketTruth,
    Impact,
    IssueFamily,
    IssueSubtype,
    Priority,
    PRIORITY_WEIGHTS,
    QueueId,
    TicketStatus,
    Urgency,
)
from ..server.episode_engine import RenderedTicket
from ..server.policy_graph import SOPTracker, compute_ujcs

__all__ = [
    # Constants
    "TERMINAL_BUSINESS_SCORE_MAX",
    "UJCS_WEIGHT",
    "COMPONENT_WEIGHTS",
    "PENALTY_INVALID_ACTION",
    "PENALTY_AVOIDABLE_REASSIGNMENT",
    "PENALTY_UNNECESSARY_ESCALATION",
    "PENALTY_SLA_HIGH",
    "PENALTY_SLA_CRITICAL",
    # Data structures
    "PerTicketScore",
    "EpisodePenalties",
    "ScoreBreakdown",
    "EpisodeScoringContext",
    # Functions
    "compute_episode_score",
    "score_ticket",
]


# ---------------------------------------------------------------------------
# Constants (section 17, exact values from CLAUDE.md)
# ---------------------------------------------------------------------------

TERMINAL_BUSINESS_SCORE_MAX: float = 0.85
"""Maximum terminal business score (sum of all 8 component weights)."""

UJCS_WEIGHT: float = 0.15
"""Weight of UJCS-OpenEnv contribution in the final score formula."""

COMPONENT_WEIGHTS: dict[str, float] = {
    "classification": 0.15,
    "impact_urgency": 0.15,
    "queue": 0.20,
    "missing_info": 0.10,
    "escalation": 0.10,
    "duplicate_non_actionable": 0.05,
    "template": 0.05,
    "terminal_status": 0.05,
}
"""Per-component weights for the terminal business score (section 17.3)."""

PENALTY_INVALID_ACTION: float = 0.03
"""Per-occurrence penalty for invalid actions (section 17.5)."""

PENALTY_AVOIDABLE_REASSIGNMENT: float = 0.05
"""Per-occurrence penalty for avoidable ticket reassignment (section 17.5)."""

PENALTY_UNNECESSARY_ESCALATION: float = 0.05
"""Per-occurrence penalty for unnecessary escalation (section 17.5)."""

PENALTY_SLA_HIGH: float = 0.05
"""SLA mishandling penalty for high-priority tickets (section 17.5)."""

PENALTY_SLA_CRITICAL: float = 0.10
"""SLA mishandling penalty for critical-priority tickets (section 17.5)."""

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerTicketScore:
    """Scoring breakdown for a single ticket.

    Each component stores its weighted contribution (component_score * weight),
    so the raw_score is the direct sum of all 8 components.
    """

    ticket_id: str
    classification: float
    impact_urgency: float
    queue: float
    missing_info: float
    escalation: float
    duplicate_non_actionable: float
    template: float
    terminal_status: float
    raw_score: float  # sum of 8 components, max 0.85
    priority_weight: float  # from PRIORITY_WEIGHTS
    ujcs: float  # from compute_ujcs, in [0, 1]
    wrong_parameterizations: int


@dataclass(frozen=True)
class EpisodePenalties:
    """Aggregated episode-level penalties (section 17.5)."""

    invalid_action_count: int
    avoidable_reassignment_count: int
    unnecessary_escalation_count: int
    sla_mishandling_penalties: float  # sum of per-ticket SLA penalties
    total_penalty: float  # sum of all penalty sources


@dataclass(frozen=True)
class ScoreBreakdown:
    """Complete scoring breakdown for an episode (section 24 trace format)."""

    per_ticket_scores: dict[str, PerTicketScore]
    terminal_business_score: float  # priority-weighted avg, max 0.85
    ujcs_openenv: float  # priority-weighted avg of per-ticket UJCS, in [0, 1]
    ujcs_contribution: float  # UJCS_WEIGHT * ujcs_openenv
    episode_penalties: EpisodePenalties
    priority_order_score: float  # section 19, trace only
    final_score: float  # clamped [0, 1]
    reassignment_count: int
    invalid_action_count: int


@dataclass
class EpisodeScoringContext:
    """All environment state needed by the scorer.

    The environment populates this at episode end and passes it to
    ``compute_episode_score()``.
    """

    tickets: list[RenderedTicket]
    ticket_states: dict[str, TicketStatus]
    ticket_classifications: dict[str, tuple[IssueFamily, IssueSubtype]]
    ticket_impact_urgency: dict[str, tuple[Impact, Urgency]]
    ticket_routed_to: dict[str, QueueId]
    ticket_escalated_to: dict[str, QueueId]
    ticket_close_reasons: dict[str, CloseReason]
    ticket_info_requested: dict[str, list[str]]
    ticket_info_received: dict[str, bool]
    ticket_merged_to: dict[str, str]
    ticket_templates_used: dict[str, list[str]]
    sop_trackers: dict[str, SOPTracker]
    invalid_action_count: int
    ticket_route_count: dict[str, int]  # times each ticket was routed
    ticket_first_substantive_step: dict[str, int]  # step number of first substantive action


# ---------------------------------------------------------------------------
# Per-ticket scoring (section 17.3)
# ---------------------------------------------------------------------------


def _score_classification(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> tuple[float, bool]:
    """Score classification correctness.

    Returns:
        (weighted_score, wrong_parameterization): the weighted contribution and
        whether the agent classified but got it wrong.
    """
    classification = ctx.ticket_classifications.get(ticket_id)
    if classification is None:
        return 0.0, False

    family, subtype = classification
    if family == ht.issue_family and subtype == ht.issue_subtype:
        return COMPONENT_WEIGHTS["classification"], False
    if family == ht.issue_family:
        return COMPONENT_WEIGHTS["classification"] * 0.5, True
    return 0.0, True


def _score_impact_urgency(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> tuple[float, bool]:
    """Score impact/urgency correctness.

    Non-actionable and duplicate tickets do not require impact/urgency
    assessment — award full marks when the action is not applicable.

    Returns:
        (weighted_score, wrong_parameterization).
    """
    # Not applicable for non-actionable, duplicate, or tickets whose SOP
    # doesn't require impact/urgency (feature requests, spam filter destinations).
    if (
        ht.non_actionable_subtype is not None
        or ht.is_duplicate
        or ht.required_queue in (QueueId.SPAM_FILTER, QueueId.SALES_OR_FEATURE_REQUESTS)
    ):
        return COMPONENT_WEIGHTS["impact_urgency"], False

    iu = ctx.ticket_impact_urgency.get(ticket_id)
    if iu is None:
        return 0.0, False

    agent_impact, agent_urgency = iu
    correct_count = (agent_impact == ht.impact) + (agent_urgency == ht.urgency)
    if correct_count == 2:
        return COMPONENT_WEIGHTS["impact_urgency"], False
    if correct_count == 1:
        return COMPONENT_WEIGHTS["impact_urgency"] * 0.5, True
    return 0.0, True


def _score_queue(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> tuple[float, bool]:
    """Score queue routing correctness.

    Checks both direct routing and escalation destination against required_queue.
    Non-actionable and duplicate tickets do not require routing — award full
    marks when the action is not applicable.

    Returns:
        (weighted_score, wrong_parameterization).
    """
    # Not applicable for non-actionable or duplicate tickets
    if ht.non_actionable_subtype is not None or ht.is_duplicate:
        return COMPONENT_WEIGHTS["queue"], False

    routed_to = ctx.ticket_routed_to.get(ticket_id)
    escalated_to = ctx.ticket_escalated_to.get(ticket_id)

    # Check direct routing first, then escalation target
    actual_queue = routed_to or escalated_to
    if actual_queue is None:
        return 0.0, False

    if actual_queue == ht.required_queue:
        return COMPONENT_WEIGHTS["queue"], False
    return 0.0, True


def _score_missing_info(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> float:
    """Score missing-info handling."""
    if not ht.required_missing_fields:
        # No missing fields required — full score
        return COMPONENT_WEIGHTS["missing_info"]

    requested = ctx.ticket_info_requested.get(ticket_id)
    received = ctx.ticket_info_received.get(ticket_id, False)

    if requested is not None and received:
        # Requested correct fields and got response
        return COMPONENT_WEIGHTS["missing_info"]
    if requested is not None and not received:
        # Per spec section 14: received=True only if requested_fields covers required fields.
        # If we get here, the agent requested info but with insufficient fields.
        return COMPONENT_WEIGHTS["missing_info"] * 0.25
    # Never requested
    return 0.0


def _score_escalation(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> float:
    """Score escalation correctness.

    Non-actionable and duplicate tickets never require escalation — award
    full marks when not applicable.
    """
    # Not applicable for non-actionable or duplicate tickets
    if ht.non_actionable_subtype is not None or ht.is_duplicate:
        return COMPONENT_WEIGHTS["escalation"]

    escalated_to = ctx.ticket_escalated_to.get(ticket_id)
    agent_escalated = escalated_to is not None

    if ht.escalation_required:
        if agent_escalated and escalated_to == ht.escalation_target:
            return COMPONENT_WEIGHTS["escalation"]
        if agent_escalated:
            return COMPONENT_WEIGHTS["escalation"] * 0.25
        return 0.0

    if not agent_escalated:
        return COMPONENT_WEIGHTS["escalation"]
    return 0.0


def _score_duplicate_non_actionable(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> float:
    """Score duplicate/non-actionable handling."""
    if ht.is_duplicate:
        merged_to = ctx.ticket_merged_to.get(ticket_id)
        if merged_to == ht.duplicate_of:
            return COMPONENT_WEIGHTS["duplicate_non_actionable"]
        return 0.0

    if ht.non_actionable_subtype is not None:
        close_reason = ctx.ticket_close_reasons.get(ticket_id)
        if close_reason == CloseReason.NON_ACTIONABLE:
            return COMPONENT_WEIGHTS["duplicate_non_actionable"]
        return 0.0

    # Neither duplicate nor non-actionable — full score (not applicable)
    return COMPONENT_WEIGHTS["duplicate_non_actionable"]


def _score_template(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> float:
    """Score template choice correctness."""
    if not ht.correct_template_ids:
        # No templates expected — full score
        return COMPONENT_WEIGHTS["template"]

    used = ctx.ticket_templates_used.get(ticket_id, [])
    if any(tmpl in ht.correct_template_ids for tmpl in used):
        return COMPONENT_WEIGHTS["template"]
    return 0.0


def _score_terminal_status(
    ctx: EpisodeScoringContext, ticket_id: str, ht: HiddenTicketTruth
) -> float:
    """Score terminal status correctness."""
    current_status = ctx.ticket_states.get(ticket_id)
    if current_status == ht.gold_terminal_status:
        return COMPONENT_WEIGHTS["terminal_status"]
    return 0.0


def score_ticket(ctx: EpisodeScoringContext, ticket: RenderedTicket) -> PerTicketScore:
    """Compute all scoring components for a single ticket.

    Args:
        ctx: Episode scoring context with all environment state.
        ticket: The rendered ticket (with hidden truth).

    Returns:
        PerTicketScore with all 8 components, raw_score, UJCS, and priority weight.
    """
    tid = ticket.ticket_id
    ht = ticket.hidden_truth

    classification, classify_wrong = _score_classification(ctx, tid, ht)
    impact_urgency, iu_wrong = _score_impact_urgency(ctx, tid, ht)
    queue, queue_wrong = _score_queue(ctx, tid, ht)
    missing_info = _score_missing_info(ctx, tid, ht)
    escalation = _score_escalation(ctx, tid, ht)
    dup_na = _score_duplicate_non_actionable(ctx, tid, ht)
    template = _score_template(ctx, tid, ht)
    terminal_status = _score_terminal_status(ctx, tid, ht)

    raw_score = (
        classification
        + impact_urgency
        + queue
        + missing_info
        + escalation
        + dup_na
        + template
        + terminal_status
    )

    # Count wrong parameterizations for UJCS
    wrong_params = int(classify_wrong) + int(iu_wrong) + int(queue_wrong)

    # Compute UJCS for this ticket
    tracker = ctx.sop_trackers.get(tid)
    if tracker is not None:
        scoring_data = tracker.get_scoring_data()
        ujcs = compute_ujcs(scoring_data, wrong_parameterizations=wrong_params)
    else:
        ujcs = 0.0

    priority_weight = PRIORITY_WEIGHTS.get(ht.priority, 1.0)

    return PerTicketScore(
        ticket_id=tid,
        classification=classification,
        impact_urgency=impact_urgency,
        queue=queue,
        missing_info=missing_info,
        escalation=escalation,
        duplicate_non_actionable=dup_na,
        template=template,
        terminal_status=terminal_status,
        raw_score=raw_score,
        priority_weight=priority_weight,
        ujcs=ujcs,
        wrong_parameterizations=wrong_params,
    )


# ---------------------------------------------------------------------------
# Episode penalties (section 17.5)
# ---------------------------------------------------------------------------


def _compute_episode_penalties(ctx: EpisodeScoringContext) -> EpisodePenalties:
    """Compute all episode-level penalties.

    Args:
        ctx: Episode scoring context.

    Returns:
        EpisodePenalties with counts and total.
    """
    # Invalid actions
    invalid_count = ctx.invalid_action_count

    # Avoidable reassignments: ticket routed more than once
    reassignment_count = sum(
        max(0, count - 1) for count in ctx.ticket_route_count.values()
    )

    # Unnecessary escalations: escalated when not required
    unnecessary_escalation_count = 0
    for ticket in ctx.tickets:
        tid = ticket.ticket_id
        ht = ticket.hidden_truth
        if not ht.escalation_required and tid in ctx.ticket_escalated_to:
            unnecessary_escalation_count += 1

    # SLA mishandling: high/critical tickets not brought to terminal status
    sla_penalties = 0.0
    for ticket in ctx.tickets:
        ht = ticket.hidden_truth
        current_status = ctx.ticket_states.get(ticket.ticket_id)
        if current_status != ht.gold_terminal_status:
            if ht.priority == Priority.CRITICAL:
                sla_penalties += PENALTY_SLA_CRITICAL
            elif ht.priority == Priority.HIGH:
                sla_penalties += PENALTY_SLA_HIGH

    total = (
        invalid_count * PENALTY_INVALID_ACTION
        + reassignment_count * PENALTY_AVOIDABLE_REASSIGNMENT
        + unnecessary_escalation_count * PENALTY_UNNECESSARY_ESCALATION
        + sla_penalties
    )

    return EpisodePenalties(
        invalid_action_count=invalid_count,
        avoidable_reassignment_count=reassignment_count,
        unnecessary_escalation_count=unnecessary_escalation_count,
        sla_mishandling_penalties=sla_penalties,
        total_penalty=total,
    )


# ---------------------------------------------------------------------------
# Priority-order score (section 19)
# ---------------------------------------------------------------------------


def _compute_priority_order_score(ctx: EpisodeScoringContext) -> float:
    """Compute the priority-order metric.

    For each pair of tickets with different gold priorities, check whether the
    higher-priority ticket received its first substantive action before the
    lower-priority one.

    Returns:
        Fraction of correctly ordered pairs in [0, 1]. Returns 1.0 if there
        are no comparable pairs (0 or 1 ticket, or all same priority).
    """
    # Build list of (ticket_id, priority, first_step).
    # Tickets with no recorded first substantive action are assigned a sentinel
    # step larger than any real step, so they are treated as "handled last".
    sentinel_step = max(ctx.ticket_first_substantive_step.values(), default=0) + 1
    ticket_data: list[tuple[str, Priority, int]] = []
    for ticket in ctx.tickets:
        tid = ticket.ticket_id
        step = ctx.ticket_first_substantive_step.get(tid, sentinel_step)
        ticket_data.append((tid, ticket.hidden_truth.priority, step))

    if len(ticket_data) <= 1:
        return 1.0

    # Priority ordering: CRITICAL > HIGH > MEDIUM > LOW
    priority_rank = {
        Priority.CRITICAL: 3,
        Priority.HIGH: 2,
        Priority.MEDIUM: 1,
        Priority.LOW: 0,
    }

    correct_pairs = 0
    total_pairs = 0

    for i in range(len(ticket_data)):
        for j in range(i + 1, len(ticket_data)):
            _, pri_i, step_i = ticket_data[i]
            _, pri_j, step_j = ticket_data[j]

            rank_i = priority_rank[pri_i]
            rank_j = priority_rank[pri_j]

            if rank_i == rank_j:
                # Same priority — not comparable
                continue

            total_pairs += 1
            # Higher priority must have strictly earlier first substantive action
            if rank_i > rank_j:
                if step_i < step_j:
                    correct_pairs += 1
            else:
                if step_j < step_i:
                    correct_pairs += 1

    if total_pairs == 0:
        return 1.0

    return correct_pairs / total_pairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_episode_score(ctx: EpisodeScoringContext) -> ScoreBreakdown:
    """Compute the complete episode score breakdown.

    This is the main entry point called by the environment at episode end.

    Formula (section 17.6):
        FinalScore = TerminalBusinessScore + 0.15 * UJCS_OpenEnv - EpisodePenalties
        Clamped to [0, 1].

    Args:
        ctx: Episode scoring context with all environment state.

    Returns:
        ScoreBreakdown with per-ticket scores, aggregated metrics, and final score.
    """
    # Score each ticket
    per_ticket: dict[str, PerTicketScore] = {}
    for ticket in ctx.tickets:
        per_ticket[ticket.ticket_id] = score_ticket(ctx, ticket)

    # Priority-weighted terminal business score (section 17.3)
    total_weight = sum(pts.priority_weight for pts in per_ticket.values())
    if total_weight > 0:
        terminal_business_score = (
            sum(pts.raw_score * pts.priority_weight for pts in per_ticket.values())
            / total_weight
        )
    else:
        terminal_business_score = 0.0

    # Priority-weighted UJCS average (section 17.4)
    if total_weight > 0:
        ujcs_openenv = (
            sum(pts.ujcs * pts.priority_weight for pts in per_ticket.values())
            / total_weight
        )
    else:
        ujcs_openenv = 0.0

    ujcs_contribution = UJCS_WEIGHT * ujcs_openenv

    # Episode penalties (section 17.5)
    penalties = _compute_episode_penalties(ctx)

    # Reassignment count
    reassignment_count = penalties.avoidable_reassignment_count

    # Priority-order score (section 19, trace only)
    priority_order_score = _compute_priority_order_score(ctx)

    # Final score formula (section 17.6)
    raw_final = terminal_business_score + ujcs_contribution - penalties.total_penalty
    final_score = max(1e-4, min(1.0 - 1e-4, raw_final))

    return ScoreBreakdown(
        per_ticket_scores=per_ticket,
        terminal_business_score=terminal_business_score,
        ujcs_openenv=ujcs_openenv,
        ujcs_contribution=ujcs_contribution,
        episode_penalties=penalties,
        priority_order_score=priority_order_score,
        final_score=final_score,
        reassignment_count=reassignment_count,
        invalid_action_count=penalties.invalid_action_count,
    )
