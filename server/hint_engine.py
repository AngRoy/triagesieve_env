"""Guided-mode hint generator (train_guided=True only).

Produces deterministic hints derived solely from hidden ticket metadata.
Never used in official scoring. See CLAUDE.md §16 for hint rules.

Implements 9 ordered predicates; first match wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    ActionType,
    HiddenTicketTruth,
    Impact,
    IssueFamily,
    IssueSubtype,
    NonActionableSubtype,
    QueueId,
    TriageSieveAction,
    TicketStatus,
    Urgency,
)

__all__ = ["HintContext", "HintEngine"]


@dataclass(frozen=True)
class HintContext:
    """Snapshot of state needed for hint evaluation.

    Built by the environment after each step, passed to HintEngine.
    """

    last_action: TriageSieveAction | None
    last_action_result: str
    ticket_status: TicketStatus | None
    hidden_truth: HiddenTicketTruth | None
    classification_set: tuple[IssueFamily, IssueSubtype] | None
    impact_urgency_set: tuple[Impact, Urgency] | None
    info_requested: bool
    info_received: bool
    routed_to: QueueId | None
    is_duplicate_truth: bool
    non_actionable_subtype: NonActionableSubtype | None


class HintEngine:
    """Deterministic hint generator for train_guided mode.

    Evaluates 9 ordered predicates against hidden truth. First match wins.
    No runtime LLM calls. Hints are derived ONLY from hidden metadata.
    """

    def generate_hint(self, ctx: HintContext) -> str | None:
        """Generate a hint based on the current context, or None.

        Args:
            ctx: Snapshot of environment state after the last action.

        Returns:
            A deterministic hint string, or None if no predicate fires.
        """
        if ctx.hidden_truth is None:
            return None

        ht = ctx.hidden_truth
        action = ctx.last_action

        # 1. Pushback detected
        if ctx.last_action_result.startswith("Pushback:"):
            return "Prerequisites must be completed before routing to specialized queues"

        # 2. Wrong classification family
        if (
            action is not None
            and action.action_type == ActionType.CLASSIFY_TICKET
            and ctx.classification_set is not None
        ):
            agent_family, agent_subtype = ctx.classification_set
            if agent_family != ht.issue_family:
                return "Review sender domain before classifying"
            # 3. Wrong subtype (correct family)
            if agent_subtype != ht.issue_subtype:
                return "Consider the specific nature of the reported problem"
            # Correct classification — no hint needed; stop evaluation
            return None

        # 4. Route/close without requesting required missing info
        if (
            action is not None
            and action.action_type in (ActionType.ROUTE_TICKET, ActionType.CLOSE_TICKET)
            and ht.required_missing_fields
            and not ctx.info_received
        ):
            first_field = ht.required_missing_fields[0]
            return f"Check thread history for {first_field}"

        # 5. Escalation without prior info request when required
        if (
            action is not None
            and action.action_type == ActionType.ESCALATE_TICKET
            and ht.required_missing_fields
            and not ctx.info_requested
        ):
            return "Escalation requires prior information request"

        # 6. Routed to wrong queue (only on the routing action itself)
        if (
            ctx.routed_to is not None
            and ctx.routed_to != ht.required_queue
            and action is not None
            and action.action_type == ActionType.ROUTE_TICKET
        ):
            return "Review routing policy for this issue family"

        # 7. Wrong impact/urgency
        if (
            action is not None
            and action.action_type == ActionType.SET_IMPACT_URGENCY
            and ctx.impact_urgency_set is not None
        ):
            agent_impact, agent_urgency = ctx.impact_urgency_set
            if agent_impact != ht.impact or agent_urgency != ht.urgency:
                return "Re-assess impact scope from thread details"

        # 8. Non-actionable ticket mishandled
        if (
            ctx.non_actionable_subtype is not None
            and action is not None
            and action.action_type in (ActionType.ROUTE_TICKET, ActionType.ESCALATE_TICKET)
        ):
            return "This ticket may not require standard resolution"

        # 9. Duplicate not merged
        if (
            ctx.is_duplicate_truth
            and action is not None
            and action.action_type in (ActionType.ROUTE_TICKET, ActionType.CLOSE_TICKET)
        ):
            return "Check for similar recent tickets in inbox"

        return None
