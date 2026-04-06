"""Scripted expert oracle policy (mandatory baseline, §22.1).

Reads hidden ground truth to follow the gold SOP graph exactly:
keyword-match classification, derived impact/urgency, correct queue + template selection,
avoids unnecessary escalation. Used for regression-test ground truth and demo traces.

Public API:
    ScriptedExpert(env)          — wraps a TriageSieveEnvironment
    ScriptedExpert.run_episode() — runs one full episode, returns structured trace dict
"""

from __future__ import annotations

from typing import Any

from ..models import (
    ActionType,
    CloseReason,
    Priority,
    QueueId,
    TriageSieveAction,
    TaskDifficulty,
)
from ..server.scorer import (
    EpisodeScoringContext,
    ScoreBreakdown,
    compute_episode_score,
)
from ..server.triagesieve_env_environment import TriageSieveEnvironment

__all__ = ["ScriptedExpert"]

# Priority sort key: higher priority → lower sort value (processed first).
_PRIORITY_ORDER: dict[Priority, int] = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 1,
    Priority.MEDIUM: 2,
    Priority.LOW: 3,
}

# Actions that count as "substantive" for priority-order scoring (§19).
_SUBSTANTIVE_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.CLASSIFY_TICKET,
    ActionType.ROUTE_TICKET,
    ActionType.CLOSE_TICKET,
})


class ScriptedExpert:
    """Oracle policy that reads hidden truth to produce optimal action sequences.

    This is NOT a fair agent — it accesses internal ground truth via
    ``env._ticket_index[ticket_id].hidden_truth``. Its purpose is to:
    1. Prove environment solvability.
    2. Produce reference traces for regression testing.
    3. Establish a score ceiling for comparison with learned policies.

    Args:
        env: A fresh (or reusable) TriageSieveEnvironment instance.
    """

    def __init__(self, env: TriageSieveEnvironment) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(
        self,
        seed: int,
        difficulty: TaskDifficulty | None = None,
    ) -> dict[str, Any]:
        """Run a full episode with oracle actions, return a structured trace.

        Args:
            seed: Deterministic seed for episode generation.
            difficulty: Task difficulty tier. If None, seed-derived.

        Returns:
            Trace dict with keys: episode_id, seed, task_difficulty, done,
            action_sequence, final_score, score_breakdown.
        """
        kwargs: dict[str, Any] = {"mode": "eval_strict"}
        if difficulty is not None:
            kwargs["difficulty"] = difficulty.value

        obs = self.env.reset(seed=seed, **kwargs)
        state = self.env.state

        # Plan ticket processing order: highest priority first (§19).
        ordered_ticket_ids = self._plan_ticket_order()

        action_sequence: list[dict[str, Any]] = []
        step_num = 0

        # Tracking for scorer context
        templates_used: dict[str, list[str]] = {}
        route_count: dict[str, int] = {}
        first_substantive_step: dict[str, int] = {}

        for ticket_id in ordered_ticket_ids:
            actions = self._plan_ticket_actions(ticket_id)
            for action in actions:
                if obs.done or obs.action_budget_remaining <= 0:
                    break
                step_num += 1
                obs = self.env.step(action)
                action_sequence.append({
                    "step": step_num,
                    "action": self._serialize_action(action),
                    "result": obs.last_action_result,
                    "step_reward": obs.reward,
                })

                # Track templates used
                tid = action.ticket_id
                if tid is not None and action.template_id is not None:
                    templates_used.setdefault(tid, []).append(action.template_id)

                # Track route count
                if tid is not None and action.action_type == ActionType.ROUTE_TICKET:
                    route_count[tid] = route_count.get(tid, 0) + 1

                # Track first substantive step
                if (
                    tid is not None
                    and action.action_type in _SUBSTANTIVE_ACTIONS
                    and tid not in first_substantive_step
                ):
                    first_substantive_step[tid] = step_num

            if obs.done:
                break

        # FINISH_EPISODE if not already done
        if not obs.done and obs.action_budget_remaining > 0:
            finish = TriageSieveAction(
                action_type=ActionType.FINISH_EPISODE,
                metadata={},
            )
            step_num += 1
            obs = self.env.step(finish)
            action_sequence.append({
                "step": step_num,
                "action": self._serialize_action(finish),
                "result": obs.last_action_result,
                "step_reward": obs.reward,
            })

        # Compute proper terminal score via scorer
        invalid_count = sum(1 for entry in action_sequence if entry["result"] != "ok")
        score_breakdown = self._compute_score(
            templates_used, route_count, first_substantive_step, invalid_count
        )

        return {
            "episode_id": state.episode_id,
            "seed": seed,
            "task_difficulty": state.task_difficulty.value,
            "done": obs.done,
            "action_sequence": action_sequence,
            "final_score": score_breakdown.final_score,
            "score_breakdown": {
                "terminal_business_score": score_breakdown.terminal_business_score,
                "ujcs_openenv": score_breakdown.ujcs_openenv,
                "episode_penalties": score_breakdown.episode_penalties.total_penalty,
                "priority_order_score": score_breakdown.priority_order_score,
                "invalid_action_count": score_breakdown.invalid_action_count,
                "reassignment_count": score_breakdown.reassignment_count,
            },
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        templates_used: dict[str, list[str]],
        route_count: dict[str, int],
        first_substantive_step: dict[str, int],
        invalid_action_count: int,
    ) -> ScoreBreakdown:
        """Build EpisodeScoringContext from environment state and compute score.

        Args:
            templates_used: Map ticket_id → list of template_ids used.
            route_count: Map ticket_id → number of route actions.
            first_substantive_step: Map ticket_id → step number of first substantive action.
            invalid_action_count: Number of actions that returned non-"ok" results.

        Returns:
            ScoreBreakdown from scorer.
        """
        env = self.env
        ctx = EpisodeScoringContext(
            tickets=list(env._ticket_index.values()),
            ticket_states=dict(env._ticket_states),
            ticket_classifications=dict(env._ticket_classifications),
            ticket_impact_urgency=dict(env._ticket_impact_urgency),
            ticket_routed_to=dict(env._ticket_routed_to),
            ticket_escalated_to=dict(env._ticket_escalated_to),
            ticket_close_reasons=dict(env._ticket_close_reasons),
            ticket_info_requested=dict(env._ticket_info_requested),
            ticket_info_received=dict(env._ticket_info_received),
            ticket_merged_to=dict(env._ticket_merged_to),
            ticket_templates_used=templates_used,
            sop_trackers=dict(env._sop_trackers),
            invalid_action_count=invalid_action_count,
            ticket_route_count=route_count,
            ticket_first_substantive_step=first_substantive_step,
        )
        return compute_episode_score(ctx)

    # ------------------------------------------------------------------
    # Ticket ordering
    # ------------------------------------------------------------------

    def _plan_ticket_order(self) -> list[str]:
        """Sort tickets by gold priority descending (critical first).

        Reads hidden truth priority to maximize §19 priority-order score.

        Returns:
            Ordered list of ticket_ids.
        """
        tickets = list(self.env._ticket_index.values())
        tickets.sort(key=lambda t: _PRIORITY_ORDER[t.hidden_truth.priority])
        return [t.ticket_id for t in tickets]

    # ------------------------------------------------------------------
    # Per-ticket action planning
    # ------------------------------------------------------------------

    def _plan_ticket_actions(self, ticket_id: str) -> list[TriageSieveAction]:
        """Plan the full oracle action sequence for a single ticket.

        Reads hidden truth and branches on:
        - Non-actionable → open + close(non_actionable)
        - Duplicate → open + merge
        - Feature request → open + classify + close(feature_request)
        - Normal flow → open, classify, set_impact_urgency, [request_info],
          route or escalate, close

        Args:
            ticket_id: Ticket to plan actions for.

        Returns:
            Ordered list of TriageSieveAction objects.
        """
        ht = self.env._ticket_index[ticket_id].hidden_truth
        actions: list[TriageSieveAction] = []

        # 1. Always open first
        actions.append(TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id=ticket_id,
            metadata={},
        ))

        # 2. Branch: non-actionable
        # Classify first so the SOP tracker advances through the "identify_*" checkpoint
        # (spam, benign, automation_false_positive, data_error archetypes all require it).
        if ht.non_actionable_subtype is not None:
            actions.append(TriageSieveAction(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id=ticket_id,
                issue_family=ht.issue_family,
                issue_subtype=ht.issue_subtype,
                metadata={},
            ))
            actions.append(TriageSieveAction(
                action_type=ActionType.CLOSE_TICKET,
                ticket_id=ticket_id,
                close_reason=CloseReason.NON_ACTIONABLE,
                metadata={},
            ))
            return actions

        # 3. Branch: duplicate
        if ht.is_duplicate and ht.duplicate_of is not None:
            actions.append(TriageSieveAction(
                action_type=ActionType.MERGE_DUPLICATE,
                ticket_id=ticket_id,
                target_ticket_id=ht.duplicate_of,
                metadata={},
            ))
            return actions

        # 4. Branch: feature request routed to sales_or_feature_requests
        # SOP requires: classify → route(sales_or_feature_requests) → close(feature_request)
        if ht.required_queue == QueueId.SALES_OR_FEATURE_REQUESTS:
            actions.append(TriageSieveAction(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id=ticket_id,
                issue_family=ht.issue_family,
                issue_subtype=ht.issue_subtype,
                metadata={},
            ))
            actions.append(TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id=ticket_id,
                queue_id=ht.required_queue,
                metadata={},
            ))
            actions.append(TriageSieveAction(
                action_type=ActionType.CLOSE_TICKET,
                ticket_id=ticket_id,
                close_reason=CloseReason.FEATURE_REQUEST,
                metadata={},
            ))
            return actions

        # 5. Normal flow: classify
        actions.append(TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id=ticket_id,
            issue_family=ht.issue_family,
            issue_subtype=ht.issue_subtype,
            metadata={},
        ))

        # 6. Set impact/urgency
        actions.append(TriageSieveAction(
            action_type=ActionType.SET_IMPACT_URGENCY,
            ticket_id=ticket_id,
            impact=ht.impact,
            urgency=ht.urgency,
            metadata={},
        ))

        # 7. Request information if missing fields
        if ht.required_missing_fields:
            template_id = ht.correct_template_ids[0] if ht.correct_template_ids else None
            actions.append(TriageSieveAction(
                action_type=ActionType.REQUEST_INFORMATION,
                ticket_id=ticket_id,
                template_id=template_id,
                requested_fields=list(ht.required_missing_fields),
                metadata={},
            ))

        # 8. Route or escalate
        if ht.escalation_required and ht.escalation_target is not None:
            # Route first, then escalate (route → escalated is valid per §12)
            actions.append(TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id=ticket_id,
                queue_id=ht.required_queue,
                metadata={},
            ))
            actions.append(TriageSieveAction(
                action_type=ActionType.ESCALATE_TICKET,
                ticket_id=ticket_id,
                queue_id=ht.escalation_target,
                reason_code="expert_escalation",
                metadata={},
            ))
        else:
            actions.append(TriageSieveAction(
                action_type=ActionType.ROUTE_TICKET,
                ticket_id=ticket_id,
                queue_id=ht.required_queue,
                metadata={},
            ))

        # 9. Close with correct template
        close_template_id = (
            ht.correct_template_ids[-1]
            if ht.correct_template_ids
            else None
        )
        actions.append(TriageSieveAction(
            action_type=ActionType.CLOSE_TICKET,
            ticket_id=ticket_id,
            close_reason=CloseReason.RESOLVED,
            template_id=close_template_id,
            metadata={},
        ))

        return actions

    # ------------------------------------------------------------------
    # Serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_action(action: TriageSieveAction) -> dict[str, Any]:
        """Serialize an action to a plain dict for traces.

        Includes only non-None fields for readability.
        """
        data: dict[str, Any] = {"action_type": action.action_type.value}
        for field_name in (
            "ticket_id",
            "issue_family",
            "issue_subtype",
            "impact",
            "urgency",
            "queue_id",
            "reason_code",
            "template_id",
            "requested_fields",
            "target_ticket_id",
            "close_reason",
        ):
            value = getattr(action, field_name, None)
            if value is not None:
                data[field_name] = value.value if hasattr(value, "value") else value
        return data
