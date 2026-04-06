"""Simple tool-using LLM baseline (§22.2).

LLM sees the observation as a structured prompt and compact policy cards.
Constrained to the typed TriageSieveAction schema. No access to hidden truth.
Uses litellm for provider-agnostic LLM access (OpenAI, Anthropic, Ollama, etc.).

Public API:
    LLMBaseline(env, model, temperature)  — wraps a TriageSieveEnvironment
    LLMBaseline.run_episode()             — runs one full episode, returns structured trace dict
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm

from ..models import (
    ActionType,
    CloseReason,
    Impact,
    IssueFamily,
    IssueSubtype,
    QueueId,
    TriageSieveAction,
    TriageSieveObservation,
    TaskDifficulty,
    Urgency,
)
from ..server.triagesieve_env_environment import TriageSieveEnvironment

__all__ = ["LLMBaseline"]

logger = logging.getLogger(__name__)

# Enum fields that need case-normalization when parsing LLM output.
_ENUM_FIELDS: dict[str, type] = {
    "action_type": ActionType,
    "issue_family": IssueFamily,
    "issue_subtype": IssueSubtype,
    "impact": Impact,
    "urgency": Urgency,
    "queue_id": QueueId,
    "close_reason": CloseReason,
}


class LLMBaseline:
    """Minimal tool-using LLM agent for TriageSieve-OpenEnv.

    Stateless per step: each step sends the current observation as a user message
    alongside a static system prompt. The observation already contains
    prior_actions_taken and thread_history, so no conversation memory is needed.

    Args:
        env: A TriageSieveEnvironment instance.
        model: litellm model string (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514").
        temperature: Sampling temperature. 0.0 for reproducibility.
    """

    def __init__(
        self,
        env: TriageSieveEnvironment,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.env = env
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(
        self,
        seed: int,
        difficulty: TaskDifficulty | None = None,
    ) -> dict[str, Any]:
        """Run a full episode, return a structured trace.

        Args:
            seed: Deterministic seed for episode generation.
            difficulty: Task difficulty tier. If None, seed-derived.

        Returns:
            Trace dict with keys: episode_id, seed, task_difficulty, done,
            action_sequence, final_score.
        """
        kwargs: dict[str, Any] = {"mode": "eval_strict"}
        if difficulty is not None:
            kwargs["difficulty"] = difficulty.value

        obs = self.env.reset(seed=seed, **kwargs)
        state = self.env.state
        system_prompt = self._build_system_prompt()

        action_sequence: list[dict[str, Any]] = []
        step_num = 0

        while not obs.done and obs.action_budget_remaining > 0:
            step_num += 1
            user_prompt = self._serialize_observation(obs)

            raw = self._call_llm(system_prompt, user_prompt)
            action = self._parse_action(raw)
            if action is None:
                logger.debug("Parse failure at step %d, falling back to skip_turn", step_num)
                action = self._fallback_action()

            obs = self.env.step(action)
            action_sequence.append({
                "step": step_num,
                "action": self._serialize_action_dict(action),
                "result": obs.last_action_result,
                "step_reward": obs.reward,
                "raw_llm": raw,
            })

        # If budget ran out but not done, send finish_episode
        if not obs.done:
            step_num += 1
            finish = TriageSieveAction(
                action_type=ActionType.FINISH_EPISODE,
                metadata={},
            )
            obs = self.env.step(finish)
            action_sequence.append({
                "step": step_num,
                "action": self._serialize_action_dict(finish),
                "result": obs.last_action_result,
                "step_reward": obs.reward,
                "raw_llm": None,
            })

        return {
            "episode_id": state.episode_id,
            "seed": seed,
            "task_difficulty": state.task_difficulty.value,
            "done": obs.done,
            "action_sequence": action_sequence,
            "final_score": obs.reward if obs.reward is not None else 0.0,
        }

    # ------------------------------------------------------------------
    # Observation → prompt serialization
    # ------------------------------------------------------------------

    def _serialize_observation(self, obs: TriageSieveObservation) -> str:
        """Convert observation to structured text for the LLM user message.

        Sections: inbox table, focused ticket (if any), policy cards,
        templates, episode context, hint (if any).
        """
        parts: list[str] = []

        # Episode context
        parts.append(
            f"=== Episode Context ===\n"
            f"Step: {obs.step_count} | Budget remaining: {obs.action_budget_remaining} | "
            f"Difficulty: {obs.task_difficulty.value} | Time: {obs.current_time}\n"
            f"Last action result: {obs.last_action_result}"
        )

        # Inbox summary
        parts.append("=== Inbox ===")
        for item in obs.inbox_summaries:
            sla = f"{item.sla_remaining_minutes}min" if item.sla_remaining_minutes is not None else "n/a"
            parts.append(
                f"- [{item.ticket_id}] {item.subject} | from: {item.sender_email} | "
                f"status: {item.status.value} | tier: {item.customer_tier.value} | "
                f"SLA: {sla} | attachment: {item.has_attachment}\n"
                f"  Preview: {item.short_preview}"
            )

        # Focused ticket
        if obs.focused_ticket is not None:
            ft = obs.focused_ticket
            parts.append(
                f"=== Focused Ticket: {ft.ticket_id} ===\n"
                f"Subject: {ft.subject}\n"
                f"Latest message: {ft.latest_message}"
            )
            if ft.thread_history:
                parts.append("Thread history:")
                for msg in ft.thread_history:
                    parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')}")
            if ft.attachments:
                parts.append(f"Attachments: {', '.join(ft.attachments)}")
            if ft.visible_internal_notes:
                parts.append(f"Internal notes: {'; '.join(ft.visible_internal_notes)}")
            if ft.prior_actions_taken:
                parts.append(f"Prior actions: {', '.join(ft.prior_actions_taken)}")

        # Legal actions
        parts.append(
            f"=== Legal Actions ===\n"
            f"{', '.join(a.value for a in obs.legal_actions)}"
        )

        # Routing policy cards
        parts.append("=== Routing Policies ===")
        for card in obs.routing_policy_cards:
            prereqs = ", ".join(card.prerequisites) if card.prerequisites else "none"
            families = ", ".join(f.value for f in card.handles_families)
            parts.append(
                f"- {card.queue_id.value}: {card.description} | "
                f"prereqs: {prereqs} | families: {families}"
            )

        # SLA policies
        parts.append("=== SLA Policies ===")
        for card in obs.sla_policy_cards:
            parts.append(
                f"- {card.tier.value}: respond {card.response_deadline_minutes}min, "
                f"resolve {card.resolution_deadline_minutes}min"
            )

        # Available templates
        if obs.available_templates:
            parts.append("=== Templates ===")
            for tpl in obs.available_templates:
                parts.append(
                    f"- {tpl.get('template_id', '?')}: {tpl.get('name', '?')} "
                    f"({tpl.get('applies_to', '?')})"
                )

        # Hint (guided mode only)
        if obs.hint:
            parts.append(f"=== Hint ===\n{obs.hint}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the static system prompt with role, action schema, and enum values."""
        return """You are a support-ticket triage agent. Your job is to process an inbox of support tickets by taking structured actions.

You must respond with EXACTLY ONE JSON object per turn. No extra text, no markdown fences, just the JSON.

== ACTION TYPES AND REQUIRED FIELDS ==

1. open_ticket: {"action_type": "open_ticket", "ticket_id": "<id>"}
2. classify_ticket: {"action_type": "classify_ticket", "ticket_id": "<id>", "issue_family": "<family>", "issue_subtype": "<subtype>"}
3. set_impact_urgency: {"action_type": "set_impact_urgency", "ticket_id": "<id>", "impact": "<impact>", "urgency": "<urgency>"}
4. route_ticket: {"action_type": "route_ticket", "ticket_id": "<id>", "queue_id": "<queue>"}
5. request_information: {"action_type": "request_information", "ticket_id": "<id>", "requested_fields": ["field1", ...], "template_id": "<optional>"}
6. escalate_ticket: {"action_type": "escalate_ticket", "ticket_id": "<id>", "queue_id": "<queue>", "reason_code": "<reason>"}
7. merge_duplicate: {"action_type": "merge_duplicate", "ticket_id": "<id>", "target_ticket_id": "<original_id>"}
8. close_ticket: {"action_type": "close_ticket", "ticket_id": "<id>", "close_reason": "<reason>", "template_id": "<optional>"}
9. skip_turn: {"action_type": "skip_turn"}
10. finish_episode: {"action_type": "finish_episode"}

== ENUM VALUES ==

issue_family: billing, technical, account, security, shipping
issue_subtype:
  billing: refund, invoice_error, failed_charge
  technical: bug_report, api_error, integration_failure
  account: password_reset, sso_issue, account_lockout
  security: suspicious_login, exposure_risk, abuse_report
  shipping: delay, tracking_problem, lost_package

impact: single_user, team, org_wide, revenue_affecting
urgency: low, medium, high, critical

queue_id: billing_team, tech_support_l1, tech_support_l2, account_team, security_team, shipping_team, refund_team, spam_filter, sales_or_feature_requests

close_reason: resolved, duplicate, non_actionable, feature_request, no_response

== PRIORITY DERIVATION (for your reasoning, not an action field) ==

Priority is derived from impact x urgency:
- single_user: low/low/medium/high
- team: low/medium/high/high
- org_wide: medium/high/high/critical
- revenue_affecting: high/high/critical/critical
(columns: low/medium/high/critical urgency)

== STRATEGY ==

1. Open tickets starting with the highest-priority ones (enterprise/critical SLA first).
2. Classify after reading the ticket content.
3. Set impact and urgency based on the ticket details.
4. Request missing information if needed before routing.
5. Route to the correct queue. Note: tech_support_l2 and security_team are gated (need classification + impact/urgency first).
6. Close with the appropriate reason and template.
7. If a ticket looks like spam or non-actionable, close it as non_actionable.
8. If a ticket is a duplicate, merge it with the original.
9. Use finish_episode when all tickets are handled.
10. Only use skip_turn if you truly cannot determine any useful action.

Respond with ONLY the JSON action object. No explanation."""

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(self, raw_text: str) -> TriageSieveAction | None:
        """Parse raw LLM output into a TriageSieveAction.

        Steps:
        1. Strip markdown fences (```json ... ```).
        2. Find first {...} in the text.
        3. Normalize enum values to lowercase.
        4. Validate via Pydantic.

        Returns None on any parse failure.
        """
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        # Fast path: entire stripped text is valid JSON
        data: dict[str, Any] | None = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Slow path: extract outermost {...} via bracket counting
        if data is None:
            start = text.find("{")
            if start == -1:
                return None
            depth = 0
            end = -1
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                return None
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None

        if not isinstance(data, dict) or "action_type" not in data:
            return None

        # Normalize enum string values to lowercase
        for field_name in _ENUM_FIELDS:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = data[field_name].lower()

        # Ensure metadata exists (required by Action base)
        if "metadata" not in data:
            data["metadata"] = {}

        try:
            return TriageSieveAction(**data)
        except (ValueError, TypeError) as exc:
            logger.debug("Action validation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_action() -> TriageSieveAction:
        """Return a safe fallback action when parsing fails."""
        return TriageSieveAction(
            action_type=ActionType.SKIP_TURN,
            metadata={},
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system: str,
        user: str,
        max_retries: int = 2,
    ) -> str:
        """Call the LLM via litellm and return the raw content string.

        Args:
            system: System prompt.
            user: User message (serialized observation).
            max_retries: Number of retries on API errors.

        Returns:
            Raw text from the LLM response.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(max_retries + 1):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=512,
                )
                content = response.choices[0].message.content
                return content if content else ""
            except (
                litellm.AuthenticationError,
                litellm.NotFoundError,
                litellm.BadRequestError,
            ) as exc:
                logger.error("Non-retryable LLM error: %s", exc)
                return ""
            except Exception as exc:
                if attempt == max_retries:
                    logger.warning("LLM call failed after %d attempts: %s", max_retries + 1, exc)
                    return ""
                logger.debug("LLM attempt %d failed: %s, retrying", attempt + 1, exc)

        return ""  # unreachable, but satisfies type checker

    # ------------------------------------------------------------------
    # Serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_action_dict(action: TriageSieveAction) -> dict[str, Any]:
        """Serialize an action to a plain dict for traces (non-None fields only)."""
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the LLM baseline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="TriageSieve LLM Baseline Agent")
    parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Task difficulty",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="litellm model string")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    env = TriageSieveEnvironment()
    difficulty = TaskDifficulty(args.difficulty) if args.difficulty else None
    baseline = LLMBaseline(env=env, model=args.model, temperature=args.temperature)
    trace = baseline.run_episode(seed=args.seed, difficulty=difficulty)

    if not args.quiet:
        for entry in trace["action_sequence"]:
            step = entry["step"]
            action_str = json.dumps(entry["action"], indent=None)
            result = entry["result"]
            print(f"  Step {step}: {action_str} → {result}")

    print(f"\nEpisode: {trace['episode_id']}")
    print(f"Difficulty: {trace['task_difficulty']}")
    print(f"Steps: {len(trace['action_sequence'])}")
    print(f"Final score: {trace['final_score']}")


if __name__ == "__main__":
    main()
