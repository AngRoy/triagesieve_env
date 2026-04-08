"""Generate SFT training data from scripted expert trajectories.

Usage:
    python scripts/generate_sft_data.py --seeds 0-99 --output data/sft_dataset.jsonl

Runs the scripted expert on each (seed, difficulty) combination, captures
(observation_text -> action_json) pairs at every step, and writes them in
HuggingFace chat format for QLoRA fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from triagesieve_env.models import (
    ActionType,
    CloseReason,
    Impact,
    IssueFamily,
    IssueSubtype,
    QueueId,
    TaskDifficulty,
    TriageSieveAction,
    TriageSieveObservation,
    Urgency,
)
from triagesieve_env.server.triagesieve_env_environment import TriageSieveEnvironment
from triagesieve_env.baseline.scripted_expert import ScriptedExpert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (same as inference.py / llm_baseline.py — model sees this)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a support-ticket triage agent. Your job is to process an inbox of support tickets by taking structured actions.

You must respond with EXACTLY ONE JSON object per turn. No extra text, no markdown fences, just the JSON.

== ACTION TYPES AND REQUIRED FIELDS ==

1. open_ticket: {"action_type": "open_ticket", "ticket_id": "<id>"}
2. classify_ticket: {"action_type": "classify_ticket", "ticket_id": "<id>", "issue_family": "<family>", "issue_subtype": "<subtype>"}
3. set_impact_urgency: {"action_type": "set_impact_urgency", "ticket_id": "<id>", "impact": "<impact>", "urgency": "<urgency>"}
4. route_ticket: {"action_type": "route_ticket", "ticket_id": "<id>", "queue_id": "<queue>"}
5. request_information: {"action_type": "request_information", "ticket_id": "<id>", "requested_fields": ["field1", ...], "template_id": "<template>"}
6. escalate_ticket: {"action_type": "escalate_ticket", "ticket_id": "<id>", "queue_id": "<queue>", "reason_code": "<reason>"}
7. merge_duplicate: {"action_type": "merge_duplicate", "ticket_id": "<id>", "target_ticket_id": "<original_id>"}
8. close_ticket: {"action_type": "close_ticket", "ticket_id": "<id>", "close_reason": "<reason>", "template_id": "<template>"}
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

== CRITICAL RULES ==

- You MUST request_information BEFORE route_ticket if required fields are missing.
- You MUST classify BEFORE routing.
- You MUST set_impact_urgency BEFORE routing to gated queues (tech_support_l2, security_team).
- Process ALL tickets. Follow: open -> classify -> set_impact_urgency -> request_info (if needed) -> route -> close.
- Process highest-priority tickets first (enterprise tier, critical SLA).
- Use finish_episode when ALL tickets are fully handled.

Respond with ONLY the JSON action object. No explanation."""


# ---------------------------------------------------------------------------
# Observation serializer (mirrors inference.py exactly)
# ---------------------------------------------------------------------------
def serialize_observation(obs: TriageSieveObservation) -> str:
    """Convert observation to the text format the model will see during inference."""
    parts: list[str] = []

    parts.append(
        f"=== Episode Context ===\n"
        f"Step: {obs.step_count} | Budget remaining: {obs.action_budget_remaining} | "
        f"Difficulty: {obs.task_difficulty.value} | Time: {obs.current_time}\n"
        f"Last action result: {obs.last_action_result}"
    )

    parts.append("=== Inbox ===")
    for item in obs.inbox_summaries:
        sla = f"{item.sla_remaining_minutes}min" if item.sla_remaining_minutes is not None else "n/a"
        parts.append(
            f"- [{item.ticket_id}] {item.subject} | from: {item.sender_email} | "
            f"status: {item.status.value} | tier: {item.customer_tier.value} | "
            f"SLA: {sla} | attachment: {item.has_attachment}\n"
            f"  Preview: {item.short_preview}"
        )

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

    parts.append(
        f"=== Legal Actions ===\n"
        f"{', '.join(a.value for a in obs.legal_actions)}"
    )

    parts.append("=== Routing Policies ===")
    for card in obs.routing_policy_cards:
        prereqs = ", ".join(card.prerequisites) if card.prerequisites else "none"
        families = ", ".join(f.value for f in card.handles_families)
        parts.append(
            f"- {card.queue_id.value}: {card.description} | "
            f"prereqs: {prereqs} | families: {families}"
        )

    if obs.available_templates:
        parts.append("=== Templates ===")
        for tpl in obs.available_templates:
            parts.append(
                f"- {tpl.get('template_id', '?')}: {tpl.get('name', '?')} "
                f"({tpl.get('applies_to', '?')})"
            )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Action serializer (produces the exact JSON the model should output)
# ---------------------------------------------------------------------------
def action_to_json(action: TriageSieveAction) -> str:
    """Serialize action to compact JSON — the target output for training."""
    d: dict[str, Any] = {"action_type": action.action_type.value}
    if action.ticket_id is not None:
        d["ticket_id"] = action.ticket_id
    if action.issue_family is not None:
        d["issue_family"] = action.issue_family.value
    if action.issue_subtype is not None:
        d["issue_subtype"] = action.issue_subtype.value
    if action.impact is not None:
        d["impact"] = action.impact.value
    if action.urgency is not None:
        d["urgency"] = action.urgency.value
    if action.queue_id is not None:
        d["queue_id"] = action.queue_id.value
    if action.reason_code is not None:
        d["reason_code"] = action.reason_code
    if action.template_id is not None:
        d["template_id"] = action.template_id
    if action.requested_fields is not None:
        d["requested_fields"] = action.requested_fields
    if action.target_ticket_id is not None:
        d["target_ticket_id"] = action.target_ticket_id
    if action.close_reason is not None:
        d["close_reason"] = action.close_reason.value
    return json.dumps(d, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Core: run expert and capture (observation, action) pairs
# ---------------------------------------------------------------------------
def generate_pairs_for_episode(
    seed: int,
    difficulty: TaskDifficulty,
) -> list[dict]:
    """Run scripted expert on one episode, return list of chat-format training examples."""
    env = TriageSieveEnvironment()
    expert = ScriptedExpert(env)

    # Reset
    obs = env.reset(seed=seed, difficulty=difficulty, mode="eval_strict")

    # Plan actions using expert's internal planning
    expert_trace = expert.run_episode(seed=seed, difficulty=difficulty)
    action_sequence = expert_trace.get("action_sequence", [])

    # Now replay: reset again and step through, capturing obs at each step
    env2 = TriageSieveEnvironment()
    obs = env2.reset(seed=seed, difficulty=difficulty, mode="eval_strict")

    pairs = []
    for step_info in action_sequence:
        # Capture current observation text
        obs_text = serialize_observation(obs)

        # Reconstruct the action from the trace
        action_data = step_info["action"]
        action_json_str = json.dumps(action_data, separators=(",", ":"))

        # Build chat-format example
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
                {"role": "assistant", "content": action_json_str},
            ],
            "metadata": {
                "seed": seed,
                "difficulty": difficulty.value,
                "step": step_info.get("step", 0),
                "reward": step_info.get("step_reward", 0.0),
            },
        }
        pairs.append(example)

        # Step environment to get next observation
        try:
            action = TriageSieveAction(**action_data)
            obs = env2.step(action)
            if obs.done:
                break
        except Exception:
            break

    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_seed_range(s: str) -> list[int]:
    """Parse '0-99' or '0,5,10' into a list of ints."""
    if "-" in s and "," not in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate SFT dataset from expert traces.")
    parser.add_argument("--seeds", type=str, default="0-99", help="Seed range, e.g. '0-99' or '0,5,10'")
    parser.add_argument("--output", type=str, default="data/sft_dataset.jsonl", help="Output JSONL path")
    args = parser.parse_args(argv)

    seeds = parse_seed_range(args.seeds)
    difficulties = [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]

    all_pairs = []
    total_episodes = 0

    for seed in seeds:
        difficulty = difficulties[seed % 3]  # cycle through difficulties
        logger.info(f"Generating: seed={seed} difficulty={difficulty.value}")
        try:
            pairs = generate_pairs_for_episode(seed, difficulty)
            all_pairs.extend(pairs)
            total_episodes += 1
        except Exception as e:
            logger.warning(f"Failed seed={seed} difficulty={difficulty.value}: {e}")

    # Write output
    from pathlib import Path
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, separators=(",", ":")) + "\n")

    logger.info(f"Generated {len(all_pairs)} training pairs from {total_episodes} episodes")
    logger.info(f"Written to {out_path}")

    # Stats
    by_diff = {}
    for p in all_pairs:
        d = p["metadata"]["difficulty"]
        by_diff[d] = by_diff.get(d, 0) + 1
    for d, c in sorted(by_diff.items()):
        logger.info(f"  {d}: {c} pairs")


if __name__ == "__main__":
    main()
