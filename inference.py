"""
Inference Script — TriageSieve-OpenEnv
========================================
MANDATORY environment variables (see pre-submission checklist):
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   The name of the local image to use for the environment
                       (required when using from_docker_image()).

Defaults are set only for API_BASE_URL and MODEL_NAME.
All LLM calls use the OpenAI client configured via these variables.
Stdout logs follow the required structured format ([START]/[STEP]/[END]).
"""

from __future__ import annotations

import asyncio
import json
import os
import re

# Load .env file if present (so HF_TOKEN, LOCAL_IMAGE_NAME etc. work without
# manually exporting in the shell). python-dotenv is already available via litellm.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

from triagesieve_env import TriageSieveEnv
from triagesieve_env.models import (
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

# ---------------------------------------------------------------------------
# Configuration (matches pre-submission checklist EXACTLY)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "triagesieve_env"
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5  # minimum final score for "success"

# Task ladder: matches episode_engine budget exactly, plus a small overflow buffer
TASK_CONFIGS = [
    {"task_name": "easy",   "seed": 0, "difficulty": "easy",   "max_steps": 8},
    {"task_name": "medium", "seed": 1, "difficulty": "medium",  "max_steps": 14},
    {"task_name": "hard",   "seed": 2, "difficulty": "hard",    "max_steps": 20},
]

# Enum fields requiring lowercase normalization when parsing LLM output
_ENUM_FIELDS: dict[str, type] = {
    "action_type": ActionType,
    "issue_family": IssueFamily,
    "issue_subtype": IssueSubtype,
    "impact": Impact,
    "urgency": Urgency,
    "queue_id": QueueId,
    "close_reason": CloseReason,
}

# ---------------------------------------------------------------------------
# Mandatory stdout logging (DO NOT MODIFY FORMAT)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation → text serialization (mirrors baseline/llm_baseline.py)
# ---------------------------------------------------------------------------


def serialize_observation(obs: TriageSieveObservation) -> str:
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

    parts.append("=== SLA Policies ===")
    for card in obs.sla_policy_cards:
        parts.append(
            f"- {card.tier.value}: respond {card.response_deadline_minutes}min, "
            f"resolve {card.resolution_deadline_minutes}min"
        )

    if obs.available_templates:
        parts.append("=== Templates ===")
        for tpl in obs.available_templates:
            parts.append(
                f"- {tpl.get('template_id', '?')}: {tpl.get('name', '?')} "
                f"({tpl.get('applies_to', '?')})"
            )

    if obs.hint:
        parts.append(f"=== Hint ===\n{obs.hint}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a support-ticket triage agent. Your job is to process an inbox of support tickets by taking structured actions.

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

    == PRIORITY DERIVATION (for your reasoning only) ==

    single_user:       low/low/medium/high     (columns: urgency low/medium/high/critical)
    team:              low/medium/high/high
    org_wide:          medium/high/high/critical
    revenue_affecting: high/high/critical/critical

    == STRATEGY ==

    1. Open tickets starting with highest-priority ones (enterprise/critical SLA first).
    2. Classify after reading the ticket content carefully.
    3. Set impact and urgency based on the ticket details.
    4. Request missing information if needed before routing.
    5. Route to the correct queue. Note: tech_support_l2 and security_team are gated (need classification + impact/urgency first).
    6. Close with the appropriate reason and template.
    7. If a ticket looks like spam or non-actionable, close it as non_actionable.
    8. If a ticket is a duplicate, merge it with the original.
    9. Use finish_episode when all tickets are fully handled.
    10. Only use skip_turn if you truly cannot determine any useful action.

    Respond with ONLY the JSON action object. No explanation.
""").strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def get_model_action(
    client: OpenAI,
    obs_text: str,
    last_reward: float,
    step: int,
) -> str:
    user_content = f"Step {step} | Last reward: {last_reward:.2f}\n\n{obs_text}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Action parsing (mirrors baseline/llm_baseline.py)
# ---------------------------------------------------------------------------


def parse_action(raw_text: str) -> Optional[TriageSieveAction]:
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    data: Optional[dict[str, Any]] = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    if data is None:
        start = text.find("{")
        if start == -1:
            return None
        depth, end = 0, -1
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
            data = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict) or "action_type" not in data:
        return None

    for field_name in _ENUM_FIELDS:
        if field_name in data and isinstance(data[field_name], str):
            data[field_name] = data[field_name].lower()

    data.setdefault("metadata", {})

    try:
        return TriageSieveAction(**data)
    except (ValueError, TypeError) as exc:
        print(f"[DEBUG] Action validation failed: {exc}", flush=True)
        return None


def action_to_str(action: TriageSieveAction) -> str:
    """Produce a concise one-token-ish string for [STEP] logging."""
    parts = [action.action_type.value]
    if action.ticket_id:
        parts.append(action.ticket_id)
    if action.queue_id:
        parts.append(action.queue_id.value)
    if action.issue_family:
        parts.append(action.issue_family.value)
    if action.close_reason:
        parts.append(action.close_reason.value)
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


async def run_task(
    client: OpenAI,
    env: TriageSieveEnv,
    task_name: str,
    seed: int,
    difficulty: str,
    max_steps: int,
) -> dict[str, Any]:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    episode_done = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=seed, difficulty=difficulty, mode="eval_strict")
        obs: TriageSieveObservation = result.observation
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if episode_done or obs.action_budget_remaining <= 0:
                break

            obs_text = serialize_observation(obs)
            raw = get_model_action(client, obs_text, last_reward, step)
            action = parse_action(raw)

            if action is None:
                print(f"[DEBUG] Parse failure at step {step}, using skip_turn", flush=True)
                action = TriageSieveAction(action_type=ActionType.SKIP_TURN, metadata={})

            result = await env.step(action)
            obs = result.observation
            reward = result.reward if result.reward is not None else 0.0
            episode_done = result.done or obs.done

            error_str = None if obs.last_action_result == "ok" else obs.last_action_result

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_to_str(action), reward=reward, done=episode_done, error=error_str)

            if episode_done:
                break

        # Send finish_episode if budget ran out but episode isn't done
        if not episode_done:
            finish = TriageSieveAction(action_type=ActionType.FINISH_EPISODE, metadata={})
            result = await env.step(finish)
            obs = result.observation
            reward = result.reward if result.reward is not None else 0.0
            episode_done = True
            steps_taken += 1
            rewards.append(reward)
            log_step(step=steps_taken, action="finish_episode", reward=reward, done=True, error=None)

        # Final score is the terminal observation.reward (already normalized to [0, 1])
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


async def create_env_from_docker(image_name: str, timeout_s: float = 120.0) -> TriageSieveEnv:
    """Start a Docker container and connect with a generous timeout.

    The default 30s from_docker_image timeout is too tight for first-start
    on some machines (Windows, CI). This helper gives 120s instead.
    """
    from openenv.core.containers.runtime.providers import LocalDockerProvider

    provider = LocalDockerProvider()
    base_url = provider.start_container(image_name)
    provider.wait_for_ready(base_url, timeout_s=timeout_s)
    client = TriageSieveEnv(base_url=base_url, provider=provider)
    await client.connect()
    return client


async def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("ERROR: HF_TOKEN environment variable is not set.")
    if not LOCAL_IMAGE_NAME:
        raise SystemExit("ERROR: LOCAL_IMAGE_NAME environment variable is not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = []
    for cfg in TASK_CONFIGS:
        env = await create_env_from_docker(LOCAL_IMAGE_NAME)
        result = await run_task(
            client=client,
            env=env,
            task_name=cfg["task_name"],
            seed=cfg["seed"],
            difficulty=cfg["difficulty"],
            max_steps=cfg["max_steps"],
        )
        results.append(result)

    print("\n=== RESULTS SUMMARY ===", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task']}: score={r['score']:.3f} steps={r['steps']} [{status}]",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
