"""Evaluate a trained (or zero-shot) model against the TriageSieve environment.

Usage:
    # Evaluate trained SFT model
    python scripts/evaluate_trained.py --model outputs/sft_model --seeds 100-129

    # Evaluate zero-shot baseline (no fine-tuning)
    python scripts/evaluate_trained.py --model Qwen/Qwen2.5-1.5B-Instruct --seeds 100-129

Runs the model on fresh episodes and reports scores per difficulty.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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

logger = logging.getLogger(__name__)

# Same system prompt used in training
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

ENUM_FIELDS: dict[str, type] = {
    "action_type": ActionType,
    "issue_family": IssueFamily,
    "issue_subtype": IssueSubtype,
    "impact": Impact,
    "urgency": Urgency,
    "queue_id": QueueId,
    "close_reason": CloseReason,
}


def serialize_observation(obs: TriageSieveObservation) -> str:
    """Same serializer used in training data generation."""
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


def parse_action(raw_text: str) -> Optional[TriageSieveAction]:
    """Parse LLM output into a TriageSieveAction."""
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
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict) or "action_type" not in data:
        return None

    for field_name in ENUM_FIELDS:
        if field_name in data and isinstance(data[field_name], str):
            data[field_name] = data[field_name].lower()

    data.setdefault("metadata", {})
    try:
        return TriageSieveAction(**data)
    except (ValueError, TypeError):
        return None


def load_model(model_path: str):
    """Load model — handles both base HF models and QLoRA checkpoints."""
    logger.info(f"Loading model from {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    path = Path(model_path)
    is_lora = (path / "adapter_config.json").exists()

    if is_lora:
        # Load adapter config to find base model
        with open(path / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        logger.info(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_action(model, tokenizer, messages: list[dict], max_new_tokens: int = 256) -> str:
    """Generate a single action from the model given chat messages."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_episode(model, tokenizer, seed: int, difficulty: TaskDifficulty) -> dict:
    """Run one episode and return results."""
    env = TriageSieveEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty, mode="eval_strict")

    valid_actions = 0
    invalid_actions = 0
    parse_failures = 0

    for step in range(1, 25):
        if obs.done or obs.action_budget_remaining <= 0:
            break

        obs_text = serialize_observation(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]

        raw = generate_action(model, tokenizer, messages)
        action = parse_action(raw)

        if action is None:
            parse_failures += 1
            action = TriageSieveAction(action_type=ActionType.SKIP_TURN, metadata={})

        obs = env.step(action)
        if obs.last_action_result == "ok":
            valid_actions += 1
        else:
            invalid_actions += 1

        if obs.done:
            break

    if not obs.done:
        obs = env.step(TriageSieveAction(action_type=ActionType.FINISH_EPISODE, metadata={}))

    final_score = obs.reward if obs.reward is not None else 0.0
    final_score = min(max(final_score, 1e-3), 1.0 - 1e-3)

    return {
        "seed": seed,
        "difficulty": difficulty.value,
        "final_score": final_score,
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
        "parse_failures": parse_failures,
    }


def parse_seed_range(s: str) -> list[int]:
    if "-" in s and "," not in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate trained model against TriageSieve environment.")
    parser.add_argument("--model", type=str, required=True, help="Model path (HF ID or local checkpoint)")
    parser.add_argument("--seeds", type=str, default="100-129", help="Seed range for evaluation")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    difficulties = [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]

    model, tokenizer = load_model(args.model)
    logger.info(f"Model loaded. Evaluating on {len(seeds)} seeds...")

    results = []
    for seed in seeds:
        difficulty = difficulties[seed % 3]
        logger.info(f"  Episode seed={seed} difficulty={difficulty.value}")
        t0 = time.time()
        result = run_episode(model, tokenizer, seed, difficulty)
        result["time_s"] = round(time.time() - t0, 1)
        results.append(result)
        logger.info(
            f"    score={result['final_score']:.3f} valid={result['valid_actions']} "
            f"invalid={result['invalid_actions']} parse_fail={result['parse_failures']} "
            f"time={result['time_s']}s"
        )

    # Summary by difficulty
    print(f"\n{'='*65}")
    print(f"  EVALUATION RESULTS: {args.model}")
    print(f"{'='*65}")
    print(f"  {'Difficulty':<10s} {'Count':>5s} {'Avg Score':>10s} {'Valid%':>8s} {'Parse%':>8s}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*8} {'-'*8}")

    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff.value]
        if not diff_results:
            continue
        avg_score = sum(r["final_score"] for r in diff_results) / len(diff_results)
        total_valid = sum(r["valid_actions"] for r in diff_results)
        total_invalid = sum(r["invalid_actions"] for r in diff_results)
        total_parse = sum(r["parse_failures"] for r in diff_results)
        total_actions = total_valid + total_invalid + total_parse
        valid_pct = 100 * total_valid / max(total_actions, 1)
        parse_pct = 100 * (1 - total_parse / max(total_actions, 1))
        print(
            f"  {diff.value:<10s} {len(diff_results):5d} {avg_score:10.3f} {valid_pct:7.1f}% {parse_pct:7.1f}%"
        )

    overall_avg = sum(r["final_score"] for r in results) / max(len(results), 1)
    print(f"\n  Overall avg score: {overall_avg:.3f}")
    print()

    # Save detailed results
    out_path = Path("outputs/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    logger.info(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
