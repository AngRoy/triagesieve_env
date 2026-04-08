---
title: TriageSieve Environment Server
emoji: 🔊
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# TriageSieve-OpenEnv

A deterministic, stateful **support-ticket triage environment** built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework for training and evaluating AI agents on real-world customer support workflows.

The agent plays the role of a tier-1 support engineer: it reads a live inbox of 1-4 tickets, inspects each one, classifies the issue type, assesses business impact and urgency, requests missing information from customers, routes tickets to the correct support queue, and closes them with the appropriate resolution template. Every decision is scored programmatically against hidden ground truth using an 8-component terminal business score, SOP path adherence (UJCS), and episode-level penalty tracking.

---

## Architecture

```mermaid
flowchart LR
    subgraph AGENT["&nbsp;&nbsp;Agent&nbsp;&nbsp;"]
        direction TB
        A1(["Observe"])
        A2(["Decide"])
        A1 --> A2
    end

    subgraph ENV["&nbsp;&nbsp;TriageSieve Environment&nbsp;&nbsp;"]
        direction TB

        subgraph GATE["&ensp;Validation Layer&ensp;"]
            direction LR
            G1["Format Gate<br/><i>schema + enum check</i>"]
            G2["State Machine<br/><i>legal action filter</i>"]
            G1 --> G2
        end

        subgraph CORE["&ensp;Execution Layer&ensp;"]
            direction LR
            C1["Action Dispatch<br/><i>10 action handlers</i>"]
            C2["SOP Tracker<br/><i>policy DAG traversal</i>"]
            C1 --> C2
        end

        subgraph SCORE["&ensp;Scoring Layer&ensp;"]
            direction LR
            S1["Step Shaping<br/><i>immediate reward</i>"]
            S2["Terminal Scorer<br/><i>8-component + UJCS</i>"]
        end

        GATE --> CORE
        CORE --> SCORE
    end

    subgraph ENGINE["&nbsp;&nbsp;Episode Engine&nbsp;&nbsp;"]
        direction TB
        D1["18 Archetypes"]
        D2["Seeded RNG"]
        D3["Hidden Truth"]
        D1 --> D2 --> D3
    end

    A2 -- "TriageSieveAction" --> G1
    S1 -- "reward + observation" --> A1
    S2 -. "final score ∈ (0,1)" .-> A1
    D3 -. "ground truth<br/>reference" .-> S2
    G1 -- "invalid → −0.02" --> A1

    style AGENT fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    style ENV fill:#fef9ef,stroke:#d97706,stroke-width:2px,color:#78350f
    style ENGINE fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    style GATE fill:#fef3c7,stroke:#d97706,color:#78350f
    style CORE fill:#fed7aa,stroke:#d97706,color:#78350f
    style SCORE fill:#fde68a,stroke:#d97706,color:#78350f
```

## Ticket Lifecycle

```mermaid
stateDiagram-v2
    direction LR

    [*] --> NEW
    NEW --> OPENED : open_ticket

    state OPENED {
        direction LR
    }
    OPENED --> CLASSIFIED : classify_ticket
    OPENED --> MERGED : merge_duplicate
    OPENED --> CLOSED : close<br/>(non-actionable only)

    state CLASSIFIED {
        direction LR
    }
    CLASSIFIED --> WAITING_FOR_INFO : request_information
    CLASSIFIED --> ROUTED : route_ticket
    CLASSIFIED --> ESCALATED : escalate_ticket
    CLASSIFIED --> MERGED : merge_duplicate
    CLASSIFIED --> CLOSED : close_ticket

    WAITING_FOR_INFO --> CLASSIFIED : re-classify
    WAITING_FOR_INFO --> ROUTED : route_ticket
    WAITING_FOR_INFO --> ESCALATED : escalate_ticket
    WAITING_FOR_INFO --> CLOSED : close_ticket

    ROUTED --> ESCALATED : escalate_ticket
    ROUTED --> CLOSED : close_ticket

    ESCALATED --> CLOSED : close_ticket

    MERGED --> [*]
    CLOSED --> [*]

    note right of ROUTED
        Gated queues (L2, Security)
        require classification +
        impact/urgency before routing
    end note

    note right of MERGED
        Terminal states:
        no further actions
    end note
```

## Scoring System

```mermaid
flowchart TB
    subgraph BIZ["Terminal Business Score &ensp; <i>max 0.85</i>"]
        direction LR
        subgraph PRIMARY["Primary Components"]
            direction TB
            C3["<b>Queue Routing</b><br/>0.20"]
            C1["<b>Classification</b><br/>0.15"]
            C2["<b>Impact / Urgency</b><br/>0.15"]
        end
        subgraph SECONDARY["Secondary Components"]
            direction TB
            C4["<b>Missing Info</b><br/>0.10"]
            C5["<b>Escalation</b><br/>0.10"]
        end
        subgraph TERTIARY["Tertiary Components"]
            direction TB
            C6["<b>Dup / Non-actionable</b><br/>0.05"]
            C7["<b>Template Choice</b><br/>0.05"]
            C8["<b>Terminal Status</b><br/>0.05"]
        end
    end

    subgraph UJCS_BOX["UJCS-OpenEnv &ensp; <i>weighted 0.15</i>"]
        direction TB
        U1["Compare agent path<br/>against gold SOP"]
        U2["+1 checkpoint visited<br/>−1 checkpoint skipped<br/>−1 illegal transition"]
        U3["Normalize to 0 – 1"]
        U1 --> U2 --> U3
    end

    subgraph PEN["Episode Penalties &ensp; <i>subtracted</i>"]
        direction TB
        P1["Invalid action &ensp; <b>−0.03</b> each"]
        P2["Avoidable reassignment &ensp; <b>−0.05</b>"]
        P3["Unnecessary escalation &ensp; <b>−0.05</b>"]
        P4["SLA mishandling &ensp; <b>−0.05 / −0.10</b>"]
    end

    BIZ --> FORMULA
    UJCS_BOX --> FORMULA
    PEN --> FORMULA

    FORMULA["<b>Final Score</b> = Business Score + 0.15 x UJCS − Penalties<br/><i>clamped to (0, 1) &ensp;|&ensp; priority-weighted across tickets</i>"]

    style BIZ fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style PRIMARY fill:#a7f3d0,stroke:#059669,color:#064e3b
    style SECONDARY fill:#a7f3d0,stroke:#059669,color:#064e3b
    style TERTIARY fill:#a7f3d0,stroke:#059669,color:#064e3b
    style UJCS_BOX fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    style PEN fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    style FORMULA fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
```

---

## Task Description

Each episode presents the agent with a support inbox. The agent must triage every ticket by following the correct Standard Operating Procedure (SOP) for its issue type. Ground truth (issue family, correct queue, urgency, customer tier, etc.) is hidden from the agent - it must infer these from visible ticket content.

### Difficulty Tiers

| Tier | Tickets | Budget | Challenge |
|------|---------|--------|-----------|
| **Easy** | 1 | 6 steps | Single-path SOP; straightforward classification and routing |
| **Medium** | 2-3 | 12 steps | Mixed issue types; SLA-sensitive tickets; possible duplicates |
| **Hard** | 3-4 | 14 steps | Urgent/security ticket + enterprise SLA; merge candidates; non-actionable distractors; gated-queue prerequisite chains |

Hard episodes are intentionally budget-constrained relative to a perfect oracle solution - partial completion is by design, testing the agent's ability to prioritize.

---

## Action Space

The agent submits a `TriageSieveAction` tagged union discriminated by `action_type`:

| Action | Required Fields | Purpose |
|--------|----------------|---------|
| `open_ticket` | `ticket_id` | Load full ticket detail into focused view |
| `classify_ticket` | `ticket_id`, `issue_family`, `issue_subtype` | Label the issue type (5 families x 3 subtypes = 15 categories) |
| `set_impact_urgency` | `ticket_id`, `impact`, `urgency` | Assess business scope and time sensitivity |
| `route_ticket` | `ticket_id`, `queue_id` | Send to one of 9 support queues |
| `request_information` | `ticket_id`, `requested_fields`, `template_id` | Ask customer for missing data before routing |
| `escalate_ticket` | `ticket_id`, `queue_id`, `reason_code` | Escalate to a higher-tier queue with justification |
| `merge_duplicate` | `ticket_id`, `target_ticket_id` | Merge a duplicate ticket into the original |
| `close_ticket` | `ticket_id`, `close_reason` | Close with reason: resolved, duplicate, non_actionable, feature_request, no_response |
| `skip_turn` | - | No-op; consumes one budget step |
| `finish_episode` | - | Explicitly end the episode early |

Two queues (`tech_support_l2`, `security_team`) are **gated** - attempting to route without meeting prerequisites returns a pushback message and a penalty.

---

## Observation Space

Each step returns a `TriageSieveObservation` containing:

| Field | Description |
|-------|-------------|
| `inbox_summaries` | All tickets with subject, sender, status, customer tier, SLA remaining, preview |
| `focused_ticket` | Full content of the last opened ticket (thread history, attachments, internal notes) |
| `routing_policy_cards` | Queue descriptions with prerequisites and handled issue families |
| `sla_policy_cards` | Customer tier SLA deadlines (response + resolution) |
| `available_templates` | Reply/closure templates the agent can reference |
| `legal_actions` | Actions currently valid given the episode state |
| `action_budget_remaining` | Steps left before forced termination |
| `last_action_result` | `"ok"` or a precise error/pushback message |
| `reward` | Step shaping reward, or the final composite score on the terminal step |
| `done` | Whether the episode has ended |
| `hint` | Guided-mode hint string (only in `train_guided` mode; never affects scoring) |

---

## Baseline Scores

### Scripted Expert (oracle with hidden truth access, seed=42)

| Tier | Score | Threshold |
|------|-------|-----------|
| Easy | **1.000** | >= 0.90 |
| Medium | **1.000** | >= 0.75 |
| Hard | **0.383** | >= 0.20 |

### LLM Baseline Results (no hidden truth access, multi-turn, seed=42)

| Model | Easy | Medium | Hard | Avg |
|-------|------|--------|------|-----|
| Llama-3.3-70B-Instruct | **1.000** | **0.771** | **0.122** | **0.631** |
| Qwen2.5-72B-Instruct | 0.720 | 0.627 | 0.155 | 0.501 |
| DeepSeek-R1 | 0.470 | 0.253 | 0.076 | 0.267 |

Llama-3.3-70B achieves a perfect score on easy and the highest average across all tiers. On hard tasks, all models struggle with budget management — they must process 3-4 tickets in 14 steps, requiring precise action ordering (request info before routing) and prioritization of high-urgency tickets. This validates the difficulty ladder design: hard tasks are genuinely challenging for frontier LLMs without fine-tuning.

### Training Validation (SFT on expert demonstrations)

To verify the environment produces a learnable reward signal, we fine-tuned Qwen2.5-0.5B-Instruct via QLoRA on 985 expert trajectory pairs (100 episodes, all difficulties). Results on 15 held-out episodes:

| Metric | Zero-shot Qwen 0.5B | SFT-trained Qwen 0.5B | Improvement |
|--------|---------------------|------------------------|-------------|
| Easy | 0.412 | **0.612** | +48% |
| Medium | 0.171 | **0.273** | +60% |
| Hard | 0.000 | 0.000 | — |
| **Overall** | 0.194 | **0.295** | **+52%** |
| Best episode | 0.500 | **0.860** | Near-expert |

The trained model achieved 97.75% token accuracy on eval (loss: 0.097), demonstrating that the observation format is learnable and the reward signal drives meaningful improvement.

---

## Quick Start

### Installation

```bash
# Requires Python >= 3.10
uv sync
# or
pip install -e ".[dev]"
```

### Run the server locally

```bash
uv run server
# or: python -m uvicorn triagesieve_env.server.app:app --reload --port 8000
```

### Connect with the Python client

```python
import asyncio
from triagesieve_env import TriageSieveEnv
from triagesieve_env.models import TriageSieveAction, ActionType

async def main():
    env = TriageSieveEnv(base_url="http://localhost:8000")
    result = await env.reset(seed=42, difficulty="easy", mode="eval_strict")
    obs = result.observation

    ticket_id = obs.inbox_summaries[0].ticket_id
    result = await env.step(TriageSieveAction(
        action_type=ActionType.OPEN_TICKET,
        ticket_id=ticket_id,
    ))
    print(result.observation.focused_ticket.subject)
    await env.close()

asyncio.run(main())
```

### Run the scripted expert baseline

```bash
python scripts/smoke_playthrough.py                    # all tiers, seed=42
python scripts/smoke_playthrough.py --difficulty easy   # single tier
python scripts/smoke_playthrough.py --seed 7 --quiet    # custom seed, no trace
```

### Run the test suite

```bash
python -m pytest tests/ -q                              # all tests (~770)
python -m pytest tests/ -m "not slow" -q                # skip real LLM tests
python -m pytest tests/test_rewards.py -q               # scoring tests only
```

---

## Docker Build and Deployment

### Build and run locally

```bash
docker build -t triagesieve_env .
docker run -p 8000:8000 triagesieve_env
```

### Deploy to Hugging Face Spaces

```bash
openenv validate
openenv push your-username/triagesieve_env
```

The deployed Space exposes:
- `/web` - Interactive web UI
- `/docs` - OpenAPI/Swagger documentation
- `/ws` - WebSocket endpoint for low-latency sessions
- `/health` - Container health check

---

## Running the Inference Script

The `inference.py` script runs an LLM agent against the Dockerized environment, producing structured logs for automated evaluation.

```bash
# Set environment variables
export HF_TOKEN="your-huggingface-token"
export LOCAL_IMAGE_NAME="triagesieve_env"

# Optional (have defaults)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Build Docker image, then run
docker build -t triagesieve_env .
python inference.py
```

Output follows the required `[START]`/`[STEP]`/`[END]` format:
```
[START] task=easy env=triagesieve_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=open_ticket:T001 reward=0.01 done=false error=null
[STEP] step=2 action=classify_ticket:T001:billing reward=0.02 done=false error=null
...
[END] success=true steps=6 score=1.000 rewards=0.01,0.02,0.01,0.03,0.01,1.00
```

---

## Project Structure

```
triagesieve_env/
    __init__.py                 Package exports
    models.py                   All Pydantic models, enums, and constants
    client.py                   Async EnvClient for HTTP/WebSocket
    inference.py                Hackathon inference script (OpenAI client)
    openenv.yaml                OpenEnv environment manifest
    pyproject.toml              Package metadata and dependencies
    Dockerfile                  Multi-stage Docker build (OpenEnv base)
    server/
        app.py                  FastAPI entrypoint via create_app()
        triagesieve_env_environment.py
                                Core Environment (step/reset/state)
        episode_engine.py       Deterministic episode generation from archetypes
        policy_graph.py         SOP DAG definitions, tracker, UJCS computation
        scorer.py               Terminal scoring, penalties, final score formula
        hint_engine.py          Guided-mode hint generation
    baseline/
        scripted_expert.py      Oracle policy (proves solvability)
        llm_baseline.py         LiteLLM-based agent (no hidden truth)
    data/
        archetypes.json         18 scenario archetypes with SOP graphs
        templates.json          Reply/closure templates
        routing_rules.json      Queue prerequisites and issue family mapping
        sla_rules.json          Customer tier SLA deadlines
        seeded_episodes.jsonl   Pre-generated episode bank (100 episodes)
    scripts/
        generate_episodes.py    Deterministic episode generator CLI
        validate_episode_bank.py
                                Validates parse, determinism, solvability
        smoke_playthrough.py    Runs scripted expert, asserts thresholds
    tests/                      770+ tests (pytest)
```

---

## Research Backing

| Concept | Source |
|---------|--------|
| Ticket categorization and hierarchical labels | [Expert Systems with Applications (2023)](https://www.sciencedirect.com/science/article/pii/S0957417423004864) |
| Priority = f(impact, urgency) matrix | [BMC Remedyforce - Creating Priorities](https://docs.bmc.com/xwiki/bin/view/More-Products/RemedyForce/BMC-Helix-Remedyforce/remforce202502/Administering/Configuring-BMC-Remedyforce/Creating-priorities/) |
| Ticket reassignment as complexity proxy | [Decision Analytics Journal (2022)](https://www.sciencedirect.com/science/article/pii/S2666827021001195) |
| Non-actionable sub-categorization (CORTEX) | [arXiv:2510.00311 (2024)](https://papers.cool/arxiv/2510.00311) |
| UJCS for policy adherence | [JourneyBench (2025)](https://researchtrend.ai/papers/2601.00596) |
| OpenEnv framework | [GitHub](https://github.com/meta-pytorch/OpenEnv) / [Docs](https://meta-pytorch.org/OpenEnv/) |

---

## License

BSD-3-Clause (same as OpenEnv)
