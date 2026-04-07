# TriageSieve-OpenEnv: Complete Setup & Usage Guide

This guide covers everything needed to run, test, build, and deploy the TriageSieve-OpenEnv environment on any system.

---

## Prerequisites

| Tool | Version | Check command |
|---|---|---|
| Python | >= 3.10 | `python --version` |
| uv (recommended) | latest | `uv --version` |
| Docker | latest | `docker --version` |
| Git | latest | `git --version` |
| openenv-core | >= 0.2.2 | `openenv --version` |

### Install uv (if not installed)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 1. Initial Setup

### Clone and install dependencies

```bash
git clone <your-repo-url> triagesieve_env
cd triagesieve_env

# Option A: using uv (recommended - fast, reproducible)
uv sync

# Option B: using pip
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Verify installation

```bash
# Should print package version info
python -c "from triagesieve_env.models import ActionType; print('OK:', list(ActionType))"
```

---

## 2. Running Tests

### Full test suite (~729 tests, ~4 minutes)

```bash
python -m pytest tests/ -q
```

### Single test file

```bash
python -m pytest tests/test_rewards.py -q
python -m pytest tests/test_transitions.py -q
python -m pytest tests/test_determinism.py -q
python -m pytest tests/test_environment_part1.py -q
python -m pytest tests/test_environment_part2.py -q
python -m pytest tests/test_scripted_expert.py -q
python -m pytest tests/test_episode_engine.py -q
python -m pytest tests/test_openenv_integration.py -q
```

### Single test by name

```bash
python -m pytest tests/test_transitions.py -k "test_route_before_classify" -q
```

### Stop on first failure with traceback

```bash
python -m pytest tests/ -x --tb=short
```

### Run with coverage

```bash
python -m pytest tests/ --cov=server --cov=baseline --cov-report=term-missing
```

### Notes

- `tests/test_llm_baseline.py` skips most tests unless a real LLM API key is set (this is expected).
- All tests are deterministic and seed-based — no flaky tests.

---

## 3. Running the Scripted Expert Baseline

The scripted expert is an oracle agent that reads hidden ground truth. It proves the environment is solvable.

### Smoke playthrough (all difficulties)

```bash
python scripts/smoke_playthrough.py
```

Output:
```
easy:   1.0000  PASS (threshold=0.90)
medium: 1.0000  PASS (threshold=0.85)
hard:   0.3833  PASS (threshold=0.35)
```

### Single difficulty

```bash
python scripts/smoke_playthrough.py --difficulty easy
python scripts/smoke_playthrough.py --difficulty medium
python scripts/smoke_playthrough.py --difficulty hard
```

### Custom seed

```bash
python scripts/smoke_playthrough.py --seed 0
python scripts/smoke_playthrough.py --seed 100 --difficulty easy
```

### Quiet mode (suppress step-by-step trace)

```bash
python scripts/smoke_playthrough.py --quiet
```

---

## 4. Running the Server Locally

### Start the FastAPI server

```bash
# Option A: uvicorn directly
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Option B: via project entry point
uv run server
```

### Verify server is running

```bash
curl http://localhost:8000/health
# Should return: {"status": "ok"}

curl http://localhost:8000/docs
# Opens Swagger UI in browser
```

### Test with the Python client

```python
import asyncio
from triagesieve_env import TriageSieveEnv
from triagesieve_env.models import TriageSieveAction, ActionType

async def main():
    env = TriageSieveEnv(base_url="http://localhost:8000")
    result = await env.reset(seed=42, difficulty="easy", mode="eval_strict")
    obs = result.observation
    print(f"Tickets: {len(obs.inbox_summaries)}, Budget: {obs.action_budget_remaining}")

    # Open first ticket
    tid = obs.inbox_summaries[0].ticket_id
    result = await env.step(TriageSieveAction(
        action_type=ActionType.OPEN_TICKET,
        ticket_id=tid,
    ))
    print(f"Result: {result.observation.last_action_result}")
    print(f"Focused ticket subject: {result.observation.focused_ticket.subject}")

asyncio.run(main())
```

---

## 5. Docker Build & Run

### Build the Docker image

```bash
# From project root (where Dockerfile is)
docker build -t triagesieve_env .
```

This uses the OpenEnv multi-stage Dockerfile:
- **Stage 1 (builder)**: Installs dependencies via `uv sync` from `uv.lock`
- **Stage 2 (runtime)**: Copies only `.venv` + code, runs uvicorn on port 8000

### Run the container

```bash
docker run -p 8000:8000 triagesieve_env
```

### Verify container health

```bash
curl http://localhost:8000/health
```

### Test with Docker client

```python
import asyncio
from triagesieve_env import TriageSieveEnv

async def main():
    # Connects to container on port 8000
    env = await TriageSieveEnv.from_docker_image("triagesieve_env")
    result = await env.reset(seed=42, difficulty="easy")
    print(f"Tickets: {len(result.observation.inbox_summaries)}")
    await env.close()

asyncio.run(main())
```

---

## 6. Running the Inference Script

The `inference.py` is the hackathon-mandated evaluation script. It runs an LLM agent against the Dockerized environment and reports scores.

### Required environment variables

```bash
# REQUIRED
export HF_TOKEN="your-huggingface-token"
export LOCAL_IMAGE_NAME="triagesieve_env"

# OPTIONAL (have defaults)
export API_BASE_URL="https://router.huggingface.co/v1"   # default
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # default
```

### Windows (PowerShell)

```powershell
$env:HF_TOKEN = "your-huggingface-token"
$env:LOCAL_IMAGE_NAME = "triagesieve_env"
```

### Windows (CMD)

```cmd
set HF_TOKEN=your-huggingface-token
set LOCAL_IMAGE_NAME=triagesieve_env
```

### Steps to run

```bash
# 1. Build the Docker image (if not already built)
docker build -t triagesieve_env .

# 2. Set env vars (see above)

# 3. Run inference
python inference.py
```

### What it does

1. For each difficulty (easy, medium, hard):
   - Spins up the environment via Docker
   - Calls `env.reset(seed, difficulty)` to get the inbox
   - Loops: serializes observation → sends to LLM → parses action → `env.step(action)`
   - Prints `[START]`, `[STEP]`, `[END]` to stdout (mandatory format for automated judging)
2. Prints a summary with scores per task

### Expected stdout format

```
[START] task=easy env=triagesieve_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=open_ticket:T001 reward=0.01 done=false error=null
[STEP] step=2 action=classify_ticket:T001:billing reward=0.03 done=false error=null
...
[END] success=true steps=5 score=0.850 rewards=0.01,0.03,0.01,0.01,0.85

[START] task=medium env=triagesieve_env model=Qwen/Qwen2.5-72B-Instruct
...
```

### Using a different model

```bash
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py
```

### Using a different API endpoint

```bash
export API_BASE_URL="http://localhost:11434/v1"  # e.g., local Ollama
export MODEL_NAME="llama3.3"
python inference.py
```

---

## 7. Episode Generation & Validation

### Generate the seeded episode bank

```bash
python scripts/generate_episodes.py --seed 0 --count 100 --difficulty all --output data/seeded_episodes.jsonl
```

Options:
- `--seed <int>`: Master seed (episodes get seeds seed+0, seed+1, ...)
- `--count <int>`: Number of episodes to generate
- `--difficulty <easy|medium|hard|all>`: Filter by difficulty (`all` = round-robin)
- `--output <path>`: Output JSONL file

### Validate the episode bank

```bash
python scripts/validate_episode_bank.py --input data/seeded_episodes.jsonl
```

Checks:
1. **Parse pass**: Valid JSON, required keys present, non-zero tickets
2. **Determinism pass**: Re-renders episodes from (seed, difficulty) and verifies they match
3. **Solvability pass**: Runs scripted expert and checks score >= threshold

---

## 8. Deploying to Hugging Face Spaces

### Validate before deploying

```bash
openenv validate
# Expected: [OK] triagesieve: Ready for multi-mode deployment
```

### Push to HF Spaces

```bash
# Login to Hugging Face (if not already)
huggingface-cli login

# Push
openenv push your-username/triagesieve_env
```

### Verify deployment

```bash
curl -X POST https://your-username-triagesieve-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
# Should return 200 with initial observation JSON
```

---

## 9. Project Structure Reference

```
triagesieve_env/
├── inference.py              # Hackathon inference script (LLM → environment)
├── Dockerfile                # Docker image definition (multi-stage, OpenEnv base)
├── .dockerignore             # Excludes .venv, .git, __pycache__ from Docker build
├── openenv.yaml              # OpenEnv manifest (name, runtime, port)
├── pyproject.toml            # Package metadata + dependencies
├── uv.lock                   # Locked dependency versions
├── README.md                 # Judge-facing documentation
├── SETUP.md                  # This file
├── CLAUDE.md                 # AI assistant instructions
├── __init__.py               # Package exports
├── models.py                 # All Pydantic models + enums (Action, Observation, State)
├── client.py                 # TriageSieveEnv(EnvClient) async client
├── server/
│   ├── app.py                # FastAPI entrypoint via create_app()
│   ├── triagesieve_env_environment.py  # Environment(step/reset/state)
│   ├── episode_engine.py     # Deterministic episode generation from archetypes
│   ├── policy_graph.py       # SOP DAG definitions + UJCS computation
│   ├── scorer.py             # Terminal scoring, penalties, final score formula
│   ├── hint_engine.py        # Guided-mode hints (train_guided only)
│   └── Dockerfile            # Copy of root Dockerfile
├── baseline/
│   ├── scripted_expert.py    # Oracle policy (reads hidden truth) — proves solvability
│   └── llm_baseline.py       # LiteLLM-based agent (no hidden truth access)
├── data/
│   ├── archetypes.json       # 18 scenario archetypes with SOP graphs
│   ├── templates.json        # Reply/closure templates
│   ├── routing_rules.json    # Queue prerequisites + issue family mapping
│   ├── sla_rules.json        # Customer tier → SLA deadline mapping
│   └── seeded_episodes.jsonl # Pre-rendered episode cache (100 episodes)
├── scripts/
│   ├── generate_episodes.py  # CLI episode generator
│   ├── validate_episode_bank.py  # Validates bank parse + determinism + solvability
│   └── smoke_playthrough.py  # Runs scripted expert, asserts thresholds
├── tests/                    # 729 tests (pytest)
│   ├── test_transitions.py   # State machine edge cases
│   ├── test_rewards.py       # Scoring regression tests
│   ├── test_determinism.py   # Seed replay tests
│   ├── test_openenv_integration.py  # reset/step/state contract
│   ├── test_environment_part1.py    # Format gate, action dispatch
│   ├── test_environment_part2.py    # Workflows, merge, close
│   ├── test_episode_engine.py       # Episode generation
│   ├── test_scripted_expert.py      # Expert correctness
│   ├── test_llm_baseline.py         # LLM baseline (skips without API key)
│   └── test_validate_episode_bank.py # Bank validation
└── outputs/
    ├── logs/                 # Runtime logs (gitignored)
    └── evals/                # Structured traces (gitignored)
```

---

## 10. Scoring Quick Reference

```
FinalScore = TerminalBusinessScore (max 0.85)
           + 0.15 x UJCS_OpenEnv
           - EpisodePenalties
           clamped to [0, 1]
```

### Terminal business score components (per ticket, weighted by priority)

| Component | Weight |
|---|---|
| Classification correctness | 0.15 |
| Impact/urgency correctness | 0.15 |
| Queue correctness | 0.20 |
| Missing-info handling | 0.10 |
| Escalation correctness | 0.10 |
| Duplicate/non-actionable handling | 0.05 |
| Template choice correctness | 0.05 |
| Terminal status correctness | 0.05 |

### Priority weights

| Priority | Weight |
|---|---|
| critical | 2.0 |
| high | 1.5 |
| medium | 1.0 |
| low | 0.5 |

### Penalties

| Penalty | Value |
|---|---|
| Invalid action | -0.03 |
| Avoidable reassignment | -0.05 |
| Unnecessary escalation | -0.05 |
| Urgent-ticket SLA mishandling | -0.05 to -0.10 |

### Task budgets

| Difficulty | Tickets | Budget |
|---|---|---|
| easy | 1 | 6 steps |
| medium | 2-3 | 12 steps |
| hard | 3-4 | 14 steps |

---

## 11. Troubleshooting

### "Module not found" errors

Make sure you're running from the project root and your venv is active:
```bash
cd triagesieve_env
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### Docker build fails

- Ensure Docker Desktop is running
- Check `.dockerignore` excludes `.venv/` (otherwise the build context is huge)
- Try: `docker build --no-cache -t triagesieve_env .`

### Tests fail with import errors

```bash
# Ensure dev dependencies are installed
uv sync
# or: pip install -e ".[dev]"
```

### inference.py can't connect to environment

```bash
# Make sure Docker image is built
docker build -t triagesieve_env .

# Make sure LOCAL_IMAGE_NAME matches
export LOCAL_IMAGE_NAME="triagesieve_env"

# Check Docker is running
docker ps
```

### openenv validate fails

```bash
# Install openenv-core
pip install "openenv-core[core]>=0.2.2"

# Run from project root (where openenv.yaml is)
cd triagesieve_env
openenv validate
```
