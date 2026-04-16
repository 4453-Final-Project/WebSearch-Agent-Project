# Run Requirements

This document describes the minimum setup needed for a full BrowserGym/WebArena GRPO run in this repo.

## Purpose

The repo now supports two practical run shapes:

1. the older single-task path for smoke tests and local debugging
2. the staged disjoint-task path for the strongest validated Liquid 350M result on Shopping

The recommended experimental path is:

1. verify the local runtime
2. run a single BrowserGym episode
3. evaluate a baseline on the intended eval family
4. collect or load warmup demos
5. run behavior-cloning warmup on one task subset
6. run GRPO on a different task subset
7. evaluate baseline, warmup-only, and post-GRPO on the same eval family

## Machine Topology

The cleanest setup is a two-machine layout.

### 1. Local Training Machine

This machine runs:

- Python scripts in this repo
- the local Qwen model
- Playwright / Chromium
- BrowserGym client code
- rollout collection
- evaluation
- warmup
- GRPO optimization

Recommended:

- Linux or WSL2
- Python 3.12
- NVIDIA GPU with enough VRAM for your chosen local model
- enough disk for the model, adapters, screenshots, videos, and rollouts

Required local directories:

```text
RL4453/
├── .venv/
├── models/
│   └── Qwen3.5-2B/
└── WebSearch-Agent-Project/
```

### 2. WebArena Host Machine

This machine runs the WebArena websites that BrowserGym talks to.

This can be:

- an EC2 instance created from a WebArena-compatible AMI
- another cloud VM
- a separate local Linux machine
- the same machine as training, if you intentionally want a single-box setup

What matters is that the websites are reachable from the training machine.

## Services That Must Be Running

The following WebArena endpoints must be live and reachable from the training machine:

- `WA_SHOPPING`
- `WA_SHOPPING_ADMIN`
- `WA_REDDIT`
- `WA_GITLAB`
- `WA_WIKIPEDIA`
- `WA_MAP`
- `WA_HOMEPAGE`

In the standard setup these map to ports like:

- `7770` shopping
- `7780` shopping admin
- `9999` reddit
- `8023` gitlab
- `8888` wikipedia
- `3000` map
- `4399` homepage

If you are using an EC2 or AMI-based WebArena host:

- keep its public DNS name or IP stable if possible
- if the hostname changes, update `.env` before running again
- make sure the security group allows inbound traffic from the training machine to the required ports

## Local Software Requirements

Install on the training machine:

- Python 3.12
- a virtual environment at `../.venv`
- repo dependencies from `requirements.txt`
- Playwright Chromium

Typical setup:

```bash
cd WebSearch-Agent-Project
source ../.venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Model Requirements

Official run target:

- model id: `Qwen/Qwen3.5-2B`
- default local path: `../models/Qwen3.5-2B`

Important:

- keep `models/` outside the repo as a sibling directory
- official comparisons should use `Qwen/Qwen3.5-2B`
- for smoke tests, debugging, and quick experiments on limited hardware, prefer smaller local models when appropriate

### Model Hot-Swapping

You can test other local checkpoints without editing code.

Every runnable model-using script supports one of:

- `--model-dir-name <dir-under-../models>`
- `--model-path <absolute-or-relative-path>`

Examples:

```bash
python scripts/test_model.py --model-dir-name Qwen3.5-2B
python scripts/run_single_task.py --policy qwen --model-path ../models/MyOtherCheckpoint
python scripts/eval_single_task.py --policy qwen --model-dir-name MyTestModel
python scripts/train_grpo_single_task.py --model-path ../models/MyTestModel
```

The override only changes which local files are loaded. It does not change repo defaults.

## Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
source .env
```

Required:

- `WA_SHOPPING`
- `WA_SHOPPING_ADMIN`
- `WA_REDDIT`
- `WA_GITLAB`
- `WA_WIKIPEDIA`
- `WA_MAP`
- `WA_HOMEPAGE`

Optional:

- `WA_FULL_RESET`

Optional for fuzzy-judged WebArena evaluations:

- `OPENAI_API_KEY`
- `OPENAI_JUDGE_MODEL`

Most exact-match tasks do not need an OpenAI API key. You only need one when the underlying WebArena task uses its LLM-based fuzzy judge or unachievable-answer judge. In this repo, the runner now overrides WebArena's old `gpt-4-1106-preview` default and uses `OPENAI_JUDGE_MODEL`, defaulting to `gpt-5-mini`. If cost matters more than judge quality, `gpt-5.4-nano` is the cheapest reasonable starting point.

The repo `.env` continues to use `WA_*` variable names. The repo now mirrors those values to WebArena's older bare names like `REDDIT` and `SHOPPING` at runtime when needed, so you do not need to export both sets manually.

You can also keep a local ignored helper like `scripts/webarena_env.local.sh`, but `.env` is the simplest shared path.

## Pre-Run Checklist

Before any evaluation or training run, confirm:

1. the repo is in `WebSearch-Agent-Project/`
2. `.venv/` exists beside the repo
3. the selected local model exists beside the repo in `models/`
4. `.env` is sourced in the current shell
5. the WebArena host websites are reachable from the training machine
6. Playwright Chromium is installed

Then run:

```bash
python scripts/check_env.py
```

Use model overrides here too if needed:

```bash
python scripts/check_env.py --model-dir-name MyTestModel
```

## Clean Run Order

### 1. Environment Verification

```bash
python scripts/check_env.py
```

### 2. Local Model Smoke Test

```bash
python scripts/test_model.py
```

### 3. One Dummy Episode

```bash
python scripts/run_single_task.py --policy dummy --task-id 310
```

### 4. One Qwen Episode

```bash
python scripts/run_single_task.py --policy qwen --task-id 310
```

### 5. Baseline Evaluation

```bash
python scripts/eval_single_task.py --policy qwen --task-id 310 --episodes 5
```

### 6. Warmup Demo Collection

```bash
python scripts/collect_warmup_demos.py --task-id 310 --episodes 4
```

### 7. Recommended Staged Shopping Run

```bash
bash scripts/local/run_liquid_shopping_disjoint.sh
```

PowerShell:

```powershell
.\scripts\local\run_liquid_shopping_disjoint.ps1
```

This helper runs the best validated staged experiment in the repo:

- matched baseline on Shopping `324-328`
- warmup on `325-326`
- GRPO on disjoint tasks `327-328`
- final eval on `324-328`

### 8. Official-Target Staged Qwen Run

```bash
bash scripts/local/run_qwen_shopping_disjoint.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_shopping_disjoint.ps1
```

This mirrors the validated Shopping staging recipe, but uses `Qwen3.5-2B` and writes to `../outputs/qwen_shopping_disjoint_v1/`.

### 9. Legacy Single-Task GRPO Run

```bash
python scripts/train_grpo_single_task.py \
  --task-id 310 \
  --warmup-demo-dir ../outputs/warmup_demos/task_0310 \
  --warmup-epochs 4 \
  --iterations 3 \
  --groups-per-iteration 2 \
  --group-size 2
```

## Artifacts Produced

Runs write outputs outside the repo source tree under the sibling `outputs/` directory.

Typical artifacts:

- episode logs
- screenshots
- videos
- evaluation metrics
- warmup summaries
- rollout traces
- LoRA adapter checkpoints
- before/after metrics tables

## Common Failure Modes

### WebArena Host Changed

Symptoms:

- URL reachability failures in `check_env.py`
- BrowserGym reset failures

Fix:

- update `.env` with the new host or DNS name
- confirm the host security rules still expose the required ports

### Playwright Browser Missing

Fix:

```bash
playwright install chromium
```

### Model Not Found

Symptoms:

- missing `config.json`
- local model path errors

Fix:

- confirm the model is stored under `../models/<name>`
- use `--model-path` if the checkpoint lives somewhere else

### Websites Reachable In Browser But Training Still Fails

Check:

- the env vars were sourced into the same shell that launches Python
- the task id is valid for BrowserGym WebArena
- the remote services are fully initialized, not just booting

## Minimal Operator Summary

If someone else needs to run this project, they need:

- one machine for training and local model inference
- one WebArena host machine or equivalent service stack
- Python 3.12, repo dependencies, and Playwright on the training machine
- the Qwen model stored locally beside the repo
- all `WA_*` URLs pointed at the live WebArena host
- a successful `python scripts/check_env.py` before any rollout or GRPO run
