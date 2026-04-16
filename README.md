# BrowserGym Local RL

This repo is a compact BrowserGym/WebArena training stack for local models with:

- environment validation
- single-task rollouts
- baseline evaluation
- behavior-cloning warmup
- staged GRPO
- LoRA adapter saving and reload

The current best validated result is the staged Shopping search/sort run on tasks `324-328` with `LFM2.5-350M`:

- baseline: `3/5`
- warmup-only: `4/5`
- warmup + disjoint GRPO: `5/5`

That run uses disjoint training splits:

- warmup on `325-326`
- GRPO on `327-328`
- final evaluation on `324-328`

## Workspace Layout

```text
RL4453/
├── .venv/
├── models/
│   ├── Qwen3.5-2B/
│   └── LFM2.5-350M/
└── WebSearch-Agent-Project/
```

Use Python `3.12`. Keep `.venv/` and `models/` as siblings of the repo.

## Model Targets

Official comparison target:

- `Qwen/Qwen3.5-2B`
- local path: `../models/Qwen3.5-2B`

Fast local experiment target:

- `LiquidAI/LFM2.5-350M`
- local path: `../models/LFM2.5-350M`

All runnable scripts accept `--model-dir-name` or `--model-path`, so you can swap local checkpoints without editing code.

## Setup

From `WebSearch-Agent-Project/`:

```bash
source ../.venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
source .env
```

Required WebArena variables:

- `WA_SHOPPING`
- `WA_SHOPPING_ADMIN`
- `WA_REDDIT`
- `WA_GITLAB`
- `WA_WIKIPEDIA`
- `WA_MAP`
- `WA_HOMEPAGE`

Optional:

- `WA_FULL_RESET`
- `OPENAI_API_KEY`
- `OPENAI_JUDGE_MODEL`

Most exact-match tasks do not need an OpenAI key. It is only needed for fuzzy-judged WebArena tasks.

## Quick Start

Validate the environment:

```bash
python scripts/check_env.py
```

Run a dummy episode:

```bash
python scripts/run_single_task.py --policy dummy --task-id 310
```

Run a local model episode:

```bash
python scripts/run_single_task.py --policy qwen --task-id 310 --model-dir-name LFM2.5-350M
```

Run a baseline evaluation:

```bash
python scripts/eval_single_task.py --policy qwen --task-id 310 --episodes 5 --model-dir-name LFM2.5-350M
```

Collect scripted warmup demos:

```bash
python scripts/collect_warmup_demos.py --task-id 310 --episodes 4
```

## Recommended Staged Run

Best validated local experiment:

```bash
bash scripts/local/run_liquid_shopping_disjoint.sh
```

PowerShell:

```powershell
.\scripts\local\run_liquid_shopping_disjoint.ps1
```

That helper runs:

- baseline on Shopping `324-328`
- warmup on `325-326`
- GRPO on disjoint tasks `327-328`
- final evaluation and figure export

Official-target helper:

```bash
bash scripts/local/run_qwen_shopping_disjoint.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_shopping_disjoint.ps1
```

## Utilities

Download the default model:

```bash
bash scripts/download_models.sh
```

Smoke test local inference:

```bash
python scripts/test_model.py
```

Benchmark local inference:

```bash
python scripts/benchmark_inference.py --profile lfm2.5-350m --model-dir-name LFM2.5-350M --repeats 5
```

Validate the saved Shopping result bundle:

```bash
python scripts/validate_final_results.py
```

## More Detail

See [RUN_REQUIREMENTS.md](./RUN_REQUIREMENTS.md) for the full setup, service, and run checklist.
