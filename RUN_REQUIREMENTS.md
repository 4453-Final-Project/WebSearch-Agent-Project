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
7. evaluate baseline, warmup-only, and post-GRPO on the holdout set and the full eval family

Once the small slice is stable, scale to a larger family. The repo now has a scripted `shopping_exact` curriculum with 32 executable exact-match tasks, which yields 26 training tasks and 6 holdout tasks under the recommended split. The broader scripted `shopping_full` curriculum covers 48 shopping tasks total and uses a 40-task recommended training pool, but the extra 16 tasks are fuzzy-judged and need `OPENAI_API_KEY`. The next cross-site expansion is `bootstrap41` with 33 training tasks and no OpenAI judge requirement, and the next post-bootstrap mixed target is `web_mix88` with 72 training tasks and 16 holdouts.

Current large-family reference points:

- full staged run: `outputs/liquid_shopping_exact_curriculum_smoke_v4/`
- untouched holdout in that run: baseline `0/6`, warmup `1/6`, post-GRPO `2/6`
- current best stack re-eval with the same GRPO adapter: `outputs/liquid_shopping_exact_holdout_reval_v5/`
- untouched holdout after parser, observation, and admin sort-normalization fixes: `6/6`
- official-target full-family run: `outputs/qwen_shopping_exact_curriculum_v1/`
- official-target full-family result: baseline `0.75`, warmup `0.765625`, post-GRPO `0.796875`
- official-target untouched holdout result: baseline `0.8333333333333334`, warmup `0.9166666666666666`, post-GRPO `0.9166666666666666`
- official-target stack re-eval with the same GRPO adapter: `outputs/qwen_shopping_exact_holdout_reval_v1/`
- official-target untouched holdout after admin sort-state normalization: `6/6`
- latest refund-history targeted re-evals on the same official adapter: task `323` succeeds in `outputs/eval_task_323_qwen_after_history_refund_detail_fix/`, and task `321` succeeds in `outputs/eval_task_321_qwen_after_history_aggregate_fix_v2/`

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
- for local `Qwen/Qwen3.5-2B` LoRA reloads, ensure the active interpreter has `peft` installed and a `transformers` build recent enough to load `qwen3_5`

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

Long family runs now reuse cached warmup demos, baseline summaries, rollout episodes, and per-task evaluation metrics. If a long run is interrupted, rerunning the same helper against the same `out_dir` will resume from the missing work rather than restarting the whole pipeline.

The family runner now fails fast for fuzzy-judged families such as `shopping_order_full` and `shopping_full` when `OPENAI_API_KEY` is missing, so large runs do not waste time collecting demos before discovering the judge is unavailable.

For mixed exact-plus-fuzzy families, `family_summary.json` now records separate judge-free and judge-gated task partitions in addition to the overall family success rate. That makes it easier to tell whether broader-family gains are actually transferring to the fuzzy-judged slice instead of only lifting the exact-match subset.

Use `python scripts/validate_family_summary.py --summary-path <family_summary.json>` to sanity-check a saved family artifact before comparing runs. If a task is a known benchmark mismatch rather than a real model miss, pass it with `--benchmark-blocker <task_id>` to get an additional blocker-aware effective success rate without hiding the raw failure.

That validator now also checks `run_provenance` when present. In practice that means split seed mismatches, malformed judge requirements, or stale recommended-split alignment metadata will fail validation before they leak into downstream compare reports.

It now also validates stage-specific `<label>_meta` blocks such as `current_stack_meta`, so merged artifacts with inconsistent override counts, override ids, or copied override metrics are rejected before they are reused in later comparisons.

If you fold targeted task re-evals back into a saved family summary with `scripts/merge_family_reevals.py`, the merged output now also writes a recomputed `<label>_holdout` section. That keeps holdout metrics aligned with the overridden stage instead of silently preserving the stale holdout numbers from the base stage.

`scripts/validate_family_summary.py` also validates those stage-specific holdout sections, so a broken `current_stack_holdout` will fail validation instead of being ignored.

For older saved artifacts that predate the newer summary schema, run `python scripts/normalize_family_summary.py --summary-path <family_summary.json>` first. That backfills `family_metadata`, stage task groups, and custom-stage holdout sections so legacy Liquid/Qwen runs can be compared on the same footing as newer outputs.

That normalization step now also backfills `run_provenance` from the paired `split_manifest` when it can find one, including older `/mnt/...` path strings from prior shells, so legacy family summaries keep split-source context instead of only raw stage metrics.

To compare two saved family artifacts directly, use `python scripts/compare_family_summaries.py --summary-a <path> --summary-b <path>`. The comparison output highlights common stages, holdout deltas, improved/regressed task ids, per-group task-group deltas, and blocker-aware effective-rate deltas when `--benchmark-blocker` is provided.

Those saved compare reports now also include `artifact_provenance` plus a `provenance_comparison` block, which helps flag cases where two artifacts differ because they came from different split sources, manifest paths, or alignment states rather than from the model changes alone.

They now also include `artifact_stage_meta` plus stage-meta key presence and comparison fields, which makes the targeted override set in a merged current-stack artifact visible directly in the saved diff.

That stage-meta output now also carries the merged stage's `base_summary` path and summarized copied base split source, so a saved compare report can show both the override set and the base artifact lineage in one place.

For one-sided meta keys such as `current_stack_meta`, the compare report now also writes `stage_meta_lineage_only_in_a` or `_in_b`, which is the quickest way to read the merged-stage lineage without inspecting the full nested stage-meta block.

When the compare command is given real artifact paths, that compact lineage block now also reports whether the merged stage's `base_summary` exactly matches the counterpart artifact path, which is the main sanity check for a current-stack refresh built on the intended base run.

That match check now normalizes Windows paths and older `/mnt/<drive>/...` paths before comparing them, so the trust signal does not flap just because two shells serialized the same artifact path differently.

If the stage names differ across the two artifacts, pass `--compare-stage <stage_in_a>=<stage_in_b>`. That is useful for comparisons like the base `warmup_plus_grpo` stage versus a merged `current_stack` stage in a re-eval artifact.

Add `--out <report.json>` to persist the comparison result into `outputs/` so the report can be reused in later notes or follow-up analysis without rerunning the comparison command.

The compare, merge, and normalize summary scripts now write their JSON outputs atomically. That prevents transient empty or truncated artifacts if a validator or another analysis step reads the file immediately after a refresh starts.

For the recurring exact-family maintenance workflow, prefer `python scripts/refresh_family_current_stack.py ...` over running merge, validate, and compare as separate commands. It refreshes the merged summary, validates it before writing, and emits the mapped compare report in one pass.

That refreshed summary now also keeps richer `<label>_meta` provenance for the merged stage, including the override count, override task ids, override metrics paths, copied base-run provenance, and the applied override metrics pulled from the targeted re-eval outputs.

For the current official Qwen exact-family stack specifically, use `scripts/local/refresh_qwen_shopping_exact_current_stack.sh` or `.ps1`. Those wrappers read the checked-in `scripts/local/qwen_shopping_exact_current_stack_manifest.json`, which keeps the current override set and output paths in one place instead of scattering them across shell history.

That helper now performs a preflight check before writing: every override metrics file must exist, and every override task id must already exist in the chosen base stage. If either check fails, it exits before touching the current-stack artifact and prints the planned override preview.

Before changing split sizes on a larger curriculum, use `python scripts/run_family_curriculum.py --family <family> --dry-run` first. The runner now validates split disjointness, checks that eval covers the full family, prints judge-free versus judge-gated partition counts, and enforces the default 20-task large-run training floor for families big enough to support it.

That dry-run now also reports `recommended_split_alignment`, which is useful when launching from `--split-manifest` or custom counts because it shows whether the current split still matches the code-defined recommended family split.

For fuzzy-judged families, that dry-run no longer needs `OPENAI_API_KEY` just to inspect the split. It prints `judge_requirements` and `launch_ready` so you can plan a `shopping_full` run before the judge key is wired in, while full runs still fail fast if the key is missing.

If you pass `--out-dir` during that dry-run, the runner now saves the preflight payload to `<out_dir>/preflight.json`. Use that when you want a checked result for the exact planned split instead of relying on terminal scrollback alone.

That saved preflight can now be fed back into `python scripts/run_family_curriculum.py --split-manifest <preflight.json> ...` so the eventual launch reuses the exact approved warmup, GRPO, and holdout partition instead of recomputing it from family defaults.

Once a real run starts, the saved `split_manifest.json` and `family_summary.json` now preserve that same provenance too: they include `split_provenance`, `judge_requirements`, and `recommended_split_alignment`, so finished artifacts still show whether the run came from a checked-in manifest, a saved preflight, the recommended split, or a custom count override.

The same pattern now covers `shopping_exact` through `scripts/local/shopping_exact_curriculum_manifest.json`, so the official Qwen and Liquid exact-family helpers also launch from a checked-in split manifest instead of implicit defaults.

For the broader `shopping_full` path, the preferred split manifest is now checked in at `scripts/local/shopping_full_curriculum_manifest.json`. The Qwen and Liquid launch helpers use that shared repo-owned manifest first, and the Qwen helper falls back to the saved `outputs/.../preflight.json` artifact only if the checked-in manifest is missing.

When the recommended broader-family split changes intentionally, refresh that checked-in manifest with `python scripts/refresh_family_split_manifest.py --family shopping_full --out scripts/local/shopping_full_curriculum_manifest.json --split-seed 42` or the local wrapper script. That keeps the portable launch manifest aligned with the curriculum code instead of drifting through manual edits.

Refresh the checked-in exact-family manifest the same way with `python scripts/refresh_family_split_manifest.py --family shopping_exact --out scripts/local/shopping_exact_curriculum_manifest.json --split-seed 42` or the paired local wrapper script.

To verify the checked-in exact/full/bootstrap/mixed manifests and their launcher wiring in one shot, run `python scripts/validate_split_manifests.py`. That checks all four portable split files against the current recommended family splits and confirms the local launcher scripts still reference the right manifest names.

The cross-site `bootstrap41` curriculum is the first non-shopping expansion path in the repo. It uses a checked-in `scripts/local/bootstrap41_curriculum_manifest.json`, launches through the paired Qwen/Liquid local helpers, and does not require `OPENAI_API_KEY` because all tasks in that family use scripted judges.

The mixed `web_mix88` curriculum is the next larger target after `bootstrap41`. It uses a checked-in `scripts/local/web_mix88_curriculum_manifest.json`, carries forward the fuzzy-judged shopping slice from `shopping_full`, and expands the training pool to `72` tasks while still keeping a disjoint `16`-task holdout.

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
- explicit holdout evaluation on `324`
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

### 9. Large-Family Curriculum Run

```bash
python scripts/run_family_curriculum.py --family shopping_exact --model-dir-name LFM2.5-350M
```

Large-family helper:

```bash
bash scripts/local/run_liquid_shopping_exact_curriculum.sh
```

This path is intended for the transition from proof-of-concept to broader training:

- total exact-match shopping tasks: `32`
- recommended warmup tasks: `14`
- recommended GRPO tasks: `12`
- recommended holdout tasks: `6`
- total training tasks: `26`

Official-target large-family helper:

```bash
bash scripts/local/run_qwen_shopping_exact_curriculum.sh
```

This writes to `../outputs/qwen_shopping_exact_curriculum_v1/` and now resumes cleanly if a long eval is interrupted.

### 10. Legacy Single-Task GRPO Run

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
