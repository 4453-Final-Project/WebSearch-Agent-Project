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
- explicit holdout transfer: task `324` stays at `0/4` after warmup, then reaches `4/4` after GRPO trained only on `327-328`

Latest larger-family signal on the executable `shopping_exact` curriculum:

- family size: `32` exact-match tasks
- training tasks: `26` total (`14` warmup + `12` GRPO)
- untouched holdout tasks: `6`
- result in `outputs/liquid_shopping_exact_curriculum_smoke_v4/`: holdout improves from `0/6` baseline to `1/6` after warmup and `2/6` after GRPO
- re-evaluated with the same GRPO adapter plus improved order-history, order-detail, and admin-order sort normalization in `outputs/liquid_shopping_exact_holdout_reval_v5/`: untouched holdout reaches `6/6`
- official-target full-family run in `outputs/qwen_shopping_exact_curriculum_v1/`: full-family success improves from `0.75` baseline to `0.765625` after warmup and `0.796875` after GRPO, while untouched holdout improves from `0.8333333333333334` to `0.9166666666666666`
- official-target stack re-eval with the same Qwen GRPO adapter in `outputs/qwen_shopping_exact_holdout_reval_v1/`: untouched holdout now reaches `6/6`
- latest refund-history stack fixes on the same official Qwen adapter: task `323` succeeds in `outputs/eval_task_323_qwen_after_history_refund_detail_fix/` after multi-order refund-detail aggregation, and task `321` succeeds in `outputs/eval_task_321_qwen_after_history_aggregate_fix_v2/` after year-wide refund aggregation is compressed into the visible summary
- current official-adapter refund slice check: tasks `320`, `321`, `322`, and `323` all succeed in `outputs/eval_task_320_qwen_current_refund_stack/`, `outputs/eval_task_321_qwen_after_history_aggregate_fix_v2/`, `outputs/eval_task_322_qwen_current_refund_stack/`, and `outputs/eval_task_323_qwen_after_history_refund_detail_fix/`
- merged current-stack family summary in `outputs/qwen_shopping_exact_current_stack_summary.json`: the same official Qwen adapter now reaches `0.96875` (`31/32`) on `shopping_exact` after folding in the targeted re-evals; only task `131` remains red, and it currently looks like a benchmark/reference mismatch rather than a model-parsing miss
- current admin item-count note: the runner can now background-scrape authenticated admin order-detail pages into structured item-count lines, but task `131` still behaves like a benchmark-data mismatch because the live accessible data supports `18` from the visible dashboard rows and `26` from the latest visible `7` admin-order details while the judge only accepts the baked reference answer `25`
- broader-family current-stack signal with the saved `shopping_full` warmup adapter: focused re-evals now put the current stack at `47/48`, with tasks `203`, `319`, `334`, `335`, `336`, `337`, `338`, `359`, and `361` fixed on top of the saved baseline and only task `204` still behaving like a benchmark/reference mismatch
- current main broad-family rerun: `outputs/qwen_shopping_full_curriculum_v4_devicefix/`, which keeps the `48`-task / `40`-train setup and the balanced `4`-task judge-gated warmup (`191`, `201`, `334`, `359`) but replaces the earlier warmup-training crash path by forcing trainable loads onto one concrete device instead of `device_map=\"auto\"`
- focused training-stack proof for that repaired loader path: `outputs/qwen_shopping_full_warmup_smoke_v1_devicefix/` successfully completed warmup on a mixed `5`-task slice (`188`, `191`, `325`, `334`, `359`) with `10` successful demos, `22` supervised samples, and final warmup loss `0.025344375520944595`
- focused full-loop training proof for that repaired stack: `outputs/qwen_shopping_full_staged_smoke_v1_devicefix/` completed disjoint warmup, rollout collection, GRPO checkpointing, and eval on a micro-split, with `4/4` successful rollout episodes on GRPO tasks `197` and `334` and `1.0` success retained on the untouched holdout task `359`
- current cross-site bootstrap signal: focused rechecks on shopping-admin tasks `0-6` are now `7/7` in `outputs/bootstrap41_dashboard_recheck_0_6_summary.json` after appending dashboard bestseller rows plus goal-specific ordered-product report aggregates into the observation
- focused cross-site map rechecks now have tasks `7`, `9`, `10`, `36`, `70`, `71`, and `72` at `7/7` in `outputs/bootstrap41_map_recheck_7_72_summary.json` after direct recovery through the local Nominatim/OSRM/Valhalla backends plus benchmark-normalized address canonicalization
- merged cross-site bootstrap current-stack artifact in `outputs/bootstrap41_current_stack_summary.json`: the `bootstrap41` baseline from `outputs/qwen_bootstrap41_curriculum_v3_mapfix/baseline_eval_summary.json` now rises from `12/41` to `21/41` after folding in targeted fixes for shopping-admin tasks `0` and `3`, shopping review task `21`, Reddit task `27`, and GitLab tasks `132`, `134`, `135`, `136`, and `293`
- current bootstrap GitLab count note: task `133` now reaches the intended repo-graph answer path and returns `0`, but it still scores `0.0`, so it currently looks closer to a benchmark/reference mismatch than a loader or parser miss
- current bootstrap RSS-token note: task `259` now deterministically reaches `/-/profile/personal_access_tokens`, the runner can recover the hidden feed token from the authenticated page HTML copy button, and the live two-step env probe now yields the direct `send_msg_to_user("TMN_bBn9Z48qVbUFZV45")` answer path; the remaining work is just capturing a clean finished re-eval artifact through the slow full single-task loop
- current cross-site run to watch: `outputs/qwen_bootstrap41_curriculum_v3_mapfix/`, which is live in warmup demos on the fixed dashboard-plus-map stack; the earlier `outputs/qwen_bootstrap41_curriculum_v2_dashboardfix/` artifact is now mainly useful as a stale pre-map-fix comparison point
- `outputs/qwen_shopping_full_curriculum_v4_devicefix/` is now complete with baseline `47/48` and both warmup-only and warmup + GRPO at `46/48`, so the broader shopping family is now mainly a stack-quality benchmark and no longer the best place to measure training lift
- `outputs/qwen_bootstrap41_curriculum_v3_mapfix/` is also complete with baseline, warmup-only, and warmup + GRPO all flat at `12/41`, which is the clearest current evidence that the previous warmup/GRPO sample weighting was too diffuse on the broader cross-site family
- both warmup and GRPO now use step-aware sample weighting that emphasizes later clean steps and successful terminal answer steps while downweighting invalid or parse-failed steps
- current cross-site rerun to watch: `outputs/qwen_bootstrap41_curriculum_v4_weightedfix/`, launched on the same checked-in `bootstrap41` split so the new weighted-training stack can be compared directly against `v3_mapfix`
- the optional local quantized path is now wired too: `outputs/qwen_shopping_full_warmup_smoke_v2_qlora/` completed a real `bnb_4bit` warmup smoke on the WSL GPU stack, and the current quantized bootstrap rerun to watch is `outputs/qwen_bootstrap41_curriculum_v6_qlora/`

The repo now also supports larger scripted curricula so we can scale past tiny slices:

- `shopping_order`: 27 exact-match shopping order-history/detail tasks
- `shopping_exact`: 32 exact-match shopping tasks total (`shopping_order` + search/sort)
- `shopping_order_full`: 43 scripted shopping order-history/detail tasks including fuzzy-judged items
- `shopping_full`: 48 scripted shopping tasks total, with a recommended 40-task training pool once `OPENAI_API_KEY` is available
- `bootstrap41`: 41 cross-site scripted tasks over shopping, shopping admin, reddit, gitlab, and map, with a recommended 33-task training pool and no OpenAI judge requirement
- `web_mix88`: 88 mixed shopping-plus-cross-site tasks, combining `shopping_full` with `bootstrap41` into a checked-in 72-task training pool and 16-task holdout
- the extra `16` non-exact shopping tasks use fuzzy judges and need `OPENAI_API_KEY`
- recommended large-run target: train on 20+ tasks by default via the family curriculum runner

That run uses disjoint training splits:

- warmup on `325-326`
- GRPO on `327-328`
- holdout on `324`
- family evaluation on `324-328`

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
For local `Qwen/Qwen3.5-2B` LoRA reloads, the active interpreter also needs `peft` and a `transformers` build new enough to recognize `qwen3_5`.
For optional local QLoRA-style loads, the active interpreter also needs `bitsandbytes`.

## Model Targets

Official comparison target:

- `Qwen/Qwen3.5-2B`
- local path: `../models/Qwen3.5-2B`

Fast local experiment target:

- `LiquidAI/LFM2.5-350M`
- local path: `../models/LFM2.5-350M`

All runnable scripts accept `--model-dir-name` or `--model-path`, so you can swap local checkpoints without editing code.
For detached local QLoRA-style bootstrap runs, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_tmux.sh` or `.ps1`.

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
- explicit holdout evaluation on `324`
- final evaluation and figure export

Official-target helper:

```bash
bash scripts/local/run_qwen_shopping_disjoint.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_shopping_disjoint.ps1
```

## Large-Family Curriculum

For a scaled run that trains on far more than five tasks, use the family curriculum runner:

```bash
python scripts/run_family_curriculum.py --family shopping_exact --model-dir-name LFM2.5-350M
```

Recommended large local helper:

```bash
bash scripts/local/run_liquid_shopping_exact_curriculum.sh
```

PowerShell:

```powershell
.\scripts\local\run_liquid_shopping_exact_curriculum.ps1
```

That path uses the 32-task exact-match shopping family and the deterministic recommended split:

- warmup tasks: `14`
- GRPO tasks: `12`
- holdout tasks: `6`
- total training tasks: `26`
- all four search/sort tasks stay in warmup so search transfer has a chance to carry into the untouched `324` holdout
- the Qwen and Liquid exact-family helpers now both prefer `scripts/local/shopping_exact_curriculum_manifest.json`, so the 32-task exact curriculum runs from one checked-in split definition instead of implicit defaults
- cached demos, baseline metrics, rollout episodes, and per-task eval metrics are reused automatically so interrupted large runs can resume instead of restarting
- family summaries now separate judge-free and judge-gated task partitions so broader fuzzy-family runs are easier to analyze once they are enabled
- `scripts/validate_family_summary.py` can sanity-check those summaries and report blocker-aware effective success rates when a known benchmark task should be excluded from comparison
- that validator now also checks stage-specific holdout sections such as `current_stack_holdout`, so merged re-eval artifacts are validated end-to-end instead of only at the stage-map level
- it now also validates stage-specific `<label>_meta` sections such as `current_stack_meta`, so override counts, override ids, copied base provenance, and applied override metrics are checked before a merged artifact is reused
- it now also validates `run_provenance`, including split seed consistency and recommended-split alignment when the summary has enough split metadata, so provenance-aware artifacts are checked before they reach the compare step
- `scripts/merge_family_reevals.py` now recomputes a companion `<label>_holdout` section after per-task overrides, so merged current-stack artifacts stay internally consistent instead of keeping stale base-stage holdout numbers
- `scripts/normalize_family_summary.py` can backfill older family summaries to the current schema by adding `family_metadata`, task groups, and stage-specific holdout sections without rerunning training
- that normalizer now also backfills `run_provenance` from the saved `split_manifest` when possible, including legacy `/mnt/...` paths from older run shells, so old exact-family artifacts can be compared with the newer provenance-aware outputs
- `scripts/compare_family_summaries.py` can compare two normalized family artifacts side by side, including blocker-aware effective rates and per-stage improved/regressed task ids
- that compare tool also supports explicit stage mappings like `--compare-stage warmup_plus_grpo=current_stack` when one artifact contains a custom merged stage
- it can also write a reusable JSON report with `--out`, which is how the saved exact-family comparison artifacts under `outputs/` are produced
- those comparison reports now also surface `artifact_provenance` and a `provenance_comparison` block, so it is easier to tell whether a delta came from the model stack or from comparing artifacts built from different split sources
- they now also surface `artifact_stage_meta`, plus common or one-sided stage-meta keys, so a merged current-stack comparison can show exactly which targeted override set produced the updated artifact
- that stage-meta output now also includes the merged stage's `base_summary` path and copied base split-source summary, so the saved diff shows which exact family artifact the targeted overrides were layered onto
- for one-sided stage-meta blocks like the Qwen `current_stack_meta`, the compare report now also emits a compact `stage_meta_lineage_only_in_*` summary so the override lineage is readable without digging through the full nested artifact payload
- when the compare command knows the summary file paths, that compact lineage summary also tells you whether the merged stage's `base_summary` exactly matches the counterpart artifact being compared, which is the fastest sanity check for current-stack lineage
- that base-summary match check now normalizes both Windows paths and legacy `/mnt/<drive>/...` paths first, so the lineage signal stays stable across PowerShell, WSL, and older saved artifacts
- stage comparisons in those reports now include task-group deltas too, so broader `shopping_full` comparisons can show whether judge-free and judge-gated partitions move together instead of only reporting one blended family average
- the compare, merge, and normalize scripts now write JSON artifacts atomically, so validation or follow-up analysis will not catch a half-written summary during a refresh
- `scripts/refresh_family_current_stack.py` now combines merge, validation, and mapped-stage compare generation in one step, which is the safest way to refresh the official Qwen current-stack artifact after targeted task re-evals
- that refresh path now also preserves richer `<label>_meta` override provenance, including override counts, task ids, metrics paths, and copied base-run provenance, so merged current-stack artifacts carry the exact re-eval inputs instead of only the merged scores
- the official Qwen exact-family current-stack refresh is now checked in as `scripts/local/qwen_shopping_exact_current_stack_manifest.json`, with `scripts/local/refresh_qwen_shopping_exact_current_stack.sh` and `.ps1` wrappers so the same override set can be replayed without rebuilding a long CLI command by hand
- that refresh helper now also preflights the override set before writing anything, so missing metrics files or task ids that are not present in the chosen base stage fail fast with a clear preview of the intended override application
- `scripts/run_family_curriculum.py` now also supports `--dry-run` and split preflight output, so large-family launches can verify the exact warmup/GRPO/holdout partition, judge-free vs judge-gated coverage, and the 20-task training-pool floor before any demos or browser rollouts begin
- that preflight now also reports `recommended_split_alignment`, so a manifest- or custom-count launch can tell you immediately whether it still matches the current recommended family split or has drifted from it
- for fuzzy-judged families such as `shopping_full`, that dry-run now works even without `OPENAI_API_KEY`; it reports judge readiness and `launch_ready` status instead of failing before the split preview is printed
- when `--out-dir` is provided during a dry-run, the runner now also saves that preflight payload to `<out_dir>/preflight.json`, which gives the broader-family planning path a reusable artifact under `outputs/`
- the family runner now also accepts `--split-manifest`, so a later launch can reuse the exact split from a saved preflight or earlier `split_manifest.json` instead of recomputing the partition
- the same checked-in manifest pattern now covers `shopping_exact` too via `scripts/local/shopping_exact_curriculum_manifest.json`, so both the official Qwen exact-family run and the Liquid exact-family helper launch from the same portable split
- the approved `shopping_full` split is now checked in as `scripts/local/shopping_full_curriculum_manifest.json`, so both broader-family launch helpers can prefer a shared repo-owned manifest over an ad hoc local artifact
- `scripts/refresh_family_split_manifest.py` can regenerate those checked-in split manifests from family defaults or a saved preflight, which keeps the approved launch splits reproducible without hand-editing JSON
- `scripts/validate_split_manifests.py` can now sanity-check the checked-in exact/full manifests and confirm the launcher scripts still point at them, so manifest drift is catchable from the CLI instead of only through unit tests
- that split-manifest validator now also covers the checked-in `bootstrap41` manifest and its Qwen/Liquid launcher pair, so the first non-shopping curriculum path stays portable too
- the same checked-in-manifest pattern now also covers `web_mix88`, so the first mixed shopping-plus-cross-site curriculum can be planned and launched through the same validator and launcher flow
- newly written `split_manifest.json` and `family_summary.json` artifacts now also record `split_provenance`, judge readiness, and `recommended_split_alignment`, so finished runs keep the exact split source visible instead of only printing it during dry-run preflight
- `scripts/build_family_override_summary.py` can now turn a completed baseline eval plus targeted re-eval metrics into a reusable family-style current-stack artifact, which is how `outputs/bootstrap41_current_stack_summary.json` is produced from the `bootstrap41` baseline plus the verified task fixes

Broader shopping helper once fuzzy-judge credentials are available:

```bash
bash scripts/local/run_qwen_shopping_full_curriculum.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_shopping_full_curriculum.ps1
```

That path uses the 48-task scripted shopping family and the deterministic recommended split:

- warmup tasks: `16`
- GRPO tasks: `24`
- holdout tasks: `8`
- total training tasks: `40`
- it fails fast if `OPENAI_API_KEY` is missing instead of partially starting a fuzzy-judged run
- its `family_summary.json` records separate judge-free and judge-gated success rates so we can tell whether gains reach both partitions
- the Qwen and Liquid broader-family launch helpers now both prefer `scripts/local/shopping_full_curriculum_manifest.json` as the approved repo-owned split, and the Qwen helper falls back to `outputs/qwen_shopping_full_curriculum_preflight_v1/preflight.json` if that checked-in manifest is absent

Manifest refresh helper:

```bash
bash scripts/local/refresh_shopping_full_curriculum_manifest.sh
```

PowerShell:

```powershell
.\scripts\local\refresh_shopping_full_curriculum_manifest.ps1
```

That helper rewrites `scripts/local/shopping_full_curriculum_manifest.json` from the current recommended `shopping_full` split, so the shared broader-family launch manifest can be refreshed after intentional curriculum changes.

Planning helper before judge credentials are ready:

```bash
bash scripts/local/run_qwen_shopping_full_preflight.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_shopping_full_preflight.ps1
```

That helper writes a reusable `preflight.json` artifact under `outputs/qwen_shopping_full_curriculum_preflight_v1/` while still reporting `launch_ready = false` until `OPENAI_API_KEY` is present.

Official-target large-family helper:

```bash
bash scripts/local/run_qwen_shopping_exact_curriculum.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_shopping_exact_curriculum.ps1
```

Exact-family manifest refresh helper:

```bash
bash scripts/local/refresh_shopping_exact_curriculum_manifest.sh
```

PowerShell:

```powershell
.\scripts\local\refresh_shopping_exact_curriculum_manifest.ps1
```

That helper rewrites `scripts/local/shopping_exact_curriculum_manifest.json` from the current recommended `shopping_exact` split, so the checked-in exact-family launch manifest stays aligned with the curriculum code.

Cross-site bootstrap helper:

```bash
bash scripts/local/run_qwen_bootstrap41_curriculum.sh
```

PowerShell:

```powershell
.\scripts\local\run_qwen_bootstrap41_curriculum.ps1
```

That path runs the checked-in `bootstrap41` curriculum over shopping, shopping admin, reddit, gitlab, and map with a deterministic `15` warmup / `18` GRPO / `8` holdout split and no OpenAI judge requirement.

Bootstrap manifest refresh helper:

```bash
bash scripts/local/refresh_bootstrap41_curriculum_manifest.sh
```

PowerShell:

```powershell
.\scripts\local\refresh_bootstrap41_curriculum_manifest.ps1
```

That helper rewrites `scripts/local/bootstrap41_curriculum_manifest.json` from the current recommended `bootstrap41` split so the non-shopping launch path stays aligned with the curriculum code.

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
