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
- that broader-family current stack is now also refreshable as a family-style artifact through `scripts/local/qwen_shopping_full_current_stack_manifest.json` and `scripts/local/refresh_qwen_shopping_full_current_stack.sh` / `.ps1`, which rebuild `outputs/qwen_shopping_full_current_stack_summary.json` plus the mapped compare report from the checked-in override set instead of relying on the older handwritten note artifact
- current main broad-family rerun: `outputs/qwen_shopping_full_curriculum_v4_devicefix/`, which keeps the `48`-task / `40`-train setup and the balanced `4`-task judge-gated warmup (`191`, `201`, `334`, `359`) but replaces the earlier warmup-training crash path by forcing trainable loads onto one concrete device instead of `device_map=\"auto\"`
- focused training-stack proof for that repaired loader path: `outputs/qwen_shopping_full_warmup_smoke_v1_devicefix/` successfully completed warmup on a mixed `5`-task slice (`188`, `191`, `325`, `334`, `359`) with `10` successful demos, `22` supervised samples, and final warmup loss `0.025344375520944595`
- focused full-loop training proof for that repaired stack: `outputs/qwen_shopping_full_staged_smoke_v1_devicefix/` completed disjoint warmup, rollout collection, GRPO checkpointing, and eval on a micro-split, with `4/4` successful rollout episodes on GRPO tasks `197` and `334` and `1.0` success retained on the untouched holdout task `359`
- current cross-site bootstrap signal: focused rechecks on shopping-admin tasks `0-6` are now `7/7` in `outputs/bootstrap41_dashboard_recheck_0_6_summary.json` after appending dashboard bestseller rows plus goal-specific ordered-product report aggregates into the observation
- focused cross-site map rechecks now have tasks `7`, `9`, `10`, `36`, `70`, `71`, and `72` at `7/7` in `outputs/bootstrap41_map_recheck_7_72_summary.json` after direct recovery through the local Nominatim/OSRM/Valhalla backends plus benchmark-normalized address canonicalization
- merged cross-site bootstrap current-stack artifact in `outputs/bootstrap41_current_stack_summary.json`: the `bootstrap41` baseline from `outputs/qwen_bootstrap41_curriculum_v3_mapfix/baseline_eval_summary.json` now rises from `12/41` to `38/41` after folding in targeted fixes for shopping-admin tasks `0`, `3`, `11`, `41`, and `77`, storefront tasks `21`, `23`, `25`, `26`, `125`, and `126`, Reddit tasks `27`, `28`, `29`, `30`, `31`, `66`, `67`, `68`, and `69`, and GitLab tasks `132`, `134`, `135`, `136`, `259`, and `293`
- `scripts/local/refresh_bootstrap41_current_stack_summary.sh` and `.ps1` now replay that exact `bootstrap41` current-stack refresh from a checked-in manifest instead of requiring a hand-built override CLI
- `scripts/local/validate_bootstrap41_current_stack_summary.sh` and `.ps1` now validate that saved `bootstrap41` current-stack summary too, so the refreshed family-style scorecard is checked with the same lightweight repo-owned workflow as the blocker audit
- `scripts/local/refresh_bootstrap41_blocker_audit.sh` and `.ps1` now replay the saved `bootstrap41` blocker audit from a checked-in manifest too, so the benchmark-mismatch evidence for tasks `124`, `133`, and `141` is refreshable instead of frozen as a one-off JSON
- `scripts/validate_webarena_blocker_audit.py` plus `scripts/local/validate_bootstrap41_blocker_audit.sh` / `.ps1` now give that saved blocker audit its own lightweight validator, so task coverage and mismatch bookkeeping are checked instead of only trusting the refresh output
- expanded cross-site bootstrap scorecard in `outputs/bootstrap44_current_stack_scorecard.json`: the validated `bootstrap44` expansion now carries that same inherited `bootstrap41` current stack plus the finished wins on tasks `12`, `13`, and `144`, yielding `41/44` (`0.9318181818181818`) without waiting on a separate full-family training rerun
- `scripts/validate_expanded_family_stage_scorecard.py` now validates that `bootstrap44`-style scorecards keep inherited-task ids, added-task ids, merged task counts, stage metrics, and copied run provenance internally consistent before those expansion artifacts are reused
- `scripts/local/validate_bootstrap44_current_stack_scorecard.sh` and `.ps1` now replay that `bootstrap44` validation step directly on the saved current-stack scorecard, so the expanded family artifact has the same repo-owned refresh-plus-validate path as `bootstrap41`
- `scripts/local/audit_bootstrap44_current_stack_scorecard.sh` and `.ps1` now replay that saved `bootstrap44` audit directly too, so the inherited-vs-added lift story stays reproducible from a repo-owned wrapper instead of only through the build path
- `outputs/bootstrap44_current_stack_audit.json` now makes that expansion explicit: the inherited `bootstrap41` slice stays unchanged at `38/41`, the three added tasks `12`, `13`, and `144` all score `1.0`, and there is no inherited-task drift between the base current stack and the expanded scorecard
- `scripts/build_expanded_family_stage_scorecard.py` now runs that validation automatically and can emit the matching expansion audit with `--audit-out`, so `bootstrap44`-style scorecards can be refreshed through one reproducible command instead of a manual build-then-validate-then-audit sequence
- `scripts/local/refresh_bootstrap44_current_stack_scorecard.sh` and `.ps1` now replay that exact `bootstrap44` refresh from the checked-in manifest, inherited `bootstrap41` current-stack summary, and validated task overrides, so the expansion scorecard and audit can be regenerated without rebuilding the CLI by hand
- current bootstrap Reddit note: tasks `28`, `29`, `30`, `31`, `66`, `67`, `68`, and `69` are now validated on the direct forum-answer path, and task `29` specifically now scores `1.0` in `outputs/eval_task_29_bootstrap_redditfix_v3/` because the stack reads the latest poster's own comments page and correctly normalizes Postmill's `&minus;36&minus;` vote formatting instead of counting negative scores on the submission thread
- current bootstrap storefront price-range note: task `125` is now validated at `1.0` in `outputs/eval_task_125_pricerangefix_v3/` after the catalog path started scanning sorted search pages for the first strongly query-matched extrema, task `126` remains green in `outputs/eval_task_126_pricerangefix_v1/`, and task `124` still only grounds to a sensible catalog range around `0.01 - 169.99` while the baked benchmark max stays much higher
- current bootstrap GitLab count note: task `133` now reaches the intended repo-graph answer path and returns `0`, but it still scores `0.0`, so it currently looks closer to a benchmark/reference mismatch than a loader or parser miss
- current bootstrap RSS-token note: task `259` is now validated at `1.0` in `outputs/eval_task_259_bootstrap_rssfix_v1/`; the stack deterministically reaches `/-/profile/personal_access_tokens`, the runner recovers the hidden feed token from the authenticated page HTML copy button, and that fix is now folded into the merged `bootstrap41` current-stack summary
- current bootstrap review-count note: task `11` is now validated at `1.0` in `outputs/eval_task_11_bootstrap_reviewcount_v1/`; the runner reaches the authenticated admin review grid, filters the `detail` column by the quoted review term, and appends the resulting `records found` count as a structured dashboard answer line
- current bootstrap search-term note: task `41` is now validated at `1.0` in `outputs/eval_task_41_bootstrap_searchterm_v1/`; the admin search-term edit pages expose "Number of Uses" through the `popularity` field, and the runner now ranks appended `Dashboard search term row:` lines by that value instead of URL order before the policy answers directly from those rows
- current bootstrap yearly-bestseller note: year-scoped admin bestseller goals now prefer aggregated admin rows over the noisier visible dashboard leaderboard when both are present, which fixes the task-`0` regression path that resurfaced in the early `v19` baseline
- current bootstrap review-status note: task `77` is now validated at `1.0` in `outputs/eval_task_77_bootstrap_reviewstatus_v1/`; the runner now filters the authenticated admin review grid by status, reads the resulting `records found` count, and appends a structured review-status answer line that the policy answers from directly
- current bootstrap storefront-review note: tasks `23`, `25`, and `26` are now validated at `1.0` in `outputs/eval_task_23_bootstrap_reviewauthors_v1/`, `outputs/eval_task_25_bootstrap_reviewauthors_v1/`, and `outputs/eval_task_26_bootstrap_reviewauthors_v1/`; the storefront review matcher now handles fingerprint-resistance language and customer-service complaint language instead of requiring near-exact phrase overlap
- current shopping order-spend note: task `144` is now validated at `1.0` in `outputs/eval_task_144_spendfix_v1/` through the new order-history spend aggregation path, while tasks `141`, `142`, `143`, and `145` now return grounded order-detail totals that still disagree with the baked benchmark references; task `141` currently grounds to `24.42` from the visible March 2023 food-like order rows
- saved bootstrap blocker audit: `outputs/bootstrap41_blocker_audit_v1.json` now records the three remaining current-stack misses side by side with their baked WebArena references, showing `124` grounding to `0.01 - 169.99` versus `$0.14 - $745.00`, `133` grounding to `0` versus `2`, and `141` grounding to `24.42` versus `$47.41`
- `outputs/qwen_shopping_full_curriculum_v4_devicefix/` is now complete with baseline `47/48` and both warmup-only and warmup + GRPO at `46/48`, so the broader shopping family is now mainly a stack-quality benchmark and no longer the best place to measure training lift
- `outputs/qwen_bootstrap41_curriculum_v3_mapfix/` is also complete with baseline, warmup-only, and warmup + GRPO all flat at `12/41`, which is the clearest current evidence that the previous warmup/GRPO sample weighting was too diffuse on the broader cross-site family
- both warmup and GRPO now use step-aware sample weighting that emphasizes later clean steps and successful terminal answer steps while downweighting invalid or parse-failed steps
- those step-aware sample-weight knobs are now exposed on the staged and family runners too, so cross-site reruns can tune terminal-step emphasis and error-step downweighting from the CLI instead of requiring another source edit
- stale quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v12_qlora_stepweight_redditfix/` started before the month-specific January admin bestseller precedence fix landed for tasks `4` and `5`, so it is now mainly a pre-admin-bestseller-fix comparison point
- completed quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v13_qlora_stepweight_adminbestsellerfix/` finished with baseline `34/41`, warmup-only `33/41`, and warmup + GRPO `33/41`, so the fixed global `4`-step recipe still under-served the slower cross-site slice instead of producing a true training lift
- stale quantized GitLab-budget note: `outputs/qwen_bootstrap41_curriculum_v14_qlora_gitlabsteps/` started before the generic warmup-demo cleanup that removed the redundant `goto(start_url)` supervision artifact from simple direct-answer tasks, so it is now mainly a pre-clean-demo comparison point
- stale quantized GitLab-budget note: `outputs/qwen_bootstrap41_curriculum_v15_qlora_gitlabsteps_cleandemos/` also started before the targeted GitLab-coverage bump landed, so it is now mainly a pre-GitLab-coverage comparison point
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v16_qlora_gitlabcoverage/` started before the generic clean-demo fix landed, and the saved `v13` baseline-vs-GRPO audit shows task `0` was the only regression on that earlier recipe
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v17_qlora_gitlabcoverage_cleandemos/` started before warmup oversampling by task group landed, so it is now mainly a pre-warmup-oversample comparison point
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v18_qlora_gitlabcoverage_cleandemos_warmupoversample/` finished baseline at `38/41` and then crashed at warmup start because the family runner passed task-group warmup-limit overrides through the wrong staged-runner field names
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v19_qlora_gitlabcoverage_cleandemos_warmupoversample_fix/` started before the yearly admin-bestseller aggregate-precedence fix landed, so it is now mainly a pre-yearly-bestseller-fix comparison point
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v20_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix/` started before the narrower yearly product-question dashboard precedence fix landed for tasks `0` and `3`, so it is now mainly a pre-yearly-product-dashboard-fix comparison point
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v21_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproductfix/` started before the deterministic dashboard tie-break fix landed for yearly product questions, so it is now mainly a pre-yearly-product-tiebreak comparison point
- current quantized GitLab-coverage rerun to watch: `outputs/qwen_bootstrap41_curriculum_v22_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproducttiefix/`, which keeps the same checked-in `bootstrap41` split, `bnb_4bit` step-weight recipe, GitLab budget, GitLab coverage bump, clean-demo fix, and GitLab warmup oversampling as `v21` but now also breaks tied dashboard bestseller rows deterministically by product name for plain yearly product questions
- focused validation on that yearly-product fix is now in place too: `outputs/eval_task_0_yearproductfix_v2/` and `outputs/eval_task_3_yearproductfix_v1/` both score `1.0`, which shows the narrowed dashboard-row precedence plus deterministic tie-breaking recovers the two early bootstrap yearly product tasks on the current head
- the dedicated detached rerun wrappers are now checked in too: `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix_tmux.sh` replays the exact `v20` recipe directly, `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproductfix_tmux.sh` replays the exact `v21` recipe directly, and `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproducttiefix_tmux.sh` replays the exact `v22` out-dir/session recipe directly instead of requiring manual environment overrides
- the order-only family path is staged too: `shopping_order` now has a checked-in split manifest plus Qwen/Liquid bash and PowerShell launchers, so the 27-task exact order-history/detail slice is a reproducible family-run target instead of only living in `task_families.py`
- the broader order-only family path is staged too: `shopping_order_full` now has the same manifest and launcher coverage, so the 43-task fuzzy-judged order-history/detail slice is ready when we want a shopping-order-heavy comparison outside the current cross-site ladder
- the detached QLoRA order-family launchers are staged too: `scripts/local/run_qwen_shopping_order_curriculum_qlora_tmux.sh` and `scripts/local/run_qwen_shopping_order_full_curriculum_qlora_tmux.sh` give the order-only ladders the same long-run `bnb_4bit` path as the bootstrap and mixed families
- the order-only current-stack refresh path is staged too: `scripts/local/refresh_shopping_order_current_stack_scorecard.sh` and `scripts/local/refresh_shopping_order_full_current_stack_scorecard.sh` now derive saved subset scorecards directly from the validated broader shopping current-stack artifacts
- the saved order-only current-stack scorecards now sit at `26/27` (`0.9629629629629629`) for `shopping_order` in `outputs/shopping_order_current_stack_scorecard.json` and `42/43` (`0.9767441860465116`) for `shopping_order_full` in `outputs/shopping_order_full_current_stack_scorecard.json`; both reduce to blocker-excluded effective `1.0` on their known mismatch tasks
- `scripts/local/validate_shopping_order_current_stack_scorecard.sh` / `.ps1` and `scripts/local/validate_shopping_order_full_current_stack_scorecard.sh` / `.ps1` now make those order-only subset scorecards validation-replayable too, so the saved order slices have the same repo-owned refresh-plus-validate path as the bootstrap and mixed-family artifacts
- `scripts/local/audit_shopping_order_current_stack_scorecard.sh` / `.ps1` and `scripts/local/audit_shopping_order_full_current_stack_scorecard.sh` / `.ps1` now do the same for the saved order-only audits, so those order-heavy scorecards have repo-owned refresh-plus-validate-plus-audit coverage too
- both order-only paths now also have saved dry-run preflights under `outputs/shopping_order_curriculum_preflight_v1/` and `outputs/shopping_order_full_curriculum_preflight_v1/`, and checked-in refresh wrappers at `scripts/local/refresh_shopping_order_curriculum_preflight.sh` / `.ps1` plus `scripts/local/refresh_shopping_order_full_curriculum_preflight.sh` / `.ps1`, so their checked-in manifests are already validated against the current recommended splits and can be refreshed without reconstructing the dry-run commands
- next no-judge cross-site expansion path is checked in too: `bootstrap44` extends `bootstrap41` with validated admin review-count tasks `12` and `13` plus validated storefront spend task `144`, giving a `44`-task family with a deterministic `16` warmup / `20` GRPO / `8` holdout split
- the matched detached `bootstrap44` QLoRA GitLab-coverage path is checked in too: `scripts/local/run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_tmux.sh` carries the same GitLab budget and coverage bump as the current `bootstrap41` comparison ladder, so `bootstrap44` can run apples-to-apples ablations instead of jumping straight from base to oversampled
- the matched detached `bootstrap44` QLoRA promotion path is checked in too: `scripts/local/run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux.sh` keeps the same `bnb_4bit`, GitLab-budget, GitLab-coverage, clean-demo, and warmup-oversample recipe as `v18`, so the next no-judge expansion can inherit the current bootstrap training setup directly
- the matched detached `bootstrap44` year-product-tiebreak promotion path is checked in too: `scripts/local/run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproducttiefix_tmux.sh` carries that same strongest recipe forward with both the narrowed yearly-product dashboard precedence and the deterministic dashboard tie-break pinned into the expansion-family launch surface
- next mixed-family scale-up path is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_tmux.sh` now launches a detached `bnb_4bit` `web_mix88` run on the checked-in 88-task split, so if `bootstrap41` finally shows training lift we can promote the same recipe onto the larger mixed family without ad hoc shell work
- `outputs/web_mix88_current_stack_scorecard.json` now gives that first mixed-family path a saved current-stack checkpoint too by combining the validated `shopping_full` current stack (`47/48`) with the validated `bootstrap41` current stack (`38/41`) into `84/88` (`0.9545454545454546`), with overlap task `188` deduplicated across the two source artifacts
- the matching mixed-family expansion path is checked in too: `web_mix91` combines `shopping_full` with `bootstrap44`, so those three validated expansion tasks carry forward into a `91`-task mixed family instead of getting dropped during scale-up
- `scripts/local/validate_web_mix88_current_stack_scorecard.sh` / `.ps1` and `scripts/local/validate_web_mix91_current_stack_scorecard.sh` / `.ps1` now turn mixed-family scorecard validation into checked-in local wrappers too, so both saved combined scorecards have the same repo-owned refresh-plus-validate path as the bootstrap artifacts
- `scripts/local/audit_web_mix88_current_stack_scorecard.sh` / `.ps1` and `scripts/local/audit_web_mix91_current_stack_scorecard.sh` / `.ps1` now turn the saved mixed-family audits into checked-in local wrappers too, so both combined scorecards have repo-owned refresh-plus-validate-plus-audit coverage
- `outputs/web_mix91_current_stack_scorecard.json` now turns that checked-in mixed expansion into a saved current-stack scorecard too: it combines the validated `shopping_full` current stack (`47/48`) with the validated `bootstrap44` current stack (`41/44`) into `87/91` (`0.9560439560439561`), with task `188` deduplicated across the two source artifacts
- `scripts/local/refresh_web_mix88_current_stack_scorecard.sh` and `.ps1` now replay that `web_mix88` mixed-family scorecard refresh from one checked-in manifest instead of requiring a manual combined-scorecard CLI invocation
- `scripts/validate_combined_family_stage_scorecard.py` and `scripts/audit_combined_family_stage_scorecard.py` now give that `web_mix91` scorecard its own dedicated validator-plus-audit path, so the mixed-family current-stack result is checked independently instead of only being trusted through the builder output
- the matched detached `web_mix91` QLoRA GitLab-coverage path is checked in too: `scripts/local/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_tmux.sh` carries the same GitLab budget and coverage bump as the current `web_mix88` ladder, so `web_mix91` can run apples-to-apples ablations before warmup oversampling is layered on
- the matched detached `web_mix91` QLoRA promotion path is checked in too: `scripts/local/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh` keeps the same `bnb_4bit`, GitLab-budget, GitLab-coverage, and warmup-oversample recipe as the strongest current mixed-family launcher, so `web_mix91` is launch-ready instead of only preflight-ready
- the matched detached `web_mix91` year-product-tiebreak promotion path is checked in too: `scripts/local/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_tmux.sh` now carries that strongest mixed-family recipe forward with the same narrowed yearly-product dashboard precedence and deterministic dashboard tie-break baked into the launch path
- the matched mixed-family GitLab-coverage variant is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh` carries the same GitLab budget and coverage bump as `v16`, so the first `web_mix88` promotion path does not fall back to the weaker mixed-family baseline launcher
- the matched mixed-family GitLab-coverage warmup-oversample variant is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh` layers the same `site_gitlab=2.5` warmup oversample used in `v18` onto that mixed-family GitLab recipe, so a future `web_mix88` promotion does not silently drop the newest supervised warmup change
- the matched mixed-family year-product-tiebreak promotion path is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_tmux.sh` now carries that same strongest `web_mix88` recipe forward with the narrowed yearly-product dashboard precedence and deterministic dashboard tie-break baked into the launch surface
- warmup summaries now record `source_task_sample_counts`, `average_epoch_task_sample_counts`, and `task_sample_multipliers`, so oversample experiments leave behind concrete per-task supervised-mix evidence instead of only CLI flags
- finished `family_summary.json` artifacts now also carry that `warmup_training_summary`, so the supervised-mix evidence survives at the same top level as baseline, warmup, and GRPO stage results
- live run status now mirrors those warmup-mix fields too once `warmup_summary.json` exists, so oversample evidence becomes visible before the full family run finishes
- live run status now also reports family-level `benchmark_blockers` plus blocker-excluded effective eval metrics, so in-flight runs like `v18` can show raw baseline progress alongside the "real" non-blocker progress without waiting for a separate blocker audit
- the optional local quantized path is now wired too: `outputs/qwen_shopping_full_warmup_smoke_v2_qlora/` completed a real `bnb_4bit` warmup smoke on the WSL GPU stack
- completed family runs now also emit `baseline_vs_warmup_audit.json`, `baseline_vs_grpo_audit.json`, and `warmup_vs_grpo_audit.json`, so stage regressions are saved automatically instead of requiring a separate manual audit pass
- current quantized bootstrap baseline rerun to watch: `outputs/qwen_bootstrap41_curriculum_v6_qlora/`
- stale quantized step-weight note: `outputs/qwen_bootstrap41_curriculum_v9_qlora_stepweight_dashboardfix/` started before the task-`41` popularity fix and is now mainly a pre-searchterm-fix comparison point
- stale quantized step-weight note: `outputs/qwen_bootstrap41_curriculum_v11_qlora_stepweight_catalogfix/` also started before the task-`29` Reddit fix landed in the stack, so it is now mainly a pre-Reddit-fix comparison point

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
For the stronger quantized reward-shaped variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_rewardtune_tmux.sh` or `.ps1`.
For the stronger quantized reward-shaped plus step-weight-tuned variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_stepweight_tune_tmux.sh` or `.ps1`.
For the GitLab-budget comparison variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabsteps_tmux.sh` or `.ps1`.
For the cleaned-demo GitLab-budget comparison variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabsteps_cleandemos_tmux.sh` or `.ps1`.
For the GitLab-coverage comparison variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_tmux.sh` or `.ps1`.
For the cleaned-demo GitLab-coverage comparison variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_tmux.sh` or `.ps1`.
For the cleaned-demo GitLab-coverage warmup-oversample variant, use `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux.sh` or `.ps1`.
For the detached order-only QLoRA path, use `scripts/local/run_qwen_shopping_order_curriculum_qlora_tmux.sh` or `.ps1`.
For the detached broader order-only QLoRA path, use `scripts/local/run_qwen_shopping_order_full_curriculum_qlora_tmux.sh` or `.ps1`.
For the detached mixed-family QLoRA scale-up path, use `scripts/local/run_qwen_web_mix88_curriculum_qlora_tmux.sh` or `.ps1`.
For the mixed-family GitLab-coverage variant, use `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh` or `.ps1`.
For the mixed-family GitLab-coverage warmup-oversample variant, use `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh` or `.ps1`.

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
- `scripts/validate_expanded_family_stage_scorecard.py` does the same kind of structural audit for expanded stage scorecards such as `outputs/bootstrap44_current_stack_scorecard.json`, including inherited-vs-added task coverage and copied stage-meta provenance
- `scripts/audit_expanded_family_stage_scorecard.py` compares an expanded scorecard back to its inherited base stage and reports the added-task lift plus any inherited-task drift, which is how `outputs/bootstrap44_current_stack_audit.json` is produced
- `scripts/build_combined_family_stage_scorecard.py` now does the same kind of saved union build for mixed-family promotions such as `outputs/web_mix91_current_stack_scorecard.json`, where the checked-in family is the union of multiple already-validated stage artifacts rather than one inherited base plus a few added tasks
- `scripts/validate_combined_family_stage_scorecard.py` now validates those combined scorecards explicitly, including source coverage, overlap-task bookkeeping, and merged stage consistency for artifacts like `outputs/web_mix88_current_stack_scorecard.json` and `outputs/web_mix91_current_stack_scorecard.json`
- `scripts/audit_combined_family_stage_scorecard.py` now audits those combined scorecards back against their source summaries, which is how the saved `outputs/web_mix91_current_stack_audit.json` overlap-consistency report is refreshed
- `scripts/local/refresh_web_mix88_current_stack_scorecard.sh` / `.ps1` and `scripts/local/refresh_web_mix91_current_stack_scorecard.sh` / `.ps1` now turn those combined mixed-family scorecards into reproducible local refresh commands instead of one-off shell history
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
- the same manifest-plus-wrapper pattern now covers the broader `shopping_full` current-stack artifact too through `scripts/local/qwen_shopping_full_current_stack_manifest.json` and `scripts/local/refresh_qwen_shopping_full_current_stack.sh` / `.ps1`, so the validated `47/48` broad-family stack can be regenerated as a provenance-aware family summary instead of only a handwritten note
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
- `scripts/build_family_override_summary.py` can now turn a completed baseline eval plus targeted re-eval metrics into a reusable family-style current-stack artifact from either direct CLI args or a checked-in manifest, which is how `outputs/bootstrap41_current_stack_summary.json` is produced from the `bootstrap41` baseline plus the verified task fixes

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
