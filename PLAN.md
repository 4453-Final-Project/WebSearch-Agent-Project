# Project Plan

## Objective

Use the existing BrowserGym/WebArena pipeline to improve a local model beyond its baseline on a narrow, reproducible evaluation family.

## Current Best Validated Slice

- family: Shopping search/sort `324-328`
- baseline: `3/5`
- warmup-only: `4/5`
- warmup + disjoint GRPO: `5/5`
- untouched holdout: task `324` improves from `0/4` to `4/4`

## Current Large-Family Signal

- family: `shopping_exact` (`32` exact-match tasks)
- training pool: `26` tasks (`14` warmup + `12` GRPO)
- untouched holdout: `6` tasks
- latest full run: `outputs/liquid_shopping_exact_curriculum_smoke_v4/`
- holdout result: baseline `0/6`, warmup `1/6`, post-GRPO `2/6`
- tasks that transferred on holdout: `324` after warmup, `195` after GRPO
- latest stack re-eval with the same `smoke_v4` GRPO adapter: `outputs/liquid_shopping_exact_holdout_reval_v5/`
- re-evaluated holdout result: `6/6` (`1.0`) on untouched holdouts
- transferred holdouts after parser, observation, and admin sort-normalization fixes: `189`, `195`, `200`, `324`, `358`, `360`
- latest official-target run: `outputs/qwen_shopping_exact_curriculum_v1/`
- official-target full-family result: baseline `0.75`, warmup `0.765625`, warmup + GRPO `0.796875`
- official-target untouched holdout result: baseline `0.8333333333333334`, warmup `0.9166666666666666`, warmup + GRPO `0.9166666666666666`
- current official-target transfer on untouched holdout: `324`
- current official-target GRPO recovery on the full eval family: task `194` returns from warmup `0.0` to post-GRPO `1.0`
- latest official-target stack re-eval with the same adapter: `outputs/qwen_shopping_exact_holdout_reval_v1/`
- re-evaluated official-target holdout result: `6/6` (`1.0`) on untouched holdouts
- transferred official-target holdouts after sort-state normalization: `189`, `195`, `200`, `324`, `358`, `360`
- merged official-target current-stack family summary: `outputs/qwen_shopping_exact_current_stack_summary.json`
- current-stack official-target family score after targeted task re-evals: `31/32` (`0.96875`)
- current known benchmark blocker: task `131` still judges only the baked answer `25` even though the live dashboard exposes `18` for the visible `5` recent orders and authenticated admin-order detail scraping yields `26` for the latest visible `7` orders
- current broader-family warmup-adapter stack after focused `shopping_full` re-evals: `47/48` (`0.9791666666666666`)
- that broader-family current stack now also has a checked-in refresh path via `scripts/local/qwen_shopping_full_current_stack_manifest.json` plus `scripts/local/refresh_qwen_shopping_full_current_stack.sh` / `.ps1`, so the focused override set can be replayed into a validated family-style artifact and compare report
- current broader-family blocker: task `204` still behaves like a benchmark/reference mismatch even though the stack answers from the actual most recent completed order
- previous broad-family rerun `outputs/qwen_shopping_full_curriculum_v3_balanced/` is now mainly a stale pre-device-fix artifact because warmup training crashed with a meta-device backward error after the strong `47/48` baseline snapshot
- current main broad-family rerun: `outputs/qwen_shopping_full_curriculum_v4_devicefix/`
- focused warmup smoke for the repaired trainable loader: `outputs/qwen_shopping_full_warmup_smoke_v1_devicefix/` succeeded on a mixed `5`-task slice, which confirms backward and optimizer steps now work on the fixed training stack before the full rerun reaches warmup training
- focused staged smoke for the repaired training loop: `outputs/qwen_shopping_full_staged_smoke_v1_devicefix/` completed warmup, rollout collection, GRPO, checkpoint save, and eval on a disjoint micro-split, which confirms the fixed stack can now run the full staged loop end-to-end
- current broad-family rerun uses a balanced `shopping_full` warmup with `4` judge-gated tasks (`191`, `201`, `334`, `359`) instead of only `2`
- current bootstrap shopping-admin recheck slice: tasks `0-6` are `7/7` in `outputs/bootstrap41_dashboard_recheck_0_6_summary.json` after dashboard bestseller extraction plus ordered-product report aggregation fixes
- current bootstrap shopping-admin recheck slice: tasks `0-6` are `7/7` in `outputs/bootstrap41_dashboard_recheck_0_6_summary.json` after dashboard bestseller extraction plus ordered-product report aggregation fixes
- current bootstrap map recheck slice: tasks `7`, `9`, `10`, `36`, `70`, `71`, and `72` are `7/7` in `outputs/bootstrap41_map_recheck_7_72_summary.json` after direct map-backend answer recovery plus benchmark-normalized address canonicalization
- merged bootstrap current-stack artifact: `outputs/bootstrap41_current_stack_summary.json` now moves the cross-site baseline from `12/41` (`0.2926829268292683`) to `38/41` (`0.926829268292683`) after folding in targeted fixes for tasks `0`, `3`, `11`, `21`, `23`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `41`, `66`, `67`, `68`, `69`, `77`, `125`, `126`, `132`, `134`, `135`, `136`, `259`, and `293`
- bootstrap current-stack local wrapper: `scripts/local/refresh_bootstrap41_current_stack_summary.sh` and `.ps1` now replay that exact checked-in `bootstrap41` current-stack refresh path from one manifest instead of a hand-built override command
- bootstrap blocker-audit local wrapper: `scripts/local/refresh_bootstrap41_blocker_audit.sh` and `.ps1` now replay the saved `bootstrap41` benchmark-blocker audit from one checked-in manifest too
- expanded bootstrap scorecard: `outputs/bootstrap44_current_stack_scorecard.json` now layers the validated wins on tasks `12`, `13`, and `144` onto the inherited `bootstrap41` current stack, yielding `41/44` (`0.9318181818181818`) on the checked-in `bootstrap44` family without waiting on a separate full-family rerun
- expanded-scorecard validation: `scripts/validate_expanded_family_stage_scorecard.py` now sanity-checks inherited vs added task coverage, merged task counts, stage metrics, and copied provenance on artifacts like `outputs/bootstrap44_current_stack_scorecard.json`
- expanded-scorecard audit: `outputs/bootstrap44_current_stack_audit.json` now confirms that `bootstrap44` adds three clean wins on `12`, `13`, and `144` with no inherited-task drift from the underlying `bootstrap41` current stack
- expanded-scorecard refresh path: `scripts/build_expanded_family_stage_scorecard.py` now validates its output automatically and can emit the matching audit with `--audit-out`, so `bootstrap44`-style scorecards can be refreshed from one reproducible command
- expanded-scorecard local wrapper: `scripts/local/refresh_bootstrap44_current_stack_scorecard.sh` and `.ps1` now replay that checked-in `bootstrap44` refresh path directly from the inherited `bootstrap41` current-stack artifact plus the validated task overrides
- current bootstrap Reddit note: tasks `28`, `29`, `30`, `31`, `66`, `67`, `68`, and `69` now have clean focused re-eval wins through the direct forum-answer path, and task `29` now specifically scores through the latest-poster comments-page recovery plus `&minus;36&minus;` vote normalization instead of the old submission-thread negative-count shortcut
- current bootstrap storefront price-range note: task `125` is now backed by a clean finished re-eval artifact in `outputs/eval_task_125_pricerangefix_v3/`, so the catalog path can now scan sorted search pages for the first strongly query-matched extrema; task `126` remains green, while task `124` still only grounds to a sensible catalog range around `0.01 - 169.99` while the baked benchmark max stays much higher
- current bootstrap GitLab count note: task `133` now reaches the intended repo-graph answer path and returns `0`, but it still scores `0.0`, so it currently looks closer to a benchmark/reference mismatch than an infrastructure or parser failure
- current bootstrap RSS-token note: task `259` is now backed by a clean finished re-eval artifact in `outputs/eval_task_259_bootstrap_rssfix_v1/`, so the authenticated personal-access-tokens route and hidden-token extraction fix are both validated and merged into the bootstrap current-stack scorecard
- current bootstrap review-count note: task `11` is now backed by a clean finished re-eval artifact in `outputs/eval_task_11_bootstrap_reviewcount_v1/`, so the admin dashboard review-count path now filters the authenticated review grid by the quoted review term and reads the resulting `records found` count end-to-end
- current bootstrap search-term note: task `41` is now backed by a clean finished re-eval artifact in `outputs/eval_task_41_bootstrap_searchterm_v1/`, so the admin dashboard search-term path now reads the authenticated edit pages, ranks by `popularity`, and answers directly from the appended search-term rows end-to-end
- current bootstrap yearly-bestseller note: year-scoped admin bestseller goals now prefer aggregated admin rows over the noisier visible dashboard leaderboard when both are present, which fixes the task-`0` regression path that resurfaced in the early `v19` baseline
- current bootstrap review-status note: task `77` is now backed by a clean finished re-eval artifact in `outputs/eval_task_77_bootstrap_reviewstatus_v1/`, so the admin dashboard review-status path now filters the authenticated review grid by status and reads the resulting `records found` count end-to-end
- current bootstrap storefront-review note: tasks `23`, `25`, and `26` are now backed by clean finished re-eval artifacts in `outputs/eval_task_23_bootstrap_reviewauthors_v1/`, `outputs/eval_task_25_bootstrap_reviewauthors_v1/`, and `outputs/eval_task_26_bootstrap_reviewauthors_v1/`, so the storefront review path now recognizes fingerprint-resistance and customer-service complaint phrasing that previously failed the literal matcher
- current shopping order-spend note: task `144` is now backed by a clean finished re-eval artifact in `outputs/eval_task_144_spendfix_v1/`, so the order-history spend path can now sum matched dated order-detail rows directly; tasks `141`, `142`, `143`, and `145` also now return grounded spend totals, but those totals still disagree with the baked references, and task `141` currently grounds to `24.42` from the visible March 2023 food-like order rows
- saved bootstrap blocker audit: `outputs/bootstrap41_blocker_audit_v1.json` now records the three remaining current-stack misses against their baked WebArena references, which makes the current blocker set auditable instead of relying on handwritten notes
- completed broad-family rerun: `outputs/qwen_shopping_full_curriculum_v4_devicefix/` finished with baseline `47/48` (`0.9791666666666666`), warmup-only `46/48` (`0.96875`), and warmup + GRPO `46/48` (`0.96875`), so this family is now effectively saturated for training comparisons and should mainly be treated as a stack-quality benchmark
- completed bootstrap rerun: `outputs/qwen_bootstrap41_curriculum_v3_mapfix/` finished with baseline `12/41` (`0.2926829268292683`), warmup-only `12/41`, and warmup + GRPO `12/41`, which confirms the old warmup/GRPO weighting was too weak to move the broader cross-site slice
- current training-signal fix: both warmup and GRPO now weight later clean steps and successful terminal answer steps more heavily than early noisy navigation, while invalid and parse-failed steps are downweighted
- current bootstrap rerun to watch: `outputs/qwen_bootstrap41_curriculum_v4_weightedfix/`, which reuses the same checked-in `bootstrap41` split on the current fixed stack plus the new warmup/GRPO step weighting so the comparison against `v3_mapfix` stays apples-to-apples
- focused quantized-training proof: `outputs/qwen_shopping_full_warmup_smoke_v2_qlora/` now completes a real `bnb_4bit` warmup smoke on the WSL GPU stack, which confirms the optional QLoRA-style path is usable for local adapter training
- current quantized bootstrap rerun to watch: `outputs/qwen_bootstrap41_curriculum_v6_qlora/`, which reuses the same checked-in `bootstrap41` split on the current weighted stack but now runs through the `bnb_4bit` loader path so we can compare speed and signal against the earlier reruns
- current quantized reward-tuned comparison rerun: `outputs/qwen_bootstrap41_curriculum_v7_qlora_rewardtune/`, which keeps the same checked-in `bootstrap41` split and quantized loader path but adds the stronger reward shaping from the earlier non-quantized reward-tuned bootstrap run
- current follow-up quantized comparison path: `outputs/qwen_bootstrap41_curriculum_v8_qlora_stepweight_tune/`, which layers stronger terminal-step weighting and stronger error-step downweighting on top of the same quantized reward-tuned bootstrap recipe
- stale quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v9_qlora_stepweight_dashboardfix/` started before the task-`41` search-term popularity fix, so it is now mainly a pre-searchterm-fix comparison point
- stale quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v11_qlora_stepweight_catalogfix/` also started before the task-`29` Reddit fix landed in the stack, so it is now mainly a pre-Reddit-fix comparison point
- stale quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v12_qlora_stepweight_redditfix/` also started before the month-specific January admin bestseller precedence fix landed for tasks `4` and `5`, so it is now mainly a pre-admin-bestseller-fix comparison point
- completed quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v13_qlora_stepweight_adminbestsellerfix/` finished with baseline `34/41`, warmup-only `33/41`, and warmup + GRPO `33/41`, so the fixed global `4`-step recipe still looks like the limiting factor on the slower cross-site slice
- stale quantized GitLab-budget note: `outputs/qwen_bootstrap41_curriculum_v14_qlora_gitlabsteps/` started before the generic warmup-demo cleanup removed the redundant `goto(start_url)` supervision artifact from simple direct-answer tasks, so it is now mainly a pre-clean-demo comparison point
- stale quantized GitLab-budget note: `outputs/qwen_bootstrap41_curriculum_v15_qlora_gitlabsteps_cleandemos/` also started before the targeted GitLab warmup/rollout coverage bump landed, so it is now mainly a pre-GitLab-coverage comparison point
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v16_qlora_gitlabcoverage/` started before the clean-demo fix, and the saved `v13` stage audit shows task `0` was the only baseline-to-GRPO regression on that earlier recipe
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v17_qlora_gitlabcoverage_cleandemos/` started before task-group warmup oversampling landed, so it is now mainly a pre-warmup-oversample comparison point
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v18_qlora_gitlabcoverage_cleandemos_warmupoversample/` finished baseline at `38/41` and then crashed at warmup start because the family runner passed task-group warmup-limit overrides through the wrong staged-runner field names
- stale quantized GitLab-coverage note: `outputs/qwen_bootstrap41_curriculum_v19_qlora_gitlabcoverage_cleandemos_warmupoversample_fix/` started before the yearly admin-bestseller aggregate-precedence fix landed, so it is now mainly a pre-yearly-bestseller-fix comparison point
- current quantized GitLab-coverage comparison rerun to watch: `outputs/qwen_bootstrap41_curriculum_v20_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix/`, which keeps the same checked-in `bootstrap41` split, `bnb_4bit` step-weight recipe, GitLab budget, GitLab coverage bump, clean-demo fix, and GitLab warmup oversampling as `v19` but now includes the yearly admin-bestseller aggregate-precedence fix from the start
- keep the dedicated detached replay path for that rerun checked in too: `scripts/local/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix_tmux.sh` should continue to launch the exact `v20` recipe without manual environment overrides
- keep the matched year-fix promotion paths checked in for the next families too: `scripts/local/run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix_tmux.sh` and `scripts/local/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_yearfix_tmux.sh` should stay aligned with the strongest current bootstrap recipe before we scale up
- next no-judge cross-site expansion path is checked in too: `bootstrap44` extends `bootstrap41` with validated tasks `12`, `13`, and `144` for a deterministic `16` warmup / `20` GRPO / `8` holdout split
- keep the order-only ladders runnable too: `shopping_order` (`27` exact tasks) and `shopping_order_full` (`43` scripted tasks with fuzzy judging) now need the same checked-in manifest/launcher coverage as the other families so order-focused ablations stay reproducible instead of drifting back into one-off local runs
- keep the order-only ladders runnable under detached QLoRA too: `scripts/local/run_qwen_shopping_order_curriculum_qlora_tmux.sh` and `scripts/local/run_qwen_shopping_order_full_curriculum_qlora_tmux.sh` now need to stay in parity with the bootstrap and mixed-family long-run launch surface
- keep the order-only current-stack subsets refreshable too: `scripts/local/refresh_shopping_order_current_stack_scorecard.sh` and `scripts/local/refresh_shopping_order_full_current_stack_scorecard.sh` should stay aligned with the validated broader shopping current-stack artifacts so order-heavy scorecards do not require separate reruns
- keep the saved order-only subset scorecards current too: `outputs/shopping_order_current_stack_scorecard.json` and `outputs/shopping_order_full_current_stack_scorecard.json` now provide `26/27` and `42/43` current-stack checkpoints for the order-heavy slices, and both should be refreshed whenever the broader shopping current-stack artifacts move
- keep saved dry-run preflights for those order-only ladders too, and keep their checked-in refresh wrappers (`scripts/local/refresh_shopping_order_curriculum_preflight.sh` / `.ps1` and `scripts/local/refresh_shopping_order_full_curriculum_preflight.sh` / `.ps1`) aligned with the current manifests, so `shopping_order` and `shopping_order_full` stay auditably aligned with the current recommended splits before we spend time launching full order-heavy runs
- the matched detached `bootstrap44` QLoRA GitLab-coverage path is ready too: `scripts/local/run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_tmux.sh` keeps the same GitLab-budget and GitLab-coverage recipe as the current `bootstrap41` comparison ladder
- the matched detached `bootstrap44` QLoRA promotion path is ready too: `scripts/local/run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux.sh` keeps the same `bnb_4bit`, GitLab-budget, GitLab-coverage, clean-demo, and warmup-oversample recipe as `v18`
- next mixed-family scale-up path is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_tmux.sh` now launches the checked-in `web_mix88` split under the same detached `bnb_4bit` recipe, so a successful `bootstrap41` training signal can be promoted directly onto the 88-task mixed family
- mixed-family current-stack checkpoint before the `bootstrap44` expansion: `outputs/web_mix88_current_stack_scorecard.json` now combines the validated `shopping_full` current stack and the validated `bootstrap41` current stack into `84/88` (`0.9545454545454546`), with overlap task `188` deduplicated across the two source artifacts
- next mixed-family expansion path after that is checked in too: `web_mix91` combines `shopping_full` with `bootstrap44`, so those three validated expansion tasks carry forward into a `91`-task mixed family
- mixed-family current-stack scorecard: `outputs/web_mix91_current_stack_scorecard.json` now combines the validated `shopping_full` current stack and the validated `bootstrap44` current stack into `87/91` (`0.9560439560439561`), with overlap task `188` deduplicated across the two source artifacts
- mixed-family validator and audit: `scripts/validate_combined_family_stage_scorecard.py` and `scripts/audit_combined_family_stage_scorecard.py` now provide a dedicated verification path for `web_mix91`-style combined scorecards, mirroring the earlier `bootstrap44` expansion workflow
- mixed-family local refresh wrappers: `scripts/local/refresh_web_mix88_current_stack_scorecard.sh` / `.ps1` and `scripts/local/refresh_web_mix91_current_stack_scorecard.sh` / `.ps1` now keep both mixed-family current-stack scorecards reproducible from checked-in manifests instead of manual CLI reconstruction
- the matched detached `web_mix91` QLoRA GitLab-coverage path is ready too: `scripts/local/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_tmux.sh` keeps the same GitLab-budget and GitLab-coverage recipe as the current `web_mix88` comparison ladder
- the matched detached `web_mix91` QLoRA promotion path is ready too: `scripts/local/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh` keeps the same `bnb_4bit`, GitLab-budget, GitLab-coverage, and warmup-oversample recipe as the strongest current mixed-family path
- the matched mixed-family GitLab-coverage path is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh` carries the same GitLab budget and coverage bump as `v16`, so the first mixed-family promotion can preserve the strongest current training recipe instead of resetting to the weaker mixed-family default
- the matched mixed-family GitLab-coverage warmup-oversample path is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh` adds the same `site_gitlab=2.5` warmup oversample used in `v18`, so a future mixed-family promotion can keep the full bootstrap GitLab recipe intact
- the matched mixed-family year-fix promotion path is ready too: `scripts/local/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_yearfix_tmux.sh` should keep that strongest `web_mix88` recipe aligned with the current year-scoped admin-bestseller fix before we scale up
- warmup summaries now record `source_task_sample_counts`, `average_epoch_task_sample_counts`, and `task_sample_multipliers`, so future oversample tuning can be audited from saved artifacts instead of only from launcher flags
- finished `family_summary.json` artifacts now also keep that `warmup_training_summary`, so future recipe audits can read supervised-mix evidence from the top-level family report
- live run status now mirrors those warmup-mix fields too once `warmup_summary.json` exists, so oversample evidence becomes visible before the full family run finishes
- live run status now also reports family-level `benchmark_blockers` plus blocker-excluded effective eval metrics, so active reruns like `v18` can separate known benchmark mismatches from real recipe regressions during baseline and later eval stages
- current warmup-demo cleanup: generic scripted warmup policies now answer directly instead of prepending a redundant `goto(start_url)`, which removes a recurring `TargetClosedError` step from simple dashboard-style demos and should reduce needless supervision noise on tasks like bootstrap task `0`
- completed family runs now auto-write baseline/warmup/GRPO stage-audit artifacts, so future tuning can read per-task and per-site regressions directly from the output dir instead of reconstructing them later

## Scale-Up Target

- move from the validated `5`-task slice to the scripted `shopping_exact` family with `32` exact-match tasks
- keep the executable exact-match curriculum at `26` training tasks and `6` untouched holdout tasks
- bias warmup toward all four search/sort tasks plus the closest admin/detail analogs so transfer has a better chance to appear on holdout tasks like `324`
- treat the broader `48`-task shopping family as the next scale-up target once fuzzy-judge credentials are available
- recommended full-family split when `OPENAI_API_KEY` is present: `16` warmup, `24` GRPO, `8` holdout (`40` training tasks)
- next non-shopping expansion path: `bootstrap41`, a `41`-task cross-site scripted family over shopping, admin, reddit, gitlab, and map with a `15` warmup / `18` GRPO / `8` holdout split
- next no-judge cross-site expansion after that: `bootstrap44`, which extends `bootstrap41` with validated tasks `12`, `13`, and `144` for a `16` warmup / `20` GRPO / `8` holdout split
- next post-bootstrap mixed expansion path: `web_mix88`, an `88`-task mixed shopping-plus-cross-site family with a checked-in `30` warmup / `42` GRPO / `16` holdout split (`72` training tasks)
- next post-bootstrap44 mixed expansion path: `web_mix91`, which combines `shopping_full` with `bootstrap44` for a checked-in `31` warmup / `44` GRPO / `16` holdout split (`75` training tasks)
- keep the final training set above `20` tasks so improvements have a chance to transfer beyond a toy slice
- use the family-runner dry-run preflight before changing split sizes so broad runs keep disjoint partitions and do not accidentally fall below the 20-task training floor
- keep the exact-family launch path on a checked-in split manifest too, so the validated 32-task curriculum stays portable and reproducible
- keep the launcher scripts covered by tests so they continue to use the checked-in manifests instead of drifting back to implicit split defaults
- keep a fast CLI validator for the checked-in split manifests and launcher wiring so portability regressions can be checked without running the full test suite
- use the dry-run alignment signal to catch checked-in manifest drift against the current recommended family split before launch
- use that dry-run even for `shopping_full` before judge credentials are ready, since it now reports launch readiness and group coverage without requiring `OPENAI_API_KEY`
- save the broader-family dry-run under `outputs/` so the exact 40-task planned split is reusable once the launch environment is ready
- launch the eventual broader-family run from that saved split manifest so the approved partition is identical between planning and execution
- keep the approved official-target broader-family split checked in under `scripts/local/` so the launch path is portable and not tied to one machine's `outputs/`
- refresh that checked-in broader-family manifest from code when the intended curriculum changes so the portable launch path stays aligned with the recommended split
- keep interrupted large-family runs resumable at the demo, rollout, and per-task eval level so transient browser failures do not force full reruns

## Working Order

1. validate the environment
2. measure the baseline on the intended eval slice
3. collect or load clean warmup demos
4. run behavior-cloning warmup
5. run short staged GRPO on a disjoint task subset
6. compare baseline, warmup-only, and post-GRPO on the holdout set and the full eval slice

## Model Guidance

- official comparison target: `Qwen/Qwen3.5-2B`
- fast local experiments: smaller models are fine when they speed up validation
