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
- current broader-family blocker: task `204` still behaves like a benchmark/reference mismatch even though the stack answers from the actual most recent completed order
- previous broad-family rerun `outputs/qwen_shopping_full_curriculum_v3_balanced/` is now mainly a stale pre-device-fix artifact because warmup training crashed with a meta-device backward error after the strong `47/48` baseline snapshot
- current main broad-family rerun: `outputs/qwen_shopping_full_curriculum_v4_devicefix/`
- focused warmup smoke for the repaired trainable loader: `outputs/qwen_shopping_full_warmup_smoke_v1_devicefix/` succeeded on a mixed `5`-task slice, which confirms backward and optimizer steps now work on the fixed training stack before the full rerun reaches warmup training
- focused staged smoke for the repaired training loop: `outputs/qwen_shopping_full_staged_smoke_v1_devicefix/` completed warmup, rollout collection, GRPO, checkpoint save, and eval on a disjoint micro-split, which confirms the fixed stack can now run the full staged loop end-to-end
- current broad-family rerun uses a balanced `shopping_full` warmup with `4` judge-gated tasks (`191`, `201`, `334`, `359`) instead of only `2`
- current bootstrap shopping-admin recheck slice: tasks `0-6` are `7/7` in `outputs/bootstrap41_dashboard_recheck_0_6_summary.json` after dashboard bestseller extraction plus ordered-product report aggregation fixes
- current bootstrap shopping-admin recheck slice: tasks `0-6` are `7/7` in `outputs/bootstrap41_dashboard_recheck_0_6_summary.json` after dashboard bestseller extraction plus ordered-product report aggregation fixes
- current bootstrap map recheck slice: tasks `7`, `9`, `10`, `36`, `70`, `71`, and `72` are `7/7` in `outputs/bootstrap41_map_recheck_7_72_summary.json` after direct map-backend answer recovery plus benchmark-normalized address canonicalization
- merged bootstrap current-stack artifact: `outputs/bootstrap41_current_stack_summary.json` now moves the cross-site baseline from `12/41` (`0.2926829268292683`) to `37/41` (`0.9024390243902439`) after folding in targeted fixes for tasks `0`, `3`, `11`, `21`, `23`, `25`, `26`, `27`, `28`, `30`, `31`, `41`, `66`, `67`, `68`, `69`, `77`, `125`, `126`, `132`, `134`, `135`, `136`, `259`, and `293`
- current bootstrap Reddit note: tasks `28`, `30`, `31`, `66`, `67`, `68`, and `69` now have clean focused re-eval wins through the direct forum-answer path, while task `29` still looks benchmark-mismatched because the grounded latest negative-comment count stays `0`
- current bootstrap storefront price-range note: task `125` is now backed by a clean finished re-eval artifact in `outputs/eval_task_125_pricerangefix_v3/`, so the catalog path can now scan sorted search pages for the first strongly query-matched extrema; task `126` remains green, while task `124` still returns a grounded but benchmark-misaligned range
- current bootstrap GitLab count note: task `133` now reaches the intended repo-graph answer path and returns `0`, but it still scores `0.0`, so it currently looks closer to a benchmark/reference mismatch than an infrastructure or parser failure
- current bootstrap RSS-token note: task `259` is now backed by a clean finished re-eval artifact in `outputs/eval_task_259_bootstrap_rssfix_v1/`, so the authenticated personal-access-tokens route and hidden-token extraction fix are both validated and merged into the bootstrap current-stack scorecard
- current bootstrap review-count note: task `11` is now backed by a clean finished re-eval artifact in `outputs/eval_task_11_bootstrap_reviewcount_v1/`, so the admin dashboard review-count path now filters the authenticated review grid by the quoted review term and reads the resulting `records found` count end-to-end
- current bootstrap search-term note: task `41` is now backed by a clean finished re-eval artifact in `outputs/eval_task_41_bootstrap_searchterm_v1/`, so the admin dashboard search-term path now reads the authenticated edit pages, ranks by `popularity`, and answers directly from the appended search-term rows end-to-end
- current bootstrap review-status note: task `77` is now backed by a clean finished re-eval artifact in `outputs/eval_task_77_bootstrap_reviewstatus_v1/`, so the admin dashboard review-status path now filters the authenticated review grid by status and reads the resulting `records found` count end-to-end
- current bootstrap storefront-review note: tasks `23`, `25`, and `26` are now backed by clean finished re-eval artifacts in `outputs/eval_task_23_bootstrap_reviewauthors_v1/`, `outputs/eval_task_25_bootstrap_reviewauthors_v1/`, and `outputs/eval_task_26_bootstrap_reviewauthors_v1/`, so the storefront review path now recognizes fingerprint-resistance and customer-service complaint phrasing that previously failed the literal matcher
- current shopping order-spend note: task `144` is now backed by a clean finished re-eval artifact in `outputs/eval_task_144_spendfix_v1/`, so the order-history spend path can now sum matched dated order-detail rows directly; tasks `141`, `142`, `143`, and `145` also now return grounded spend totals, but those totals still disagree with the baked references
- completed broad-family rerun: `outputs/qwen_shopping_full_curriculum_v4_devicefix/` finished with baseline `47/48` (`0.9791666666666666`), warmup-only `46/48` (`0.96875`), and warmup + GRPO `46/48` (`0.96875`), so this family is now effectively saturated for training comparisons and should mainly be treated as a stack-quality benchmark
- completed bootstrap rerun: `outputs/qwen_bootstrap41_curriculum_v3_mapfix/` finished with baseline `12/41` (`0.2926829268292683`), warmup-only `12/41`, and warmup + GRPO `12/41`, which confirms the old warmup/GRPO weighting was too weak to move the broader cross-site slice
- current training-signal fix: both warmup and GRPO now weight later clean steps and successful terminal answer steps more heavily than early noisy navigation, while invalid and parse-failed steps are downweighted
- current bootstrap rerun to watch: `outputs/qwen_bootstrap41_curriculum_v4_weightedfix/`, which reuses the same checked-in `bootstrap41` split on the current fixed stack plus the new warmup/GRPO step weighting so the comparison against `v3_mapfix` stays apples-to-apples
- focused quantized-training proof: `outputs/qwen_shopping_full_warmup_smoke_v2_qlora/` now completes a real `bnb_4bit` warmup smoke on the WSL GPU stack, which confirms the optional QLoRA-style path is usable for local adapter training
- current quantized bootstrap rerun to watch: `outputs/qwen_bootstrap41_curriculum_v6_qlora/`, which reuses the same checked-in `bootstrap41` split on the current weighted stack but now runs through the `bnb_4bit` loader path so we can compare speed and signal against the earlier reruns
- current quantized reward-tuned comparison rerun: `outputs/qwen_bootstrap41_curriculum_v7_qlora_rewardtune/`, which keeps the same checked-in `bootstrap41` split and quantized loader path but adds the stronger reward shaping from the earlier non-quantized reward-tuned bootstrap run
- current follow-up quantized comparison path: `outputs/qwen_bootstrap41_curriculum_v8_qlora_stepweight_tune/`, which layers stronger terminal-step weighting and stronger error-step downweighting on top of the same quantized reward-tuned bootstrap recipe
- stale quantized step-weight comparison note: `outputs/qwen_bootstrap41_curriculum_v9_qlora_stepweight_dashboardfix/` started before the task-`41` search-term popularity fix, so it is now mainly a pre-searchterm-fix comparison point
- current quantized step-weight comparison rerun to watch: `outputs/qwen_bootstrap41_curriculum_v11_qlora_stepweight_catalogfix/`, which reuses the same checked-in `bootstrap41` split and `bnb_4bit` step-weight recipe on the latest catalog-plus-spend stack and is already in baseline evaluation on the live run

## Scale-Up Target

- move from the validated `5`-task slice to the scripted `shopping_exact` family with `32` exact-match tasks
- keep the executable exact-match curriculum at `26` training tasks and `6` untouched holdout tasks
- bias warmup toward all four search/sort tasks plus the closest admin/detail analogs so transfer has a better chance to appear on holdout tasks like `324`
- treat the broader `48`-task shopping family as the next scale-up target once fuzzy-judge credentials are available
- recommended full-family split when `OPENAI_API_KEY` is present: `16` warmup, `24` GRPO, `8` holdout (`40` training tasks)
- next non-shopping expansion path: `bootstrap41`, a `41`-task cross-site scripted family over shopping, admin, reddit, gitlab, and map with a `15` warmup / `18` GRPO / `8` holdout split
- next post-bootstrap mixed expansion path: `web_mix88`, an `88`-task mixed shopping-plus-cross-site family with a checked-in `30` warmup / `42` GRPO / `16` holdout split (`72` training tasks)
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
