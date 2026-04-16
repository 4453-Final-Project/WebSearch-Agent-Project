# Project Plan

## Objective

Use the existing BrowserGym/WebArena pipeline to improve a local model beyond its baseline on a narrow, reproducible evaluation family.

## Current Best Validated Slice

- family: Shopping search/sort `324-328`
- baseline: `3/5`
- warmup-only: `4/5`
- warmup + disjoint GRPO: `5/5`

## Working Order

1. validate the environment
2. measure the baseline on the intended eval slice
3. collect or load clean warmup demos
4. run behavior-cloning warmup
5. run short staged GRPO on a disjoint task subset
6. compare baseline, warmup-only, and post-GRPO on the same eval slice

## Model Guidance

- official comparison target: `Qwen/Qwen3.5-2B`
- fast local experiments: smaller models are fine when they speed up validation
