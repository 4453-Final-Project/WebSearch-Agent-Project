from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_model_id, get_output_dir, resolve_model_path  # noqa: E402
from src.utils.local_inference import (  # noqa: E402
    build_generation_kwargs,
    build_model_inputs,
    get_model_device,
    move_inputs_to_model_device,
    synchronize_if_needed,
)
from src.utils.model_profiles import get_model_profile, list_model_profiles  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local text generation throughput for a local model.")
    parser.add_argument("--profile", choices=list_model_profiles(), default="lfm2.5-350m")
    parser.add_argument("--model-dir-name", default=None, help="Optional local model directory name under ../models/.")
    parser.add_argument("--model-path", default=None, help="Optional explicit local model path.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt override.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", default=str(get_output_dir("inference_benchmarks", create=True)))
    parser.add_argument("--label", default=None, help="Optional output label.")
    return parser


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_arg_parser().parse_args()
    profile = get_model_profile(args.profile)
    model_path = resolve_model_path(
        model_path=args.model_path,
        model_dir_name=args.model_dir_name or profile.local_dir_name,
    )
    model_id = profile.huggingface_id if args.profile != "qwen-default" else get_model_id()
    prompt = args.prompt or profile.prompt
    system_prompt = args.system_prompt if args.system_prompt is not None else profile.system_prompt
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else profile.max_new_tokens
    temperature = args.temperature if args.temperature is not None else profile.temperature
    top_k = args.top_k if args.top_k is not None else profile.top_k
    repetition_penalty = (
        args.repetition_penalty if args.repetition_penalty is not None else profile.repetition_penalty
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}\n"
            f"Download it with: hf download {model_id} --local-dir {model_path}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_tokenizer_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer_load_sec = perf_counter() - load_tokenizer_start

    load_model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model_load_sec = perf_counter() - load_model_start

    inputs = build_model_inputs(
        tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
        use_chat_template=profile.use_chat_template,
    )
    inputs = move_inputs_to_model_device(inputs, model)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    device = get_model_device(model)
    generation_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    run_records: list[dict[str, float | int | str]] = []
    total_runs = max(0, args.warmup_runs) + max(1, args.repeats)
    for run_idx in range(total_runs):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        synchronize_if_needed(device)
        generation_start = perf_counter()
        outputs = model.generate(**inputs, **generation_kwargs)
        synchronize_if_needed(device)
        generation_sec = perf_counter() - generation_start

        generated_tokens = int(outputs[0].shape[-1] - prompt_tokens)
        record = {
            "run_index": run_idx,
            "phase": "warmup" if run_idx < args.warmup_runs else "measured",
            "generation_sec": generation_sec,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "decode_tokens_per_sec": (
                (generated_tokens / generation_sec) if generation_sec > 0.0 else 0.0
            ),
        }
        if device.type == "cuda" and torch.cuda.is_available():
            record["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            record["peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        run_records.append(record)

    measured = [record for record in run_records if record["phase"] == "measured"]
    decode_rates = [float(record["decode_tokens_per_sec"]) for record in measured]
    durations = [float(record["generation_sec"]) for record in measured]
    generated = [int(record["generated_tokens"]) for record in measured]

    summary = {
        "profile": profile.name,
        "model_id": model_id,
        "model_path": str(model_path),
        "device": str(device),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "tokenizer_load_sec": tokenizer_load_sec,
        "model_load_sec": model_load_sec,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "average_generation_sec": statistics.mean(durations) if durations else 0.0,
        "median_generation_sec": statistics.median(durations) if durations else 0.0,
        "average_generated_tokens": statistics.mean(generated) if generated else 0.0,
        "average_decode_tokens_per_sec": statistics.mean(decode_rates) if decode_rates else 0.0,
        "max_decode_tokens_per_sec": max(decode_rates) if decode_rates else 0.0,
        "min_decode_tokens_per_sec": min(decode_rates) if decode_rates else 0.0,
        "runs": run_records,
    }

    output_name = args.label or f"{profile.name}.json"
    if not output_name.endswith(".json"):
        output_name += ".json"
    output_path = out_dir / output_name
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote benchmark summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
