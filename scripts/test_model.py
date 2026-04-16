from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_model_id, resolve_model_path  # noqa: E402
from src.utils.local_inference import build_generation_kwargs, build_model_inputs, move_inputs_to_model_device  # noqa: E402
from src.utils.model_profiles import get_model_profile, list_model_profiles  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a local model and run one short generation.")
    parser.add_argument("--profile", choices=list_model_profiles(), default="qwen-default")
    parser.add_argument("--model-dir-name", default=None, help="Optional local model directory name under ../models/.")
    parser.add_argument("--model-path", default=None, help="Optional explicit local model path.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt override.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    return parser


def main() -> None:
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

    overall_start = perf_counter()
    print(f"Profile: {profile.name}")
    print(f"Model id: {model_id}")
    print(f"Model path: {model_path}")
    print(f"Prompt: {prompt}")

    tokenizer_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f"Tokenizer loaded in {perf_counter() - tokenizer_start:.1f}s", flush=True)

    model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"Model loaded in {perf_counter() - model_start:.1f}s", flush=True)

    inputs = build_model_inputs(
        tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
        use_chat_template=profile.use_chat_template,
    )
    inputs = move_inputs_to_model_device(inputs, model)
    generation_start = perf_counter()
    outputs = model.generate(
        **inputs,
        **build_generation_kwargs(
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ),
    )
    print(f"Generation finished in {perf_counter() - generation_start:.1f}s", flush=True)

    prompt_token_count = int(inputs["input_ids"].shape[-1])
    model_output = tokenizer.decode(outputs[0][prompt_token_count:], skip_special_tokens=True)
    print(f"Model Output: {model_output}")
    print(f"Total time: {perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
