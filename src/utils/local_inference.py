from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


def get_model_device(model: Any) -> torch.device:
    """Best-effort lookup of the device that should receive prompt tensors."""

    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        return torch.device(model_device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    """Move tokenized inputs to the model device when possible."""

    try:
        return inputs.to(get_model_device(model))
    except Exception:
        return inputs


def build_generation_kwargs(
    tokenizer: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    """Build a compact generation kwargs dictionary shared by smoke tests and benchmarks."""

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        kwargs["pad_token_id"] = eos_token_id

    if temperature > 0.0:
        kwargs["temperature"] = temperature
    if top_k is not None:
        kwargs["top_k"] = top_k
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    return kwargs


def build_model_inputs(
    tokenizer: Any,
    *,
    prompt: str,
    system_prompt: str | None = None,
    use_chat_template: bool = True,
):
    """Build prompt tensors, preferring the tokenizer chat template when available."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if use_chat_template and callable(apply_chat_template):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            return tokenizer(rendered, return_tensors="pt")

    return tokenizer(prompt, return_tensors="pt")


def render_prompt_text(
    tokenizer: Any,
    *,
    prompt: str,
    system_prompt: str | None = None,
    use_chat_template: bool = True,
) -> str:
    """Render the exact prompt text used for generation and supervised scoring."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if use_chat_template and callable(apply_chat_template):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors=None,
                tokenize=False,
            )
        if isinstance(rendered, str):
            return rendered

    return prompt


@contextmanager
def fast_generation_mode(model: Any):
    """Temporarily prefer cached inference over training-time memory settings."""

    was_training = bool(getattr(model, "training", False))
    had_use_cache = hasattr(getattr(model, "config", None), "use_cache")
    previous_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    had_gradient_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))

    if had_gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if had_use_cache:
        model.config.use_cache = True
    if hasattr(model, "eval"):
        model.eval()

    try:
        yield model
    finally:
        if had_use_cache:
            model.config.use_cache = previous_use_cache
        if had_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "train"):
            model.train(was_training)


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA timing when benchmarking generation."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
