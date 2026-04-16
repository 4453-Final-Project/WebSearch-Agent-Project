from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.training.peft_setup import get_adapter_base_model_path, is_adapter_checkpoint


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Small profile describing how to smoke-test and benchmark a local model."""

    name: str
    huggingface_id: str
    local_dir_name: str
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.0
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int = 64
    use_chat_template: bool = True


DEFAULT_QWEN_PROFILE = ModelProfile(
    name="qwen-default",
    huggingface_id="Qwen/Qwen3.5-2B",
    local_dir_name="Qwen3.5-2B",
    prompt="Explain in one short paragraph what reinforcement learning is.",
    temperature=0.0,
    max_new_tokens=64,
)

LFM25_350M_PROFILE = ModelProfile(
    name="lfm2.5-350m",
    huggingface_id="LiquidAI/LFM2.5-350M",
    local_dir_name="LFM2.5-350M",
    prompt="What is C. elegans?",
    system_prompt="You are a helpful assistant trained by Liquid AI.",
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
    max_new_tokens=128,
)

MODEL_PROFILES = {
    DEFAULT_QWEN_PROFILE.name: DEFAULT_QWEN_PROFILE,
    LFM25_350M_PROFILE.name: LFM25_350M_PROFILE,
}


def get_model_profile(name: str) -> ModelProfile:
    """Return a named local inference profile."""

    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model profile {name!r}. Valid profiles: {', '.join(sorted(MODEL_PROFILES))}"
        ) from exc


def list_model_profiles() -> list[str]:
    """Return the stable list of supported profile names."""

    return sorted(MODEL_PROFILES)


def find_model_profile_by_dir_name(local_dir_name: str | None) -> ModelProfile | None:
    """Return the first profile whose local directory matches ``local_dir_name``."""

    if not local_dir_name:
        return None
    for profile in MODEL_PROFILES.values():
        if profile.local_dir_name == local_dir_name:
            return profile
    return None


def find_model_profile_by_path(model_path: str | Path | None) -> ModelProfile | None:
    """Infer a profile from a local checkpoint or adapter path."""

    if not model_path:
        return None

    resolved_path = Path(model_path).expanduser()
    candidate_dir_name = resolved_path.name
    if is_adapter_checkpoint(resolved_path):
        base_model_path = get_adapter_base_model_path(resolved_path)
        if base_model_path:
            candidate_dir_name = Path(base_model_path).expanduser().name

    return find_model_profile_by_dir_name(candidate_dir_name)
