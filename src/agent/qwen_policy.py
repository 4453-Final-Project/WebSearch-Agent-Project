"""Task 3 Qwen policy skeleton with a backend-driven generation flow."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .compat import load_policy_config, normalize_observation
from .parsing import make_decision, make_fallback_decision
from .prompting import build_full_prompt, build_retry_prompt
from .types import AgentDecision, NormalizedObservation, PolicyConfig


class PolicyBackend(Protocol):
    """Minimal backend interface required by the policy wrapper."""

    def generate(self, prompt: str) -> str:
        """Generate a raw text response for a prompt."""


class _UnavailableBackend:
    """Placeholder backend used until real model loading is implemented."""

    def generate(self, prompt: str) -> str:
        """Fail clearly when no backend was provided."""

        raise RuntimeError(
            "No policy backend is configured. Provide a custom backend until model loading is implemented."
        )


class TransformersGPTQBackend:
    """Lazy backend contract for future GPTQ-backed Transformers inference."""

    def __init__(self, config: PolicyConfig) -> None:
        """Store policy config and validate the model path early."""

        self.config = config
        self.model_path = Path(config.model_path).expanduser() if config.model_path else None
        if self.model_path is None:
            raise FileNotFoundError(
                "PolicyConfig.model_path is empty. Provide a valid local model path for TransformersGPTQBackend."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Configured model path does not exist: {self.model_path}"
            )

    def generate(self, prompt: str) -> str:
        """Lazily check runtime dependencies and fail clearly until loading is enabled."""

        self._import_runtime_dependencies()
        raise RuntimeError(
            "TransformersGPTQBackend is a future-loading contract only. "
            "Model loading is not yet possible in the current environment."
        )

    def _import_runtime_dependencies(self) -> tuple[object, object]:
        """Import heavyweight runtime dependencies only when generation is attempted."""

        try:
            import torch  # type: ignore
            import transformers  # type: ignore
        except ImportError as exc:
            missing_name = getattr(exc, "name", None) or "transformers/torch"
            raise RuntimeError(
                "TransformersGPTQBackend requires 'transformers' and 'torch' to generate. "
                f"Missing dependency: {missing_name}"
            ) from exc
        return torch, transformers


class QwenPolicy:
    """Reusable Task 3 policy wrapper with one retry and safe fallback."""

    def __init__(
        self,
        backend: PolicyBackend | None = None,
        config: PolicyConfig | None = None,
    ) -> None:
        """Initialize the policy with an optional backend and config."""

        self.backend: PolicyBackend = backend if backend is not None else _UnavailableBackend()
        self.config = config if config is not None else load_policy_config()

    def act(self, observation: dict | NormalizedObservation, step_idx: int) -> AgentDecision:
        """Generate, parse, retry once, and safely fall back if needed."""

        normalized = (
            normalize_observation(observation)
            if isinstance(observation, dict)
            else observation
        )

        prompt = build_full_prompt(normalized, step_idx)
        first_raw_text = self.backend.generate(prompt)
        first_decision = make_decision(first_raw_text)
        if first_decision.parse_error is None:
            return first_decision

        retry_prompt = build_retry_prompt(
            normalized,
            step_idx,
            previous_raw_text=first_decision.raw_text,
            parse_error=first_decision.parse_error,
        )
        second_raw_text = self.backend.generate(retry_prompt)
        second_decision = make_decision(second_raw_text)
        if second_decision.parse_error is None:
            return second_decision

        return make_fallback_decision(
            raw_text=second_decision.raw_text,
            reason=second_decision.parse_error or "Unknown parse failure.",
        )
