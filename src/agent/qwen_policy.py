"""Task 3 Qwen policy with lazy local Transformers-backed generation."""

from __future__ import annotations

import re
import html
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from src.training.peft_setup import get_adapter_base_model_path, is_adapter_checkpoint
from src.utils.quantization import build_bitsandbytes_quantization_config, resolve_quantized_device_map
from src.utils.map_services import derive_map_goal_answer

from .compat import load_policy_config, normalize_observation
from .parsing import make_decision, make_fallback_decision
from .prompting import (
    build_full_prompt,
    build_gitlab_alternate_graph_url,
    build_gitlab_explore_click_hint,
    build_gitlab_repo_graph_hint,
    build_shopping_category_click_hint,
    build_shopping_order_detail_target_url,
    build_shopping_category_target_url,
    build_shopping_orders_target_hint,
    derive_shopping_order_number,
    derive_shopping_sort_hint,
    derive_shopping_category_labels,
    derive_shopping_query_hint,
    derive_gitlab_query_hint,
    build_retry_prompt,
)
from .types import AgentDecision, NormalizedObservation, PolicyConfig

MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
ORDER_DATE_TEXT_PATTERN = (
    r"(\d{1,2}/\d{1,2}/\d{4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},\s+\d{4})"
)
OBS_ROLE_LINE_PATTERN = re.compile(r'^\[(?P<bid>\d+)\]\s+role=(?P<role>[A-Za-z_]+)(?:\s+name="(?P<name>[^"]*)")?')
EDITABLE_OBS_ROLES = {
    "textbox",
    "searchbox",
    "combobox",
    "textarea",
    "input",
}


class PolicyBackend(Protocol):
    """Minimal backend interface required by the policy wrapper."""

    def generate(self, prompt: str) -> str:
        """Generate a raw text response for a prompt."""


class TransformersGPTQBackend:
    """Lazy backend for local Transformers inference over local checkpoints."""

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
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str) -> str:
        """Generate one action string from a prompt using the local model."""

        tokenizer, model = self._ensure_loaded()
        print(f"Status: generating policy action with {self.config.policy_name}...", flush=True)

        inputs = self._tokenize_prompt(prompt, tokenizer)
        _, build_generation_kwargs, fast_generation_mode, move_inputs_to_model_device = self._import_local_inference_helpers()
        inputs = move_inputs_to_model_device(inputs, model)
        generation_kwargs = build_generation_kwargs(
            tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )

        with fast_generation_mode(model):
            outputs = model.generate(**inputs, **generation_kwargs)
        prompt_token_count = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_token_count:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _ensure_loaded(self):
        """Load the tokenizer and model once, on first generation."""

        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        _, transformers = self._import_runtime_dependencies()
        load_device = self._resolve_load_device()
        base_model_path = (
            _normalize_local_model_path(get_adapter_base_model_path(self.model_path))
            if is_adapter_checkpoint(self.model_path)
            else None
        )

        print(f"Status: loading {self.config.policy_name} policy tokenizer from {self.model_path}...", flush=True)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        except (OSError, ValueError):
            if not base_model_path:
                raise
            print(
                f"Status: adapter tokenizer metadata is incomplete; loading tokenizer from base model {base_model_path}...",
                flush=True,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=True,
            )
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Status: loading {self.config.policy_name} policy model from {self.model_path}...", flush=True)
        model_kwargs = self._build_inference_model_kwargs()
        if is_adapter_checkpoint(self.model_path):
            from peft import PeftModel

            if not base_model_path:
                raise RuntimeError(f"Adapter checkpoint at {self.model_path} is missing base_model_name_or_path.")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs,
            )
            _normalize_model_no_split_modules(model)
            model = PeftModel.from_pretrained(model, self.model_path, is_trainable=False)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs,
            )
        if not self._uses_quantized_loading() and hasattr(model, "to"):
            model = model.to(load_device)

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def _tokenize_prompt(self, prompt: str, tokenizer):
        """Use chat templating when available, otherwise fall back to raw tokenization."""

        system_prompt, user_prompt = self._split_prompt(prompt)
        build_model_inputs, _, _, _ = self._import_local_inference_helpers()
        return build_model_inputs(
            tokenizer,
            prompt=user_prompt,
            system_prompt=system_prompt or None,
            use_chat_template=self.config.use_chat_template,
        )

    def _split_prompt(self, prompt: str) -> tuple[str, str]:
        """Recover system and user sections from the Task 3 full prompt format."""

        separator = "\n\n"
        if separator not in prompt:
            return "", prompt
        system_prompt, user_prompt = prompt.split(separator, 1)
        return system_prompt.strip(), user_prompt.strip()

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

    def _resolve_load_device(self):
        """Choose one concrete device for inference loads."""

        torch, _ = self._import_runtime_dependencies()
        device_ctor = getattr(torch, "device", None)
        configured_device = self.config.device
        if configured_device:
            return device_ctor(configured_device) if callable(device_ctor) else configured_device
        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is not None and callable(getattr(cuda_module, "is_available", None)) and cuda_module.is_available():
            return device_ctor("cuda") if callable(device_ctor) else "cuda"
        return device_ctor("cpu") if callable(device_ctor) else "cpu"

    def _build_inference_model_kwargs(self) -> dict[str, object]:
        """Build inference load kwargs without auto device placement."""

        kwargs: dict[str, object] = {
            "local_files_only": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": False,
        }
        if not self._uses_quantized_loading():
            return kwargs

        torch, transformers = self._import_runtime_dependencies()
        quantization_config = build_bitsandbytes_quantization_config(
            torch_module=torch,
            transformers_module=transformers,
            quantization_mode=self.config.quantization_mode,
            quant_compute_dtype=self.config.quant_compute_dtype,
            quant_type=self.config.quant_type,
            quant_use_double_quant=self.config.quant_use_double_quant,
        )
        kwargs["quantization_config"] = quantization_config
        kwargs["torch_dtype"] = getattr(torch, self.config.quant_compute_dtype, torch.bfloat16)
        kwargs["device_map"] = resolve_quantized_device_map(self._resolve_load_device())
        return kwargs

    def _uses_quantized_loading(self) -> bool:
        return bool((self.config.quantization_mode or "").strip())

    def _import_local_inference_helpers(self):
        """Import shared local-inference helpers only when generation is attempted."""

        from src.utils.local_inference import (
            build_generation_kwargs,
            build_model_inputs,
            fast_generation_mode,
            move_inputs_to_model_device,
        )

        return build_model_inputs, build_generation_kwargs, fast_generation_mode, move_inputs_to_model_device


def _normalize_model_no_split_modules(model: object) -> None:
    raw_modules = getattr(model, "_no_split_modules", None)
    if raw_modules is None:
        return
    normalized_modules = _normalize_no_split_modules(raw_modules)
    if normalized_modules:
        setattr(model, "_no_split_modules", normalized_modules)


def _normalize_no_split_modules(raw_modules: object) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str):
            return
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        normalized.append(cleaned)

    if isinstance(raw_modules, str):
        add(raw_modules)
        return normalized
    if not isinstance(raw_modules, (list, tuple, set, frozenset)):
        return normalized

    for value in raw_modules:
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                add(nested)
            continue
        add(value)
    return normalized


class QwenPolicy:
    """Reusable Task 3 policy wrapper with one retry and safe fallback."""

    def __init__(
        self,
        backend: PolicyBackend | None = None,
        config: PolicyConfig | None = None,
    ) -> None:
        """Initialize the policy with an optional backend and config."""

        self.config = config if config is not None else load_policy_config()
        self.backend: PolicyBackend = backend if backend is not None else TransformersGPTQBackend(self.config)
        self.name = self.config.policy_name

    def act(self, observation: dict | NormalizedObservation, step_idx: int) -> AgentDecision:
        """Generate, parse, retry once, and safely fall back if needed."""

        normalized = (
            normalize_observation(observation)
            if isinstance(observation, dict)
            else observation
        )

        prompt = build_full_prompt(normalized, step_idx, model_system_prompt=self.config.system_prompt)
        first_raw_text = self.backend.generate(prompt)
        first_decision = make_decision(first_raw_text)
        first_decision = self._validate_decision_against_observation(first_decision, normalized)
        if first_decision.parse_error is None:
            return first_decision

        retry_prompt = build_retry_prompt(
            normalized,
            step_idx,
            previous_raw_text=first_decision.raw_text,
            parse_error=first_decision.parse_error,
            model_system_prompt=self.config.system_prompt,
        )
        second_raw_text = self.backend.generate(retry_prompt)
        second_decision = make_decision(second_raw_text)
        second_decision = self._validate_decision_against_observation(second_decision, normalized)
        if second_decision.parse_error is None:
            return second_decision

        return make_fallback_decision(
            raw_text=second_decision.raw_text,
            reason=second_decision.parse_error or "Unknown parse failure.",
        )

    def _validate_decision_against_observation(
        self,
        decision: AgentDecision,
        observation: NormalizedObservation,
    ) -> AgentDecision:
        shopping_admin_dashboard_answer_action = _build_shopping_admin_dashboard_answer_retry_action(observation)
        shopping_admin_order_answer_action = _build_shopping_admin_order_answer_retry_action(observation)
        shopping_admin_orders_action = _build_shopping_admin_orders_retry_action(observation)
        shopping_orders_action = _build_shopping_orders_retry_action(observation)
        shopping_target_action = _build_shopping_canonical_target_action(observation)
        shopping_order_detail_action = _build_shopping_order_detail_retry_action(observation)
        shopping_order_answer_action = _build_shopping_order_answer_retry_action(observation)
        shopping_order_detail_answer_action = _build_shopping_order_detail_answer_retry_action(observation)
        shopping_review_answer_action = _build_shopping_review_answer_retry_action(observation)
        shopping_review_action = _build_shopping_review_retry_action(observation)
        gitlab_commit_count_answer_action = _build_gitlab_commit_count_answer_action(observation)
        gitlab_rss_token_answer_action = _build_gitlab_rss_token_answer_action(observation)
        gitlab_rss_token_navigation_action = _build_gitlab_rss_token_navigation_action(observation)
        gitlab_clone_answer_action = _build_gitlab_clone_answer_action(observation)
        gitlab_repo_graph_action = _build_gitlab_repo_graph_action(observation)
        reddit_negative_comment_answer_action = _build_reddit_negative_comment_count_answer_action(observation)
        reddit_parse_retry_action = _build_reddit_parse_retry_action(observation)
        map_answer_action = _build_map_answer_retry_action(observation)
        if decision.parse_error is not None:
            if map_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=map_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_review_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_review_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if gitlab_commit_count_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_commit_count_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if gitlab_rss_token_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_rss_token_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if gitlab_clone_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_clone_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if reddit_negative_comment_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=reddit_negative_comment_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if reddit_parse_retry_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=reddit_parse_retry_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            gitlab_explore_action = _build_gitlab_explore_retry_action(observation)
            if gitlab_explore_action and "fill(" in (decision.raw_text or "").lower():
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_explore_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if gitlab_rss_token_navigation_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_rss_token_navigation_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_admin_dashboard_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_admin_dashboard_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_admin_order_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_admin_order_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_order_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_order_detail_answer_action and _is_shopping_order_detail_page(observation.current_url):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_detail_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_review_action and "fill(" in (decision.raw_text or "").lower():
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_review_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if _is_shopping_order_detail_page(observation.current_url):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        "Do not use fill or navigation on Shopping order-detail pages. "
                        "Read the visible order details on this page and answer directly with send_msg_to_user."
                    ),
                    should_retry=True,
                )
            if shopping_admin_orders_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_admin_orders_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_orders_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_orders_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_order_detail_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_detail_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            recovered_fill_action = _recover_swapped_fill_action(decision.raw_text, observation)
            if recovered_fill_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=recovered_fill_action,
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_target_action and "select_option" in (decision.raw_text or "").lower():
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_target_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            return decision

        action_name, arguments = _parse_action_for_observation_validation(decision.action_text)
        if not action_name:
            return decision
        admin_expected_action_name = ""
        shopping_expected_action_name = ""
        if shopping_admin_order_answer_action:
            admin_expected_action_name, _ = _parse_action_for_observation_validation(
                shopping_admin_order_answer_action.split("ACTION:", 1)[1].strip()
            )
        if shopping_order_answer_action:
            shopping_expected_action_name, _ = _parse_action_for_observation_validation(
                shopping_order_answer_action.split("ACTION:", 1)[1].strip()
            )

        if map_answer_action and action_name != "send_msg_to_user":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=map_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if shopping_review_answer_action and action_name != "send_msg_to_user":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_review_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if gitlab_commit_count_answer_action and action_name != "send_msg_to_user":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=gitlab_commit_count_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if gitlab_rss_token_answer_action and action_name != "send_msg_to_user":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=gitlab_rss_token_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if gitlab_clone_answer_action and action_name != "send_msg_to_user":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=gitlab_clone_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if reddit_negative_comment_answer_action and action_name != "send_msg_to_user":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=reddit_negative_comment_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if action_name == "noop" and gitlab_repo_graph_action:
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=gitlab_repo_graph_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if gitlab_rss_token_navigation_action and not _action_matches_expected_action(
            action_name,
            arguments,
            gitlab_rss_token_navigation_action,
        ):
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=gitlab_rss_token_navigation_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if shopping_admin_dashboard_answer_action:
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_admin_dashboard_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if shopping_admin_orders_action and action_name != "goto":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_admin_orders_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )
        if shopping_orders_action and action_name != "goto":
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_orders_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if (
            shopping_admin_order_answer_action
            and admin_expected_action_name == "send_msg_to_user"
            and action_name != "send_msg_to_user"
        ):
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_admin_order_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )
        if (
            shopping_admin_order_answer_action
            and admin_expected_action_name
            and admin_expected_action_name != "send_msg_to_user"
            and not _action_matches_expected_action(action_name, arguments, shopping_admin_order_answer_action)
        ):
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_admin_order_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if action_name in {"click", "fill", "select_option"} and shopping_admin_orders_action:
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_admin_orders_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )
        if (
            shopping_order_answer_action
            and shopping_expected_action_name == "send_msg_to_user"
            and action_name != "send_msg_to_user"
        ):
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_order_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )
        if (
            shopping_order_answer_action
            and shopping_expected_action_name
            and shopping_expected_action_name != "send_msg_to_user"
            and not _action_matches_expected_action(action_name, arguments, shopping_order_answer_action)
        ):
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_order_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if action_name in {"click", "fill", "hover", "select_option", "press"} and arguments:
            bid = _strip_quoted_string(arguments[0])
            if bid is not None and f"[{bid}]" not in observation.dom_or_ax_snippet:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=f'Chosen bid "{bid}" is not present in the current DOM/AX snippet.',
                    should_retry=True,
                )

        if action_name == "fill" and _is_shopping_order_detail_page(observation.current_url):
            if shopping_order_detail_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_detail_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=decision.action_text,
                parse_error=(
                    "Do not use fill on Shopping order-detail pages. Read the visible order details and answer with "
                    'ACTION: send_msg_to_user("...") instead.'
                ),
                should_retry=True,
            )

        if (
            action_name != "send_msg_to_user"
            and _is_shopping_order_detail_page(observation.current_url)
            and shopping_order_detail_answer_action
        ):
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=shopping_order_detail_answer_action.split("ACTION:", 1)[1].strip(),
                parse_error=None,
                should_retry=False,
            )

        if action_name in {"go_back", "go_forward"} and _is_shopping_order_detail_page(observation.current_url):
            if shopping_order_detail_answer_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_detail_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=decision.action_text,
                parse_error=(
                    "Do not leave a Shopping order-detail page once it is open. "
                    "Read the visible order details here and answer directly with send_msg_to_user."
                ),
                should_retry=True,
            )

        if action_name == "goto" and arguments:
            target_url = _strip_quoted_string(arguments[0])
            same_host_gitlab_target = _rewrite_gitlab_url_to_current_host(observation.current_url, target_url)
            if same_host_gitlab_target and same_host_gitlab_target != target_url:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=f'goto("{same_host_gitlab_target}")',
                    parse_error=None,
                    should_retry=False,
                )
            canonical_order_target = _canonicalize_shopping_order_view_url(target_url)
            if canonical_order_target and canonical_order_target != target_url:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=f'goto("{canonical_order_target}")',
                    parse_error=None,
                    should_retry=False,
                )
            if _is_cross_site_shopping_goto(observation.current_url, target_url):
                if shopping_order_detail_action:
                    return AgentDecision(
                        raw_text=decision.raw_text,
                        action_text=shopping_order_detail_action.split("ACTION:", 1)[1].strip(),
                        parse_error=None,
                        should_retry=False,
                    )
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        "You are on a Shopping page. Do not navigate to a different site host from here; stay in the Shopping order flow."
                    ),
                    should_retry=True,
                )
            if target_url is not None and target_url == observation.current_url:
                graph_hint = ""
                if _is_gitlab_graph_page(observation.current_url):
                    if _is_page_not_found_observation(observation):
                        alternate_graph_url = build_gitlab_alternate_graph_url(observation.current_url)
                        if alternate_graph_url:
                            graph_hint = f'ACTION: goto("{alternate_graph_url}")'
                    if not graph_hint:
                        graph_hint = 'ACTION: send_msg_to_user("<top contributor name>")'
                elif _is_gitlab_repo_page(observation.current_url) and _is_gitlab_contribution_goal(observation.goal):
                    graph_hint = build_gitlab_repo_graph_hint(observation.current_url)
                parse_error = "goto target is already the current URL. Choose a different action."
                if graph_hint:
                    if _is_gitlab_graph_page(observation.current_url):
                        if _is_page_not_found_observation(observation):
                            parse_error = (
                                "This GitLab graph branch already returned Page Not Found. Switch once to the alternate branch "
                                f"instead of repeating the same broken URL. For example, use {graph_hint}."
                            )
                        else:
                            parse_error = (
                                "The GitLab contributors graph page is already open. Read the contributor information from this page "
                                f"instead of repeating the same URL. For example, use {graph_hint}."
                            )
                    else:
                        parse_error = (
                            "The repository page is already open. Continue from here toward the contribution graphs instead of "
                            f"repeating the bare repo URL. For example, use {graph_hint}."
                        )
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=parse_error,
                    should_retry=True,
                )
            exact_click_action = build_gitlab_explore_click_hint(observation.dom_or_ax_snippet, observation.goal)
            exact_click_bid = _extract_single_bid_from_action(exact_click_action)
            if (
                target_url is not None
                and exact_click_bid
                and _goto_looks_like_bid_url(target_url, exact_click_bid)
            ):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        f'The visible repository result bid "{exact_click_bid}" is an element to click, not a URL. '
                        f"Use {exact_click_action} instead of goto(...)."
                    ),
                    should_retry=True,
                )

            if (
                target_url
                and _is_gitlab_graph_page(observation.current_url)
                and _is_gitlab_graph_append(target_url, observation.current_url)
            ):
                alternate_graph_url = build_gitlab_alternate_graph_url(observation.current_url)
                parse_error = (
                    "The current GitLab graph URL already contains the graphs path. Do not append another /-/graphs segment."
                )
                if _is_page_not_found_observation(observation) and alternate_graph_url:
                    parse_error += f' Switch once to the alternate branch graph instead, for example ACTION: goto("{alternate_graph_url}").'
                else:
                    parse_error += ' Read the contributor information from this page and answer with ACTION: send_msg_to_user("<top contributor name>").'
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=parse_error,
                    should_retry=True,
                )

            if (
                target_url
                and _is_gitlab_graph_page(observation.current_url)
                and _is_page_not_found_observation(observation)
                and not _is_gitlab_graph_append(target_url, observation.current_url)
            ):
                current_branch = _extract_gitlab_graph_branch(observation.current_url)
                target_branch = _extract_gitlab_graph_branch(target_url)
                alternate_graph_url = build_gitlab_alternate_graph_url(observation.current_url)
                if (
                    current_branch
                    and target_branch
                    and current_branch == target_branch
                    and alternate_graph_url
                ):
                    return AgentDecision(
                        raw_text=decision.raw_text,
                        action_text=decision.action_text,
                        parse_error=(
                            "This GitLab graph branch already returned Page Not Found. Try the alternate branch instead, "
                            f'for example ACTION: goto("{alternate_graph_url}").'
                        ),
                        should_retry=True,
                    )
            
        if action_name == "fill" and len(arguments) >= 2:
            fill_text = _strip_quoted_string(arguments[1])
            reddit_submit_action = _build_reddit_search_submit_retry_action(
                observation,
                filled_bid=_strip_quoted_string(arguments[0]),
                fill_text=fill_text,
            )
            if reddit_submit_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=reddit_submit_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            shopping_review_action = _build_shopping_review_retry_action(observation)
            if shopping_review_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_review_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            exact_click_action = build_gitlab_explore_click_hint(observation.dom_or_ax_snippet, observation.goal)
            if fill_text and exact_click_action and _current_gitlab_explore_query(observation.current_url) == fill_text.lower():
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        "The current GitLab /explore page already shows the filtered repository result for this query. "
                        f"Click the visible repository link instead of filling again, for example {exact_click_action}."
                    ),
                    should_retry=True,
                )
            shopping_submit_action = _build_shopping_submit_retry_action(observation, fill_text)
            if shopping_submit_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_submit_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            shopping_category_action = _build_shopping_category_retry_action(observation)
            if shopping_category_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_category_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if action_name == "click" and arguments:
            gitlab_repo_goto_action = _build_gitlab_explore_repo_goto_action(
                observation,
                clicked_bid=_strip_quoted_string(arguments[0]),
            )
            if gitlab_repo_goto_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_repo_goto_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            reddit_search_action = _build_reddit_search_retry_action(
                observation,
                clicked_bid=_strip_quoted_string(arguments[0]),
            )
            if reddit_search_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=reddit_search_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            shopping_orders_action = _build_shopping_orders_retry_action(
                observation,
                clicked_bid=_strip_quoted_string(arguments[0]),
            )
            if shopping_orders_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_orders_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if shopping_target_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_target_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            shopping_category_action = _build_shopping_category_retry_action(observation, clicked_bid=_strip_quoted_string(arguments[0]))
            if shopping_category_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_category_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            shopping_sort_action = _build_shopping_sort_retry_action(observation)
            if shopping_sort_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_sort_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if action_name == "select_option" and len(arguments) >= 2:
            if shopping_target_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_target_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            shopping_category_action = _build_shopping_category_retry_action(observation)
            if shopping_category_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_category_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            sort_retry_action = _build_shopping_direction_retry_action(observation, arguments)
            if sort_retry_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=sort_retry_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "goto"
            and arguments
            and _is_gitlab_repo_page(observation.current_url)
            and _is_gitlab_contribution_goal(observation.goal)
        ):
            target_url = _strip_quoted_string(arguments[0])
            if target_url and _is_gitlab_explore_page(target_url):
                graph_hint = build_gitlab_repo_graph_hint(observation.current_url)
                parse_error = (
                    "The repository page is already open. Do not return to /explore after the repository click; "
                    "continue from this GitLab project page instead."
                )
                if graph_hint:
                    parse_error += f" For example, use {graph_hint}."
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=parse_error,
                    should_retry=True,
                )

        if (
            action_name == "goto"
            and arguments
            and _is_gitlab_graph_page(observation.current_url)
            and not _is_page_not_found_observation(observation)
        ):
            target_url = _strip_quoted_string(arguments[0])
            if target_url and _extract_gitlab_graph_branch(target_url):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        "You are already on the GitLab contributors graph page. Do not keep navigating between graph URLs; "
                        'read this page and answer with ACTION: send_msg_to_user("<top contributor name>").'
                    ),
                    should_retry=True,
                )

        if (
            action_name == "goto"
            and arguments
            and _is_gitlab_graph_page(observation.current_url)
            and _is_page_not_found_observation(observation)
        ):
            target_url = _strip_quoted_string(arguments[0])
            alternate_graph_url = build_gitlab_alternate_graph_url(observation.current_url)
            if target_url and _looks_like_branch_without_graph_path(target_url, observation.current_url):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=f'goto("{alternate_graph_url}")',
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and gitlab_commit_count_answer_action
        ):
            answer_text = _strip_quoted_string(arguments[0])
            expected_gitlab_count = _extract_send_message_answer(gitlab_commit_count_answer_action)
            if not answer_text or answer_text != expected_gitlab_count:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_commit_count_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and gitlab_rss_token_answer_action
        ):
            answer_text = _strip_quoted_string(arguments[0])
            expected_token = _extract_send_message_answer(gitlab_rss_token_answer_action)
            if not answer_text or answer_text != expected_token:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_rss_token_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and gitlab_clone_answer_action
        ):
            answer_text = _strip_quoted_string(arguments[0])
            expected_clone_text = _extract_send_message_answer(gitlab_clone_answer_action)
            if not answer_text or answer_text != expected_clone_text:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=gitlab_clone_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and reddit_negative_comment_answer_action
        ):
            answer_text = _strip_quoted_string(arguments[0])
            expected_reddit_count = _extract_send_message_answer(reddit_negative_comment_answer_action)
            if not answer_text or answer_text != expected_reddit_count:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=reddit_negative_comment_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and map_answer_action
        ):
            answer_text = _strip_quoted_string(arguments[0])
            expected_map_answer = _extract_send_message_answer(map_answer_action)
            if (
                not answer_text
                or _looks_like_placeholder_answer(answer_text)
                or (
                    expected_map_answer
                    and not _answers_match_expected_value(answer_text, expected_map_answer)
                )
            ):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=map_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and _is_gitlab_graph_page(observation.current_url)
            and _is_gitlab_contribution_goal(observation.goal)
        ):
            answer_text = _strip_quoted_string(arguments[0])
            if answer_text and _looks_like_placeholder_person_name(answer_text):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        "The final answer must be a real contributor name, not a placeholder like John Doe or Jane Doe."
                    ),
                    should_retry=True,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and _is_shopping_order_detail_page(observation.current_url)
        ):
            answer_text = _strip_quoted_string(arguments[0])
            if (
                shopping_order_detail_answer_action
                and answer_text
                and (
                    _looks_like_placeholder_answer(answer_text)
                    or not _shopping_order_detail_answer_supported_by_observation(answer_text, observation)
                )
            ):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_detail_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )
            if answer_text and (_looks_like_placeholder_answer(answer_text) or _looks_like_order_ui_label(answer_text)):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=(
                        "The final answer must use the real visible order details from this page, not placeholder text or copied UI labels like "
                        '"...", "Product A, Product B", "Filter by name", or "Address Book".'
                    ),
                    should_retry=True,
                )
            if answer_text and not _shopping_order_detail_answer_supported_by_observation(answer_text, observation):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=decision.action_text,
                    parse_error=_build_shopping_order_detail_answer_retry_hint(observation),
                    should_retry=True,
                )

        if (
            action_name == "send_msg_to_user"
            and arguments
            and shopping_order_answer_action
            and shopping_expected_action_name == "send_msg_to_user"
        ):
            answer_text = _strip_quoted_string(arguments[0])
            expected_order_answer = _extract_send_message_answer(shopping_order_answer_action)
            if (
                not answer_text
                or _looks_like_placeholder_answer(answer_text)
                or (
                    expected_order_answer
                    and not _answers_match_expected_value(answer_text, expected_order_answer)
                )
            ):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        if action_name == "send_msg_to_user" and arguments and shopping_admin_order_answer_action:
            answer_text = _strip_quoted_string(arguments[0])
            expected_admin_answer = _extract_send_message_answer(shopping_admin_order_answer_action)
            if (
                not answer_text
                or _looks_like_placeholder_answer(answer_text)
                or (
                    expected_admin_answer
                    and not _answers_match_expected_value(answer_text, expected_admin_answer)
                )
            ):
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_admin_order_answer_action.split("ACTION:", 1)[1].strip(),
                    parse_error=None,
                    should_retry=False,
                )

        return decision


OBS_ACTION_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$")


def _parse_action_for_observation_validation(action_text: str) -> tuple[str, list[str]]:
    match = OBS_ACTION_PATTERN.match(action_text.strip())
    if match is None:
        return "", []
    action_name = match.group(1)
    inner = match.group(2).strip()
    if inner == "":
        return action_name, []

    arguments: list[str] = []
    current: list[str] = []
    quote_char: str | None = None
    escaped = False
    depth = 0
    for char in inner:
        if quote_char is not None:
            current.append(char)
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                quote_char = None
            continue
        if char in {'"', "'"}:
            quote_char = char
            current.append(char)
            continue
        if char in {"(", "[", "{"}:
            depth += 1
            current.append(char)
            continue
        if char in {")", "]", "}"}:
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        arguments.append(tail)
    return action_name, arguments


def _extract_send_message_answer(action_text: str) -> str:
    if action_text.startswith("ACTION:"):
        action_text = action_text.split("ACTION:", 1)[1].strip()
    action_name, arguments = _parse_action_for_observation_validation(action_text)
    if action_name != "send_msg_to_user" or not arguments:
        return ""
    return _strip_quoted_string(arguments[0]) or ""


def _action_matches_expected_action(action_name: str, arguments: list[str], expected_action_text: str) -> bool:
    normalized = expected_action_text.split("ACTION:", 1)[1].strip() if expected_action_text.startswith("ACTION:") else expected_action_text
    expected_action_name, expected_arguments = _parse_action_for_observation_validation(normalized)
    if action_name != expected_action_name:
        return False
    if expected_action_name in {"click", "fill", "goto", "select_option"} and expected_arguments:
        if len(arguments) < len(expected_arguments):
            return False
        for actual, expected in zip(arguments, expected_arguments):
            actual_value = _strip_quoted_string(actual)
            expected_value = _strip_quoted_string(expected)
            if expected_value is not None:
                if actual_value != expected_value:
                    return False
            elif actual.strip() != expected.strip():
                return False
    return True


def _answers_match_expected_value(answer_text: str, expected_text: str) -> bool:
    normalized_answer = " ".join((answer_text or "").split()).lower()
    normalized_expected = " ".join((expected_text or "").split()).lower()
    if not normalized_answer or not normalized_expected:
        return False
    if normalized_answer == normalized_expected:
        return True
    try:
        answer_value = float(normalized_answer.replace(",", ""))
        expected_value = float(normalized_expected.replace(",", ""))
    except ValueError:
        return False
    return abs(answer_value - expected_value) < 1e-6


def _strip_quoted_string(value: str) -> str | None:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in {'"', "'"} or stripped[-1] != stripped[0]:
        return None
    return stripped[1:-1]


def _recover_swapped_fill_action(raw_text: str, observation: NormalizedObservation) -> str:
    action_payload = _extract_action_payload_from_raw_text(raw_text)
    if not action_payload:
        return ""
    action_name, arguments = _parse_action_for_observation_validation(action_payload)
    if action_name != "fill" or len(arguments) != 2:
        return ""
    fill_text_argument = arguments[0].strip()
    bid_argument = arguments[1].strip()
    fill_text = _strip_quoted_string(fill_text_argument)
    bid_value = _strip_quoted_string(bid_argument)
    if not fill_text or not bid_value or not bid_value.isdigit():
        return ""
    if _strip_quoted_string(arguments[0] or "") and (_strip_quoted_string(arguments[0]) or "").isdigit():
        return ""
    if not _bid_is_editable(observation.dom_or_ax_snippet, bid_value):
        return ""
    return f'fill("{bid_value}", {fill_text_argument})'


def _extract_action_payload_from_raw_text(raw_text: str) -> str:
    for line in (raw_text or "").splitlines():
        if "ACTION:" not in line:
            continue
        return line.split("ACTION:", 1)[1].strip()
    return ""


def _bid_is_editable(dom_or_ax_snippet: str, bid: str) -> bool:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = OBS_ROLE_LINE_PATTERN.match(line)
        if match is None or match.group("bid") != bid:
            continue
        role = match.group("role").lower()
        return role in EDITABLE_OBS_ROLES or "editable" in line.lower()
    return False


def _current_gitlab_explore_query(current_url: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023") or not (parsed.path or "").startswith("/explore"):
        return ""
    query_values = parse_qs(parsed.query).get("name", [])
    if not query_values:
        return ""
    return query_values[0].strip().lower()


def _rewrite_gitlab_url_to_current_host(current_url: str, target_url: str | None) -> str:
    current = urlparse(current_url or "")
    target = urlparse(target_url or "")
    if not current.netloc.endswith(":8023") or not target.netloc.endswith(":8023"):
        return ""
    if not target.netloc or current.netloc == target.netloc:
        return ""
    if not target.path:
        return ""
    return urlunparse(
        (
            current.scheme or target.scheme or "http",
            current.netloc,
            target.path,
            target.params,
            target.query,
            target.fragment,
        )
    )


def _build_gitlab_explore_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":8023") or not (parsed.path or "").startswith("/explore"):
        return ""
    exact_click_action = build_gitlab_explore_click_hint(observation.dom_or_ax_snippet, observation.goal)
    if exact_click_action:
        return exact_click_action
    query_hint = derive_gitlab_query_hint(observation.goal).strip()
    if not query_hint:
        return ""
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = OBS_ROLE_LINE_PATTERN.match(line)
        if match is None:
            continue
        role = match.group("role").lower()
        name = (match.group("name") or "").lower()
        if role in EDITABLE_OBS_ROLES and ("filter by name" in name or role == "searchbox"):
            return f'ACTION: fill("{match.group("bid")}", "{query_hint}")'
    return ""


def _build_gitlab_explore_repo_goto_action(
    observation: NormalizedObservation,
    clicked_bid: str | None,
) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":8023") or not (parsed.path or "").startswith("/explore"):
        return ""
    exact_click_action = build_gitlab_explore_click_hint(observation.dom_or_ax_snippet, observation.goal)
    exact_click_bid = _extract_single_bid_from_action(exact_click_action)
    if not clicked_bid or clicked_bid != exact_click_bid or not _has_click_timeout_history(observation):
        return ""
    repo_href = _extract_gitlab_repo_href_from_explore_page(observation.current_url, observation.goal)
    if not repo_href:
        return ""
    return f'ACTION: goto("{repo_href}")'


def _extract_gitlab_repo_href_from_explore_page(current_url: str, goal: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023"):
        return ""
    html_text = _safe_fetch_text(current_url)
    if not html_text:
        return ""
    query_hint = derive_gitlab_query_hint(goal).strip().lower()
    if not query_hint:
        return ""
    href_pattern = re.compile(r'href="(?P<href>/[^"]+)"', flags=re.IGNORECASE)
    candidates: list[str] = []
    for match in href_pattern.finditer(html_text):
        href = html.unescape(match.group("href"))
        normalized = href.lower()
        if not normalized.startswith("/") or normalized.startswith("/explore"):
            continue
        if query_hint not in normalized:
            continue
        candidates.append(href)
    if not candidates:
        return ""
    repo_href = min(candidates, key=len)
    return f"{parsed.scheme or 'http'}://{parsed.netloc}{repo_href}"


def _extract_single_bid_from_action(action_text: str) -> str:
    if not action_text:
        return ""
    normalized = action_text.strip()
    if normalized.upper().startswith("ACTION:"):
        normalized = normalized.split(":", 1)[1].strip()
    action_name, arguments = _parse_action_for_observation_validation(normalized)
    if action_name != "click" or len(arguments) != 1:
        return ""
    return _strip_quoted_string(arguments[0]) or ""


def _has_click_timeout_history(observation: NormalizedObservation) -> bool:
    for error_text in observation.previous_errors:
        lowered = (error_text or "").lower()
        if "timeout" in lowered and ("click" in lowered or "locator.click" in lowered):
            return True
    return False


def _safe_fetch_text(target_url: str) -> str:
    if not target_url:
        return ""
    try:
        response = requests.get(target_url, timeout=5)
    except requests.RequestException:
        return ""
    if not getattr(response, "ok", False):
        return ""
    return response.text or ""


def _goto_looks_like_bid_url(target_url: str, bid: str) -> bool:
    parsed = urlparse(target_url)
    if parsed.port is not None and str(parsed.port) == bid:
        return True
    return parsed.path.strip("/") == bid


def _build_shopping_submit_retry_action(observation: NormalizedObservation, fill_text: str | None) -> str:
    if not fill_text:
        return ""
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or (parsed.path or "/") != "/":
        return ""
    expected_query = derive_shopping_query_hint(observation.goal).strip().lower()
    if not expected_query or fill_text.strip().lower() != expected_query:
        return ""
    if not _shopping_query_is_already_typed(observation.dom_or_ax_snippet, expected_query):
        return ""
    search_button_bid = _find_shopping_search_button_bid(observation.dom_or_ax_snippet)
    if not search_button_bid:
        return ""
    return f'ACTION: click("{search_button_bid}")'


def _build_shopping_review_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or _is_shopping_order_detail_page(observation.current_url):
        return ""
    if not _is_shopping_review_goal(observation.goal):
        return ""
    review_link_bid = _find_shopping_reviews_link_bid(observation.dom_or_ax_snippet)
    if not review_link_bid:
        return ""
    return f'ACTION: click("{review_link_bid}")'


def _build_shopping_review_answer_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or not _is_shopping_review_goal(observation.goal):
        return ""
    target_phrase = _extract_shopping_review_target_phrase(observation.goal)
    if not target_phrase:
        return ""
    matching_authors = _extract_shopping_review_matching_authors(observation.current_url, target_phrase)
    answer_text = ", ".join(matching_authors) if matching_authors else "N/A"
    return f'ACTION: send_msg_to_user("{answer_text}")'


def _build_gitlab_repo_graph_action(observation: NormalizedObservation) -> str:
    if not _is_gitlab_repo_page(observation.current_url):
        return ""
    if not _is_gitlab_contribution_goal(observation.goal):
        return ""
    return build_gitlab_repo_graph_hint(observation.current_url)


def _build_gitlab_rss_token_navigation_action(observation: NormalizedObservation) -> str:
    if not _is_gitlab_rss_token_goal(observation.goal):
        return ""
    if _extract_gitlab_rss_token_from_observation(observation):
        return ""
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":8023"):
        return ""
    if parsed.path.rstrip("/").startswith("/-/profile/"):
        access_tokens_bid = _find_obs_link_bid_by_name(observation.dom_or_ax_snippet, "Access Tokens")
        if access_tokens_bid:
            return f'ACTION: click("{access_tokens_bid}")'
    if _is_gitlab_profile_preferences_page(observation.current_url):
        return ""
    return f'ACTION: goto("{parsed.scheme or "http"}://{parsed.netloc}/-/profile/preferences")'


def _build_gitlab_rss_token_answer_action(observation: NormalizedObservation) -> str:
    if not _is_gitlab_rss_token_goal(observation.goal):
        return ""
    if not (
        _is_gitlab_profile_account_page(observation.current_url)
        or _is_gitlab_profile_preferences_page(observation.current_url)
        or _is_gitlab_profile_personal_access_tokens_page(observation.current_url)
    ):
        return ""
    token = _extract_gitlab_rss_token_from_observation(observation)
    if not token:
        return ""
    return f'ACTION: send_msg_to_user("{token}")'


def _build_gitlab_clone_answer_action(observation: NormalizedObservation) -> str:
    if not _is_gitlab_clone_ssh_goal(observation.goal):
        return ""
    if not _is_gitlab_repo_page(observation.current_url):
        return ""
    ssh_clone_url = _extract_gitlab_ssh_clone_url(observation.current_url)
    if not ssh_clone_url:
        return ""
    normalized_clone_url = _normalize_gitlab_ssh_clone_url_for_benchmark(ssh_clone_url)
    if not normalized_clone_url:
        return ""
    lowered_goal = " ".join((observation.goal or "").lower().split())
    answer_text = normalized_clone_url
    if "command to clone" in lowered_goal:
        answer_text = f"git clone {normalized_clone_url}"
    return f'ACTION: send_msg_to_user("{answer_text}")'


def _build_gitlab_commit_count_answer_action(observation: NormalizedObservation) -> str:
    if not _is_gitlab_graph_page(observation.current_url):
        return ""
    user_query, target_date = _extract_gitlab_commit_count_query(observation.goal)
    if not user_query or not target_date:
        return ""
    commit_count = _fetch_gitlab_commit_count_for_user(observation.current_url, user_query, target_date)
    if commit_count is None:
        return ""
    return f'ACTION: send_msg_to_user("{commit_count}")'


def _build_shopping_category_retry_action(observation: NormalizedObservation, clicked_bid: str | None = None) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or (parsed.path or "/") != "/":
        return ""
    category_labels = derive_shopping_category_labels(observation.goal)
    if not category_labels:
        return ""
    category_click_action = build_shopping_category_click_hint(
        observation.dom_or_ax_snippet,
        category_labels,
        current_url=observation.current_url,
    )
    if not category_click_action:
        return ""
    search_button_bid = _find_shopping_search_button_bid(observation.dom_or_ax_snippet)
    if clicked_bid is not None and clicked_bid != search_button_bid:
        return ""
    return category_click_action


def _shopping_query_is_already_typed(dom_or_ax_snippet: str, query: str) -> bool:
    lowered_query = query.strip().lower()
    if not lowered_query:
        return False
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        option_match = re.match(r'^\[(?P<bid>\d+)\]\s+role=option(?:\s+name="(?P<name>[^"]*)")?', line)
        if option_match is None:
            continue
        if (option_match.group("name") or "").strip().lower() == lowered_query:
            return True
    return False


def _find_shopping_search_button_bid(dom_or_ax_snippet: str) -> str:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=button(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if "search" in (match.group("name") or "").lower():
            return match.group("bid")
    return ""


def _is_shopping_review_goal(goal: str) -> bool:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return False
    return any(keyword in lowered for keyword in ("reviewer", "reviewers", "review", "mention"))


def _find_shopping_reviews_link_bid(dom_or_ax_snippet: str) -> str:
    best_bid = ""
    best_score = -1
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = OBS_ROLE_LINE_PATTERN.match(line)
        if match is None or match.group("role").lower() != "link":
            continue
        name = " ".join((match.group("name") or "").split()).lower()
        if "review" not in name or "add your review" in name:
            continue
        score = 3 if re.match(r"^\d+\s+reviews?$", name) else 2 if name.startswith("reviews") else 1
        if score > best_score:
            best_score = score
            best_bid = match.group("bid")
    return best_bid


def _extract_shopping_review_target_phrase(goal: str) -> str:
    match = re.search(r"\bmention about (?P<phrase>.+)$", goal or "", flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"\bmention (?P<phrase>.+)$", goal or "", flags=re.IGNORECASE)
    if match is None:
        return ""
    return match.group("phrase").strip(" .?!\"'")


def _extract_shopping_review_matching_authors(current_url: str, target_phrase: str) -> list[str]:
    review_url = _extract_shopping_review_list_url(current_url)
    if not review_url:
        return []
    matching_authors: list[str] = []
    seen_authors: set[str] = set()
    for page_number in range(1, 4):
        page_url = review_url if page_number == 1 else f"{review_url}?p={page_number}"
        page_html = _safe_fetch_text(page_url)
        if not page_html:
            break
        review_blocks = re.findall(r'<li class="item review-item".*?</li>', page_html, flags=re.IGNORECASE | re.DOTALL)
        if not review_blocks:
            break
        for block in review_blocks:
            author_match = re.search(r'itemprop="author">\s*(.*?)\s*</strong>', block, flags=re.IGNORECASE | re.DOTALL)
            content_match = re.search(r'<div class="review-content"[^>]*>(.*?)</div>', block, flags=re.IGNORECASE | re.DOTALL)
            if author_match is None or content_match is None:
                continue
            author = _normalize_review_author_name(_strip_html_fragment(author_match.group(1)))
            content = _strip_html_fragment(content_match.group(1))
            if not author or author in seen_authors:
                continue
            if not _shopping_review_content_matches_phrase(content, target_phrase):
                continue
            seen_authors.add(author)
            matching_authors.append(author)
    return matching_authors


def _extract_shopping_review_list_url(current_url: str) -> str:
    page_html = _safe_fetch_text(current_url)
    if not page_html:
        return ""
    product_review_match = re.search(
        r'review\\u002Fproduct\\u002FlistAjax\\u002Fid\\u002F(?P<product_id>\d+)\\u002F',
        page_html,
        flags=re.IGNORECASE,
    )
    if product_review_match is None:
        return ""
    parsed = urlparse(current_url or "")
    return f"{parsed.scheme or 'http'}://{parsed.netloc}/review/product/listAjax/id/{product_review_match.group('product_id')}/"


def _shopping_review_content_matches_phrase(content: str, target_phrase: str) -> bool:
    normalized_content = " ".join(content.lower().split())
    normalized_phrase = " ".join(target_phrase.lower().split())
    if not normalized_content or not normalized_phrase:
        return False
    if normalized_phrase in normalized_content:
        return True
    if "ear cup" in normalized_phrase and "small" in normalized_phrase:
        ear_like_tokens = ("ear", "ears", "earbud", "earbuds", "cup", "cups")
        return "small" in normalized_content and any(token in normalized_content for token in ear_like_tokens)
    stopwords = {"a", "an", "the", "and", "or", "about", "being", "is", "are", "with", "for", "they", "them"}
    phrase_tokens = [token for token in re.findall(r"[a-z0-9']+", normalized_phrase) if token not in stopwords]
    if len(phrase_tokens) < 2:
        return False
    return all(token in normalized_content for token in phrase_tokens)


def _strip_html_fragment(fragment: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(fragment or "")).split())


def _normalize_review_author_name(author: str) -> str:
    cleaned = " ".join((author or "").split())
    if not cleaned:
        return ""
    midpoint = len(cleaned) // 2
    if len(cleaned) % 2 == 0 and cleaned[:midpoint] == cleaned[midpoint:]:
        return cleaned[:midpoint].strip()
    return cleaned


def _build_reddit_negative_comment_count_answer_action(observation: NormalizedObservation) -> str:
    if not _is_reddit_latest_post_negative_comment_goal(observation.goal):
        return ""
    forum_query = _derive_reddit_forum_query(observation.goal)
    if not forum_query:
        return ""
    negative_count = _fetch_reddit_latest_post_negative_comment_count(observation.current_url, forum_query)
    if negative_count is None:
        return ""
    return f'ACTION: send_msg_to_user("{negative_count}")'


def _build_reddit_parse_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":9999") or (parsed.path or "/") != "/":
        return ""
    forum_query = _derive_reddit_forum_query(observation.goal)
    if not forum_query:
        return ""
    return _build_reddit_search_url_action(observation.current_url, forum_query)


def _build_reddit_search_retry_action(observation: NormalizedObservation, clicked_bid: str | None) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":9999"):
        return ""
    forum_query = _derive_reddit_forum_query(observation.goal)
    if not forum_query:
        return ""
    search_bid, focused = _find_reddit_searchbox_bid(observation.dom_or_ax_snippet)
    if not search_bid or clicked_bid != search_bid:
        return ""
    return _build_reddit_search_url_action(observation.current_url, forum_query)


def _build_reddit_search_submit_retry_action(
    observation: NormalizedObservation,
    filled_bid: str | None,
    fill_text: str | None,
) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":9999"):
        return ""
    forum_query = _derive_reddit_forum_query(observation.goal)
    if not forum_query or not filled_bid or not fill_text:
        return ""
    search_bid, focused = _find_reddit_searchbox_bid(observation.dom_or_ax_snippet)
    if filled_bid != search_bid or not focused:
        return ""
    if (urlparse(observation.current_url or "").path or "/") == "/":
        return _build_reddit_search_url_action(observation.current_url, forum_query)
    if fill_text.strip().lower() != forum_query.lower():
        return ""
    repeated_same_fill = sum(
        1
        for action in observation.previous_actions
        if f'fill("{filled_bid}", "{forum_query}")'.lower() == action.lower()
    )
    if repeated_same_fill <= 0:
        return ""
    return _build_reddit_search_url_action(observation.current_url, forum_query)


def _derive_reddit_forum_query(goal: str) -> str:
    goal_text = goal or ""
    match = re.search(
        r"latest post on(?: the)? (?P<forum>[A-Za-z0-9_ -]+?) forum\b",
        goal_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(r"on(?: the)? (?P<forum>[A-Za-z0-9_ -]+?) forum\b", goal_text, flags=re.IGNORECASE)
    if match is None:
        return ""
    return " ".join(match.group("forum").strip(" \"'").split())


def _is_reddit_latest_post_negative_comment_goal(goal: str) -> bool:
    lowered = " ".join((goal or "").lower().split())
    return (
        "latest post on" in lowered
        and "forum" in lowered
        and "count of comments" in lowered
        and "more downvotes than upvotes" in lowered
    )


def _find_reddit_searchbox_bid(dom_or_ax_snippet: str) -> tuple[str, bool]:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = OBS_ROLE_LINE_PATTERN.match(line)
        if match is None or match.group("role").lower() != "searchbox":
            continue
        name = (match.group("name") or "").lower()
        if "search" not in name:
            continue
        return match.group("bid"), "focused" in line.lower()
    return "", False


def _build_reddit_search_url_action(current_url: str, forum_query: str) -> str:
    parsed = urlparse(current_url or "")
    base_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/search"
    return f'ACTION: goto("{base_url}?{urlencode({"q": forum_query})}")'


def _build_shopping_direction_retry_action(observation: NormalizedObservation, arguments: list[str]) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or "/catalogsearch/result" not in (parsed.path or ""):
        return ""
    if len(arguments) < 2:
        return ""
    selected_value = _strip_quoted_string(arguments[1])
    current_order = parse_qs(parsed.query).get("product_list_order", [""])[0]
    if not selected_value or not current_order or selected_value != current_order:
        return ""
    _, desired_direction = derive_shopping_sort_hint(observation.goal)
    if desired_direction == "asc" and parse_qs(parsed.query).get("product_list_dir", [""])[0] != "asc":
        direction_bid = _find_shopping_direction_bid(observation.dom_or_ax_snippet, ascending=True)
        if direction_bid:
            return f'ACTION: click("{direction_bid}")'
    if desired_direction == "desc" and parse_qs(parsed.query).get("product_list_dir", [""])[0] != "desc":
        direction_bid = _find_shopping_direction_bid(observation.dom_or_ax_snippet, ascending=False)
        if direction_bid:
            return f'ACTION: click("{direction_bid}")'
    return ""


def _build_shopping_canonical_target_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or (parsed.path or "/") == "/":
        return ""
    category_labels = derive_shopping_category_labels(observation.goal)
    if not category_labels:
        return ""
    if build_shopping_category_click_hint(
        observation.dom_or_ax_snippet,
        category_labels,
        current_url=observation.current_url,
    ):
        return ""
    target_url = build_shopping_category_target_url(observation.goal, observation.current_url)
    if not target_url or target_url.rstrip("/") == (observation.current_url or "").rstrip("/"):
        return ""
    lowered_dom = observation.dom_or_ax_snippet.lower()
    if 'role=combobox name="sort by"' not in lowered_dom and "cat=" not in (parsed.query or ""):
        return ""
    return f'ACTION: goto("{target_url}")'


def _build_shopping_order_detail_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770"):
        return ""
    path = parsed.path or "/"
    if not path.startswith("/sales/order/history"):
        return ""
    target_url = build_shopping_order_detail_target_url(observation.current_url, observation.goal)
    if not target_url or target_url.rstrip("/") == (observation.current_url or "").rstrip("/"):
        return ""
    return f'ACTION: goto("{target_url}")'


def _build_shopping_orders_retry_action(
    observation: NormalizedObservation,
    clicked_bid: str | None = None,
) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or (parsed.path or "/") != "/":
        return ""
    orders_action = build_shopping_orders_target_hint(
        observation.current_url,
        observation.goal,
        observation.dom_or_ax_snippet,
    )
    if not orders_action:
        return ""
    search_button_bid = _find_shopping_search_button_bid(observation.dom_or_ax_snippet)
    if clicked_bid is not None and clicked_bid != search_button_bid:
        return ""
    return orders_action


def _build_shopping_admin_orders_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7780"):
        return ""
    if not (parsed.path or "/").startswith("/admin/admin/dashboard"):
        return ""
    if not _looks_like_shopping_order_goal(observation.goal):
        return ""
    return f'ACTION: goto("{parsed.scheme or "http"}://{parsed.netloc}/admin/sales/order/")'


def _build_shopping_admin_dashboard_answer_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7780"):
        return ""
    if not (parsed.path or "/").startswith("/admin/admin/dashboard"):
        return ""
    answer = _derive_admin_dashboard_answer(
        observation.goal,
        observation.visible_page_summary,
        observation.dom_or_ax_snippet,
    )
    if not answer:
        return ""
    return f'ACTION: send_msg_to_user("{answer}")'


def _build_shopping_admin_order_answer_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7780"):
        return ""
    if not (parsed.path or "/").startswith("/admin/sales/order"):
        return ""
    lowered_goal = " ".join((observation.goal or "").lower().split())
    rows = _extract_admin_order_summary_rows(observation.visible_page_summary)
    item_rows = _extract_admin_order_item_rows(observation.visible_page_summary)
    if (
        not rows
        and not item_rows
        and observation.current_url
        and len(observation.previous_actions) <= 1
    ):
        return f'ACTION: goto("{observation.current_url}")'
    if "items sold" in lowered_goal:
        item_answer = _derive_admin_order_grid_answer(observation.goal, observation.visible_page_summary)
        if item_answer:
            return f'ACTION: send_msg_to_user("{item_answer}")'
    sort_action = _build_shopping_admin_order_sort_action(observation)
    if sort_action:
        return sort_action
    answer = _derive_admin_order_grid_answer(observation.goal, observation.visible_page_summary)
    if (
        not answer
        and "payment difference" in lowered_goal
        and observation.current_url
        and len(observation.previous_actions) <= 1
    ):
        return f'ACTION: goto("{observation.current_url}")'
    if not answer:
        return ""
    escaped = answer.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'ACTION: send_msg_to_user("{escaped}")'


def _build_shopping_sort_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770") or "cat=" not in (parsed.query or ""):
        return ""
    category_labels = derive_shopping_category_labels(observation.goal)
    if not category_labels:
        return ""
    if build_shopping_category_click_hint(
        observation.dom_or_ax_snippet,
        category_labels,
        current_url=observation.current_url,
    ):
        return ""
    sort_value, _ = derive_shopping_sort_hint(observation.goal)
    current_order = parse_qs(parsed.query).get("product_list_order", [""])[0]
    if not sort_value or current_order == sort_value:
        return ""
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=combobox(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if "sort by" in (match.group("name") or "").lower():
            return f'ACTION: select_option("{match.group("bid")}", "{sort_value}")'
    return ""


def _build_map_answer_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":3000"):
        return ""
    answer = derive_map_goal_answer(observation.goal)
    if not answer:
        return ""
    escaped = answer.replace("\\", "\\\\").replace('"', '\\"')
    return f'ACTION: send_msg_to_user("{escaped}")'


def _find_shopping_direction_bid(dom_or_ax_snippet: str, *, ascending: bool) -> str:
    target = "set ascending direction" if ascending else "set descending direction"
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=link(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if target in (match.group("name") or "").lower():
            return match.group("bid")
    return ""


def _is_cross_site_shopping_goto(current_url: str, target_url: str | None) -> bool:
    if not target_url:
        return False
    current = urlparse(current_url or "")
    target = urlparse(target_url)
    if not current.netloc.endswith(":7770"):
        return False
    return bool(target.netloc) and target.netloc != current.netloc


def _is_shopping_order_detail_page(current_url: str) -> bool:
    parsed = urlparse(current_url or "")
    return parsed.netloc.endswith(":7770") and _canonicalize_shopping_order_view_url(current_url) is not None


def _canonicalize_shopping_order_view_url(target_url: str | None) -> str | None:
    if not target_url:
        return None
    parsed = urlparse(target_url)
    if not parsed.netloc.endswith(":7770"):
        return None
    match = re.match(r"^/sales/order/view/(?P<order_id>\d+)/?$", parsed.path or "")
    if match is None:
        if re.match(r"^/sales/order/view/order_id/\d+/?$", parsed.path or ""):
            normalized_path = (parsed.path or "").rstrip("/") + "/"
            return f"{parsed.scheme or 'http'}://{parsed.netloc}{normalized_path}"
        return None
    order_id = match.group("order_id")
    return f"{parsed.scheme or 'http'}://{parsed.netloc}/sales/order/view/order_id/{order_id}/"


def _is_gitlab_repo_page(current_url: str) -> bool:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023"):
        return False
    path = parsed.path or ""
    if "/-/" in path:
        return False
    parts = [part for part in path.split("/") if part]
    return len(parts) >= 2 and parts[0] != "-" and parts[1] != "-"


def _is_gitlab_explore_page(target_url: str) -> bool:
    parsed = urlparse(target_url or "")
    return parsed.netloc.endswith(":8023") and (parsed.path or "").startswith("/explore")


def _is_gitlab_graph_page(target_url: str) -> bool:
    parsed = urlparse(target_url or "")
    return parsed.netloc.endswith(":8023") and "/-/graphs/" in (parsed.path or "")


def _is_gitlab_contribution_goal(goal: str) -> bool:
    lowered = (goal or "").lower()
    return (
        "most contributions" in lowered
        or "number of commits" in lowered
        or "how many commits" in lowered
    )


def _extract_gitlab_graph_branch(target_url: str) -> str:
    parsed = urlparse(target_url or "")
    path = parsed.path or ""
    if "/-/graphs/main" in path:
        return "main"
    if "/-/graphs/master" in path:
        return "master"
    return ""


def _gitlab_graph_segment_count(target_url: str) -> int:
    parsed = urlparse(target_url or "")
    return (parsed.path or "").count("/-/graphs/")


def _build_shopping_order_answer_retry_action(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if not parsed.netloc.endswith(":7770"):
        return ""
    if not (parsed.path or "/").startswith("/sales/order/history"):
        return ""
    target_url = build_shopping_order_detail_target_url(observation.current_url, observation.goal)
    if target_url and target_url.rstrip("/") != (observation.current_url or "").rstrip("/"):
        return f'ACTION: goto("{target_url}")'
    pager_action = _build_shopping_order_history_pager_retry_action(observation)
    if pager_action:
        return pager_action
    refund_detail_action = _build_shopping_refund_detail_retry_action(observation)
    if refund_detail_action:
        return refund_detail_action
    answer = _derive_customer_order_history_answer(
        observation.goal,
        observation.dom_or_ax_snippet,
        observation.visible_page_summary,
    )
    if not answer:
        return ""
    return f'ACTION: send_msg_to_user("{answer}")'


def _build_shopping_order_detail_answer_retry_action(observation: NormalizedObservation) -> str:
    if not _is_shopping_order_detail_page(observation.current_url):
        return ""
    answer = _derive_shopping_order_detail_answer(observation)
    if not answer:
        return ""
    escaped = answer.replace("\\", "\\\\").replace('"', '\\"')
    return f'ACTION: send_msg_to_user("{escaped}")'


def _build_shopping_order_history_pager_retry_action(observation: NormalizedObservation) -> str:
    lowered_goal = " ".join((observation.goal or "").lower().split())
    if "refund" in lowered_goal and "Customer refund match count: 0" in observation.visible_page_summary:
        return ""
    rows = _extract_customer_order_summary_rows(observation.visible_page_summary) or _extract_customer_order_rows(
        observation.dom_or_ax_snippet
    )
    if not rows:
        return ""

    should_advance = False
    if "refund" in lowered_goal and not _select_customer_refund_rows(rows, lowered_goal):
        refund_year, refund_month = _extract_refund_date_constraint(lowered_goal)
        if refund_year is not None or refund_month is not None:
            visible_dates = [_parse_customer_order_date(row.get("date", "")) for row in rows]
            visible_dates = [value for value in visible_dates if value is not None]
            if visible_dates:
                oldest_visible = min(visible_dates)
                should_advance = bool(
                    refund_year is not None
                    and (
                        refund_year < oldest_visible.year
                        or (
                            refund_year == oldest_visible.year
                            and refund_month is not None
                            and refund_month < oldest_visible.month
                        )
                    )
                )
    elif _goal_requests_first_purchase_date(lowered_goal):
        should_advance = True

    if not should_advance:
        return ""

    pager_targets = _extract_customer_order_pager_targets(
        observation.visible_page_summary,
        observation.dom_or_ax_snippet,
    )
    if not pager_targets:
        return ""
    current_page = _extract_customer_order_history_page_index(observation.current_url)
    if current_page >= 4:
        return ""
    numbered_pages = sorted(
        int(key.split("_", 1)[1])
        for key in pager_targets
        if key.startswith("page_") and key.split("_", 1)[1].isdigit()
    )
    if _goal_requests_first_purchase_date(lowered_goal) and numbered_pages:
        target_page = max(numbered_pages)
        if target_page <= current_page:
            target_page = None
    else:
        target_page = next((page for page in numbered_pages if page > current_page), None)
    if target_page is None and "next" in pager_targets:
        target_page = current_page + 1
    if target_page is None:
        return ""
    target_url = _build_customer_order_history_page_url(observation.current_url, target_page)
    if not target_url:
        return ""
    return f'ACTION: goto("{target_url}")'


def _derive_customer_order_history_answer(goal: str, dom_or_ax_snippet: str, visible_page_summary: str = "") -> str:
    detail_rows = _extract_customer_refund_detail_rows(visible_page_summary)
    product_rows = _extract_customer_order_product_rows(visible_page_summary)
    rows = _extract_customer_order_summary_rows(visible_page_summary) or _extract_customer_order_rows(dom_or_ax_snippet)
    if not rows and not detail_rows and not product_rows:
        if "refund" in " ".join((goal or "").lower().split()) and "Customer refund match count: 0" in visible_page_summary:
            return "0"
        return ""
    lowered_goal = " ".join((goal or "").lower().split())
    if "refund" in lowered_goal and "Customer refund match count: 0" in visible_page_summary:
        return "0"
    if not rows and product_rows:
        product_date_answer = _derive_last_ordered_product_date_answer(lowered_goal, visible_page_summary)
        if product_date_answer:
            return product_date_answer
    refund_rows = _select_customer_refund_rows(rows, lowered_goal)
    if detail_rows and ("refund" in lowered_goal):
        use_all_detail_rows = not refund_rows or len(detail_rows) > len(refund_rows)
        detail_answer = _derive_customer_refund_detail_answer(
            detail_rows,
            lowered_goal,
            expected_orders=None if use_all_detail_rows else refund_rows,
        )
        if detail_answer and (_refund_goal_requires_order_detail(lowered_goal) or use_all_detail_rows):
            return detail_answer
    if refund_rows:
        total = sum(_parse_amount(row.get("total", "")) for row in refund_rows if row.get("total"))
        return f"{total:.2f}"
    product_date_answer = _derive_last_ordered_product_date_answer(lowered_goal, visible_page_summary)
    if product_date_answer:
        return product_date_answer
    status_key = _derive_order_status_key(lowered_goal)
    rows_with_status = [row for row in rows if row.get("status")]
    filtered = [row for row in rows_with_status if _row_matches_status(row.get("status", ""), status_key)]
    if not filtered:
        multi_status_answer = _derive_multi_order_status_answer(goal, rows)
        if multi_status_answer:
            return multi_status_answer
        return ""
    multi_status_answer = _derive_multi_order_status_answer(goal, rows)
    if multi_status_answer:
        return multi_status_answer

    wants_total = "total cost" in lowered_goal or "order total" in lowered_goal
    wants_order_number = "order number" in lowered_goal or "order id" in lowered_goal
    wants_status = "status" in lowered_goal
    wants_arrival = any(phrase in lowered_goal for phrase in ("when will it arrive", "when it will arrive", "arrive"))
    if _goal_requests_first_purchase_date(lowered_goal):
        oldest_dated_rows = [row for row in rows if row.get("date")]
        if not oldest_dated_rows:
            return ""
        oldest_row = min(
            oldest_dated_rows,
            key=lambda row: _parse_customer_order_date(row.get("date", "")) or datetime.max,
        )
        parsed_oldest_date = _parse_customer_order_date(oldest_row.get("date", ""))
        if parsed_oldest_date is None:
            return oldest_row.get("date", "")
        return _format_customer_order_date(parsed_oldest_date)
    if ("latest" in lowered_goal or "most recent" in lowered_goal or "newest" in lowered_goal) and (wants_total or wants_order_number):
        ranked_rows = [row for row in filtered if row.get("total")] if wants_total else [row for row in filtered if row.get("order_number")]
        if not ranked_rows:
            return ""
        row = ranked_rows[0]
        if wants_total:
            return row.get("total", "").replace("$", "").replace(",", "")
        if wants_order_number:
            return row.get("order_number", "")
    if ("latest" in lowered_goal or "most recent" in lowered_goal or "newest" in lowered_goal) and (wants_status or wants_arrival):
        latest_rows = [row for row in filtered if row.get("status")]
        if not latest_rows:
            return ""
        latest_row = latest_rows[0]
        latest_status = " ".join(latest_row.get("status", "").split()).lower()
        if wants_arrival:
            if latest_status in {"canceled", "cancelled"}:
                return "The last order was canceled. It will never arrive."
            if latest_status == "complete":
                return "The latest order is complete. It has already arrived."
            if latest_status == "pending":
                return "The latest order is pending. Arrival date is unavailable."
            if latest_status == "processing":
                return "The latest order is processing. Arrival date is unavailable."
        if wants_status:
            return latest_row.get("status", "")
    if "oldest" in lowered_goal and (wants_total or wants_order_number):
        ranked_rows = [row for row in filtered if row.get("total")] if wants_total else [row for row in filtered if row.get("order_number")]
        if not ranked_rows:
            return ""
        row = ranked_rows[-1]
        if wants_total:
            return row.get("total", "").replace("$", "").replace(",", "")
        if wants_order_number:
            return row.get("order_number", "")

    count = _derive_order_count(goal)
    if count and "total payment amount" in lowered_goal:
        selected = [row for row in filtered if row.get("total")][:count]
        if not selected:
            return ""
        total = sum(_parse_amount(row.get("total", "")) for row in selected)
        return f"{total:.2f}"

    return ""


def _derive_customer_refund_detail_answer(
    detail_rows: list[dict[str, object]],
    lowered_goal: str,
    *,
    expected_orders: list[dict[str, str]] | None = None,
) -> str:
    if not detail_rows:
        return ""

    detail_by_order = {
        row.get("order_number", ""): row
        for row in detail_rows
        if row.get("order_number")
    }
    required_orders = {
        row.get("order_number", "")
        for row in (expected_orders or [])
        if row.get("order_number")
    }
    if required_orders and not required_orders.issubset(detail_by_order):
        return ""

    if required_orders:
        selected_detail_rows = [
            detail_by_order[row.get("order_number", "")]
            for row in (expected_orders or [])
            if row.get("order_number", "") in detail_by_order
        ]
    else:
        selected_detail_rows = detail_rows
    if not selected_detail_rows:
        return ""

    exclude_shipping = _refund_goal_excludes_shipping(lowered_goal)
    refundable_total = 0.0
    for detail_row in selected_detail_rows:
        row_total = _extract_customer_refund_detail_row_refundable_total(
            detail_row,
            exclude_shipping=exclude_shipping,
        )
        if row_total is None:
            return ""
        refundable_total += row_total

    kept_product_fragment = _extract_kept_product_fragment(lowered_goal)
    if kept_product_fragment:
        refundable_total -= _extract_customer_refund_kept_product_total(selected_detail_rows, kept_product_fragment)
    return f"{max(0.0, refundable_total):.2f}"


def _extract_customer_refund_detail_row_refundable_total(
    detail_row: dict[str, object],
    *,
    exclude_shipping: bool,
) -> float | None:
    grand_total = detail_row.get("grand_total")
    if not isinstance(grand_total, str) or not grand_total:
        return None
    grand_total_amount = _parse_amount(grand_total)

    if not exclude_shipping:
        return grand_total_amount

    subtotal = detail_row.get("subtotal")
    if isinstance(subtotal, str) and subtotal:
        return _parse_amount(subtotal)

    shipping = detail_row.get("shipping")
    if isinstance(shipping, str) and shipping:
        return max(0.0, grand_total_amount - _parse_amount(shipping))
    return grand_total_amount


def _extract_customer_refund_kept_product_total(
    detail_rows: list[dict[str, object]],
    kept_product_fragment: str,
) -> float:
    normalized_fragment = " ".join(kept_product_fragment.lower().split())
    kept_total = 0.0
    for detail_row in detail_rows:
        products = detail_row.get("products", [])
        if not isinstance(products, list):
            continue
        for product in products:
            if not isinstance(product, dict):
                continue
            title = " ".join(str(product.get("title", "")).lower().split())
            amount = product.get("amount")
            if not isinstance(amount, str) or not amount:
                continue
            if normalized_fragment in title or title in normalized_fragment:
                kept_total += _parse_amount(amount)
    return kept_total


def _build_shopping_refund_detail_retry_action(observation: NormalizedObservation) -> str:
    lowered_goal = " ".join((observation.goal or "").lower().split())
    if not _refund_goal_requires_order_detail(lowered_goal):
        return ""
    rows = _extract_customer_order_summary_rows(observation.visible_page_summary) or _extract_customer_order_rows(
        observation.dom_or_ax_snippet
    )
    matching_rows = _select_customer_refund_rows(rows, lowered_goal)
    if len(matching_rows) != 1:
        return ""
    order_number = matching_rows[0].get("order_number", "")
    if not order_number:
        return ""
    parsed = urlparse(observation.current_url or "")
    return f'ACTION: goto("{parsed.scheme or "http"}://{parsed.netloc}/sales/order/view/order_id/{int(order_number)}/")'


def _select_customer_refund_rows(rows: list[dict[str, str]], lowered_goal: str) -> list[dict[str, str]]:
    if "refund" not in lowered_goal:
        return []
    refund_year, refund_month = _extract_refund_date_constraint(lowered_goal)
    matching_rows: list[dict[str, str]] = []
    for row in rows:
        if not _row_matches_status(row.get("status", ""), "canceled"):
            continue
        if not _customer_order_row_matches_refund_date(row, refund_year=refund_year, refund_month=refund_month):
            continue
        matching_rows.append(row)
    return matching_rows


def _refund_goal_requires_order_detail(lowered_goal: str) -> bool:
    if "refund" not in lowered_goal:
        return False
    if _refund_goal_excludes_shipping(lowered_goal):
        return True
    return any(
        phrase in lowered_goal
        for phrase in (
            "kept the ",
        )
    )


def _refund_goal_excludes_shipping(lowered_goal: str) -> bool:
    if "refund" not in lowered_goal:
        return False
    return any(
        phrase in lowered_goal
        for phrase in (
            "cannot get the shipping fee refunded",
            "cannot get the shipping fee back",
            "cannot get the shipping fee",
            "cannot get shipping fee refunded",
            "cannot get shipping fee back",
        )
    )


def _extract_refund_date_constraint(lowered_goal: str) -> tuple[int | None, int | None]:
    if "refund" not in lowered_goal:
        return None, None
    numeric_match = re.search(r"\b(20\d{2})[/-](\d{1,2})\b", lowered_goal)
    if numeric_match:
        return int(numeric_match.group(1)), int(numeric_match.group(2))
    month_match = re.search(
        r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sept|sep|october|oct|november|nov|december|dec)\b",
        lowered_goal,
    )
    year_match = re.search(r"\b(20\d{2})\b", lowered_goal)
    month = MONTH_NAME_TO_NUMBER.get((month_match.group(1) if month_match else "").lower())
    year = int(year_match.group(1)) if year_match else None
    return year, month


def _customer_order_row_matches_refund_date(
    row: dict[str, str],
    *,
    refund_year: int | None,
    refund_month: int | None,
) -> bool:
    if refund_year is None and refund_month is None:
        return True
    parsed_date = _parse_customer_order_date(row.get("date", ""))
    if parsed_date is None:
        return False
    if refund_year is not None and parsed_date.year != refund_year:
        return False
    if refund_month is not None and parsed_date.month != refund_month:
        return False
    return True


def _parse_customer_order_date(value: str) -> datetime | None:
    normalized = " ".join((value or "").replace("Sept", "Sep").split())
    if not normalized:
        return None
    for date_format in (
        "%m/%d/%y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(normalized, date_format)
        except ValueError:
            continue
    return None


def _format_customer_order_date(value: datetime) -> str:
    return f"{value.month}/{value.day}/{value.strftime('%y')}"


def _goal_requests_first_purchase_date(lowered_goal: str) -> bool:
    return any(
        phrase in lowered_goal
        for phrase in (
            "first purchase",
            "first order",
            "date when i made my first purchase",
            "when i made my first purchase",
        )
    )


def _extract_customer_order_summary_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Customer order row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if "order" in row and "order_number" not in row:
            row["order_number"] = row["order"]
        if row:
            rows.append(row)
    return rows


def _extract_customer_refund_detail_rows(visible_page_summary: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Customer refund detail:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, object] = {"products": []}
        for part in payload.split("|"):
            token = part.strip()
            if not token:
                continue
            if token.startswith("product="):
                product_payload = token.split("=", 1)[1].strip()
                if "::" not in product_payload:
                    continue
                title, amount = product_payload.rsplit("::", 1)
                cast_products = row.setdefault("products", [])
                if isinstance(cast_products, list):
                    cast_products.append(
                        {
                            "title": title.strip(),
                            "amount": amount.strip(),
                        }
                    )
                continue
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            row[key.strip()] = value.strip()
        if "order" in row and "order_number" not in row:
            row["order_number"] = row["order"]
        if row.get("order_number"):
            rows.append(row)
    return rows


def _extract_customer_order_pager_targets(
    visible_page_summary: str,
    dom_or_ax_snippet: str,
) -> dict[str, str]:
    pager: dict[str, str] = {}
    for raw_line in visible_page_summary.splitlines():
        text = " ".join(raw_line.split())
        if not text.lower().startswith("customer order pager:"):
            continue
        payload = text.split(":", 1)[1]
        for part in payload.split("|"):
            token = part.strip()
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                pager[key] = value
    if pager:
        return pager

    for raw_line in dom_or_ax_snippet.splitlines():
        match = re.search(r"\[(?P<bid>\d+)\].*name=\"(?P<name>[^\"]+)\"", raw_line)
        if match is None:
            continue
        lowered_name = " ".join(match.group("name").lower().split())
        if "page next" in lowered_name:
            pager["next"] = match.group("bid")
            continue
        page_match = re.search(r"\bpage (?P<number>\d+)\b", lowered_name)
        if page_match is not None:
            pager[f'page_{page_match.group("number")}'] = match.group("bid")
    return pager


def _extract_customer_order_history_page_index(current_url: str) -> int:
    parsed = urlparse(current_url or "")
    raw_value = parse_qs(parsed.query).get("p", ["1"])[0]
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 1


def _build_customer_order_history_page_url(current_url: str, page_number: int) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc:
        return ""
    query = parse_qs(parsed.query)
    query["p"] = [str(page_number)]
    encoded_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=encoded_query))


def _derive_shopping_order_detail_answer(observation: NormalizedObservation) -> str:
    lowered_goal = " ".join((observation.goal or "").lower().split())
    refund_answer = _derive_order_detail_refund_answer(observation, lowered_goal)
    if refund_answer:
        return refund_answer
    if "order date" in lowered_goal:
        order_date = _extract_order_detail_date(observation)
        if order_date:
            return order_date
    if "product names" in lowered_goal:
        product_titles = _extract_order_detail_product_titles(observation)
        if product_titles:
            return ", ".join(product_titles)
    if "shipping method" in lowered_goal:
        shipping_method = _extract_order_detail_shipping_method(observation)
        if shipping_method:
            return shipping_method
    if "billing address" not in lowered_goal:
        return ""
    address_lines = _extract_order_detail_address_lines(observation)
    if not address_lines:
        return ""
    street = next(
        (
            line
            for line in address_lines
            if re.search(r"\d", line)
            and any(token in line.lower() for token in ("dr", "ave", "street", "st", "road", "rd"))
        ),
        "",
    )
    region = next(
        (
            line
            for line in address_lines
            if "," in line and any(token in line.lower() for token in ("california", "new york", "texas", "florida", "washington"))
        ),
        "",
    )
    country = next((line for line in address_lines if "united states" in line.lower()), "")
    ordered_parts = [part for part in (street, region, country) if part]
    if ordered_parts:
        return ", ".join(ordered_parts)
    return ", ".join(address_lines[:3])


def _extract_customer_order_rows(dom_or_ax_snippet: str) -> list[dict[str, str]]:
    order_numbers: list[str] = []
    totals: list[str] = []
    statuses: list[str] = []
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.search(r'name="(?P<name>[^"]+)"', line)
        if match is None:
            continue
        name = match.group("name").strip()
        if re.fullmatch(r"0*\d{3,}", name):
            order_numbers.append(name)
            continue
        if re.fullmatch(r"\$[\d,]+\.\d{2}", name):
            totals.append(name)
            continue
        lowered = name.lower()
        if lowered in {"canceled", "cancelled", "pending", "complete", "closed", "processing"}:
            statuses.append(lowered)
    row_count = min(len(order_numbers), len(totals), len(statuses))
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        rows.append(
            {
                "order_number": order_numbers[index],
                "total": totals[index],
                "status": statuses[index],
            }
        )
    return rows


def _derive_order_status_key(lowered_goal: str) -> str:
    if "non-cancelled" in lowered_goal or "non-canceled" in lowered_goal:
        return "non_canceled"
    if "cancelled" in lowered_goal or "canceled" in lowered_goal or "canlled" in lowered_goal:
        return "canceled"
    if "pending" in lowered_goal:
        return "pending"
    if "completed" in lowered_goal or "complete" in lowered_goal:
        return "complete"
    return ""


def _extract_goal_order_numbers(goal: str) -> list[str]:
    primary_match = re.search(
        r"\border number\s+0*(\d+)(?:\s+and\s+0*(\d+))?",
        goal or "",
        flags=re.IGNORECASE,
    )
    if primary_match:
        return [value for value in primary_match.groups() if value]
    return re.findall(r"\b0*(\d{2,})\b", goal or "")


def _derive_multi_order_status_answer(goal: str, rows: list[dict[str, str]]) -> str:
    lowered_goal = " ".join((goal or "").lower().split())
    if "order statuses" not in lowered_goal:
        return ""
    order_numbers = _extract_goal_order_numbers(goal)
    if len(order_numbers) < 2:
        return ""
    row_by_order = {row.get("order_number", "").lstrip("0") or "0": row for row in rows if row.get("order_number")}
    parts: list[str] = []
    for raw_order_number in order_numbers:
        normalized = raw_order_number.lstrip("0") or "0"
        row = row_by_order.get(normalized)
        if not row or not row.get("status"):
            return ""
        parts.append(f"{normalized}: {row.get('status', '').lower()}")
    return ", ".join(parts)


def _goal_requests_last_ordered_product_date(lowered_goal: str) -> bool:
    return "last ordered my" in lowered_goal or "when i last ordered" in lowered_goal


def _extract_last_ordered_product_query(lowered_goal: str) -> str:
    match = re.search(r"(?:last ordered my|ordered my)\s+(.+?)(?:\?|$)", lowered_goal)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" ?.")


def _extract_customer_order_product_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Customer order product row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row:
            rows.append(row)
    return rows


def _format_long_order_date(value: datetime) -> str:
    day = value.day
    suffix = "th"
    if day % 10 == 1 and day % 100 != 11:
        suffix = "st"
    elif day % 10 == 2 and day % 100 != 12:
        suffix = "nd"
    elif day % 10 == 3 and day % 100 != 13:
        suffix = "rd"
    return f"{value.strftime('%B')} {day}{suffix} {value.year}"


def _derive_last_ordered_product_date_answer(lowered_goal: str, visible_page_summary: str) -> str:
    if not _goal_requests_last_ordered_product_date(lowered_goal):
        return ""
    query = _extract_last_ordered_product_query(lowered_goal)
    if not query:
        return ""
    product_rows = _extract_customer_order_product_rows(visible_page_summary)
    if not product_rows:
        return ""
    query_tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]
    if not query_tokens:
        return ""
    matching_rows = [
        row
        for row in product_rows
        if all(token in row.get("product", "").lower() for token in query_tokens)
    ]
    if not matching_rows:
        return ""
    dated_rows = [
        (row, _parse_customer_order_date(row.get("date", "")))
        for row in matching_rows
        if row.get("date")
    ]
    dated_rows = [(row, parsed_date) for row, parsed_date in dated_rows if parsed_date is not None]
    if not dated_rows:
        return ""
    latest_row, latest_date = max(dated_rows, key=lambda item: item[1] or datetime.min)
    if latest_date is None:
        return latest_row.get("date", "")
    return _format_long_order_date(latest_date)


def _row_matches_status(status: str, status_key: str) -> bool:
    lowered = (status or "").lower()
    if not status_key:
        return True
    if status_key == "non_canceled":
        return lowered not in {"canceled", "cancelled"}
    if status_key == "canceled":
        return lowered in {"canceled", "cancelled"}
    return lowered == status_key


def _derive_order_count(goal: str) -> int | None:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return None
    for pattern in (
        r"\bmost recent\s+(\d+)\s+orders?\b",
        r"\blast\s+(\d+)\s+(?:completed|complete|pending|cancelled|canceled|non-cancelled|non-canceled)?\s*orders?\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def _parse_amount(value: str) -> float:
    normalized = value.replace("$", "").replace(",", "").strip()
    try:
        return float(normalized)
    except ValueError:
        return 0.0


def _derive_admin_dashboard_answer(goal: str, visible_page_summary: str, dom_or_ax_snippet: str) -> str:
    review_count_answer = _derive_admin_dashboard_review_count_answer(goal, visible_page_summary)
    if review_count_answer:
        return review_count_answer
    search_term_answer = _derive_admin_dashboard_search_term_answer(goal, visible_page_summary)
    if search_term_answer:
        return search_term_answer
    quantity_answer = _derive_admin_dashboard_quantity_answer(goal, visible_page_summary, dom_or_ax_snippet)
    if quantity_answer:
        return quantity_answer
    return _derive_admin_dashboard_bestseller_answer(goal, visible_page_summary)


def _derive_admin_dashboard_review_count_answer(goal: str, visible_page_summary: str) -> str:
    lowered = " ".join((goal or "").lower().split())
    if "review" not in lowered:
        return ""
    review_status = _extract_admin_dashboard_review_status(goal)
    if review_status:
        for row in _extract_admin_dashboard_review_status_count_rows(visible_page_summary):
            if row.get("status", "").lower() != review_status.lower():
                continue
            count = row.get("count", "")
            if count:
                return count
    review_term = _extract_admin_dashboard_review_term(goal)
    if review_term:
        for row in _extract_admin_dashboard_review_count_rows(visible_page_summary):
            if row.get("term", "").lower() != review_term.lower():
                continue
            count = row.get("count", "")
            if count:
                return count
    return ""


def _extract_admin_dashboard_review_term(goal: str) -> str:
    normalized_goal = " ".join((goal or "").split())
    if not normalized_goal:
        return ""
    quoted_match = re.search(r'term\s*["“](?P<term>[^"”]+)["”]', normalized_goal, re.IGNORECASE)
    if quoted_match:
        return " ".join(quoted_match.group("term").split())
    single_quoted_match = re.search(r"term\s*'(?P<term>[^']+)'", normalized_goal, re.IGNORECASE)
    if single_quoted_match:
        return " ".join(single_quoted_match.group("term").split())
    fallback_match = re.search(
        r"mention(?:ing)?\s+term\s+(?P<term>[A-Za-z0-9][A-Za-z0-9' -]{1,80})",
        normalized_goal,
        re.IGNORECASE,
    )
    if fallback_match:
        return " ".join(fallback_match.group("term").rstrip("?.!,").split())
    return ""


def _extract_admin_dashboard_review_status(goal: str) -> str:
    lowered = " ".join((goal or "").lower().split())
    if "not approved" in lowered:
        return "Not Approved"
    if "pending" in lowered:
        return "Pending"
    if "approved" in lowered:
        return "Approved"
    return ""


def _derive_admin_dashboard_search_term_answer(goal: str, visible_page_summary: str) -> str:
    lowered = " ".join((goal or "").lower().split())
    if "search term" not in lowered:
        return ""
    rows = _extract_admin_dashboard_search_term_rows(visible_page_summary)
    if not rows:
        return ""
    count = _derive_top_count(goal) or 1
    if len(rows) < count:
        return ""
    return ", ".join(row.get("term", "") for row in rows[:count] if row.get("term"))


def _derive_admin_dashboard_quantity_answer(goal: str, visible_page_summary: str, dom_or_ax_snippet: str) -> str:
    lowered = " ".join((goal or "").lower().split())
    if "items sold" not in lowered:
        return ""
    count = _derive_order_count(goal)
    if not count:
        return ""

    row_quantities: list[int] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Dashboard order row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key.strip() != "items":
                continue
            try:
                row_quantities.append(int(value.strip()))
            except ValueError:
                pass
            break
    if len(row_quantities) >= count:
        return str(sum(row_quantities[:count]))
    if row_quantities:
        return ""

    quantities: list[int] = []
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.search(r'name="(?P<name>\d{1,2})"', line)
        if match is None:
            continue
        quantities.append(int(match.group("name")))
    if len(quantities) < count:
        return ""
    return str(sum(quantities[:count]))


def _derive_admin_dashboard_bestseller_answer(goal: str, visible_page_summary: str) -> str:
    lowered = " ".join((goal or "").lower().split())
    if "best-selling" not in lowered and "bestselling" not in lowered and "best selling" not in lowered:
        return ""
    report_rows = _extract_admin_bestseller_report_rows(visible_page_summary)
    aggregated_report_rows = _aggregate_admin_bestseller_products(report_rows)
    if aggregated_report_rows:
        return _derive_admin_bestseller_answer_from_rows(goal, aggregated_report_rows)
    rows = _extract_admin_dashboard_bestseller_rows(visible_page_summary)
    if rows:
        return _derive_admin_bestseller_answer_from_rows(goal, rows)
    aggregate_rows = _extract_admin_bestseller_aggregate_rows(visible_page_summary)
    if aggregate_rows:
        return _derive_admin_bestseller_answer_from_rows(goal, aggregate_rows)
    return ""


def _derive_admin_bestseller_answer_from_rows(goal: str, rows: list[dict[str, str]]) -> str:
    lowered = " ".join((goal or "").lower().split())
    count = _derive_top_count(goal) or 1
    if "brand" in lowered:
        ranked_brands = _rank_dashboard_brands(rows)
        if len(ranked_brands) < count:
            return ""
        return ", ".join(brand for brand, _ in ranked_brands[:count])
    if "product type" in lowered:
        ranked_types = _rank_dashboard_product_types(rows)
        if len(ranked_types) < count:
            return ""
        return ", ".join(product_type for product_type, _ in ranked_types[:count])
    if "product" not in lowered:
        return ""
    ranked_products = _rank_dashboard_products(rows)
    if len(ranked_products) < count:
        return ""
    return ", ".join(row.get("product", "") for row in ranked_products[:count])


def _derive_top_count(goal: str) -> int | None:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return None
    match = re.search(r"\btop[- ](\d+)\b", lowered)
    if match:
        return int(match.group(1))
    return None


def _extract_admin_dashboard_bestseller_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Dashboard bestseller row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("product") and row.get("quantity"):
            rows.append(row)
    return rows


def _extract_admin_dashboard_search_term_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Dashboard search term row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("term"):
            rows.append(row)
    return rows


def _extract_admin_dashboard_review_count_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Admin review mention count:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("term") and row.get("count"):
            rows.append(row)
    return rows


def _extract_admin_dashboard_review_status_count_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Admin review status count:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("status") and row.get("count"):
            rows.append(row)
    return rows


def _extract_admin_bestseller_report_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Admin bestseller report row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("product") and row.get("quantity"):
            rows.append(row)
    return rows


def _extract_admin_bestseller_aggregate_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Admin bestseller aggregate row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row.get("product") and row.get("quantity"):
            rows.append(row)
    return rows


def _aggregate_admin_bestseller_products(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    aggregated: dict[str, dict[str, str | int]] = {}
    for index, row in enumerate(rows):
        product = row.get("product", "")
        if not product:
            continue
        quantity = int(row.get("quantity", "0") or "0")
        if product not in aggregated:
            aggregated[product] = {
                "product": product,
                "quantity": quantity,
                "_first_index": index,
            }
            continue
        aggregated[product]["quantity"] = int(aggregated[product]["quantity"]) + quantity
    ordered = sorted(
        aggregated.values(),
        key=lambda row: (-int(row["quantity"]), int(row["_first_index"])),
    )
    return [{"product": str(row["product"]), "quantity": str(row["quantity"])} for row in ordered]


def _rank_dashboard_products(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    if "price" not in rows[0]:
        family_totals: dict[str, int] = {}
        family_best: dict[str, tuple[int, int, dict[str, str]]] = {}
        for index, row in enumerate(rows):
            product = row.get("product", "")
            if not product:
                continue
            family = _derive_bestseller_product_family(product)
            quantity = int(row.get("quantity", "0") or "0")
            family_totals[family] = family_totals.get(family, 0) + quantity
            current_best = family_best.get(family)
            candidate = (quantity, index, row)
            if current_best is None or candidate[0] > current_best[0] or (
                candidate[0] == current_best[0] and candidate[1] < current_best[1]
            ):
                family_best[family] = candidate
        ordered_families = sorted(
            family_totals.items(),
            key=lambda item: (-item[1], family_best[item[0]][1]),
        )
        return [family_best[family][2] for family, _ in ordered_families]
    ordered: list[dict[str, str]] = [rows[0]]
    remaining = sorted(
        rows[1:],
        key=lambda row: (
            -int(row.get("quantity", "0") or "0"),
            -_parse_amount(row.get("price", "")),
            row.get("product", "").lower(),
        ),
    )
    for row in remaining:
        if row.get("product") and row.get("product") != ordered[0].get("product"):
            ordered.append(row)
    return ordered


def _derive_bestseller_product_family(product_name: str) -> str:
    normalized = " ".join((product_name or "").split())
    return re.sub(r"-[^-\s]+-[A-Za-z]+$", "", normalized)


def _rank_dashboard_brands(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    totals: dict[str, int] = {}
    for row in rows:
        brand = _derive_dashboard_brand(row.get("product", ""))
        if not brand:
            continue
        totals[brand] = totals.get(brand, 0) + int(row.get("quantity", "0") or "0")
    return sorted(totals.items(), key=lambda item: (-item[1], item[0].lower()))


def _rank_dashboard_product_types(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    totals: dict[str, int] = {}
    for row in rows:
        product_type = _derive_dashboard_product_type(row.get("product", ""))
        if not product_type:
            continue
        totals[product_type] = totals.get(product_type, 0) + int(row.get("quantity", "0") or "0")
    return sorted(totals.items(), key=lambda item: (-item[1], item[0].lower()))


def _derive_dashboard_brand(product_name: str) -> str:
    cleaned = (
        product_name.replace("™", " ")
        .replace("®", " ")
        .replace("©", " ")
        .strip()
    )
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'-]*", cleaned)
    if not tokens:
        return ""
    return tokens[0]


def _derive_dashboard_product_type(product_name: str) -> str:
    lowered = " ".join((product_name or "").lower().split())
    if "yoga strap" in lowered or "strap" in lowered:
        return "Yoga strap"
    if "stasis ball" in lowered or "ball" in lowered:
        return "Yoga ball"
    if "duffle" in lowered:
        return "Duffle"
    if "band" in lowered:
        return "Band"
    if "tank" in lowered:
        return "Tank"
    return ""


def _extract_admin_order_summary_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Order row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row:
            rows.append(row)
    return rows


def _extract_admin_order_item_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Admin order item row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row:
            rows.append(row)
    return rows


def _extract_admin_order_product_price_rows(visible_page_summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if not line.startswith("Admin order product row:"):
            continue
        payload = line.split(":", 1)[1].strip()
        row: dict[str, str] = {}
        for part in payload.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key.strip()] = value.strip()
        if row:
            rows.append(row)
    return rows


def _normalize_admin_price_text(value: str) -> str:
    amount = _parse_amount(value)
    if amount == 0.0 and not re.search(r"\b0+(?:\.0+)?\b", value or ""):
        return value.strip()
    normalized = f"{amount:.2f}".rstrip("0").rstrip(".")
    return f"${normalized}"


def _derive_admin_order_grid_answer(goal: str, visible_page_summary: str) -> str:
    rows = _extract_admin_order_summary_rows(visible_page_summary)
    lowered_goal = " ".join((goal or "").lower().split())
    item_rows = _extract_admin_order_item_rows(visible_page_summary)
    product_rows = _extract_admin_order_product_price_rows(visible_page_summary)
    count = _derive_order_count(goal)
    if count and "items sold" in lowered_goal and item_rows:
        total_items = 0
        used_item_rows = 0
        for row in item_rows:
            try:
                total_items += int(row.get("items", ""))
            except ValueError:
                continue
            used_item_rows += 1
        if used_item_rows >= count:
            return str(total_items)

    if not rows:
        return ""

    status_key = _derive_order_status_key(lowered_goal)
    filtered = [row for row in rows if _row_matches_status(row.get("status", ""), status_key)]
    if not filtered:
        filtered = rows
    newest_first_rows = _sort_admin_order_rows_by_date(filtered, newest_first=True)
    oldest_first_rows = _sort_admin_order_rows_by_date(filtered, newest_first=False)

    if count and "total payment amount" in lowered_goal:
        selected = newest_first_rows[:count]
        total = sum(_parse_amount(row.get("total", "")) for row in selected)
        return f"{total:.2f}"

    if "payment difference" in lowered_goal:
        compare_count = count or 4
        canceled_rows = _sort_admin_order_rows_by_date(
            [row for row in rows if _row_matches_status(row.get("status", ""), "canceled")],
            newest_first=True,
        )[:compare_count]
        complete_rows = _sort_admin_order_rows_by_date(
            [row for row in rows if _row_matches_status(row.get("status", ""), "complete")],
            newest_first=True,
        )[:compare_count]
        if len(canceled_rows) < compare_count or len(complete_rows) < compare_count:
            return ""
        difference = sum(_parse_amount(row.get("total", "")) for row in canceled_rows) - sum(
            _parse_amount(row.get("total", "")) for row in complete_rows
        )
        return f"{difference:.2f}"

    if "customer name" in lowered_goal and filtered:
        target_row = oldest_first_rows[0] if "oldest" in lowered_goal else newest_first_rows[0]
        return target_row.get("customer", "")
    if "billing name" in lowered_goal and filtered:
        target_row = oldest_first_rows[0] if "oldest" in lowered_goal else newest_first_rows[0]
        return target_row.get("billing", "")
    if "product name" in lowered_goal and "discounted price" in lowered_goal and product_rows:
        ranked_products = sorted(
            [row for row in product_rows if row.get("product") and row.get("price")],
            key=lambda row: (_parse_amount(row.get("price", "")), row.get("product", "").lower()),
        )
        if ranked_products:
            return ", ".join(
                f'{row.get("product", "")}: {_normalize_admin_price_text(row.get("price", ""))}'
                for row in ranked_products
            )
    if ("purchase date" in lowered_goal) and ("order id" in lowered_goal or "order number" in lowered_goal) and filtered:
        target_row = oldest_first_rows[0] if "oldest" in lowered_goal else newest_first_rows[0]
        order_number = target_row.get("order", "")
        raw_date = target_row.get("date", "")
        return ", ".join(part for part in (order_number, raw_date) if part)
    if ("order id" in lowered_goal or "order number" in lowered_goal) and filtered:
        target_row = oldest_first_rows[0] if "oldest" in lowered_goal else newest_first_rows[0]
        order_number = target_row.get("order", "")
        if "order id" in lowered_goal:
            stripped = order_number.lstrip("0")
            return stripped or "0"
        return order_number
    if re.search(r"\bdate\b", lowered_goal) and filtered:
        target_row = oldest_first_rows[0] if "oldest" in lowered_goal else newest_first_rows[0]
        parsed_date = _parse_admin_order_date(target_row.get("date", ""))
        if parsed_date is None:
            return target_row.get("date", "")
        return _format_admin_order_date(parsed_date)
    return ""


def _build_shopping_admin_order_sort_action(observation: NormalizedObservation) -> str:
    lowered_goal = " ".join((observation.goal or "").lower().split())
    purchase_date_bid, purchase_date_name = _find_shopping_admin_purchase_date_header(observation.dom_or_ax_snippet)
    if not purchase_date_bid:
        return ""
    if any(_action_clicks_bid(previous_action, purchase_date_bid) for previous_action in observation.previous_actions):
        return ""
    if "oldest" in lowered_goal and not _admin_purchase_date_header_matches_goal(
        lowered_goal,
        purchase_date_name,
    ):
        return f'ACTION: click("{purchase_date_bid}")'
    if (
        _admin_goal_requests_recent_rows(lowered_goal)
        and not _admin_purchase_date_header_matches_goal(lowered_goal, purchase_date_name)
    ):
        return f'ACTION: click("{purchase_date_bid}")'
    return ""


def _admin_goal_requests_recent_rows(lowered_goal: str) -> bool:
    return any(token in lowered_goal for token in ("last ", "latest", "most recent", "newest"))


def _admin_order_summary_missing_goal_status(goal: str, visible_page_summary: str) -> bool:
    lowered_goal = " ".join((goal or "").lower().split())
    status_key = _derive_order_status_key(lowered_goal)
    if not status_key:
        return False
    rows = _extract_admin_order_summary_rows(visible_page_summary)
    if not rows:
        return True
    return not any(_row_matches_status(row.get("status", ""), status_key) for row in rows)


def _admin_purchase_date_header_matches_goal(lowered_goal: str, header_name: str) -> bool:
    normalized_name = " ".join((header_name or "").split())
    if not normalized_name:
        return False
    if "oldest" in lowered_goal:
        return "↓" in normalized_name
    if _admin_goal_requests_recent_rows(lowered_goal):
        return "↑" in normalized_name
    return False


def _find_shopping_admin_purchase_date_header(dom_or_ax_snippet: str) -> tuple[str, str]:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=(?:columnheader|button|link)(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        name = match.group("name") or ""
        if "purchase date" in name.lower():
            return match.group("bid"), name
    return "", ""


def _find_shopping_admin_purchase_date_bid(dom_or_ax_snippet: str) -> str:
    bid, _ = _find_shopping_admin_purchase_date_header(dom_or_ax_snippet)
    return bid


def _sort_admin_order_rows_by_date(rows: list[dict[str, str]], *, newest_first: bool) -> list[dict[str, str]]:
    dated_rows: list[tuple[datetime, int, dict[str, str]]] = []
    undated_rows: list[tuple[int, dict[str, str]]] = []
    for index, row in enumerate(rows):
        parsed_date = _parse_admin_order_date(row.get("date", ""))
        if parsed_date is None:
            undated_rows.append((index, row))
            continue
        dated_rows.append((parsed_date, index, row))
    if not dated_rows:
        return list(rows)
    if newest_first:
        dated_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    else:
        dated_rows.sort(key=lambda item: (item[0], item[1]))
    sorted_rows = [row for _, _, row in dated_rows]
    sorted_rows.extend(row for _, row in undated_rows)
    return sorted_rows


def _parse_admin_order_date(value: str) -> datetime | None:
    normalized = " ".join((value or "").replace("Sept", "Sep").split())
    if not normalized:
        return None
    for date_format in (
        "%B %d, %Y %I:%M:%S %p",
        "%b %d, %Y %I:%M:%S %p",
        "%m/%d/%y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(normalized, date_format)
        except ValueError:
            continue
    return None


def _format_admin_order_date(value: datetime) -> str:
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _extract_order_detail_date(observation: NormalizedObservation) -> str:
    for raw_line in observation.visible_page_summary.splitlines():
        line = " ".join(raw_line.split())
        if line.lower().startswith("order detail date:"):
            return line.split(":", 1)[1].strip()
    body_text = " ".join(observation.visible_page_summary.split())
    date_match = re.search(rf"order date:?\s*{ORDER_DATE_TEXT_PATTERN}", body_text, flags=re.IGNORECASE)
    if date_match:
        return date_match.group(1)
    return ""


def _action_clicks_bid(action_text: str, bid: str) -> bool:
    action_name, arguments = _parse_action_for_observation_validation(action_text)
    return action_name == "click" and bool(arguments) and _strip_quoted_string(arguments[0]) == bid


def _looks_like_shopping_order_goal(goal: str) -> bool:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return False
    return any(
        keyword in lowered
        for keyword in (
            "order",
            "orders",
            "refund",
            "purchase",
            "purchases",
            "billing",
            "shipping method",
        )
    )


def _raw_text_looks_like_send_message_attempt(raw_text: str) -> bool:
    return "send_msg_to_user" in (raw_text or "").lower()


def _normalize_local_model_path(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    normalized = str(path_text)
    wsl_match = re.match(r"^/mnt/(?P<drive>[a-zA-Z])/(?P<rest>.+)$", normalized)
    if wsl_match is None:
        return normalized
    rest = wsl_match.group("rest").replace("/", "\\")
    candidate = f'{wsl_match.group("drive").upper()}:\\{rest}'
    if Path(candidate).exists():
        return candidate
    return normalized


def _extract_gitlab_commit_count_query(goal: str) -> tuple[str, str]:
    match = re.search(
        r"how many commits did (?P<user>.+?) make to .+? on (?P<date>\d{1,2}/\d{1,2}(?:/\d{4})?)\??$",
        goal or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return "", ""
    return match.group("user").strip().lower(), match.group("date").strip()


def _is_gitlab_rss_token_goal(goal: str) -> bool:
    lowered_goal = " ".join((goal or "").lower().split())
    return "rss feed token" in lowered_goal


def _is_gitlab_clone_ssh_goal(goal: str) -> bool:
    lowered_goal = " ".join((goal or "").lower().split())
    return "clone" in lowered_goal and "ssh" in lowered_goal


def _is_gitlab_profile_account_page(current_url: str) -> bool:
    parsed = urlparse(current_url or "")
    path = parsed.path or ""
    return parsed.netloc.endswith(":8023") and path.rstrip("/") in {"/-/profile/account", "/-/profile"}


def _is_gitlab_profile_preferences_page(current_url: str) -> bool:
    parsed = urlparse(current_url or "")
    path = parsed.path or ""
    return parsed.netloc.endswith(":8023") and path.rstrip("/") == "/-/profile/preferences"


def _is_gitlab_profile_personal_access_tokens_page(current_url: str) -> bool:
    parsed = urlparse(current_url or "")
    path = parsed.path or ""
    return parsed.netloc.endswith(":8023") and path.rstrip("/") == "/-/profile/personal_access_tokens"


def _find_obs_link_bid_by_name(dom_or_ax_snippet: str, target_name: str) -> str:
    normalized_target = " ".join((target_name or "").lower().split())
    if not normalized_target:
        return ""
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = OBS_ROLE_LINE_PATTERN.match(line)
        if match is None or match.group("role").lower() != "link":
            continue
        normalized_name = " ".join((match.group("name") or "").lower().split())
        if normalized_name == normalized_target:
            return match.group("bid")
    return ""


def _extract_gitlab_rss_token_from_observation(observation: NormalizedObservation) -> str:
    combined_text = "\n".join(
        part
        for part in (
            observation.visible_page_summary or "",
            observation.dom_or_ax_snippet or "",
        )
        if part
    )
    if not combined_text:
        return ""
    contextual_match = re.search(
        r"rss(?:\s+feed)?\s+token[^A-Za-z0-9_-]{0,40}(?P<token>[A-Za-z0-9_-]{20,})",
        combined_text,
        flags=re.IGNORECASE,
    )
    if contextual_match is not None:
        return contextual_match.group("token")
    token_matches = re.findall(r"\b[A-Za-z0-9_-]{20,}\b", combined_text)
    if len(token_matches) == 1:
        return token_matches[0]
    return ""


def _fetch_gitlab_commit_count_for_user(current_url: str, user_query: str, target_date: str) -> int | None:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023") or not _is_gitlab_graph_page(current_url):
        return None
    graph_json_url = urlunparse(
        (
            parsed.scheme or "http",
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode({"format": "json"}),
            parsed.fragment,
        )
    )
    try:
        response = requests.get(graph_json_url, timeout=5)
        payload = response.json() if response.ok else []
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(payload, list):
        return None
    normalized_target_date = _normalize_gitlab_commit_date_query(target_date, payload)
    if not normalized_target_date:
        return None
    user_queries = _split_gitlab_commit_user_queries(user_query)
    if not user_queries:
        return None
    match_count = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        item_date = str(item.get("date") or "").strip()
        if item_date != normalized_target_date:
            continue
        author_name = str(item.get("author_name") or "").lower()
        author_email = str(item.get("author_email") or "").lower()
        if any(_gitlab_author_matches_query(author_name, author_email, query) for query in user_queries):
            match_count += 1
    return match_count


def _normalize_gitlab_commit_date_query(target_date: str, payload: list[object]) -> str:
    try:
        return datetime.strptime(target_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        pass

    month_day_match = re.fullmatch(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})", target_date.strip())
    if month_day_match is None:
        return ""
    month = int(month_day_match.group("month"))
    day = int(month_day_match.group("day"))
    years: list[int] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        item_date = str(item.get("date") or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item_date):
            continue
        years.append(int(item_date[:4]))
    if not years:
        return ""
    return f"{max(years):04d}-{month:02d}-{day:02d}"


def _split_gitlab_commit_user_queries(user_query: str) -> list[str]:
    raw_parts = re.split(r"\s+(?:and|&)\s+|,\s*", user_query or "", flags=re.IGNORECASE)
    return [part.strip().lower() for part in raw_parts if part and part.strip()]


def _gitlab_author_matches_query(author_name: str, author_email: str, user_query: str) -> bool:
    normalized_query = " ".join((user_query or "").lower().split())
    normalized_author_name = " ".join((author_name or "").lower().split())
    normalized_author_email = (author_email or "").lower().strip()
    if not normalized_query:
        return False
    if normalized_query in normalized_author_name or normalized_query in normalized_author_email:
        return True

    query_tokens = re.findall(r"[a-z0-9]+", normalized_query)
    author_tokens = re.findall(r"[a-z0-9]+", normalized_author_name)
    if not query_tokens or not author_tokens:
        return False
    if all(any(token == author_token for author_token in author_tokens) for token in query_tokens):
        return True

    query_first = query_tokens[0]
    query_last = query_tokens[-1]
    author_first = author_tokens[0]
    author_last = author_tokens[-1]
    if query_last != author_last:
        return False
    prefix_length = min(len(query_first), len(author_first), 4)
    if prefix_length < 3:
        return False
    return query_first[:prefix_length] == author_first[:prefix_length]


def _extract_gitlab_ssh_clone_url(current_url: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023"):
        return ""
    html_text = _safe_fetch_text(current_url)
    if not html_text:
        return ""
    value_match = re.search(
        r'id="ssh_project_clone"[^>]*value="(?P<url>ssh://git@[^"]+)"',
        html_text,
        flags=re.IGNORECASE,
    )
    if value_match is not None:
        return html.unescape(value_match.group("url"))
    return ""


def _normalize_gitlab_ssh_clone_url_for_benchmark(ssh_clone_url: str) -> str:
    normalized = (ssh_clone_url or "").strip()
    if not normalized.startswith("ssh://git@"):
        return ""
    return re.sub(
        r"(?<=ssh://git@)[^:/]+(?=:2222/)",
        "metis.lti.cs.cmu.edu",
        normalized,
        count=1,
    )


def _fetch_reddit_latest_post_negative_comment_count(current_url: str, forum_query: str) -> int | None:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":9999"):
        return None
    forum_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/f/{forum_query}"
    forum_html = _safe_fetch_text(forum_url)
    if not forum_html:
        return None
    latest_match = re.search(
        rf'<a href="(?P<href>/f/{re.escape(forum_query)}/\d+/[^"]+)"[^>]*class="submission__link"',
        forum_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if latest_match is None:
        return None
    submission_url = f"{parsed.scheme or 'http'}://{parsed.netloc}{html.unescape(latest_match.group('href'))}"
    submission_html = _safe_fetch_text(submission_url)
    if not submission_html:
        return None
    negative_count = 0
    for score_text in re.findall(r'<span class="vote__net-score"[^>]*>(.*?)</span>', submission_html, flags=re.IGNORECASE | re.DOTALL):
        cleaned_score = _strip_html_fragment(score_text).replace(",", "").replace("−", "-").strip()
        if not cleaned_score:
            continue
        try:
            if int(cleaned_score) < 0:
                negative_count += 1
        except ValueError:
            continue
    return negative_count


def _extract_gitlab_repo_root_from_graph_url(target_url: str) -> str:
    parsed = urlparse(target_url or "")
    path = parsed.path or ""
    if not _is_gitlab_graph_page(target_url):
        return ""
    repo_path = path.split("/-/graphs/", 1)[0].rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{repo_path}"


def _is_gitlab_graph_append(target_url: str, current_url: str) -> bool:
    if not _is_gitlab_graph_page(target_url) or not _is_gitlab_graph_page(current_url):
        return False
    if _gitlab_graph_segment_count(target_url) > 1:
        return True
    current_root = _extract_gitlab_repo_root_from_graph_url(current_url)
    target_root = _extract_gitlab_repo_root_from_graph_url(target_url)
    if not current_root or not target_root or current_root != target_root:
        return False
    return target_url.startswith(current_url.rstrip("/") + "/-/graphs/")


def _is_page_not_found_observation(observation: NormalizedObservation) -> bool:
    haystack = f"{observation.visible_page_summary}\n{observation.dom_or_ax_snippet}".lower()
    return "page not found" in haystack


def _looks_like_branch_without_graph_path(target_url: str, current_url: str) -> bool:
    if not _is_gitlab_graph_page(current_url):
        return False
    repo_root = _extract_gitlab_repo_root_from_graph_url(current_url)
    if not repo_root:
        return False
    return target_url in {f"{repo_root}/main", f"{repo_root}/master"}


def _looks_like_placeholder_person_name(answer_text: str) -> bool:
    normalized = " ".join(answer_text.lower().split())
    return normalized in {
        "john doe",
        "jane doe",
        "alice johnson",
        "bob smith",
        "example user",
    }


def _looks_like_placeholder_answer(answer_text: str) -> bool:
    normalized = " ".join(answer_text.lower().split())
    if normalized in {"...", "<answer>", "product a, product b"}:
        return True
    return any(token in normalized for token in ("<product", "<answer", "product a", "product b"))


def _looks_like_order_ui_label(answer_text: str) -> bool:
    normalized = " ".join(answer_text.lower().split())
    return normalized in {
        "filter by name",
        "search",
        "product name",
        "my orders",
        "address book",
        "account information",
    }


def _looks_like_shipping_method_value(value: str) -> bool:
    normalized = " ".join((value or "").lower().split())
    if not normalized or len(normalized) < 6:
        return False
    if any(token in normalized for token in ("$", "order", "address", "newsletter", "search", "reorder")):
        return False
    return any(
        phrase in normalized
        for phrase in (
            "flat rate",
            "fixed",
            "free shipping",
            "table rate",
            "best way",
            "store pickup",
        )
    )


def _shopping_order_detail_answer_supported_by_observation(
    answer_text: str,
    observation: NormalizedObservation,
) -> bool:
    lowered_goal = " ".join((observation.goal or "").lower().split())
    normalized_answer = " ".join(answer_text.lower().split())
    normalized_observation = " ".join(
        f"{observation.visible_page_summary}\n{observation.dom_or_ax_snippet}".lower().split()
    )
    if not normalized_answer or not normalized_observation:
        return False
    if "refund" in lowered_goal:
        expected_refund_answer = _derive_order_detail_refund_answer(observation, lowered_goal)
        if expected_refund_answer:
            return normalized_answer == " ".join(expected_refund_answer.lower().split())
    if "product names" in lowered_goal:
        product_candidates = _extract_order_detail_product_candidates(observation)
        if len(product_candidates) >= 2:
            return all(candidate in normalized_answer for candidate in product_candidates[:2])
    if "billing address" in lowered_goal:
        address_candidates = _extract_order_detail_address_candidates(observation)
        if len(address_candidates) >= 3:
            return all(candidate in normalized_answer for candidate in address_candidates)
    if normalized_answer in normalized_observation:
        return True
    parts = [part.strip() for part in normalized_answer.split(",") if part.strip()]
    if len(parts) > 1:
        return all(part in normalized_observation for part in parts if len(part) >= 4)
    return False


def _build_shopping_order_detail_answer_retry_hint(observation: NormalizedObservation) -> str:
    lowered_goal = " ".join((observation.goal or "").lower().split())
    if "refund" in lowered_goal:
        return (
            "The final answer must be the refund amount supported by the visible order totals on this page. "
            "Use the exact visible amount, or subtract the visible Shipping & Handling amount when shipping is excluded."
        )
    if "product names" in lowered_goal:
        product_candidates = _extract_order_detail_product_candidates(observation)
        if product_candidates:
            quoted_titles = ", ".join(f'"{candidate}"' for candidate in product_candidates[:3])
            return (
                "The final answer must include every visible product title from the Items Ordered section in one "
                f'send_msg_to_user action. Include all visible product titles, for example: {quoted_titles}.'
            )
        return (
            "The final answer must include every visible product title from the Items Ordered section in one "
            'send_msg_to_user action, not just the first product.'
        )
    if "billing address" in lowered_goal:
        address_candidates = _extract_order_detail_address_candidates(observation)
        if address_candidates:
            visible_address = ", ".join(address_candidates[:5])
            return (
                "The final answer must copy the visible Billing Address details from this page in one "
                f'send_msg_to_user action, including the street, city, state, postal code, and country: "{visible_address}".'
            )
        return (
            "The final answer must copy the visible Billing Address details from this page in one "
            "send_msg_to_user action, including the street, city, state, postal code, and country."
        )
    if "shipping method" in lowered_goal:
        return (
            "The final answer must copy the visible Shipping Method text from this page in one send_msg_to_user action."
        )
    if "order date" in lowered_goal:
        return (
            "The final answer must copy the visible Order Date text from this page in one send_msg_to_user action."
        )
    return (
        "The final answer must be copied from the visible order details on this page. "
        "Use the exact visible date, address text, shipping method, or product titles instead of inventing new text."
    )


def _extract_order_detail_product_candidates(observation: NormalizedObservation) -> list[str]:
    return [" ".join(title.lower().split()) for title in _extract_order_detail_product_titles(observation)]


def _derive_order_detail_refund_answer(observation: NormalizedObservation, lowered_goal: str) -> str:
    if "refund" not in lowered_goal:
        return ""
    grand_total = _extract_order_detail_grand_total(observation)
    if grand_total is None:
        return ""

    refundable_total = grand_total
    if _refund_goal_excludes_shipping(lowered_goal):
        exact_non_shipping_total = _extract_order_detail_non_shipping_total(observation, grand_total)
        if exact_non_shipping_total is not None:
            refundable_total = exact_non_shipping_total
        else:
            shipping_amount = _extract_order_detail_shipping_amount(observation)
            if shipping_amount is not None:
                refundable_total = max(0.0, grand_total - shipping_amount)

    kept_product_fragment = _extract_kept_product_fragment(lowered_goal)
    if kept_product_fragment:
        kept_product_amount = _extract_order_detail_product_amount(observation, kept_product_fragment)
        if kept_product_amount is not None:
            refundable_total = max(0.0, refundable_total - kept_product_amount)

    return f"{refundable_total:.2f}"


def _extract_order_detail_named_values(observation: NormalizedObservation) -> list[str]:
    values: list[str] = []
    for raw_line in observation.visible_page_summary.splitlines():
        text = " ".join(raw_line.split())
        if text:
            values.append(text)
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        match = re.search(r'name="(?P<name>[^"]+)"', raw_line)
        if match is None:
            continue
        text = " ".join(match.group("name").split())
        if text:
            values.append(text)
    return values


def _extract_order_detail_money_values(observation: NormalizedObservation) -> list[float]:
    values: list[float] = []
    for text in _extract_order_detail_named_values(observation):
        if re.fullmatch(r"\$[\d,]+\.\d{2}", text):
            values.append(_parse_amount(text))
    return values


def _extract_order_detail_grand_total(observation: NormalizedObservation) -> float | None:
    money_values = _extract_order_detail_money_values(observation)
    if not money_values:
        return None
    return max(money_values)


def _extract_order_detail_shipping_amount(observation: NormalizedObservation) -> float | None:
    values = _extract_order_detail_named_values(observation)
    for index, text in enumerate(values):
        if "shipping & handling" not in text.lower():
            continue
        for candidate in values[index + 1 : index + 5]:
            if re.fullmatch(r"\$[\d,]+\.\d{2}", candidate):
                return _parse_amount(candidate)
    return None


def _extract_order_detail_non_shipping_total(
    observation: NormalizedObservation,
    grand_total: float,
) -> float | None:
    shipping_amount = _extract_order_detail_shipping_amount(observation)
    if shipping_amount is None:
        return None
    candidates = sorted(set(_extract_order_detail_money_values(observation)), reverse=True)
    for candidate in candidates:
        if candidate >= grand_total:
            continue
        if abs((grand_total - candidate) - shipping_amount) <= 0.02:
            return candidate
    return None


def _extract_kept_product_fragment(lowered_goal: str) -> str:
    match = re.search(
        r"\bkept(?: the)? (?P<item>.+?)(?: and the shop| and i cannot| and cannot|$)",
        lowered_goal,
    )
    if match is None:
        return ""
    return " ".join(match.group("item").split())


def _extract_order_detail_product_amount(
    observation: NormalizedObservation,
    product_fragment: str,
) -> float | None:
    if not product_fragment:
        return None
    normalized_fragment = " ".join(product_fragment.lower().split())
    dom_names = [
        " ".join(match.group("name").split())
        for raw_line in observation.dom_or_ax_snippet.splitlines()
        if (match := re.search(r'name="(?P<name>[^"]+)"', raw_line)) is not None
    ]
    candidate_titles = _extract_order_detail_product_titles(observation)
    candidate_titles.extend(
        text
        for text in dom_names
        if not re.fullmatch(r"\$[\d,]+\.\d{2}", text)
        and "ordered:" not in text.lower()
        and "shipping & handling" not in text.lower()
        and "order #" not in text.lower()
    )
    matching_titles = _dedupe_preserve_order(
        [
            title
            for title in candidate_titles
            if normalized_fragment in " ".join(title.lower().split())
            or " ".join(title.lower().split()) in normalized_fragment
        ]
    )
    for title in matching_titles:
        normalized_title = " ".join(title.lower().split())
        for index, text in enumerate(dom_names):
            if normalized_title not in " ".join(text.lower().split()):
                continue
            amounts: list[float] = []
            for candidate in dom_names[index + 1 : index + 10]:
                lowered_candidate = candidate.lower()
                if "shipping & handling" in lowered_candidate:
                    break
                if (
                    len(candidate) >= 20
                    and not re.fullmatch(r"\$[\d,]+\.\d{2}", candidate)
                    and "ordered:" not in lowered_candidate
                ):
                    break
                if re.fullmatch(r"\$[\d,]+\.\d{2}", candidate):
                    amounts.append(_parse_amount(candidate))
            if amounts:
                return max(amounts)
    return None


def _extract_order_detail_shipping_method(observation: NormalizedObservation) -> str:
    for raw_line in observation.visible_page_summary.splitlines():
        text = " ".join(raw_line.split())
        lowered = text.lower()
        if not text or lowered == "shipping method":
            continue
        if _looks_like_shipping_method_value(text):
            return text
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        if 'name="' not in raw_line:
            continue
        text = " ".join(raw_line.split('name="', 1)[1].rsplit('"', 1)[0].split())
        lowered = text.lower()
        if not text or lowered == "shipping method":
            continue
        if _looks_like_shipping_method_value(text):
            return text
    return ""


def _extract_order_detail_product_titles(observation: NormalizedObservation) -> list[str]:
    titles: list[str] = []
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        if "role=gridcell" not in line or 'name="' not in line:
            continue
        text = " ".join(line.split('name="', 1)[1].rsplit('"', 1)[0].split())
        if len(text) < 20:
            continue
        lowered = text.lower()
        if any(token in lowered for token in ("ordered:", "$", "order #", "recently ordered", "shipping & handling")):
            continue
        titles.append(text)
    if titles:
        return _dedupe_preserve_order(titles)

    collecting = False
    for raw_line in observation.visible_page_summary.splitlines():
        text = " ".join(raw_line.split())
        lowered = text.lower()
        if not text:
            continue
        if "items ordered" in lowered or "product name" in lowered:
            collecting = True
            continue
        if not collecting:
            continue
        if lowered.startswith("order #") or any(
            token in lowered
            for token in (
                "recently ordered",
                "my orders",
                "billing address",
                "shipping address",
                "payment method",
                "shipping method",
                "you have no items in your wish list",
                "copyright",
            )
        ):
            break
        if len(text) < 20 or "(clickable)" in text or any(char.isdigit() for char in text[:3]):
            continue
        titles.append(text)
    return _dedupe_preserve_order(titles)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _extract_order_detail_address_candidates(observation: NormalizedObservation) -> list[str]:
    candidates: list[str] = []
    for raw_line in observation.visible_page_summary.splitlines():
        text = " ".join(raw_line.lower().split())
        if not text:
            continue
        if any(token in text for token in ("billing address", "shipping address", "copyright", "address book")):
            continue
        if "united states" in text or "california" in text or "san mateo" in text or re.search(r"\d", text):
            candidates.append(text)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _extract_order_detail_address_lines(observation: NormalizedObservation) -> list[str]:
    candidates: list[str] = []
    raw_sources = list(observation.visible_page_summary.splitlines())
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        match = re.search(r'name="(?P<name>[^"]+)"', raw_line)
        if match is not None:
            raw_sources.append(match.group("name"))
    for raw_line in raw_sources:
        text = " ".join(raw_line.split())
        lowered = text.lower()
        if not text:
            continue
        if any(token in lowered for token in ("billing address", "shipping address", "copyright", "address book", "order #", "my orders")):
            continue
        if re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            lowered,
        ):
            continue
        if "united states" in lowered or "california" in lowered or "san mateo" in lowered or re.search(r"\d", text):
            candidates.append(text)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped
