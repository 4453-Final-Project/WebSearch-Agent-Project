"""Task 3 Qwen policy with lazy local Transformers-backed generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from src.training.peft_setup import get_adapter_base_model_path, is_adapter_checkpoint

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
    derive_shopping_sort_hint,
    derive_shopping_category_labels,
    derive_shopping_query_hint,
    build_retry_prompt,
)
from .types import AgentDecision, NormalizedObservation, PolicyConfig


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

        print(f"Status: loading {self.config.policy_name} policy tokenizer from {self.model_path}...", flush=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Status: loading {self.config.policy_name} policy model from {self.model_path}...", flush=True)
        model_kwargs: dict[str, object] = {
            "local_files_only": True,
            "device_map": self.config.device or "auto",
            "torch_dtype": "auto",
        }
        if is_adapter_checkpoint(self.model_path):
            from peft import PeftModel

            base_model_path = get_adapter_base_model_path(self.model_path)
            if not base_model_path:
                raise RuntimeError(f"Adapter checkpoint at {self.model_path} is missing base_model_name_or_path.")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs,
            )
            model = PeftModel.from_pretrained(model, self.model_path, is_trainable=False)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs,
            )

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

    def _import_local_inference_helpers(self):
        """Import shared local-inference helpers only when generation is attempted."""

        from src.utils.local_inference import (
            build_generation_kwargs,
            build_model_inputs,
            fast_generation_mode,
            move_inputs_to_model_device,
        )

        return build_model_inputs, build_generation_kwargs, fast_generation_mode, move_inputs_to_model_device


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
        shopping_target_action = _build_shopping_canonical_target_action(observation)
        shopping_order_detail_action = _build_shopping_order_detail_retry_action(observation)
        if decision.parse_error is not None:
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
            if shopping_order_detail_action:
                return AgentDecision(
                    raw_text=decision.raw_text,
                    action_text=shopping_order_detail_action.split("ACTION:", 1)[1].strip(),
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
            return AgentDecision(
                raw_text=decision.raw_text,
                action_text=decision.action_text,
                parse_error=(
                    "Do not use fill on Shopping order-detail pages. Read the visible order details and answer with "
                    'ACTION: send_msg_to_user("...") instead.'
                ),
                should_retry=True,
            )

        if action_name in {"go_back", "go_forward"} and _is_shopping_order_detail_page(observation.current_url):
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


def _strip_quoted_string(value: str) -> str | None:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in {'"', "'"} or stripped[-1] != stripped[0]:
        return None
    return stripped[1:-1]


def _current_gitlab_explore_query(current_url: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023") or not (parsed.path or "").startswith("/explore"):
        return ""
    query_values = parse_qs(parsed.query).get("name", [])
    if not query_values:
        return ""
    return query_values[0].strip().lower()


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
    return "most contributions" in lowered or "number of commits" in lowered


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
    candidates: list[str] = []
    for raw_line in observation.dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        if "role=gridcell" not in line:
            continue
        if 'name="' not in line:
            continue
        text = " ".join(line.split('name="', 1)[1].rsplit('"', 1)[0].lower().split())
        if len(text) < 20:
            continue
        if any(token in text for token in ("ordered:", "$", "order #", "recently ordered")):
            continue
        candidates.append(text)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
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
