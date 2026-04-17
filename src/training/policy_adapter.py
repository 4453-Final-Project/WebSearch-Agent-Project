from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from src.agent.compat import load_policy_config, normalize_observation
from src.agent.parsing import make_decision, make_fallback_decision
from src.agent.prompting import build_full_prompt, build_retry_prompt
from src.agent.qwen_policy import QwenPolicy
from src.agent.types import AgentDecision, NormalizedObservation, PolicyConfig
from src.utils.local_inference import (
    build_generation_kwargs,
    build_model_inputs,
    fast_generation_mode,
    move_inputs_to_model_device,
    render_prompt_text,
)

from .peft_setup import LoRAConfig, get_adapter_base_model_path, is_adapter_checkpoint, prepare_lora_model


@dataclass(slots=True)
class PolicyAdapterConfig:
    """Config for a trainable Qwen policy with LoRA adapters."""

    model_path: str
    policy_name: str = "qwen-grpo"
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_k: int | None = None
    repetition_penalty: float | None = None
    system_prompt: str | None = None
    use_chat_template: bool = True
    device: str | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    score_batch_size: int = 1
    max_supervised_tokens: int = 512
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class TrainableQwenPolicy:
    """Trainable policy wrapper reused for rollout collection and evaluation."""

    def __init__(self, config: PolicyAdapterConfig | None = None) -> None:
        base_config = load_policy_config()
        self.config = config if config is not None else PolicyAdapterConfig(
            model_path=base_config.model_path,
            policy_name=base_config.policy_name,
            max_new_tokens=base_config.max_new_tokens,
            temperature=max(base_config.temperature, 0.2),
            top_k=base_config.top_k,
            repetition_penalty=base_config.repetition_penalty,
            system_prompt=base_config.system_prompt,
            use_chat_template=base_config.use_chat_template,
            device=base_config.device,
        )
        self.name = self.config.policy_name

        self.model_path = Path(self.config.model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Configured model path does not exist: {self.model_path}")

        self.tokenizer = None
        self.model = None
        self.optimizer = None

    def act(self, observation: dict[str, Any] | NormalizedObservation, step_idx: int) -> AgentDecision:
        normalized = normalize_observation(observation) if isinstance(observation, dict) else observation
        prompt = build_full_prompt(normalized, step_idx, model_system_prompt=self.config.system_prompt)
        first_raw_text = self.generate_raw(prompt)
        first_decision = make_decision(first_raw_text)
        first_decision = QwenPolicy._validate_decision_against_observation(self, first_decision, normalized)
        if first_decision.parse_error is None:
            return first_decision

        retry_prompt = build_retry_prompt(
            normalized,
            step_idx,
            previous_raw_text=first_decision.raw_text,
            parse_error=first_decision.parse_error,
            model_system_prompt=self.config.system_prompt,
        )
        second_raw_text = self.generate_raw(retry_prompt)
        second_decision = make_decision(second_raw_text)
        second_decision = QwenPolicy._validate_decision_against_observation(self, second_decision, normalized)
        if second_decision.parse_error is None:
            return second_decision

        return make_fallback_decision(
            raw_text=second_decision.raw_text,
            reason=second_decision.parse_error or "Unknown parse failure.",
        )

    def generate_raw(self, prompt: str) -> str:
        tokenizer, model = self._ensure_loaded()
        generation_kwargs = build_generation_kwargs(
            tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )

        inputs = self._tokenize_prompt(prompt, tokenizer)
        inputs = move_inputs_to_model_device(inputs, model)
        with fast_generation_mode(model), torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        prompt_token_count = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_token_count:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def score_responses(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        use_reference: bool = False,
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """Compute mean completion logprobs for prompt/response pairs."""

        if not prompts:
            return torch.empty(0, dtype=torch.float32, device=self._model_device())

        tokenizer, model = self._ensure_loaded()
        context_manager = self._reference_context() if use_reference else nullcontext()
        grad_manager = torch.no_grad() if use_reference or not requires_grad else nullcontext()
        model.train(requires_grad and not use_reference)

        results: list[torch.Tensor] = []
        chunk_size = max(1, self.config.score_batch_size)
        with context_manager, grad_manager:
            for batch_start in range(0, len(prompts), chunk_size):
                batch_end = batch_start + chunk_size
                batch = self._tokenize_supervised_batch(
                    prompts[batch_start:batch_end],
                    responses[batch_start:batch_end],
                    tokenizer,
                )
                batch = {key: value.to(self._model_device()) for key, value in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                valid_mask = labels != -100
                safe_labels = labels.masked_fill(~valid_mask, 0)
                token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                token_logprobs = token_logprobs * valid_mask
                token_counts = valid_mask.sum(dim=1).clamp(min=1)
                results.append(token_logprobs.sum(dim=1) / token_counts)
                del outputs, logits, labels, valid_mask, safe_labels, token_logprobs, token_counts, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return torch.cat(results, dim=0)

    def zero_grad(self) -> None:
        self._ensure_optimizer()
        self.optimizer.zero_grad(set_to_none=True)

    def step_optimizer(self) -> None:
        self._ensure_optimizer()
        self.optimizer.step()

    def clip_grad_norm_(self, max_grad_norm: float) -> float:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm))

    def save_adapter(self, output_dir: str | Path) -> None:
        _, model = self._ensure_loaded()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def _ensure_loaded(self):
        if self.tokenizer is not None and self.model is not None:
            return self.tokenizer, self.model

        import transformers

        load_device = self._resolve_load_device()
        model_kwargs = self._build_trainable_model_kwargs()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if is_adapter_checkpoint(self.model_path):
            from peft import PeftModel

            base_model_path = get_adapter_base_model_path(self.model_path)
            if not base_model_path:
                raise RuntimeError(f"Adapter checkpoint at {self.model_path} is missing base_model_name_or_path.")
            model = transformers.AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
            model = PeftModel.from_pretrained(model, self.model_path, is_trainable=True)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
            model = prepare_lora_model(model, self.config.lora)
        model = model.to(load_device)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        self.tokenizer = tokenizer
        self.model = model
        return tokenizer, model

    def _ensure_optimizer(self) -> None:
        self._ensure_loaded()
        if self.optimizer is not None:
            return
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _move_inputs_to_model_device(self, inputs):
        return inputs.to(self._model_device())

    def _tokenize_prompt(self, prompt: str, tokenizer):
        system_prompt, user_prompt = self._split_prompt(prompt)
        return build_model_inputs(
            tokenizer,
            prompt=user_prompt,
            system_prompt=system_prompt or None,
            use_chat_template=self.config.use_chat_template,
        )

    def _model_device(self) -> torch.device:
        _, model = self._ensure_loaded()
        if hasattr(model, "device") and str(model.device) != "meta":
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _reference_context(self):
        _, model = self._ensure_loaded()
        if hasattr(model, "disable_adapter"):
            return model.disable_adapter()
        return nullcontext()

    def _resolve_load_device(self) -> torch.device:
        configured_device = self.config.device
        if configured_device:
            return torch.device(configured_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_trainable_model_kwargs(self) -> dict[str, Any]:
        return {
            "local_files_only": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": False,
        }

    def _split_prompt(self, prompt: str) -> tuple[str, str]:
        separator = "\n\n"
        if separator not in prompt:
            return "", prompt
        system_prompt, user_prompt = prompt.split(separator, 1)
        return system_prompt.strip(), user_prompt.strip()

    def _tokenize_supervised_batch(self, prompts: Sequence[str], responses: Sequence[str], tokenizer) -> dict[str, torch.Tensor]:
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        input_id_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []

        for prompt, response in zip(prompts, responses):
            system_prompt, user_prompt = self._split_prompt(prompt)
            rendered_prompt = render_prompt_text(
                tokenizer,
                prompt=user_prompt,
                system_prompt=system_prompt or None,
                use_chat_template=self.config.use_chat_template,
            )
            prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            if eos_token_id is not None:
                response_ids = list(response_ids) + [eos_token_id]

            max_total_tokens = max(8, self.config.max_supervised_tokens)
            available_prompt_tokens = max(0, max_total_tokens - len(response_ids))
            if len(prompt_ids) > available_prompt_tokens:
                prompt_ids = prompt_ids[-available_prompt_tokens:] if available_prompt_tokens > 0 else []

            input_ids = list(prompt_ids) + list(response_ids)
            labels = ([-100] * len(prompt_ids)) + list(response_ids)
            attention = [1] * len(input_ids)

            input_id_rows.append(input_ids)
            label_rows.append(labels)
            attention_rows.append(attention)

        max_length = max(len(row) for row in input_id_rows)
        padded_input_ids: list[list[int]] = []
        padded_labels: list[list[int]] = []
        padded_attention: list[list[int]] = []

        for input_ids, labels, attention in zip(input_id_rows, label_rows, attention_rows):
            pad_length = max_length - len(input_ids)
            padded_input_ids.append(input_ids + ([pad_token_id] * pad_length))
            padded_labels.append(labels + ([-100] * pad_length))
            padded_attention.append(attention + ([0] * pad_length))

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        }
