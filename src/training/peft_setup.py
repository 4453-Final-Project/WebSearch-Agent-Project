from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LoRAConfig:
    """LoRA configuration for adapter-based fine-tuning."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )


def is_adapter_checkpoint(path: str | Path) -> bool:
    """Return ``True`` when ``path`` looks like a saved PEFT adapter directory."""

    return (Path(path).expanduser() / "adapter_config.json").exists()


def get_adapter_base_model_path(path: str | Path) -> str | None:
    """Read the base model path recorded in a saved adapter directory."""

    adapter_config_path = Path(path).expanduser() / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    value = data.get("base_model_name_or_path")
    return str(value) if value else None


def prepare_lora_model(model: Any, config: LoRAConfig) -> Any:
    """Wrap a loaded causal LM with PEFT LoRA adapters."""

    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model, prepare_model_for_kbit_training

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # Some GPTQ runtimes do not expose the full k-bit hooks. Continue with LoRA wrapping.
        pass

    task_type = getattr(TaskType, config.task_type, TaskType.CAUSAL_LM)
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        task_type=task_type,
        target_modules=list(config.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
