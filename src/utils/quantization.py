from __future__ import annotations

from typing import Any


def build_bitsandbytes_quantization_config(
    *,
    torch_module: Any,
    transformers_module: Any,
    quantization_mode: str | None,
    quant_compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    quant_use_double_quant: bool = True,
) -> object | None:
    """Build an optional bitsandbytes quantization config for Transformers loads."""

    normalized_mode = (quantization_mode or "").strip().lower()
    if not normalized_mode:
        return None

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Quantized loading requires the optional 'bitsandbytes' package. "
            "Install bitsandbytes in the training environment before enabling TASK3_QUANTIZATION_MODE."
        ) from exc

    bitsandbytes_config = getattr(transformers_module, "BitsAndBytesConfig", None)
    if bitsandbytes_config is None:
        raise RuntimeError(
            "Quantized loading requires transformers.BitsAndBytesConfig, but it is unavailable "
            "in the active Transformers installation."
        )

    resolved_compute_dtype = _resolve_torch_dtype(torch_module, quant_compute_dtype)

    if normalized_mode in {"bnb_4bit", "4bit", "qlora"}:
        return bitsandbytes_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolved_compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=quant_use_double_quant,
        )
    if normalized_mode in {"bnb_8bit", "8bit"}:
        return bitsandbytes_config(load_in_8bit=True)

    raise ValueError(
        f"Unsupported quantization mode: {quantization_mode!r}. "
        "Supported values are bnb_4bit/qlora or bnb_8bit."
    )


def resolve_quantized_device_map(load_device: Any) -> dict[str, Any]:
    """Return a single-device map for quantized model loads."""

    if isinstance(load_device, str):
        normalized = load_device.strip().lower()
        if normalized.startswith("cuda"):
            if ":" in normalized:
                _, raw_index = normalized.split(":", 1)
                if raw_index.isdigit():
                    return {"": int(raw_index)}
            return {"": 0}
        return {"": load_device}

    device_type = getattr(load_device, "type", None)
    device_index = getattr(load_device, "index", None)
    if device_type == "cuda":
        return {"": device_index if device_index is not None else 0}
    return {"": str(load_device)}


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> object:
    normalized_name = (dtype_name or "").strip().lower()
    if not normalized_name:
        normalized_name = "bfloat16"
    resolved = getattr(torch_module, normalized_name, None)
    if resolved is not None:
        return resolved
    fallback = getattr(torch_module, "bfloat16", None)
    if fallback is not None:
        return fallback
    return getattr(torch_module, "float16")
