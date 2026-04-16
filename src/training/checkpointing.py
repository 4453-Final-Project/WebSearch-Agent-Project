from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CheckpointRecord:
    """Metadata emitted for one saved training checkpoint."""

    iteration: int
    optimizer_step: int
    checkpoint_path: str
    metrics_path: str


class CheckpointManager:
    """Persist LoRA adapters and training metadata in deterministic paths."""

    def __init__(self, base_dir: str | Path, save_every_iterations: int = 1) -> None:
        self.base_dir = Path(base_dir)
        self.save_every_iterations = max(1, int(save_every_iterations))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, iteration_idx: int) -> bool:
        return (iteration_idx + 1) % self.save_every_iterations == 0

    def save(
        self,
        policy: Any,
        *,
        iteration_idx: int,
        optimizer_step: int,
        metrics: dict[str, Any],
    ) -> CheckpointRecord:
        checkpoint_dir = self.base_dir / f"iter_{iteration_idx:04d}"
        adapter_dir = checkpoint_dir / "adapter"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy.save_adapter(adapter_dir)

        metrics_path = checkpoint_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

        record = CheckpointRecord(
            iteration=iteration_idx,
            optimizer_step=optimizer_step,
            checkpoint_path=str(adapter_dir),
            metrics_path=str(metrics_path),
        )
        record_path = checkpoint_dir / "checkpoint.json"
        record_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
        return record

