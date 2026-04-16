"""Plot the main Liquid 350M experiment outcomes from saved output folders."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"


@dataclass(frozen=True)
class SuccessSeries:
    label: str
    stages: list[str]
    values: list[float]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_success_series() -> list[SuccessSeries]:
    task310_summary = _load_json(OUTPUTS_ROOT / "liquid_grpo_task310_v1" / "summary.json")
    weighted_replay_grpo_310 = _load_json(
        OUTPUTS_ROOT / "liquid_staged_e2_weighted_replay_v1" / "fair_liquid_grpo_310" / "metrics.json"
    )
    weighted_replay_warmup_310 = _load_json(
        OUTPUTS_ROOT / "liquid_staged_e2_weighted_replay_v1" / "fair_liquid_warmup_310" / "metrics.json"
    )
    map_base_72 = _load_json(OUTPUTS_ROOT / "map_transfer_probe" / "base_72" / "metrics.json")
    map_e1_72 = _load_json(OUTPUTS_ROOT / "map_transfer_probe_e1_promptfix" / "eval_72" / "metrics.json")
    map_e1_70 = _load_json(OUTPUTS_ROOT / "map_transfer_probe_e1_promptfix" / "eval_70" / "metrics.json")
    gitlab_312_base = _load_json(OUTPUTS_ROOT / "gitlab_contrib_probe312_base" / "metrics.json")
    gitlab_312_direct = _load_json(OUTPUTS_ROOT / "gitlab_contrib_probe_v1" / "eval_312" / "metrics.json")
    gitlab_312_search = _load_json(OUTPUTS_ROOT / "gitlab_contrib_search_probe_v1" / "eval_312" / "metrics.json")

    return [
        SuccessSeries(
            label="Task 310",
            stages=["baseline", "warmup", "grpo"],
            values=[
                float(task310_summary["before_t06"]["success_rate"]),
                float(task310_summary["after_warmup_t06"]["success_rate"]),
                float(task310_summary["after_grpo_t06"]["success_rate"]),
            ],
        ),
        SuccessSeries(
            label="Task 310 staged replay",
            stages=["warmup", "grpo"],
            values=[
                float(weighted_replay_warmup_310["success_rate"]),
                float(weighted_replay_grpo_310["success_rate"]),
            ],
        ),
        SuccessSeries(
            label="Map transfer",
            stages=["baseline t72", "warmup t70", "warmup t72"],
            values=[
                float(map_base_72["success_rate"]),
                float(map_e1_70["success_rate"]),
                float(map_e1_72["success_rate"]),
            ],
        ),
        SuccessSeries(
            label="GitLab held-out 312",
            stages=["baseline", "direct warmup", "search warmup"],
            values=[
                float(gitlab_312_base["success_rate"]),
                float(gitlab_312_direct["success_rate"]),
                float(gitlab_312_search["success_rate"]),
            ],
        ),
    ]


def build_grpo_diagnostics() -> tuple[list[str], list[float], list[float]]:
    collapsed = _load_json(OUTPUTS_ROOT / "liquid_staged_e2_full_v1" / "grpo_summary.json")[0]
    replay = _load_json(OUTPUTS_ROOT / "liquid_staged_e2_weighted_replay_v1" / "grpo_summary.json")[0]
    labels = ["collapsed", "weighted replay"]
    approx_kl = [float(collapsed["approx_kl"]), float(replay["approx_kl"])]
    ratio_mean = [float(collapsed["ratio_mean"]), float(replay["ratio_mean"])]
    return labels, approx_kl, ratio_mean


def plot_success_rates(out_dir: Path) -> Path:
    series_list = build_success_series()
    fig, axes = plt.subplots(len(series_list), 1, figsize=(10, 10), constrained_layout=True)
    if len(series_list) == 1:
        axes = [axes]

    for axis, series in zip(axes, series_list):
        axis.bar(series.stages, series.values, color="#3b82f6")
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Success")
        axis.set_title(series.label)
        for idx, value in enumerate(series.values):
            axis.text(idx, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    output_path = out_dir / "liquid_success_rates.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_grpo_diagnostics(out_dir: Path) -> Path:
    labels, approx_kl, ratio_mean = build_grpo_diagnostics()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].bar(labels, approx_kl, color="#ef4444")
    axes[0].set_title("Approx KL")
    for idx, value in enumerate(approx_kl):
        axes[0].text(idx, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, ratio_mean, color="#10b981")
    axes[1].set_title("Ratio Mean")
    for idx, value in enumerate(ratio_mean):
        axes[1].text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    output_path = out_dir / "liquid_grpo_diagnostics.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> int:
    out_dir = OUTPUTS_ROOT / "liquid_report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    success_plot = plot_success_rates(out_dir)
    diagnostics_plot = plot_grpo_diagnostics(out_dir)
    print(json.dumps({"success_plot": str(success_plot), "diagnostics_plot": str(diagnostics_plot)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
