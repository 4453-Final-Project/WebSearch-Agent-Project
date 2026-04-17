"""Plot and summarize a staged Shopping search/sort result bundle."""

from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"


@dataclass(frozen=True)
class FamilyStage:
    label: str
    per_task: dict[str, float]

    @property
    def family_success_rate(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(self.per_task.values()) / len(self.per_task)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_stage(source: dict[str, dict[str, object]]) -> dict[str, float]:
    return {task_id: float(metrics["success_rate"]) for task_id, metrics in sorted(source.items(), key=lambda item: int(item[0]))}


def _load_baseline_eval(source_dir: Path) -> dict[str, float]:
    per_task: dict[str, float] = {}
    for metrics_path in sorted(source_dir.glob("task_*/metrics.json")):
        task_id = metrics_path.parent.name.removeprefix("task_")
        metrics = _load_json(metrics_path)
        per_task[task_id] = float(metrics["success_rate"])
    return dict(sorted(per_task.items(), key=lambda item: int(item[0])))


def _mean_metric(source: dict[str, dict[str, object]], metric_key: str) -> float:
    if not source:
        return 0.0
    values = [float(metrics[metric_key]) for metrics in source.values()]
    return sum(values) / len(values)


def build_family_stages(disjoint_dir: Path) -> list[FamilyStage]:
    baseline = _load_baseline_eval(disjoint_dir / "eval_baseline")
    compare = _load_json(disjoint_dir / "eval_compare.json")

    warmup = _normalize_stage(compare["warmup_eval"])
    grpo = _normalize_stage(compare["grpo_eval"])

    return [
        FamilyStage(label="baseline", per_task=baseline),
        FamilyStage(label="warmup_only", per_task=warmup),
        FamilyStage(label="warmup_plus_grpo", per_task=grpo),
    ]


def write_stage_csv(out_dir: Path, stages: list[FamilyStage]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shopping_searchsort_stage_metrics.csv"
    task_ids = sorted({task_id for stage in stages for task_id in stage.per_task}, key=int)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "family_success_rate", *[f"task_{task_id}" for task_id in task_ids]])
        for stage in stages:
            writer.writerow(
                [
                    stage.label,
                    f"{stage.family_success_rate:.4f}",
                    *[f"{stage.per_task[task_id]:.4f}" for task_id in task_ids],
                ]
            )
    return path


def write_summary(out_dir: Path, stages: list[FamilyStage]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        stage.label: {
            "family_success_rate": stage.family_success_rate,
            "per_task_success_rate": stage.per_task,
        }
        for stage in stages
    }
    path = out_dir / "shopping_searchsort_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_run_card(out_dir: Path, stages: list[FamilyStage], *, disjoint_dir: Path, run_name: str, model_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    warmup_summary = _load_json(disjoint_dir / "warmup_summary.json")
    rollout_summary = _load_json(disjoint_dir / "rollout_summary.json")
    grpo_summary = _load_json(disjoint_dir / "grpo_summary.json")
    eval_compare = _load_json(disjoint_dir / "eval_compare.json")
    holdout_task_ids = warmup_summary.get("holdout_task_ids", [])
    if not holdout_task_ids:
        train_task_ids = set(warmup_summary["warmup_task_ids"]) | set(warmup_summary["grpo_task_ids"])
        holdout_task_ids = [task_id for task_id in warmup_summary["eval_task_ids"] if task_id not in train_task_ids]

    run_card = {
        "run_name": run_name,
        "model": model_name,
        "task_family": "shopping_searchsort_324_328",
        "task_split": {
            "warmup_task_ids": warmup_summary["warmup_task_ids"],
            "grpo_task_ids": warmup_summary["grpo_task_ids"],
            "eval_task_ids": warmup_summary["eval_task_ids"],
            "holdout_task_ids": holdout_task_ids,
        },
        "warmup": {
            "selected_demo_count": warmup_summary["selected_demo_count"],
            "selected_demo_task_counts": warmup_summary["selected_demo_task_counts"],
            "epochs": warmup_summary["warmup_metrics"]["epochs"],
            "final_loss": warmup_summary["warmup_metrics"]["final_loss"],
            "success_demo_count": warmup_summary["warmup_metrics"]["success_demo_count"],
            "family_success_rate": stages[1].family_success_rate,
            "average_steps": _mean_metric(eval_compare["warmup_eval"], "average_steps"),
        },
        "rollout_collection": {
            "episode_count": rollout_summary["episode_count"],
            "success_count": rollout_summary["success_count"],
            "success_rate": rollout_summary["success_count"] / rollout_summary["episode_count"],
            "task_success_counts": rollout_summary["task_success_counts"],
            "average_steps_taken": sum(float(episode["steps_taken"]) for episode in rollout_summary["episodes"]) / rollout_summary["episode_count"],
        },
        "grpo": {
            "iteration_count": len(grpo_summary),
            "last_iteration": grpo_summary[-1],
            "family_success_rate": stages[2].family_success_rate,
            "average_steps": _mean_metric(eval_compare["grpo_eval"], "average_steps"),
        },
        "baseline": {
            "family_success_rate": stages[0].family_success_rate,
            "per_task_success_rate": stages[0].per_task,
        },
        "final_eval": {
            "warmup_only_per_task": stages[1].per_task,
            "warmup_plus_grpo_per_task": stages[2].per_task,
        },
        "holdout": _build_holdout_run_card(stages, holdout_task_ids),
    }
    path = out_dir / "shopping_searchsort_run_card.json"
    path.write_text(json.dumps(run_card, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_holdout_run_card(stages: list[FamilyStage], holdout_task_ids: list[int]) -> dict[str, object]:
    if not holdout_task_ids:
        return {
            "task_ids": [],
            "baseline_success_rate": 0.0,
            "warmup_success_rate": 0.0,
            "grpo_success_rate": 0.0,
            "gain_vs_baseline": 0.0,
            "gain_vs_warmup": 0.0,
            "per_task": {},
        }

    holdout_keys = [str(task_id) for task_id in holdout_task_ids]
    baseline = _subset_mean(stages[0].per_task, holdout_keys)
    warmup = _subset_mean(stages[1].per_task, holdout_keys)
    grpo = _subset_mean(stages[2].per_task, holdout_keys)
    per_task = {
        task_id: {
            "baseline_success_rate": stages[0].per_task[task_id],
            "warmup_success_rate": stages[1].per_task[task_id],
            "grpo_success_rate": stages[2].per_task[task_id],
            "gain_vs_baseline": stages[2].per_task[task_id] - stages[0].per_task[task_id],
            "gain_vs_warmup": stages[2].per_task[task_id] - stages[1].per_task[task_id],
        }
        for task_id in holdout_keys
    }
    return {
        "task_ids": holdout_task_ids,
        "baseline_success_rate": baseline,
        "warmup_success_rate": warmup,
        "grpo_success_rate": grpo,
        "gain_vs_baseline": grpo - baseline,
        "gain_vs_warmup": grpo - warmup,
        "per_task": per_task,
    }


def _subset_mean(per_task: dict[str, float], task_ids: list[str]) -> float:
    if not task_ids:
        return 0.0
    values = [per_task[task_id] for task_id in task_ids]
    return sum(values) / len(values)


def plot_family_success(out_dir: Path, stages: list[FamilyStage], *, title_suffix: str) -> Path:
    labels = [stage.label for stage in stages]
    values = [stage.family_success_rate for stage in stages]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(labels, values, color=["#6b7280", "#2563eb", "#16a34a"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Shopping Search/Sort Family Success ({title_suffix})")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    path = out_dir / "shopping_searchsort_family_success.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_per_task_success(out_dir: Path, stages: list[FamilyStage]) -> Path:
    task_ids = sorted(stages[0].per_task, key=int)
    x_positions = list(range(len(task_ids)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    colors = ["#6b7280", "#2563eb", "#16a34a"]
    stage_count = len(stages)
    offsets = [(idx - (stage_count - 1) / 2) * width for idx in range(stage_count)]
    for color, offset, stage in zip(colors, offsets, stages):
        values = [stage.per_task[task_id] for task_id in task_ids]
        bars = [position + offset for position in x_positions]
        ax.bar(bars, values, width=width, label=stage.label, color=color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_ids)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Success Rate")
    ax.set_title("Shopping Search/Sort Per-Task Success")
    ax.legend()

    path = out_dir / "shopping_searchsort_per_task_success.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot and summarize a staged Shopping search/sort run.")
    parser.add_argument("--run-dir", default=str(OUTPUTS_ROOT / "liquid_shopping_disjoint_v1"))
    parser.add_argument("--out-dir", default=str(OUTPUTS_ROOT / "shopping_report_figures"))
    parser.add_argument("--run-name", default="liquid_shopping_disjoint_v1")
    parser.add_argument("--model-name", default="LFM2.5-350M")
    parser.add_argument("--title-suffix", default="Disjoint Run")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    disjoint_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = build_family_stages(disjoint_dir)
    summary_path = write_summary(out_dir, stages)
    csv_path = write_stage_csv(out_dir, stages)
    run_card_path = write_run_card(
        out_dir,
        stages,
        disjoint_dir=disjoint_dir,
        run_name=args.run_name,
        model_name=args.model_name,
    )
    family_plot = plot_family_success(out_dir, stages, title_suffix=args.title_suffix)
    per_task_plot = plot_per_task_success(out_dir, stages)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "csv": str(csv_path),
                "run_card": str(run_card_path),
                "family_plot": str(family_plot),
                "per_task_plot": str(per_task_plot),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
