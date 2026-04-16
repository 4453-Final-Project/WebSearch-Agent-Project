from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from src.agent.compat import normalize_observation
from src.agent.prompting import build_site_hints, build_system_prompt, describe_editable_controls

from .trajectory import EpisodeTrajectory, load_episode_trajectory


@dataclass(slots=True)
class WarmupConfig:
    """Configuration for supervised behavior-cloning warm-start."""

    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    success_only: bool = True
    shuffle_seed: int = 0
    max_visible_chars: int = 320
    max_dom_chars: int = 480
    max_history_items: int = 3


@dataclass(slots=True)
class WarmupMetrics:
    """Compact metrics emitted after warmup."""

    epochs: int
    episodes_used: int
    samples_used: int
    success_demo_count: int
    average_demo_reward: float
    final_loss: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class WarmupSample:
    prompt: str
    response_text: str


def discover_episode_dirs(paths: Sequence[str | Path]) -> list[Path]:
    """Find runner-format episode directories under the provided paths."""

    episode_dirs: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Warmup demo path does not exist: {path}")

        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path.parent]
        else:
            candidates = [path, *path.rglob("*")]

        for candidate in candidates:
            if not candidate.is_dir():
                continue
            episode_json = candidate / "episode.json"
            steps_jsonl = candidate / "steps.jsonl"
            if episode_json.exists() and steps_jsonl.exists():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    episode_dirs.append(candidate)
    return sorted(episode_dirs)


def load_demo_trajectories(paths: Sequence[str | Path], *, success_only: bool = True) -> list[EpisodeTrajectory]:
    """Load runner artifacts and optionally keep only successful demos."""

    trajectories = [load_episode_trajectory(path) for path in discover_episode_dirs(paths)]
    if success_only:
        trajectories = [trajectory for trajectory in trajectories if trajectory.success]
    return trajectories


def build_warmup_samples(
    trajectories: Sequence[EpisodeTrajectory],
    *,
    model_system_prompt: str | None = None,
) -> list[WarmupSample]:
    """Convert demo trajectories into supervised prompt/response pairs."""

    samples: list[WarmupSample] = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            if not step.response_text:
                continue
            samples.append(
                WarmupSample(
                    prompt=build_compact_warmup_prompt(step, model_system_prompt=model_system_prompt),
                    response_text=step.response_text,
                )
            )
    return samples


def run_supervised_warmup(policy, trajectories: Sequence[EpisodeTrajectory], config: WarmupConfig) -> dict[str, float | int]:
    """Warm-start the trainable policy with behavior cloning over demo traces."""

    if config.epochs <= 0:
        return WarmupMetrics(
            epochs=0,
            episodes_used=0,
            samples_used=0,
            success_demo_count=0,
            average_demo_reward=0.0,
            final_loss=0.0,
        ).to_dict()

    usable_trajectories = [trajectory for trajectory in trajectories if trajectory.success or not config.success_only]
    model_system_prompt = getattr(getattr(policy, "config", None), "system_prompt", None)
    samples = build_warmup_samples(usable_trajectories, model_system_prompt=model_system_prompt)
    if not usable_trajectories or not samples:
        raise RuntimeError("Warmup requested but no usable demonstration trajectories were found.")

    rng = random.Random(config.shuffle_seed)
    final_loss = 0.0
    accumulation_steps = max(1, config.gradient_accumulation_steps)
    for _ in range(config.epochs):
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        shuffled = [samples[index] for index in indices]
        policy.zero_grad()
        pending_steps = 0

        for batch_start in range(0, len(shuffled), config.batch_size):
            batch = shuffled[batch_start : batch_start + config.batch_size]
            prompts = [sample.prompt for sample in batch]
            responses = [sample.response_text for sample in batch]
            logprobs = policy.score_responses(prompts, responses, use_reference=False)
            loss = -torch.mean(logprobs)
            scaled_loss = loss / accumulation_steps

            scaled_loss.backward()
            final_loss = float(loss.detach().item())
            pending_steps += 1

            is_last_batch = batch_start + config.batch_size >= len(shuffled)
            if pending_steps >= accumulation_steps or is_last_batch:
                policy.clip_grad_norm_(config.max_grad_norm)
                policy.step_optimizer()
                policy.zero_grad()
                pending_steps = 0

    success_demo_count = sum(int(trajectory.success) for trajectory in usable_trajectories)
    average_demo_reward = sum(trajectory.reward for trajectory in usable_trajectories) / max(1, len(usable_trajectories))
    return WarmupMetrics(
        epochs=config.epochs,
        episodes_used=len(usable_trajectories),
        samples_used=len(samples),
        success_demo_count=success_demo_count,
        average_demo_reward=float(average_demo_reward),
        final_loss=final_loss,
    ).to_dict()


def build_compact_warmup_prompt(step, *, model_system_prompt: str | None = None) -> str:
    """Create a shorter prompt for supervised warmup to keep memory manageable."""

    observation = step.observation or {}
    previous_actions = observation.get("previous_actions", [])
    previous_errors = observation.get("previous_errors", [])
    sections = [
        f"Step: {step.step_idx}",
        f"Task Goal: {_truncate_text(str(observation.get('goal', '')), 240)}",
        f"Current URL: {observation.get('current_url', '')}",
        "Visible Page Summary:",
        _render_block(_truncate_text(str(observation.get("visible_page_summary", "")), 320)),
        "Relevant DOM or AX-Tree Snippet:",
        _render_block(_truncate_text(str(observation.get("dom_or_ax_snippet", "")), 480)),
        "Previous Actions:",
        _render_history(previous_actions, max_items=3),
        "Previous Errors:",
        _render_history(previous_errors, max_items=3),
    ]
    last_action_error = observation.get("last_action_error")
    if last_action_error:
        sections.extend(
            [
                "Last Action Error:",
                _render_block(_truncate_text(str(last_action_error), 240)),
            ]
        )

    sections.extend(
        [
            "Output Contract:",
            "Respond with exactly one line.",
            "Use this exact format and nothing else:",
            "ACTION: <browser action string>",
            "For click, fill, hover, select_option, and press, use the bid from the DOM or AX-Tree snippet, not the label text.",
            "Only use fill on editable controls such as role=textbox, role=searchbox, role=combobox, input, textarea, or elements explicitly marked editable.",
            "Never use fill on headings, generic containers, buttons, links, list items, or non-editable search regions.",
            "Use only these action names:",
            "click, fill, hover, scroll, press, goto, go_back, go_forward, select_option, send_msg_to_user, report_infeasible, noop",
            "Do not use tab_focus or tab_close.",
            "Do not guess or invent URLs.",
            "If you use goto, stay on the WebArena host and avoid made-up GitLab paths.",
        ]
    )
    editable_hints = describe_editable_controls(str(observation.get("dom_or_ax_snippet", "")))
    if editable_hints:
        sections.extend(
            [
                "Editable Controls In Current DOM:",
                editable_hints,
            ]
        )
    site_hints = build_site_hints(normalize_observation(observation))
    if site_hints:
        sections.extend(
            [
                "Navigation Hints:",
                site_hints,
            ]
        )
    return build_system_prompt(model_system_prompt) + "\n\n" + "\n".join(sections)


def _truncate_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _render_block(text: str) -> str:
    return text if text else "(empty)"


def _render_history(items, *, max_items: int) -> str:
    if not items:
        return "(none)"
    trimmed = list(items)[-max_items:]
    return "\n".join(f"{index}. {item}" for index, item in enumerate(trimmed, start=1))
