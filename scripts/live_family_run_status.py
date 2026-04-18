"""Summarize live progress for a family curriculum run directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic  # noqa: E402
from src.training.task_families import get_family_known_benchmark_blockers, get_family_task_groups  # noqa: E402

NON_MONITORING_METADATA_FILENAMES = {
    "live_status.json",
    "run.pid",
    "tmux_session.txt",
    "run_commit.txt",
    "run_branch.txt",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize live family-run progress from output artifacts.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--write-out", type=Path, default=None)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _windows_path_to_wsl(path: Path) -> str | None:
    path_str = str(path)
    if len(path_str) >= 3 and path_str[1:3] == ":\\":
        drive = path_str[0].lower()
        rest = path_str[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return None


def _probe_pid(run_pid: int) -> dict[str, object]:
    if os.name == "nt":
        command = ["wsl.exe", "-e", "bash", "-lc", f"ps -p {run_pid} -o pid=,stat=,cmd="]
        backend = "wsl_ps"
    else:
        command = ["ps", "-p", str(run_pid), "-o", "pid=,stat=,command="]
        backend = "ps"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "run_pid": run_pid,
            "run_pid_alive": None,
            "probe_backend": backend,
            "probe_output": None,
        }

    output = (completed.stdout or "").strip()
    return {
        "run_pid": run_pid,
        "run_pid_alive": completed.returncode == 0 and bool(output),
        "probe_backend": backend,
        "probe_output": output or None,
    }


def _discover_run_process(*, family_name: str | None, out_dir: Path) -> dict[str, object] | None:
    if not isinstance(family_name, str):
        return None

    if os.name == "nt":
        command = ["wsl.exe", "-e", "bash", "-lc", "ps -ef"]
        backend = "wsl_ps_scan"
    else:
        command = ["ps", "-ef"]
        backend = "ps_scan"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = (completed.stdout or "").strip()
    if completed.returncode != 0 or not output:
        return None

    out_dir_variants = {str(out_dir)}
    wsl_out_dir = _windows_path_to_wsl(out_dir)
    if wsl_out_dir:
        out_dir_variants.add(wsl_out_dir)

    for line in output.splitlines():
        if "scripts/run_family_curriculum.py" not in line:
            continue
        if f"--family {family_name}" not in line:
            continue
        if not any(out_dir_variant in line for out_dir_variant in out_dir_variants):
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        try:
            discovered_pid = int(parts[1])
        except ValueError:
            continue
        return {
            "run_pid": discovered_pid,
            "run_pid_alive": True,
            "probe_backend": backend,
            "probe_output": line.strip(),
        }

    return None


def _summarize_run_process(run_pid_path: Path, *, family_name: str | None, out_dir: Path) -> dict[str, object]:
    if not run_pid_path.exists():
        discovered = _discover_run_process(family_name=family_name, out_dir=out_dir)
        if discovered is None:
            return {
                "run_pid_present": False,
                "saved_run_pid": None,
                "run_pid": None,
                "run_pid_alive": None,
                "run_pid_source": "missing",
                "probe_backend": None,
                "probe_output": None,
            }
        return {
            "run_pid_present": False,
            "saved_run_pid": None,
            "run_pid": discovered["run_pid"],
            "run_pid_alive": True,
            "run_pid_source": "scan",
            "probe_backend": discovered["probe_backend"],
            "probe_output": discovered["probe_output"],
        }

    raw_pid = run_pid_path.read_text(encoding="utf-8").strip()
    try:
        saved_run_pid: int | str | None = int(raw_pid)
    except ValueError:
        return {
            "run_pid_present": True,
            "saved_run_pid": raw_pid,
            "run_pid": raw_pid,
            "run_pid_alive": None,
            "run_pid_source": "invalid",
            "probe_backend": "invalid",
            "probe_output": None,
        }

    probed = _probe_pid(saved_run_pid)
    if probed["run_pid_alive"]:
        return {
            "run_pid_present": True,
            "saved_run_pid": saved_run_pid,
            "run_pid": saved_run_pid,
            "run_pid_alive": True,
            "run_pid_source": "pid_file",
            "probe_backend": probed["probe_backend"],
            "probe_output": probed["probe_output"],
        }

    discovered = _discover_run_process(family_name=family_name, out_dir=out_dir)
    if discovered is not None:
        return {
            "run_pid_present": True,
            "saved_run_pid": saved_run_pid,
            "run_pid": discovered["run_pid"],
            "run_pid_alive": True,
            "run_pid_source": "scan",
            "probe_backend": discovered["probe_backend"],
            "probe_output": discovered["probe_output"],
        }

    return {
        "run_pid_present": True,
        "saved_run_pid": saved_run_pid,
        "run_pid": saved_run_pid,
        "run_pid_alive": False,
        "run_pid_source": "pid_file",
        "probe_backend": probed["probe_backend"],
        "probe_output": probed["probe_output"],
    }


def _summarize_tmux_session(tmux_session_path: Path) -> dict[str, object]:
    if not tmux_session_path.exists():
        return {
            "session_name_present": False,
            "session_name": None,
            "session_alive": None,
            "probe_backend": None,
            "probe_output": None,
            "pane_pid": None,
            "pane_command": None,
        }

    session_name = tmux_session_path.read_text(encoding="utf-8").strip()
    if not session_name:
        return {
            "session_name_present": True,
            "session_name": "",
            "session_alive": None,
            "probe_backend": "invalid",
            "probe_output": None,
            "pane_pid": None,
            "pane_command": None,
        }

    if os.name == "nt":
        command = [
            "wsl.exe",
            "-e",
            "bash",
            "-lc",
            f"tmux list-panes -t {session_name} -F '#{{pane_pid}}\t#{{pane_current_command}}'",
        ]
        backend = "wsl_tmux"
    else:
        command = ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_pid}\t#{pane_current_command}"]
        backend = "tmux"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "session_name_present": True,
            "session_name": session_name,
            "session_alive": None,
            "probe_backend": backend,
            "probe_output": None,
            "pane_pid": None,
            "pane_command": None,
        }

    output = (completed.stdout or "").strip()
    if completed.returncode != 0 or not output:
        return {
            "session_name_present": True,
            "session_name": session_name,
            "session_alive": False,
            "probe_backend": backend,
            "probe_output": output or None,
            "pane_pid": None,
            "pane_command": None,
        }

    first_line = output.splitlines()[0]
    pane_pid = None
    pane_command = None
    if "\t" in first_line:
        raw_pid, pane_command = first_line.split("\t", 1)
        try:
            pane_pid = int(raw_pid)
        except ValueError:
            pane_pid = None
    return {
        "session_name_present": True,
        "session_name": session_name,
        "session_alive": True,
        "probe_backend": backend,
        "probe_output": output,
        "pane_pid": pane_pid,
        "pane_command": pane_command,
    }


def _summarize_run_provenance(*, commit_path: Path, branch_path: Path) -> dict[str, object]:
    commit = None
    branch = None
    if commit_path.exists():
        raw_commit = commit_path.read_text(encoding="utf-8").strip()
        commit = raw_commit or None
    if branch_path.exists():
        raw_branch = branch_path.read_text(encoding="utf-8").strip()
        branch = raw_branch or None
    return {
        "commit_path_present": commit_path.exists(),
        "branch_path_present": branch_path.exists(),
        "run_commit": commit,
        "run_branch": branch,
    }


def _read_current_repo_provenance(repo_root: Path) -> dict[str, object]:
    try:
        head_completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        branch_completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "repo_head_commit": None,
            "repo_head_branch": None,
            "head_readable": False,
            "branch_readable": False,
        }

    head = (head_completed.stdout or "").strip() or None
    branch = (branch_completed.stdout or "").strip() or None
    return {
        "repo_head_commit": head if head_completed.returncode == 0 else None,
        "repo_head_branch": branch if branch_completed.returncode == 0 else None,
        "head_readable": head_completed.returncode == 0,
        "branch_readable": branch_completed.returncode == 0,
    }


def _compare_run_to_current_repo(
    *,
    run_provenance: dict[str, object],
    current_repo_provenance: dict[str, object],
) -> dict[str, object]:
    run_commit = run_provenance.get("run_commit")
    run_branch = run_provenance.get("run_branch")
    repo_commit = current_repo_provenance.get("repo_head_commit")
    repo_branch = current_repo_provenance.get("repo_head_branch")
    commit_match = isinstance(run_commit, str) and isinstance(repo_commit, str) and run_commit == repo_commit
    branch_match = isinstance(run_branch, str) and isinstance(repo_branch, str) and run_branch == repo_branch
    return {
        "run_matches_current_repo_commit": commit_match if run_commit and repo_commit else None,
        "run_matches_current_repo_branch": branch_match if run_branch and repo_branch else None,
        "run_is_stale_vs_current_repo": (not commit_match) if run_commit and repo_commit else None,
    }


def _summarize_latest_activity(
    out_dir: Path,
    *,
    ignored_names: set[str] | None = None,
) -> dict[str, object]:
    ignored_names = ignored_names or set()
    latest_path: Path | None = None
    latest_mtime: float | None = None
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in ignored_names:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_path = path
            latest_mtime = mtime

    if latest_path is None or latest_mtime is None:
        return {
            "path": None,
            "mtime_epoch": None,
            "age_seconds": None,
        }

    return {
        "path": str(latest_path),
        "mtime_epoch": latest_mtime,
        "age_seconds": max(0.0, time.time() - latest_mtime),
    }


def _summarize_in_progress_task_activity(root: Path, task_ids: list[int]) -> dict[str, object]:
    if not root.exists() or not task_ids:
        return {}

    summary: dict[str, object] = {}
    for task_id in task_ids:
        task_dir = root / f"task_{task_id}"
        if not task_dir.exists():
            task_dir = root / f"task_{task_id:04d}"
        if not task_dir.exists():
            continue
        summary[str(task_id)] = _summarize_latest_activity(task_dir)
    return summary


def _summarize_log_tail(path: Path, *, max_lines: int = 8) -> dict[str, object]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "size_bytes": 0,
            "mtime_epoch": None,
            "age_seconds": None,
            "last_lines": [],
            "last_nonempty_line": None,
        }

    try:
        stat_result = path.stat()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "exists": True,
            "path": str(path),
            "size_bytes": None,
            "mtime_epoch": None,
            "age_seconds": None,
            "last_lines": [],
            "last_nonempty_line": None,
        }

    lines = text.splitlines()
    last_lines = lines[-max_lines:]
    last_nonempty_line = next((line for line in reversed(lines) if line.strip()), None)
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat_result.st_size,
        "mtime_epoch": stat_result.st_mtime,
        "age_seconds": max(0.0, time.time() - stat_result.st_mtime),
        "last_lines": last_lines,
        "last_nonempty_line": last_nonempty_line,
    }


def _count_task_dir_status(root: Path, *, summary_name: str) -> dict[str, object]:
    if not root.exists():
        return {
            "started_task_count": 0,
            "completed_task_count": 0,
            "started_task_ids": [],
            "completed_task_ids": [],
            "in_progress_task_ids": [],
        }
    task_dirs = sorted(path for path in root.glob("task_*") if path.is_dir())
    started_task_ids = [int(path.name.split("_", 1)[1]) for path in task_dirs]
    completed_task_ids = [
        int(path.name.split("_", 1)[1])
        for path in task_dirs
        if (path / summary_name).exists()
    ]
    return {
        "started_task_count": len(started_task_ids),
        "completed_task_count": len(completed_task_ids),
        "started_task_ids": started_task_ids,
        "completed_task_ids": completed_task_ids,
        "in_progress_task_ids": sorted(set(started_task_ids) - set(completed_task_ids)),
    }


def _summarize_completed_eval_metrics(root: Path) -> dict[str, object]:
    if not root.exists():
        return {
            "completed_task_count": 0,
            "average_success_rate": 0.0,
            "success_task_count": 0,
            "per_task_success_rate": {},
        }

    per_task_success_rate: dict[str, float] = {}
    for task_dir in sorted(path for path in root.glob("task_*") if path.is_dir()):
        metrics_path = task_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        task_id = task_dir.name.split("_", 1)[1]
        metrics = _load_json(metrics_path)
        per_task_success_rate[str(int(task_id))] = float(metrics.get("success_rate", 0.0))

    if not per_task_success_rate:
        return {
            "completed_task_count": 0,
            "average_success_rate": 0.0,
            "success_task_count": 0,
            "per_task_success_rate": {},
        }

    values = list(per_task_success_rate.values())
    return {
        "completed_task_count": len(per_task_success_rate),
        "average_success_rate": sum(values) / len(values),
        "success_task_count": sum(1 for value in values if value >= 1.0),
        "per_task_success_rate": per_task_success_rate,
    }


def _with_benchmark_blocker_metrics(
    metrics: dict[str, object],
    *,
    family_name: str | None,
) -> dict[str, object]:
    if not isinstance(family_name, str):
        metrics = dict(metrics)
        metrics["benchmark_blockers"] = []
        return metrics

    benchmark_blockers = sorted(get_family_known_benchmark_blockers(family_name))
    metrics = dict(metrics)
    metrics["benchmark_blockers"] = benchmark_blockers
    per_task_success_rate = metrics.get("per_task_success_rate", {})
    if not isinstance(per_task_success_rate, dict):
        return metrics

    blocker_set = {str(task_id) for task_id in benchmark_blockers}
    filtered = {
        str(task_id): float(value)
        for task_id, value in per_task_success_rate.items()
        if str(task_id) not in blocker_set
    }
    if not filtered:
        metrics["effective_completed_task_count_excluding_blockers"] = 0
        metrics["effective_average_success_rate_excluding_blockers"] = 0.0
        metrics["effective_success_task_count_excluding_blockers"] = 0
        metrics["effective_unresolved_task_ids_excluding_blockers"] = []
        return metrics

    filtered_values = list(filtered.values())
    metrics["effective_completed_task_count_excluding_blockers"] = len(filtered)
    metrics["effective_average_success_rate_excluding_blockers"] = sum(filtered_values) / len(filtered_values)
    metrics["effective_success_task_count_excluding_blockers"] = sum(1 for value in filtered_values if value >= 1.0)
    metrics["effective_unresolved_task_ids_excluding_blockers"] = sorted(
        (int(task_id) for task_id, value in filtered.items() if value < 1.0),
        key=int,
    )
    return metrics


def _summarize_completed_eval_task_groups(
    per_task_success_rate: dict[str, float],
    *,
    family_name: str | None,
    split_preview: dict[str, object],
) -> dict[str, object]:
    if not per_task_success_rate:
        return {}

    if not isinstance(family_name, str):
        return {}
    task_groups = get_family_task_groups(family_name)
    split_task_ids = {int(task_id) for task_id in split_preview.get("task_ids", [])}
    groups = {
        group_name: {str(task_id) for task_id in task_ids if task_id in split_task_ids}
        for group_name, task_ids in task_groups.items()
    }
    summary: dict[str, object] = {}
    for group_name, task_ids in groups.items():
        completed = {
            task_id: success
            for task_id, success in per_task_success_rate.items()
            if task_id in task_ids
        }
        if not completed:
            summary[group_name] = {
                "completed_task_count": 0,
                "average_success_rate": 0.0,
                "success_task_count": 0,
                "completed_task_ids": [],
            }
            continue
        values = list(completed.values())
        summary[group_name] = {
            "completed_task_count": len(completed),
            "average_success_rate": sum(values) / len(values),
            "success_task_count": sum(1 for value in values if value >= 1.0),
            "completed_task_ids": sorted((int(task_id) for task_id in completed), key=int),
        }
    return summary


def _summarize_task_status_groups(
    task_status: dict[str, object],
    *,
    family_name: str | None,
    split_preview: dict[str, object],
) -> dict[str, object]:
    if not isinstance(family_name, str):
        return {}

    split_task_ids = {int(task_id) for task_id in split_preview.get("task_ids", [])}
    if not split_task_ids:
        return {}

    task_groups = get_family_task_groups(family_name)
    started_task_ids = {int(task_id) for task_id in task_status.get("started_task_ids", [])}
    completed_task_ids = {int(task_id) for task_id in task_status.get("completed_task_ids", [])}
    in_progress_task_ids = {int(task_id) for task_id in task_status.get("in_progress_task_ids", [])}

    summary: dict[str, object] = {}
    for group_name, task_ids in task_groups.items():
        group_task_ids = {int(task_id) for task_id in task_ids if int(task_id) in split_task_ids}
        summary[group_name] = {
            "started_task_count": len(group_task_ids & started_task_ids),
            "completed_task_count": len(group_task_ids & completed_task_ids),
            "in_progress_task_count": len(group_task_ids & in_progress_task_ids),
            "started_task_ids": sorted(group_task_ids & started_task_ids),
            "completed_task_ids": sorted(group_task_ids & completed_task_ids),
            "in_progress_task_ids": sorted(group_task_ids & in_progress_task_ids),
        }
    return summary


def _summarize_rollout_tree(root: Path) -> dict[str, object]:
    if not root.exists():
        return {
            "started_episode_count": 0,
            "completed_episode_count": 0,
            "in_progress_episode_count": 0,
        }

    episode_dirs = sorted(path for path in root.glob("task_*/group_*/episode_*") if path.is_dir())
    started_episode_count = len(episode_dirs)
    completed_episode_count = sum(1 for path in episode_dirs if (path / "episode.json").exists())
    return {
        "started_episode_count": started_episode_count,
        "completed_episode_count": completed_episode_count,
        "in_progress_episode_count": started_episode_count - completed_episode_count,
    }


def _summarize_warmup_training(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    metrics = payload.get("warmup_metrics", {})
    return {
        "selected_demo_count": payload.get("selected_demo_count", 0),
        "selected_demo_task_counts": payload.get("selected_demo_task_counts", {}),
        "source_task_sample_counts": metrics.get("source_task_sample_counts", {}),
        "average_epoch_task_sample_counts": metrics.get(
            "average_epoch_task_sample_counts",
            {},
        ),
        "task_sample_multipliers": metrics.get("task_sample_multipliers", {}),
        "average_demo_reward": float(metrics.get("average_demo_reward", 0.0)),
        "final_loss": float(metrics.get("final_loss", 0.0)),
        "epochs": int(metrics.get("epochs", 0)),
        "success_demo_count": int(metrics.get("success_demo_count", 0)),
    }


def _summarize_rollout_collection(summary_path: Path, root: Path) -> dict[str, object]:
    tree_summary = _summarize_rollout_tree(root)
    if not summary_path.exists():
        return tree_summary
    payload = _load_json(summary_path)
    episode_count = int(payload.get("episode_count", 0))
    success_count = int(payload.get("success_count", 0))
    return {
        **tree_summary,
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": (success_count / episode_count) if episode_count else 0.0,
        "task_success_counts": payload.get("task_success_counts", {}),
    }


def _summarize_grpo_training(summary_path: Path, checkpoints_dir: Path) -> dict[str, object]:
    checkpoint_dirs = sorted(path for path in checkpoints_dir.glob("iter_*") if path.is_dir()) if checkpoints_dir.exists() else []
    if not summary_path.exists():
        return {
            "checkpoint_count": len(checkpoint_dirs),
            "last_checkpoint": str(checkpoint_dirs[-1]) if checkpoint_dirs else None,
        }

    payload = _load_json(summary_path)
    if isinstance(payload, list):
        iterations = payload
    else:
        iterations = [payload]
    last_iteration = iterations[-1] if iterations else {}
    return {
        "iteration_count": len(iterations),
        "checkpoint_count": len(checkpoint_dirs),
        "last_checkpoint": str(checkpoint_dirs[-1]) if checkpoint_dirs else None,
        "last_iteration": {
            "iteration": last_iteration.get("iteration"),
            "success_rate": float(last_iteration.get("success_rate", 0.0)),
            "episode_reward": float(last_iteration.get("episode_reward", 0.0)),
            "loss": float(last_iteration.get("loss", 0.0)),
            "approx_kl": float(last_iteration.get("approx_kl", 0.0)),
        },
    }


def _summarize_eval_stage(
    root: Path,
    *,
    family_name: str | None,
    split_preview: dict[str, object],
) -> dict[str, object]:
    eval_status = _count_task_dir_status(root, summary_name="metrics.json")
    metrics = _with_benchmark_blocker_metrics(
        _summarize_completed_eval_metrics(root),
        family_name=family_name,
    )
    return {
        "eval": eval_status,
        "metrics": metrics,
        "task_groups": _summarize_completed_eval_task_groups(
            family_name=family_name,
            per_task_success_rate=metrics["per_task_success_rate"],
            split_preview=split_preview,
        ),
    }


def build_live_family_run_status(out_dir: Path) -> dict[str, object]:
    preflight_path = out_dir / "preflight.json"
    split_manifest_path = out_dir / "split_manifest.json"
    run_pid_path = out_dir / "run.pid"
    tmux_session_path = out_dir / "tmux_session.txt"
    run_commit_path = out_dir / "run_commit.txt"
    run_branch_path = out_dir / "run_branch.txt"
    run_stdout_log_path = out_dir / "run.stdout.log"
    if not run_stdout_log_path.exists():
        legacy_run_stdout_log_path = out_dir / "run.log"
        if legacy_run_stdout_log_path.exists():
            run_stdout_log_path = legacy_run_stdout_log_path
    run_stderr_log_path = out_dir / "run.stderr.log"
    warmup_demos_summary_path = out_dir / "warmup_demos" / "summary.json"
    baseline_summary_path = out_dir / "baseline_eval_summary.json"
    warmup_summary_path = out_dir / "warmup_summary.json"
    rollout_summary_path = out_dir / "rollout_summary.json"
    grpo_summary_path = out_dir / "grpo_summary.json"
    eval_compare_path = out_dir / "eval_compare.json"
    family_summary_path = out_dir / "family_summary.json"
    rollouts_root = out_dir / "rollouts" / "iteration_0000"
    checkpoints_dir = out_dir / "checkpoints"
    eval_warmup_root = out_dir / "eval_warmup"
    eval_grpo_root = out_dir / "eval_grpo"

    split_preview: dict[str, object] = {}
    family_name = None
    if split_manifest_path.exists():
        split_payload = _load_json(split_manifest_path)
        split_preview = dict(split_payload.get("split_preview", {}))
        family_name = split_payload.get("family_name")
    elif preflight_path.exists():
        preflight_payload = _load_json(preflight_path)
        split_preview = dict(preflight_payload.get("split_preview", {}))
        family_name = preflight_payload.get("family_name")

    warmup_demos = _count_task_dir_status(out_dir / "warmup_demos", summary_name="summary.json")
    warmup_task_groups = _summarize_task_status_groups(
        warmup_demos,
        family_name=family_name,
        split_preview=split_preview,
    )
    baseline_eval = _count_task_dir_status(out_dir / "eval_baseline", summary_name="metrics.json")
    baseline_metrics = _with_benchmark_blocker_metrics(
        _summarize_completed_eval_metrics(out_dir / "eval_baseline"),
        family_name=family_name,
    )
    baseline_task_groups = _summarize_completed_eval_task_groups(
        family_name=family_name,
        per_task_success_rate=baseline_metrics["per_task_success_rate"],
        split_preview=split_preview,
    )
    baseline_progress_task_groups = _summarize_task_status_groups(
        baseline_eval,
        family_name=family_name,
        split_preview=split_preview,
    )
    warmup_in_progress_activity = _summarize_in_progress_task_activity(
        out_dir / "warmup_demos",
        warmup_demos["in_progress_task_ids"],
    )
    baseline_in_progress_activity = _summarize_in_progress_task_activity(
        out_dir / "eval_baseline",
        baseline_eval["in_progress_task_ids"],
    )
    warmup_training = _summarize_warmup_training(warmup_summary_path)
    rollout_collection = _summarize_rollout_collection(rollout_summary_path, rollouts_root)
    grpo_training = _summarize_grpo_training(grpo_summary_path, checkpoints_dir)
    warmup_eval = _summarize_eval_stage(
        eval_warmup_root,
        family_name=family_name,
        split_preview=split_preview,
    )
    grpo_eval = _summarize_eval_stage(
        eval_grpo_root,
        family_name=family_name,
        split_preview=split_preview,
    )
    run_provenance = _summarize_run_provenance(
        commit_path=run_commit_path,
        branch_path=run_branch_path,
    )
    current_repo_provenance = _read_current_repo_provenance(REPO_ROOT)

    if family_summary_path.exists():
        current_stage = "complete"
    elif eval_compare_path.exists():
        current_stage = "final_evaluation_complete"
    elif warmup_eval["eval"]["started_task_count"] > 0 or grpo_eval["eval"]["started_task_count"] > 0:
        current_stage = "final_evaluation"
    elif grpo_summary_path.exists():
        current_stage = "post_grpo_evaluation"
    elif checkpoints_dir.exists() and any(checkpoints_dir.glob("iter_*")):
        current_stage = "grpo_training"
    elif rollout_summary_path.exists():
        current_stage = "grpo_training"
    elif rollout_collection["started_episode_count"] > 0:
        current_stage = "rollout_collection"
    elif warmup_summary_path.exists():
        current_stage = "rollout_collection"
    elif baseline_eval["in_progress_task_ids"]:
        current_stage = "baseline_evaluation"
    elif baseline_summary_path.exists():
        current_stage = "warmup_training"
    elif baseline_eval["started_task_count"] > 0:
        current_stage = "baseline_evaluation"
    elif warmup_demos_summary_path.exists() or warmup_demos["started_task_count"] > 0:
        current_stage = "warmup_demos"
    elif preflight_path.exists() or split_manifest_path.exists():
        current_stage = "initialized"
    else:
        current_stage = "not_started"

    return {
        "out_dir": str(out_dir),
        "family_name": family_name,
        "current_stage": current_stage,
        "run_process": _summarize_run_process(
            run_pid_path,
            family_name=family_name,
            out_dir=out_dir,
        ),
        "run_provenance": run_provenance,
        "current_repo_provenance": current_repo_provenance,
        "run_repo_alignment": _compare_run_to_current_repo(
            run_provenance=run_provenance,
            current_repo_provenance=current_repo_provenance,
        ),
        "tmux_session": _summarize_tmux_session(tmux_session_path),
        "run_stdout_log": _summarize_log_tail(run_stdout_log_path),
        "run_stderr_log": _summarize_log_tail(run_stderr_log_path),
        "latest_activity": _summarize_latest_activity(out_dir),
        "latest_non_monitoring_activity": _summarize_latest_activity(
            out_dir,
            ignored_names=NON_MONITORING_METADATA_FILENAMES,
        ),
        "split_preview": split_preview,
        "artifact_presence": {
            "preflight": preflight_path.exists(),
            "split_manifest": split_manifest_path.exists(),
            "warmup_demos_summary": warmup_demos_summary_path.exists(),
            "baseline_eval_summary": baseline_summary_path.exists(),
            "warmup_summary": warmup_summary_path.exists(),
            "rollout_summary": rollout_summary_path.exists(),
            "grpo_summary": grpo_summary_path.exists(),
            "eval_compare": eval_compare_path.exists(),
            "family_summary": family_summary_path.exists(),
        },
        "warmup_demos": warmup_demos,
        "warmup_in_progress_activity": warmup_in_progress_activity,
        "warmup_task_groups": warmup_task_groups,
        "baseline_eval": baseline_eval,
        "baseline_metrics": baseline_metrics,
        "baseline_task_groups": baseline_task_groups,
        "baseline_progress_task_groups": baseline_progress_task_groups,
        "baseline_in_progress_activity": baseline_in_progress_activity,
        "warmup_training": warmup_training,
        "rollout_collection": rollout_collection,
        "grpo_training": grpo_training,
        "warmup_eval": warmup_eval,
        "grpo_eval": grpo_eval,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = build_live_family_run_status(args.out_dir)
    if args.write_out is not None:
        write_json_atomic(args.write_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
