from __future__ import annotations

import argparse
import importlib.resources
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic  # noqa: E402


ANSWER_ACTION_RE = re.compile(r'send_msg_to_user\("(?P<answer>.*)"\)')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit saved WebArena eval artifacts against the baked task references."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="TASK_ID=ARTIFACT_DIR",
        help="Saved eval artifact directory for a task, e.g. 124=outputs/eval_task_124_pricerangefix_v2",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def parse_artifact_spec(spec: str) -> tuple[int, Path]:
    task_id_text, sep, artifact_dir_text = spec.partition("=")
    if not sep or not task_id_text.strip() or not artifact_dir_text.strip():
        raise ValueError(
            f"Invalid artifact spec {spec!r}. Expected TASK_ID=ARTIFACT_DIR."
        )
    return int(task_id_text.strip()), Path(artifact_dir_text.strip())


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def _resolve_manifest_path(manifest_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _merge_manifest_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.manifest is None:
        return args
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    if getattr(args, "_out_explicit", False) is False and "out" in manifest:
        args.out = _resolve_manifest_path(manifest_path, str(manifest["out"]))
    manifest_artifacts = manifest.get("artifacts", {})
    if not isinstance(manifest_artifacts, dict):
        raise ValueError(f"manifest.artifacts must be an object: {manifest_path}")
    merged_artifacts = {
        str(task_id): str(_resolve_manifest_path(manifest_path, str(artifact_dir)))
        for task_id, artifact_dir in manifest_artifacts.items()
    }
    cli_artifacts = dict(parse_artifact_spec(spec) for spec in args.artifact)
    for task_id, artifact_dir in cli_artifacts.items():
        merged_artifacts[str(task_id)] = str(artifact_dir)
    args.artifact = [
        f"{task_id}={artifact_dir}"
        for task_id, artifact_dir in sorted(merged_artifacts.items(), key=lambda item: int(item[0]))
    ]
    return args


def load_webarena_task_configs() -> dict[int, dict[str, Any]]:
    import webarena

    config_path = importlib.resources.files(webarena).joinpath("test.raw.json")
    all_configs = json.loads(config_path.read_text())
    return {int(config["task_id"]): config for config in all_configs}


def _extract_answer_from_action_text(action_text: str) -> str | None:
    if not action_text:
        return None
    match = ANSWER_ACTION_RE.fullmatch(action_text.strip())
    if match is None:
        return None
    return match.group("answer")


def load_artifact_result(artifact_dir: Path) -> dict[str, Any]:
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    episode = json.loads(
        (artifact_dir / "episode_0" / "episode.json").read_text(encoding="utf-8")
    )
    steps_path = artifact_dir / "episode_0" / "steps.jsonl"
    final_step: dict[str, Any] | None = None
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            final_step = json.loads(line)
    if final_step is None:
        raise ValueError(f"No steps found in {steps_path}.")

    parsed_action = str(final_step.get("parsed_action") or "")
    raw_agent_output = str(final_step.get("raw_agent_output") or "")
    answer = (
        _extract_answer_from_action_text(parsed_action)
        or _extract_answer_from_action_text(raw_agent_output.removeprefix("ACTION:").strip())
        or _extract_answer_from_action_text(raw_agent_output)
    )
    return {
        "artifact_dir": str(artifact_dir),
        "metrics": metrics,
        "episode": episode,
        "final_step": final_step,
        "grounded_answer": answer,
    }


def build_blocker_audit_entry(
    *,
    task_id: int,
    task_config: dict[str, Any],
    artifact_result: dict[str, Any],
) -> dict[str, Any]:
    eval_config = task_config.get("eval") or {}
    must_include = [str(item) for item in (eval_config.get("reference_answers") or {}).get("must_include", [])]
    grounded_answer = artifact_result.get("grounded_answer")
    grounded_answer_text = str(grounded_answer) if grounded_answer is not None else ""
    return {
        "task_id": task_id,
        "intent": task_config.get("intent"),
        "sites": task_config.get("sites"),
        "artifact_dir": artifact_result["artifact_dir"],
        "success": bool((artifact_result.get("metrics") or {}).get("success_rate", 0.0) > 0.0),
        "success_rate": (artifact_result.get("metrics") or {}).get("success_rate"),
        "grounded_answer": grounded_answer,
        "reference_answer_raw_annotation": eval_config.get("reference_answer_raw_annotation"),
        "reference_must_include": must_include,
        "grounded_answer_matches_reference": bool(
            grounded_answer_text
            and must_include
            and all(required in grounded_answer_text for required in must_include)
        ),
        "final_step": {
            "parsed_action": artifact_result["final_step"].get("parsed_action"),
            "raw_agent_output": artifact_result["final_step"].get("raw_agent_output"),
            "reward": artifact_result["final_step"].get("reward"),
            "terminated": artifact_result["final_step"].get("terminated"),
            "truncated": artifact_result["final_step"].get("truncated"),
        },
        "episode_failure_reasons": (artifact_result.get("episode") or {}).get("failure_reasons", []),
    }


def audit_webarena_blockers(
    artifact_specs: list[tuple[int, Path]],
    *,
    task_configs: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    task_configs = task_configs or load_webarena_task_configs()
    entries: list[dict[str, Any]] = []
    for task_id, artifact_dir in artifact_specs:
        task_config = task_configs[task_id]
        artifact_result = load_artifact_result(artifact_dir)
        entries.append(
            build_blocker_audit_entry(
                task_id=task_id,
                task_config=task_config,
                artifact_result=artifact_result,
            )
        )
    return {
        "task_count": len(entries),
        "tasks": entries,
        "mismatch_task_ids": [
            entry["task_id"] for entry in entries if not entry["grounded_answer_matches_reference"]
        ],
    }


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args._out_explicit = "--out" in sys.argv
    args = _merge_manifest_args(args)
    if args.out is None:
        raise ValueError("An output path is required via --out or --manifest")
    artifact_specs = [parse_artifact_spec(spec) for spec in args.artifact]
    payload = audit_webarena_blockers(artifact_specs)
    write_json_atomic(args.out, payload)
    print(json.dumps({"out": str(args.out), **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
