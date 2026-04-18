from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a saved WebArena blocker-audit artifact.")
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--expected-task-id", action="append", default=[])
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Audit payload must be a JSON object: {path}")
    return payload


def validate_webarena_blocker_audit(
    audit_payload: dict[str, Any],
    *,
    expected_task_ids: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []

    tasks = audit_payload.get("tasks")
    if not isinstance(tasks, list):
        errors.append("audit.tasks: expected list")
        tasks = []

    derived_tasks: list[dict[str, Any]] = []
    task_ids: list[int] = []
    mismatch_task_ids_from_tasks: list[int] = []

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"audit.tasks[{index}]: expected object")
            continue
        task_id = task.get("task_id")
        if not isinstance(task_id, int):
            errors.append(f"audit.tasks[{index}].task_id: expected integer")
            continue
        task_ids.append(task_id)

        artifact_dir = task.get("artifact_dir")
        if not isinstance(artifact_dir, str) or not artifact_dir:
            errors.append(f"audit.tasks[{index}].artifact_dir: expected non-empty string")

        if not isinstance(task.get("grounded_answer_matches_reference"), bool):
            errors.append(
                f"audit.tasks[{index}].grounded_answer_matches_reference: expected boolean"
            )
        elif not task["grounded_answer_matches_reference"]:
            mismatch_task_ids_from_tasks.append(task_id)

        if "grounded_answer" not in task:
            errors.append(f"audit.tasks[{index}].grounded_answer: expected field to be present")

        reference_must_include = task.get("reference_must_include")
        if not isinstance(reference_must_include, list):
            errors.append(f"audit.tasks[{index}].reference_must_include: expected list")

        sites = task.get("sites")
        if not isinstance(sites, list):
            errors.append(f"audit.tasks[{index}].sites: expected list")

        final_step = task.get("final_step")
        if not isinstance(final_step, dict):
            errors.append(f"audit.tasks[{index}].final_step: expected object")

        derived_tasks.append(
            {
                "task_id": task_id,
                "artifact_dir": artifact_dir,
                "grounded_answer_matches_reference": task.get("grounded_answer_matches_reference"),
                "success_rate": task.get("success_rate"),
            }
        )

    task_count = audit_payload.get("task_count")
    if not isinstance(task_count, int):
        errors.append("audit.task_count: expected integer")
    elif task_count != len(tasks):
        errors.append(f"audit.task_count: expected {len(tasks)}, found {task_count}")

    mismatch_task_ids = audit_payload.get("mismatch_task_ids")
    if not isinstance(mismatch_task_ids, list):
        errors.append("audit.mismatch_task_ids: expected list")
        mismatch_task_ids = []
    else:
        try:
            normalized_mismatch_task_ids = [int(task_id) for task_id in mismatch_task_ids]
        except Exception:
            errors.append("audit.mismatch_task_ids: expected integer task ids")
            normalized_mismatch_task_ids = []
        else:
            if normalized_mismatch_task_ids != mismatch_task_ids_from_tasks:
                errors.append(
                    "audit.mismatch_task_ids: expected "
                    f"{mismatch_task_ids_from_tasks}, found {normalized_mismatch_task_ids}"
                )

    if len(task_ids) != len(set(task_ids)):
        errors.append(f"audit.tasks: duplicate task ids found: {task_ids}")

    expected_task_id_set = {int(task_id) for task_id in expected_task_ids}
    if expected_task_id_set and set(task_ids) != expected_task_id_set:
        errors.append(
            f"audit.tasks: expected task ids {sorted(expected_task_id_set)}, found {sorted(task_ids)}"
        )

    derived = {
        "task_count": len(tasks),
        "task_ids": task_ids,
        "mismatch_task_ids": mismatch_task_ids_from_tasks,
        "tasks": derived_tasks,
    }
    return errors, derived


def main() -> int:
    args = build_arg_parser().parse_args()
    audit_path = Path(args.audit_path)
    payload = _load_json(audit_path)
    errors, derived = validate_webarena_blocker_audit(
        payload,
        expected_task_ids=args.expected_task_id,
    )
    response = {
        "ok": not errors,
        "audit_path": str(audit_path),
        "derived": derived,
    }
    if errors:
        response["errors"] = errors
        print(json.dumps(response, indent=2, sort_keys=True))
        return 1
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
