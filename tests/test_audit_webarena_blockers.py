from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_webarena_blockers import _merge_manifest_args, audit_webarena_blockers, build_arg_parser


class AuditWebArenaBlockersTests(unittest.TestCase):
    def test_reports_reference_mismatch_from_saved_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact_dir = tmp_path / "eval_task_141"
            episode_dir = artifact_dir / "episode_0"
            episode_dir.mkdir(parents=True)
            (artifact_dir / "metrics.json").write_text(
                json.dumps({"task_id": 141, "success_rate": 0.0}),
                encoding="utf-8",
            )
            (episode_dir / "episode.json").write_text(
                json.dumps({"failure_reasons": ["terminated_without_positive_reward"]}),
                encoding="utf-8",
            )
            (episode_dir / "steps.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"parsed_action": 'goto("http://example.com")'}),
                        json.dumps(
                            {
                                "parsed_action": 'send_msg_to_user("24.42")',
                                "raw_agent_output": 'ACTION: send_msg_to_user("N/A")',
                                "reward": 0.0,
                                "terminated": True,
                                "truncated": False,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = audit_webarena_blockers(
                [(141, artifact_dir)],
                task_configs={
                    141: {
                        "task_id": 141,
                        "intent": "How much I spent on food-related shopping during March 2023",
                        "sites": ["shopping"],
                        "eval": {
                            "reference_answers": {"must_include": ["47.41"]},
                            "reference_answer_raw_annotation": "$47.41",
                        },
                    }
                },
            )

            self.assertEqual(report["task_count"], 1)
            self.assertEqual(report["mismatch_task_ids"], [141])
            task = report["tasks"][0]
            self.assertEqual(task["grounded_answer"], "24.42")
            self.assertEqual(task["reference_answer_raw_annotation"], "$47.41")
            self.assertFalse(task["grounded_answer_matches_reference"])
            self.assertEqual(task["episode_failure_reasons"], ["terminated_without_positive_reward"])

    def test_reports_reference_match_when_grounded_answer_contains_all_required_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact_dir = tmp_path / "eval_task_124"
            episode_dir = artifact_dir / "episode_0"
            episode_dir.mkdir(parents=True)
            (artifact_dir / "metrics.json").write_text(
                json.dumps({"task_id": 124, "success_rate": 1.0}),
                encoding="utf-8",
            )
            (episode_dir / "episode.json").write_text(
                json.dumps({"failure_reasons": []}),
                encoding="utf-8",
            )
            (episode_dir / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "parsed_action": 'send_msg_to_user("0.14 - 745.00")',
                        "raw_agent_output": 'ACTION: send_msg_to_user("0.14 - 745.00")',
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                    }
                ),
                encoding="utf-8",
            )

            report = audit_webarena_blockers(
                [(124, artifact_dir)],
                task_configs={
                    124: {
                        "task_id": 124,
                        "intent": "What is the price range of wireless earphone in the One Stop Market?",
                        "sites": ["shopping"],
                        "eval": {
                            "reference_answers": {"must_include": ["0.14", "745.00"]},
                            "reference_answer_raw_annotation": "$0.14 - $745.00",
                        },
                    }
                },
            )

            self.assertEqual(report["mismatch_task_ids"], [])
            task = report["tasks"][0]
            self.assertTrue(task["grounded_answer_matches_reference"])
            self.assertEqual(task["grounded_answer"], "0.14 - 745.00")

    def test_manifest_merges_checked_in_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact_dir = tmp_path / "eval_task_133"
            episode_dir = artifact_dir / "episode_0"
            episode_dir.mkdir(parents=True)
            (artifact_dir / "metrics.json").write_text(
                json.dumps({"task_id": 133, "success_rate": 0.0}),
                encoding="utf-8",
            )
            (episode_dir / "episode.json").write_text(
                json.dumps({"failure_reasons": ["terminated_without_positive_reward"]}),
                encoding="utf-8",
            )
            (episode_dir / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "parsed_action": 'send_msg_to_user("0")',
                        "raw_agent_output": 'ACTION: send_msg_to_user("0")',
                        "reward": 0.0,
                        "terminated": True,
                        "truncated": False,
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = tmp_path / "bootstrap41_blocker_audit_manifest.json"
            out_path = tmp_path / "bootstrap41_blocker_audit.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "out": "bootstrap41_blocker_audit.json",
                        "artifacts": {"133": "eval_task_133"},
                    }
                ),
                encoding="utf-8",
            )

            parser = build_arg_parser()
            args = parser.parse_args(["--manifest", str(manifest_path)])
            args._out_explicit = False
            args = _merge_manifest_args(args)

            report = audit_webarena_blockers(
                [(int(task_id), Path(path)) for task_id, path in (spec.split("=", 1) for spec in args.artifact)],
                task_configs={
                    133: {
                        "task_id": 133,
                        "intent": "How many commits did Steve make?",
                        "sites": ["gitlab"],
                        "eval": {
                            "reference_answers": {"must_include": ["2"]},
                            "reference_answer_raw_annotation": "2",
                        },
                    }
                },
            )
            out_path.write_text(json.dumps(report), encoding="utf-8")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_count"], 1)
            self.assertEqual(payload["mismatch_task_ids"], [133])
            self.assertEqual(payload["tasks"][0]["grounded_answer"], "0")


if __name__ == "__main__":
    unittest.main()
