from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_webarena_blockers import audit_webarena_blockers


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


if __name__ == "__main__":
    unittest.main()
