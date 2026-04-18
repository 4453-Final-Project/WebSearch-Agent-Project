from __future__ import annotations

import unittest

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_webarena_blocker_audit import validate_webarena_blocker_audit


class ValidateWebArenaBlockerAuditTests(unittest.TestCase):
    def test_accepts_valid_blocker_audit(self) -> None:
        payload = {
            "task_count": 2,
            "mismatch_task_ids": [124],
            "tasks": [
                {
                    "task_id": 124,
                    "artifact_dir": "outputs/eval_task_124_pricerangefix_v2",
                    "grounded_answer": "0.01 - 169.99",
                    "grounded_answer_matches_reference": False,
                    "reference_must_include": ["0.14", "745.00"],
                    "sites": ["shopping"],
                    "final_step": {"parsed_action": 'send_msg_to_user("0.01 - 169.99")'},
                    "success_rate": 0.0,
                },
                {
                    "task_id": 133,
                    "artifact_dir": "outputs/eval_task_133_bootstrap_gitlabcountfix_v1",
                    "grounded_answer": "2",
                    "grounded_answer_matches_reference": True,
                    "reference_must_include": ["2"],
                    "sites": ["gitlab"],
                    "final_step": {"parsed_action": 'send_msg_to_user("2")'},
                    "success_rate": 1.0,
                },
            ],
        }

        errors, derived = validate_webarena_blocker_audit(
            payload,
            expected_task_ids=["124", "133"],
        )

        self.assertEqual(errors, [])
        self.assertEqual(derived["task_ids"], [124, 133])
        self.assertEqual(derived["mismatch_task_ids"], [124])

    def test_rejects_mismatch_list_drift(self) -> None:
        payload = {
            "task_count": 1,
            "mismatch_task_ids": [],
            "tasks": [
                {
                    "task_id": 141,
                    "artifact_dir": "outputs/eval_task_141_spendfix_v1",
                    "grounded_answer": "24.42",
                    "grounded_answer_matches_reference": False,
                    "reference_must_include": ["47.41"],
                    "sites": ["shopping"],
                    "final_step": {"parsed_action": 'send_msg_to_user("24.42")'},
                    "success_rate": 0.0,
                }
            ],
        }

        errors, _ = validate_webarena_blocker_audit(payload)

        self.assertTrue(
            any("audit.mismatch_task_ids" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
