from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_final_results import (  # noqa: E402
    EXPECTED_PER_TASK,
    EXPECTED_RUN_CARD,
    EXPECTED_STAGE_RATES,
    validate_final_results,
)


def _build_summary() -> dict[str, object]:
    return {
        stage: {
            "family_success_rate": family_success_rate,
            "per_task_success_rate": copy.deepcopy(EXPECTED_PER_TASK[stage]),
        }
        for stage, family_success_rate in EXPECTED_STAGE_RATES.items()
    }


def _build_run_card() -> dict[str, object]:
    run_card = copy.deepcopy(EXPECTED_RUN_CARD)
    run_card["baseline"] = {"family_success_rate": EXPECTED_STAGE_RATES["baseline"]}
    run_card["final_eval"] = {
        "warmup_only_per_task": copy.deepcopy(EXPECTED_PER_TASK["warmup_only"]),
        "warmup_plus_grpo_per_task": copy.deepcopy(EXPECTED_PER_TASK["warmup_plus_grpo"]),
    }
    return run_card


class ValidateFinalResultsTests(unittest.TestCase):
    def test_validate_final_results_accepts_expected_metrics(self) -> None:
        errors = validate_final_results(_build_summary(), _build_run_card())
        self.assertEqual(errors, [])

    def test_validate_final_results_reports_stage_rate_mismatch(self) -> None:
        summary = _build_summary()
        summary["warmup_only"]["family_success_rate"] = 0.75

        errors = validate_final_results(summary, _build_run_card())

        self.assertTrue(any("summary.warmup_only.family_success_rate" in error for error in errors))

    def test_validate_final_results_reports_run_card_rollout_mismatch(self) -> None:
        run_card = _build_run_card()
        run_card["rollout_collection"]["success_count"] = 18

        errors = validate_final_results(_build_summary(), run_card)

        self.assertTrue(any("run_card.rollout_collection.success_count" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
