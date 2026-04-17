from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.validate_split_manifests import (
    _validate_launchers,
    _validate_manifest,
    validate_checked_in_split_manifests,
)


class ValidateSplitManifestsTests(unittest.TestCase):
    def test_validate_manifest_accepts_checked_in_bootstrap41_manifest(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "bootstrap41_curriculum_manifest.json"

        errors, payload = _validate_manifest("bootstrap41", manifest_path)

        self.assertEqual(errors, [])
        self.assertEqual(payload["task_count"], 41)
        self.assertEqual(payload["training_task_count"], 33)

    def test_validate_manifest_accepts_checked_in_web_mix88_manifest(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "web_mix88_curriculum_manifest.json"

        errors, payload = _validate_manifest("web_mix88", manifest_path)

        self.assertEqual(errors, [])
        self.assertEqual(payload["task_count"], 88)
        self.assertEqual(payload["training_task_count"], 72)

    def test_validate_manifest_accepts_checked_in_exact_manifest(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "shopping_exact_curriculum_manifest.json"

        errors, payload = _validate_manifest("shopping_exact", manifest_path)

        self.assertEqual(errors, [])
        self.assertEqual(payload["task_count"], 32)
        self.assertEqual(payload["training_task_count"], 26)

    def test_validate_manifest_reports_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "shopping_exact_curriculum_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_exact",
                        "task_ids": [1, 2, 3],
                        "warmup_task_ids": [1],
                        "grpo_task_ids": [2],
                        "holdout_task_ids": [3],
                        "eval_task_ids": [1, 2, 3],
                        "split_seed": 42,
                    }
                ),
                encoding="utf-8",
            )

            errors, _payload = _validate_manifest("shopping_exact", manifest_path)

        self.assertIn("task_ids do not match recommended split", errors)

    def test_validate_launchers_accepts_full_family_scripts(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "shopping_full_curriculum_manifest.json"

        errors, payload = _validate_launchers("shopping_full", manifest_path)

        self.assertEqual(errors, [])
        self.assertTrue(payload["run_qwen_shopping_full_curriculum.sh"]["ok"])
        self.assertTrue(payload["run_liquid_shopping_full_curriculum.sh"]["ok"])

    def test_validate_launchers_accepts_bootstrap41_scripts(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "bootstrap41_curriculum_manifest.json"

        errors, payload = _validate_launchers("bootstrap41", manifest_path)

        self.assertEqual(errors, [])
        self.assertTrue(payload["run_qwen_bootstrap41_curriculum.sh"]["ok"])
        self.assertTrue(payload["run_liquid_bootstrap41_curriculum.sh"]["ok"])

    def test_validate_launchers_accepts_web_mix88_scripts(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "web_mix88_curriculum_manifest.json"

        errors, payload = _validate_launchers("web_mix88", manifest_path)

        self.assertEqual(errors, [])
        self.assertTrue(payload["run_qwen_web_mix88_curriculum.sh"]["ok"])
        self.assertTrue(payload["run_liquid_web_mix88_curriculum.sh"]["ok"])

    def test_validate_checked_in_split_manifests_can_limit_family(self) -> None:
        errors, payload = validate_checked_in_split_manifests(["shopping_exact"])

        self.assertEqual(errors, [])
        self.assertEqual(list(payload), ["shopping_exact"])

    def test_validate_checked_in_split_manifests_defaults_include_bootstrap41(self) -> None:
        errors, payload = validate_checked_in_split_manifests([])

        self.assertEqual(errors, [])
        self.assertIn("bootstrap41", payload)
        self.assertIn("web_mix88", payload)

    def test_validate_launchers_reports_missing_manifest_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            launcher_path = Path(tmp_dir) / "run_qwen_shopping_exact_curriculum.sh"
            launcher_path.write_text("python scripts/run_family_curriculum.py --family shopping_exact\n", encoding="utf-8")
            with mock.patch(
                "scripts.validate_split_manifests.DEFAULT_LAUNCHERS",
                {"shopping_exact": (launcher_path,)},
            ):
                manifest_path = Path(__file__).resolve().parents[1] / "scripts" / "local" / "shopping_exact_curriculum_manifest.json"
                errors, payload = _validate_launchers("shopping_exact", manifest_path)

        self.assertIn("missing manifest reference shopping_exact_curriculum_manifest.json", errors[0])
        self.assertFalse(payload["run_qwen_shopping_exact_curriculum.sh"]["ok"])


if __name__ == "__main__":
    unittest.main()
