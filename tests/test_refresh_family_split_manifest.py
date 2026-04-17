from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.refresh_family_split_manifest import build_arg_parser, resolve_split


class RefreshFamilySplitManifestTests(unittest.TestCase):
    def test_resolve_split_uses_recommended_family_split_by_default(self) -> None:
        args = build_arg_parser().parse_args(
            ["--family", "shopping_full", "--out", "C:\\tmp\\manifest.json"]
        )

        split = resolve_split(args)

        self.assertEqual(split.family_name, "shopping_full")
        self.assertEqual(len(split.training_task_ids), 40)
        self.assertEqual(len(split.holdout_task_ids), 8)

    def test_resolve_split_can_load_saved_preflight_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "preflight.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "family_name": "shopping_full",
                            "task_ids": [96, 128, 189, 324],
                            "warmup_task_ids": [128],
                            "grpo_task_ids": [96, 189],
                            "holdout_task_ids": [324],
                            "eval_task_ids": [96, 128, 189, 324],
                            "split_seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = build_arg_parser().parse_args(
                [
                    "--family",
                    "shopping_full",
                    "--out",
                    str(Path(tmp_dir) / "manifest.json"),
                    "--source-manifest",
                    str(manifest_path),
                ]
            )

            split = resolve_split(args)

        self.assertEqual(split.family_name, "shopping_full")
        self.assertEqual(split.holdout_task_ids, (324,))

    def test_resolve_split_can_build_custom_counts(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--family",
                "shopping_exact",
                "--out",
                "C:\\tmp\\manifest.json",
                "--warmup-task-count",
                "10",
                "--holdout-task-count",
                "6",
            ]
        )

        split = resolve_split(args)

        self.assertEqual(split.family_name, "shopping_exact")
        self.assertEqual(len(split.warmup_task_ids), 10)
        self.assertEqual(len(split.holdout_task_ids), 6)
        self.assertEqual(len(split.training_task_ids), 26)


if __name__ == "__main__":
    unittest.main()
