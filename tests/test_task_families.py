from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_family_curriculum import _load_split_from_manifest
from src.training.task_families import (
    build_task_split,
    family_requires_openai_judge,
    get_family_task_groups,
    recommend_task_split,
)


class TaskFamiliesTests(unittest.TestCase):
    def test_recommended_order_split_uses_20_plus_training_tasks(self) -> None:
        split = recommend_task_split("shopping_order", split_seed=7)

        self.assertEqual(split.family_name, "shopping_order")
        self.assertEqual(len(split.task_ids), 27)
        self.assertEqual(len(split.warmup_task_ids), 8)
        self.assertEqual(len(split.holdout_task_ids), 5)
        self.assertGreaterEqual(len(split.training_task_ids), 20)
        self.assertEqual(len(set(split.task_ids)), len(split.task_ids))
        self.assertTrue(set(split.warmup_task_ids).isdisjoint(split.grpo_task_ids))
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))

    def test_recommended_exact_split_uses_large_training_pool(self) -> None:
        split = recommend_task_split("shopping_exact", split_seed=11)

        self.assertEqual(len(split.task_ids), 32)
        self.assertEqual(len(split.warmup_task_ids), 14)
        self.assertEqual(len(split.holdout_task_ids), 6)
        self.assertGreaterEqual(len(split.training_task_ids), 20)
        self.assertEqual(split.eval_task_ids, split.task_ids)
        self.assertIn(324, split.holdout_task_ids)
        self.assertTrue({325, 326, 327, 328}.issubset(set(split.warmup_task_ids)))
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))

    def test_checked_in_exact_manifest_matches_recommended_split(self) -> None:
        manifest_path = PROJECT_ROOT / "scripts" / "local" / "shopping_exact_curriculum_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked_in_split = _load_split_from_manifest(manifest_path, expected_family_name="shopping_exact")
        split = recommend_task_split("shopping_exact", split_seed=checked_in_split.split_seed)

        self.assertEqual(payload["family_name"], split.family_name)
        self.assertEqual(payload["split_provenance"]["split_source"], "recommended")
        self.assertTrue(payload["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(checked_in_split.task_ids, split.task_ids)
        self.assertEqual(checked_in_split.warmup_task_ids, split.warmup_task_ids)
        self.assertEqual(checked_in_split.grpo_task_ids, split.grpo_task_ids)
        self.assertEqual(checked_in_split.holdout_task_ids, split.holdout_task_ids)
        self.assertEqual(checked_in_split.eval_task_ids, split.eval_task_ids)

    def test_recommended_full_split_scales_to_40_training_tasks(self) -> None:
        split = recommend_task_split("shopping_full", split_seed=11)
        task_groups = get_family_task_groups("shopping_full")
        warmup_judge_gated = set(split.warmup_task_ids) & set(task_groups["judge_gated"])

        self.assertEqual(len(split.task_ids), 48)
        self.assertEqual(len(split.warmup_task_ids), 16)
        self.assertEqual(len(split.holdout_task_ids), 8)
        self.assertEqual(len(split.training_task_ids), 40)
        self.assertIn(324, split.holdout_task_ids)
        self.assertTrue({325, 326, 327, 328}.issubset(set(split.warmup_task_ids)))
        self.assertGreaterEqual(len(warmup_judge_gated), 4)
        self.assertTrue({191, 201, 334, 359}.issubset(set(split.warmup_task_ids)))
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))
        self.assertTrue(family_requires_openai_judge("shopping_full"))

    def test_checked_in_qwen_full_manifest_matches_recommended_split(self) -> None:
        manifest_path = PROJECT_ROOT / "scripts" / "local" / "shopping_full_curriculum_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked_in_split = _load_split_from_manifest(manifest_path, expected_family_name="shopping_full")
        split = recommend_task_split("shopping_full", split_seed=checked_in_split.split_seed)

        self.assertEqual(payload["family_name"], split.family_name)
        self.assertEqual(payload["split_provenance"]["split_source"], "recommended")
        self.assertTrue(payload["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(checked_in_split.task_ids, split.task_ids)
        self.assertEqual(checked_in_split.warmup_task_ids, split.warmup_task_ids)
        self.assertEqual(checked_in_split.grpo_task_ids, split.grpo_task_ids)
        self.assertEqual(checked_in_split.holdout_task_ids, split.holdout_task_ids)
        self.assertEqual(checked_in_split.eval_task_ids, split.eval_task_ids)

    def test_checked_in_web_mix88_manifest_matches_recommended_split(self) -> None:
        manifest_path = PROJECT_ROOT / "scripts" / "local" / "web_mix88_curriculum_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked_in_split = _load_split_from_manifest(manifest_path, expected_family_name="web_mix88")
        split = recommend_task_split("web_mix88", split_seed=checked_in_split.split_seed)

        self.assertEqual(payload["family_name"], split.family_name)
        self.assertEqual(payload["split_provenance"]["split_source"], "recommended")
        self.assertTrue(payload["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(checked_in_split.task_ids, split.task_ids)
        self.assertEqual(checked_in_split.warmup_task_ids, split.warmup_task_ids)
        self.assertEqual(checked_in_split.grpo_task_ids, split.grpo_task_ids)
        self.assertEqual(checked_in_split.holdout_task_ids, split.holdout_task_ids)
        self.assertEqual(checked_in_split.eval_task_ids, split.eval_task_ids)

    def test_checked_in_bootstrap44_manifest_matches_recommended_split(self) -> None:
        manifest_path = PROJECT_ROOT / "scripts" / "local" / "bootstrap44_curriculum_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked_in_split = _load_split_from_manifest(manifest_path, expected_family_name="bootstrap44")
        split = recommend_task_split("bootstrap44", split_seed=checked_in_split.split_seed)

        self.assertEqual(payload["family_name"], split.family_name)
        self.assertEqual(payload["split_provenance"]["split_source"], "recommended")
        self.assertTrue(payload["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(checked_in_split.task_ids, split.task_ids)
        self.assertEqual(checked_in_split.warmup_task_ids, split.warmup_task_ids)
        self.assertEqual(checked_in_split.grpo_task_ids, split.grpo_task_ids)
        self.assertEqual(checked_in_split.holdout_task_ids, split.holdout_task_ids)
        self.assertEqual(checked_in_split.eval_task_ids, split.eval_task_ids)

    def test_checked_in_web_mix91_manifest_matches_recommended_split(self) -> None:
        manifest_path = PROJECT_ROOT / "scripts" / "local" / "web_mix91_curriculum_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked_in_split = _load_split_from_manifest(manifest_path, expected_family_name="web_mix91")
        split = recommend_task_split("web_mix91", split_seed=checked_in_split.split_seed)

        self.assertEqual(payload["family_name"], split.family_name)
        self.assertEqual(payload["split_provenance"]["split_source"], "recommended")
        self.assertTrue(payload["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(checked_in_split.task_ids, split.task_ids)
        self.assertEqual(checked_in_split.warmup_task_ids, split.warmup_task_ids)
        self.assertEqual(checked_in_split.grpo_task_ids, split.grpo_task_ids)
        self.assertEqual(checked_in_split.holdout_task_ids, split.holdout_task_ids)
        self.assertEqual(checked_in_split.eval_task_ids, split.eval_task_ids)

    def test_recommended_bootstrap41_split_balances_sites(self) -> None:
        split = recommend_task_split("bootstrap41", split_seed=11)
        task_groups = get_family_task_groups("bootstrap41")

        self.assertEqual(len(split.task_ids), 41)
        self.assertEqual(len(split.warmup_task_ids), 15)
        self.assertEqual(len(split.holdout_task_ids), 8)
        self.assertEqual(len(split.training_task_ids), 33)
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))
        self.assertFalse(family_requires_openai_judge("bootstrap41"))
        self.assertEqual(len(task_groups["site_shopping_admin"]), 9)
        self.assertEqual(len(task_groups["site_shopping"]), 9)
        self.assertEqual(len(task_groups["site_reddit"]), 9)
        self.assertEqual(len(task_groups["site_gitlab"]), 7)
        self.assertEqual(len(task_groups["site_map"]), 7)
        self.assertTrue({0, 21, 27, 132, 7}.issubset(set(split.warmup_task_ids)))
        self.assertTrue({41, 141, 68, 259, 71}.issubset(set(split.holdout_task_ids)))

    def test_recommended_bootstrap44_split_expands_validated_admin_and_spend_tasks(self) -> None:
        split = recommend_task_split("bootstrap44", split_seed=11)
        task_groups = get_family_task_groups("bootstrap44")

        self.assertEqual(len(split.task_ids), 44)
        self.assertEqual(len(split.warmup_task_ids), 16)
        self.assertEqual(len(split.holdout_task_ids), 8)
        self.assertEqual(len(split.training_task_ids), 36)
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))
        self.assertFalse(family_requires_openai_judge("bootstrap44"))
        self.assertTrue({12}.issubset(set(split.warmup_task_ids)))
        self.assertTrue({13, 144}.issubset(set(split.grpo_task_ids)))
        self.assertTrue({41, 77, 141, 188, 68, 69, 259, 71}.issubset(set(split.holdout_task_ids)))
        self.assertEqual(len(task_groups["site_shopping_admin"]), 11)
        self.assertEqual(len(task_groups["site_shopping"]), 10)
        self.assertEqual(len(task_groups["site_reddit"]), 9)
        self.assertEqual(len(task_groups["site_gitlab"]), 7)
        self.assertEqual(len(task_groups["site_map"]), 7)

    def test_recommended_web_mix88_split_scales_with_cross_site_coverage(self) -> None:
        split = recommend_task_split("web_mix88", split_seed=11)
        task_groups = get_family_task_groups("web_mix88")

        self.assertEqual(len(split.task_ids), 88)
        self.assertEqual(len(split.warmup_task_ids), 30)
        self.assertEqual(len(split.holdout_task_ids), 16)
        self.assertEqual(len(split.training_task_ids), 72)
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))
        self.assertTrue(family_requires_openai_judge("web_mix88"))
        self.assertTrue({0, 21, 27, 132, 7}.issubset(set(split.warmup_task_ids)))
        self.assertTrue({191, 201, 334, 359}.issubset(set(split.warmup_task_ids)))
        self.assertTrue({41, 77, 141, 188, 68, 69, 259, 71}.issubset(set(split.holdout_task_ids)))
        self.assertTrue({96, 117, 189, 195, 200, 324, 358, 360}.issubset(set(split.holdout_task_ids)))
        self.assertEqual(len(task_groups["judge_gated"]), 16)
        self.assertEqual(len(task_groups["site_shopping_full"]), 48)
        self.assertEqual(len(task_groups["site_shopping_admin"]), 9)
        self.assertEqual(len(task_groups["site_shopping"]), 9)
        self.assertEqual(len(task_groups["site_reddit"]), 9)
        self.assertEqual(len(task_groups["site_gitlab"]), 7)
        self.assertEqual(len(task_groups["site_map"]), 7)

    def test_recommended_web_mix91_split_scales_with_bootstrap44_expansion(self) -> None:
        split = recommend_task_split("web_mix91", split_seed=11)
        task_groups = get_family_task_groups("web_mix91")

        self.assertEqual(len(split.task_ids), 91)
        self.assertEqual(len(split.warmup_task_ids), 31)
        self.assertEqual(len(split.holdout_task_ids), 16)
        self.assertEqual(len(split.training_task_ids), 75)
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))
        self.assertTrue(family_requires_openai_judge("web_mix91"))
        self.assertTrue({12}.issubset(set(split.warmup_task_ids)))
        self.assertTrue({13, 144}.issubset(set(split.grpo_task_ids)))
        self.assertTrue({41, 77, 141, 188, 68, 69, 259, 71}.issubset(set(split.holdout_task_ids)))
        self.assertTrue({96, 117, 189, 195, 200, 324, 358, 360}.issubset(set(split.holdout_task_ids)))
        self.assertEqual(len(task_groups["judge_gated"]), 16)
        self.assertEqual(len(task_groups["site_shopping_full"]), 48)
        self.assertEqual(len(task_groups["site_shopping_admin"]), 11)
        self.assertEqual(len(task_groups["site_shopping"]), 10)
        self.assertEqual(len(task_groups["site_reddit"]), 9)
        self.assertEqual(len(task_groups["site_gitlab"]), 7)
        self.assertEqual(len(task_groups["site_map"]), 7)

    def test_recommended_full_order_split_uses_large_training_pool(self) -> None:
        split = recommend_task_split("shopping_order_full", split_seed=11)

        self.assertEqual(len(split.task_ids), 43)
        self.assertEqual(len(split.warmup_task_ids), 14)
        self.assertEqual(len(split.holdout_task_ids), 8)
        self.assertEqual(len(split.training_task_ids), 35)
        self.assertTrue(set(split.holdout_task_ids).isdisjoint(split.training_task_ids))
        self.assertTrue(family_requires_openai_judge("shopping_order_full"))

    def test_full_family_task_groups_split_exact_and_fuzzy_tasks(self) -> None:
        task_groups = get_family_task_groups("shopping_full")

        self.assertEqual(len(task_groups["judge_free"]), 32)
        self.assertEqual(len(task_groups["judge_gated"]), 16)
        self.assertTrue(set(task_groups["judge_free"]).isdisjoint(task_groups["judge_gated"]))
        self.assertIn(324, task_groups["judge_free"])
        self.assertIn(96, task_groups["judge_gated"])

    def test_search_sort_split_matches_validated_recipe(self) -> None:
        split = recommend_task_split("shopping_search_sort", split_seed=123)

        self.assertEqual(split.warmup_task_ids, (325, 326))
        self.assertEqual(split.grpo_task_ids, (327, 328))
        self.assertEqual(split.holdout_task_ids, (324,))

    def test_build_task_split_rejects_exhausted_partition(self) -> None:
        with self.assertRaises(ValueError):
            build_task_split((1, 2, 3), family_name="tiny", warmup_count=1, holdout_count=2, split_seed=42)


if __name__ == "__main__":
    unittest.main()
