from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_profiles import (
    find_model_profile_by_dir_name,
    find_model_profile_by_path,
    get_model_profile,
    list_model_profiles,
)


class ModelProfileTests(unittest.TestCase):
    def test_profile_list_contains_qwen_and_lfm(self) -> None:
        profiles = list_model_profiles()
        self.assertIn("qwen-default", profiles)
        self.assertIn("lfm2.5-350m", profiles)

    def test_lfm_profile_fields_are_stable(self) -> None:
        profile = get_model_profile("lfm2.5-350m")
        self.assertEqual(profile.huggingface_id, "LiquidAI/LFM2.5-350M")
        self.assertEqual(profile.local_dir_name, "LFM2.5-350M")
        self.assertGreater(profile.max_new_tokens, 0)

    def test_profile_lookup_by_local_directory_name(self) -> None:
        profile = find_model_profile_by_dir_name("LFM2.5-350M")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "lfm2.5-350m")

    def test_profile_lookup_by_model_path(self) -> None:
        profile = find_model_profile_by_path("/tmp/models/LFM2.5-350M")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "lfm2.5-350m")

    def test_profile_lookup_by_adapter_path_uses_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/tmp/models/LFM2.5-350M"}),
                encoding="utf-8",
            )

            profile = find_model_profile_by_path(adapter_dir)

        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "lfm2.5-350m")


if __name__ == "__main__":
    unittest.main()
