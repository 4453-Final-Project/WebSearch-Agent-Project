from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.peft_setup import get_adapter_base_model_path, is_adapter_checkpoint


class PeftSetupTests(unittest.TestCase):
    def test_adapter_checkpoint_helpers_detect_and_read_base_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir)
            adapter_dir.joinpath("adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/models/LFM2.5-350M"}),
                encoding="utf-8",
            )

            self.assertTrue(is_adapter_checkpoint(adapter_dir))
            self.assertEqual(get_adapter_base_model_path(adapter_dir), "/models/LFM2.5-350M")

    def test_non_adapter_directory_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(is_adapter_checkpoint(temp_dir))
            self.assertIsNone(get_adapter_base_model_path(temp_dir))


if __name__ == "__main__":
    unittest.main()
