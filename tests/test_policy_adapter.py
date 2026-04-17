from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.policy_adapter import PolicyAdapterConfig, TrainableQwenPolicy


class PolicyAdapterTests(unittest.TestCase):
    def test_trainable_model_load_kwargs_disable_auto_device_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy = TrainableQwenPolicy(
                PolicyAdapterConfig(
                    model_path=tmp_dir,
                    device="cuda:0",
                )
            )

        kwargs = policy._build_trainable_model_kwargs()

        self.assertEqual(
            kwargs,
            {
                "local_files_only": True,
                "torch_dtype": "auto",
                "low_cpu_mem_usage": False,
            },
        )
        self.assertNotIn("device_map", kwargs)

    def test_resolve_load_device_prefers_configured_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy = TrainableQwenPolicy(
                PolicyAdapterConfig(
                    model_path=tmp_dir,
                    device="cpu",
                )
            )

        self.assertEqual(str(policy._resolve_load_device()), "cpu")


if __name__ == "__main__":
    unittest.main()
