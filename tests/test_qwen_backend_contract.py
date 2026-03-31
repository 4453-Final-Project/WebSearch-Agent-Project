"""Contract tests for the lazy GPTQ backend interface."""

from __future__ import annotations

import builtins
import importlib
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.fake_backend import FakeBackend
from src.agent.qwen_policy import QwenPolicy, TransformersGPTQBackend
from src.agent.types import PolicyConfig


class QwenBackendContractTests(unittest.TestCase):
    """Tests for the lazy backend contract without real model loading."""

    def test_backend_class_exists(self) -> None:
        self.assertTrue(callable(TransformersGPTQBackend))

    def test_missing_model_path_raises_clear_exception(self) -> None:
        config = PolicyConfig(model_path="d:/RL4453/WebSearch-Agent-Project/does-not-exist")

        with self.assertRaises(FileNotFoundError) as context:
            TransformersGPTQBackend(config)

        self.assertIn("Configured model path does not exist", str(context.exception))

    def test_qwen_policy_accepts_custom_backend_object(self) -> None:
        backend = FakeBackend(['ACTION: stop("N/A")'])
        policy = QwenPolicy(backend=backend, config=PolicyConfig(model_path="fake-model"))

        self.assertIs(policy.backend, backend)

    def test_module_import_does_not_eagerly_require_heavyweight_imports(self) -> None:
        module_name = "src.agent.qwen_policy"
        original_import = builtins.__import__
        removed_modules: dict[str, object] = {}

        for name in [module_name, "src.agent", "src"]:
            if name in sys.modules:
                removed_modules[name] = sys.modules.pop(name)

        attempted_heavy_imports: list[str] = []

        def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            root_name = name.split(".")[0]
            if root_name in {"torch", "transformers"}:
                attempted_heavy_imports.append(root_name)
                raise AssertionError(f"Unexpected eager import: {root_name}")
            return original_import(name, globals, locals, fromlist, level)

        try:
            builtins.__import__ = guarded_import
            importlib.import_module(module_name)
        finally:
            builtins.__import__ = original_import
            for name in [module_name, "src.agent", "src"]:
                sys.modules.pop(name, None)
            sys.modules.update(removed_modules)

        self.assertEqual(attempted_heavy_imports, [])

    def test_existing_directory_still_fails_cleanly_without_real_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = TransformersGPTQBackend(PolicyConfig(model_path=temp_dir))

            with self.assertRaises(RuntimeError) as context:
                backend.generate("test prompt")

        message = str(context.exception)
        self.assertTrue(
            "requires 'transformers' and 'torch'" in message
            or "Model loading is not yet possible in the current environment." in message
        )


if __name__ == "__main__":
    unittest.main()
