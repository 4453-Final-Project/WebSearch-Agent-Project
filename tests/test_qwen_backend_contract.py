"""Contract tests for the lazy GPTQ backend interface."""

from __future__ import annotations

import builtins
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
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

    def test_existing_directory_can_generate_with_patched_runtime(self) -> None:
        class FakeIds(list):
            def __init__(self, values):
                super().__init__(values)
                self.shape = (1, len(values))

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            def __init__(self) -> None:
                self.pad_token_id = 0
                self.eos_token_id = 99
                self.eos_token = "<eos>"

            def __call__(self, prompt: str, return_tensors: str):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def apply_chat_template(
                self,
                messages,
                add_generation_prompt: bool,
                return_tensors: str,
                tokenize: bool,
            ):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
                return 'ACTION: click("Pricing")'

        class FakeModel:
            def __init__(self) -> None:
                self.device = "cpu"
                self.generate_calls: list[dict[str, object]] = []

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return [[11, 22, 33, 44]]

        fake_model = FakeModel()

        class FakeTokenizerLoader:
            calls = 0

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls += 1
                return FakeTokenizer()

        class FakeModelLoader:
            calls = 0

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls += 1
                return fake_model

        fake_transformers = SimpleNamespace(
            AutoTokenizer=FakeTokenizerLoader,
            AutoModelForCausalLM=FakeModelLoader,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = TransformersGPTQBackend(PolicyConfig(model_path=temp_dir))
            backend._import_runtime_dependencies = lambda: (object(), fake_transformers)  # type: ignore[method-assign]

            result_one = backend.generate("test prompt")
            result_two = backend.generate("test prompt again")

        self.assertEqual(result_one, 'ACTION: click("Pricing")')
        self.assertEqual(result_two, 'ACTION: click("Pricing")')
        self.assertEqual(FakeTokenizerLoader.calls, 1)
        self.assertEqual(FakeModelLoader.calls, 1)
        self.assertEqual(len(fake_model.generate_calls), 2)

    def test_quantized_inference_kwargs_include_quantization_config_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = TransformersGPTQBackend(
                PolicyConfig(
                    model_path=temp_dir,
                    device="cuda:0",
                    quantization_mode="bnb_4bit",
                    quant_compute_dtype="float16",
                    quant_type="nf4",
                    quant_use_double_quant=False,
                )
            )

            fake_torch = SimpleNamespace(float16="float16", bfloat16="bfloat16", device=lambda value: value)
            fake_transformers = SimpleNamespace(BitsAndBytesConfig=object)
            backend._import_runtime_dependencies = lambda: (fake_torch, fake_transformers)  # type: ignore[method-assign]

            fake_quant_config = object()
            with mock.patch(
                "src.agent.qwen_policy.build_bitsandbytes_quantization_config",
                return_value=fake_quant_config,
            ):
                kwargs = backend._build_inference_model_kwargs()

        self.assertEqual(kwargs["quantization_config"], fake_quant_config)
        self.assertEqual(kwargs["device_map"], {"": 0})
        self.assertEqual(kwargs["torch_dtype"], "float16")

    def test_adapter_directory_loads_base_model_then_attaches_adapter(self) -> None:
        class FakeIds(list):
            def __init__(self, values):
                super().__init__(values)
                self.shape = (1, len(values))

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            def __init__(self) -> None:
                self.pad_token_id = 0
                self.eos_token_id = 99
                self.eos_token = "<eos>"

            def __call__(self, prompt: str, return_tensors: str):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def apply_chat_template(
                self,
                messages,
                add_generation_prompt: bool,
                return_tensors: str,
                tokenize: bool,
            ):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
                return 'ACTION: click("58")'

        class FakeModel:
            def __init__(self) -> None:
                self.device = "cpu"
                self.generate_calls: list[dict[str, object]] = []

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return [[11, 22, 33, 44]]

        fake_model = FakeModel()

        class FakeTokenizerLoader:
            calls = 0

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls += 1
                return FakeTokenizer()

        class FakeModelLoader:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append(args[0])
                return fake_model

        class FakePeftModel:
            calls = []

            @classmethod
            def from_pretrained(cls, model, adapter_path, is_trainable: bool = False):
                cls.calls.append((model, adapter_path, is_trainable))
                return model

        fake_transformers = SimpleNamespace(
            AutoTokenizer=FakeTokenizerLoader,
            AutoModelForCausalLM=FakeModelLoader,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            base_model_dir = Path(temp_dir) / "base-model"
            base_model_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "' + str(base_model_dir).replace("\\", "\\\\") + '"}',
                encoding="utf-8",
            )
            backend = TransformersGPTQBackend(PolicyConfig(model_path=str(adapter_dir)))
            backend._import_runtime_dependencies = lambda: (object(), fake_transformers)  # type: ignore[method-assign]

            original_peft = sys.modules.get("peft")
            sys.modules["peft"] = SimpleNamespace(PeftModel=FakePeftModel)
            try:
                result = backend.generate("test prompt")
            finally:
                if original_peft is None:
                    sys.modules.pop("peft", None)
                else:
                    sys.modules["peft"] = original_peft

        self.assertEqual(result, 'ACTION: click("58")')
        self.assertEqual(FakeModelLoader.calls, [str(base_model_dir)])
        self.assertEqual(len(FakePeftModel.calls), 1)
        self.assertEqual(FakePeftModel.calls[0][1], backend.model_path)
        self.assertFalse(FakePeftModel.calls[0][2])

    def test_adapter_directory_falls_back_to_base_tokenizer_when_adapter_tokenizer_is_invalid(self) -> None:
        class FakeIds(list):
            def __init__(self, values):
                super().__init__(values)
                self.shape = (1, len(values))

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            def __init__(self) -> None:
                self.pad_token_id = 0
                self.eos_token_id = 99
                self.eos_token = "<eos>"

            def __call__(self, prompt: str, return_tensors: str):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def apply_chat_template(
                self,
                messages,
                add_generation_prompt: bool,
                return_tensors: str,
                tokenize: bool,
            ):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
                return 'ACTION: click("58")'

        class FakeModel:
            def __init__(self) -> None:
                self.device = "cpu"

            def generate(self, **kwargs):
                return [[11, 22, 33, 44]]

        fake_model = FakeModel()

        class FakeTokenizerLoader:
            calls: list[str] = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append(args[0])
                if len(cls.calls) == 1:
                    raise ValueError("Tokenizer class TokenizersBackend does not exist")
                return FakeTokenizer()

        class FakeModelLoader:
            calls: list[str] = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append(args[0])
                return fake_model

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, adapter_path, is_trainable: bool = False):
                return model

        fake_transformers = SimpleNamespace(
            AutoTokenizer=FakeTokenizerLoader,
            AutoModelForCausalLM=FakeModelLoader,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            base_model_dir = Path(temp_dir) / "base-model"
            base_model_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "' + str(base_model_dir).replace("\\", "\\\\") + '"}',
                encoding="utf-8",
            )
            backend = TransformersGPTQBackend(PolicyConfig(model_path=str(adapter_dir)))
            backend._import_runtime_dependencies = lambda: (object(), fake_transformers)  # type: ignore[method-assign]

            original_peft = sys.modules.get("peft")
            sys.modules["peft"] = SimpleNamespace(PeftModel=FakePeftModel)
            try:
                result = backend.generate("test prompt")
            finally:
                if original_peft is None:
                    sys.modules.pop("peft", None)
                else:
                    sys.modules["peft"] = original_peft

        self.assertEqual(result, 'ACTION: click("58")')
        self.assertEqual([str(path) for path in FakeTokenizerLoader.calls], [str(adapter_dir), str(base_model_dir)])
        self.assertEqual(FakeModelLoader.calls, [str(base_model_dir)])

    def test_adapter_directory_normalizes_wsl_base_model_paths_on_windows(self) -> None:
        class FakeIds(list):
            def __init__(self, values):
                super().__init__(values)
                self.shape = (1, len(values))

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            def __init__(self) -> None:
                self.pad_token_id = 0
                self.eos_token_id = 99
                self.eos_token = "<eos>"

            def __call__(self, prompt: str, return_tensors: str):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def apply_chat_template(
                self,
                messages,
                add_generation_prompt: bool,
                return_tensors: str,
                tokenize: bool,
            ):
                return FakeBatch({"input_ids": FakeIds([11, 22])})

            def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
                return 'ACTION: click("58")'

        class FakeModel:
            def __init__(self) -> None:
                self.device = "cpu"

            def generate(self, **kwargs):
                return [[11, 22, 33, 44]]

        fake_model = FakeModel()

        class FakeTokenizerLoader:
            calls: list[str] = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append(str(args[0]))
                return FakeTokenizer()

        class FakeModelLoader:
            calls: list[str] = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append(str(args[0]))
                return fake_model

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, adapter_path, is_trainable: bool = False):
                return model

        fake_transformers = SimpleNamespace(
            AutoTokenizer=FakeTokenizerLoader,
            AutoModelForCausalLM=FakeModelLoader,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            base_model_dir = Path(temp_dir) / "base-model"
            base_model_dir.mkdir()
            drive = base_model_dir.drive.rstrip(":").lower()
            relative_parts = base_model_dir.as_posix().split(":/", 1)[1]
            wsl_base_model_dir = f"/mnt/{drive}/{relative_parts}"
            (adapter_dir / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "' + wsl_base_model_dir + '"}',
                encoding="utf-8",
            )
            backend = TransformersGPTQBackend(PolicyConfig(model_path=str(adapter_dir)))
            backend._import_runtime_dependencies = lambda: (object(), fake_transformers)  # type: ignore[method-assign]

            original_peft = sys.modules.get("peft")
            sys.modules["peft"] = SimpleNamespace(PeftModel=FakePeftModel)
            try:
                result = backend.generate("test prompt")
            finally:
                if original_peft is None:
                    sys.modules.pop("peft", None)
                else:
                    sys.modules["peft"] = original_peft

        self.assertEqual(result, 'ACTION: click("58")')
        self.assertEqual(FakeTokenizerLoader.calls, [str(adapter_dir)])
        self.assertEqual(FakeModelLoader.calls, [str(base_model_dir)])


if __name__ == "__main__":
    unittest.main()
