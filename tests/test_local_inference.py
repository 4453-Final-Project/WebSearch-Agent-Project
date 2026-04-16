from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.local_inference import build_generation_kwargs, build_model_inputs, fast_generation_mode, render_prompt_text


class FakeBatch(dict):
    pass


class FakeTokenizer:
    pad_token_id = 7
    eos_token_id = 9

    def __call__(self, text: str, return_tensors: str | None = None):
        return FakeBatch({"input_ids": [[1, 2, 3]], "text": text, "return_tensors": return_tensors})

    def apply_chat_template(self, messages, add_generation_prompt: bool, return_tensors=None, tokenize: bool = True):
        if tokenize:
            return FakeBatch(
                {
                    "messages": messages,
                    "add_generation_prompt": add_generation_prompt,
                    "return_tensors": return_tensors,
                    "tokenize": tokenize,
                }
            )
        rendered_messages = [f"{message['role']}: {message['content']}" for message in messages]
        return "\n".join(rendered_messages) + "\nassistant:"


class LocalInferenceHelperTests(unittest.TestCase):
    def test_build_generation_kwargs_respects_sampling_controls(self) -> None:
        tokenizer = FakeTokenizer()
        kwargs = build_generation_kwargs(
            tokenizer,
            max_new_tokens=32,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
        )
        self.assertEqual(kwargs["max_new_tokens"], 32)
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["top_k"], 50)
        self.assertEqual(kwargs["repetition_penalty"], 1.05)
        self.assertEqual(kwargs["pad_token_id"], 7)

    def test_build_model_inputs_uses_chat_template_when_available(self) -> None:
        tokenizer = FakeTokenizer()
        batch = build_model_inputs(
            tokenizer,
            prompt="What is C. elegans?",
            system_prompt="You are helpful.",
            use_chat_template=True,
        )
        self.assertEqual(batch["messages"][0]["role"], "system")
        self.assertEqual(batch["messages"][1]["role"], "user")

    def test_build_model_inputs_falls_back_to_raw_tokenization(self) -> None:
        tokenizer = FakeTokenizer()
        batch = build_model_inputs(
            tokenizer,
            prompt="Hello",
            system_prompt=None,
            use_chat_template=False,
        )
        self.assertEqual(batch["text"], "Hello")

    def test_render_prompt_text_uses_chat_template_when_available(self) -> None:
        tokenizer = FakeTokenizer()
        rendered = render_prompt_text(
            tokenizer,
            prompt="Open the pricing page",
            system_prompt="You are helpful.",
            use_chat_template=True,
        )
        self.assertIn("system: You are helpful.", rendered)
        self.assertIn("user: Open the pricing page", rendered)
        self.assertTrue(rendered.endswith("assistant:"))

    def test_render_prompt_text_falls_back_to_raw_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        rendered = render_prompt_text(
            tokenizer,
            prompt="Hello",
            system_prompt=None,
            use_chat_template=False,
        )
        self.assertEqual(rendered, "Hello")

    def test_fast_generation_mode_enables_use_cache_and_restores_state(self) -> None:
        class FakeConfig:
            def __init__(self) -> None:
                self.use_cache = False

        class FakeModel:
            def __init__(self) -> None:
                self.config = FakeConfig()
                self.training = True
                self.is_gradient_checkpointing = True
                self.disabled = 0
                self.enabled = 0

            def eval(self) -> None:
                self.training = False

            def train(self, mode: bool = True) -> None:
                self.training = mode

            def gradient_checkpointing_disable(self) -> None:
                self.disabled += 1
                self.is_gradient_checkpointing = False

            def gradient_checkpointing_enable(self) -> None:
                self.enabled += 1
                self.is_gradient_checkpointing = True

        model = FakeModel()
        with fast_generation_mode(model):
            self.assertTrue(model.config.use_cache)
            self.assertFalse(model.training)
            self.assertEqual(model.disabled, 1)

        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.training)
        self.assertEqual(model.enabled, 1)


if __name__ == "__main__":
    unittest.main()
