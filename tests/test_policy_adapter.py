from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.policy_adapter import PolicyAdapterConfig, TrainableQwenPolicy
from src.training.rewards import build_optimization_samples
from src.training.trajectory import EpisodeTrajectory, TrajectoryStep


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = False):
        tokens = [index + 1 for index, _ in enumerate(text.split())]
        return {"input_ids": tokens}


class PolicyAdapterTests(unittest.TestCase):
    def test_supervised_batch_truncates_prompt_prefix_before_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = TrainableQwenPolicy(
                PolicyAdapterConfig(
                    model_path=temp_dir,
                    policy_name="test-policy",
                    use_chat_template=False,
                    max_supervised_tokens=10,
                )
            )
            tokenizer = FakeTokenizer()
            prompt = " ".join(f"prompt{i}" for i in range(12))
            response = "ACTION: click(1)"

            batch = policy._tokenize_supervised_batch([prompt], [response], tokenizer)

            self.assertEqual(batch["input_ids"].shape[1], 10)
            labels = batch["labels"][0].tolist()
            self.assertIn(99, labels)
            response_token_count = len(tokenizer(response)["input_ids"]) + 1
            self.assertEqual(sum(1 for value in labels if value != -100), response_token_count)

    def test_optimization_samples_use_compact_prompt_when_observation_exists(self) -> None:
        episode = EpisodeTrajectory(
            task_id=325,
            seed=42,
            raw_goal='Show me the "mouth night guard" listings by descending price.',
            success=True,
            reward=1.0,
            steps_taken=2,
            invalid_action_count=0,
            parse_failure_count=0,
            terminated=True,
            truncated=False,
            failure_reasons=[],
            output_dir="episode",
            steps_path="steps.jsonl",
            episode_path="episode.json",
            policy_name="test",
            steps=[
                TrajectoryStep(
                    step_idx=0,
                    prompt="FULL PROMPT SHOULD NOT BE REUSED",
                    response_text='ACTION: fill("274", "mouth night guard")',
                    action_text='fill("274", "mouth night guard")',
                    raw_text='ACTION: fill("274", "mouth night guard")',
                    parse_error=None,
                    last_action_error=None,
                    reward=0.0,
                    done=False,
                    terminated=False,
                    truncated=False,
                    observation={
                        "goal": 'Show me the "mouth night guard" listings by descending price.',
                        "current_url": "http://3.14.148.71:7770/",
                        "visible_page_summary": "Shopping home page",
                        "dom_or_ax_snippet": '[274] role=combobox name="Search" clickable\n[279] role=button name="Search" clickable',
                        "previous_actions": [],
                        "previous_errors": [],
                        "last_action_error": None,
                    },
                    info={},
                )
            ],
        )

        samples = build_optimization_samples(
            episode,
            group_id=0,
            episode_score=2.0,
            advantage=1.0,
        )

        self.assertEqual(len(samples), 1)
        self.assertNotEqual(samples[0].prompt, "FULL PROMPT SHOULD NOT BE REUSED")
        self.assertIn("Relevant DOM or AX-Tree Snippet:", samples[0].prompt)
        self.assertIn('ACTION: click("279")', samples[0].prompt)

    def test_trainable_policy_applies_shopping_submit_autocorrection(self) -> None:
        class StubTrainablePolicy(TrainableQwenPolicy):
            def __init__(self) -> None:
                super().__init__(
                    PolicyAdapterConfig(
                        model_path=tempfile.gettempdir(),
                        policy_name="test-policy",
                        use_chat_template=False,
                    )
                )
                self.outputs = ['ACTION: fill("272", "mouth night guard")']

            def generate_raw(self, prompt: str) -> str:
                return self.outputs.pop(0)

        policy = StubTrainablePolicy()
        observation = {
            "goal": 'Show me the "mouth night guard" listings by descending price.',
            "current_url": "http://3.14.148.71:7770/",
            "visible_page_summary": "mouth night guard (clickable)\nSearch (clickable)",
            "dom_or_ax_snippet": (
                '[1916] role=option name="mouth night guard" clickable\n'
                '[272] role=combobox name="Search" clickable focused\n'
                '[277] role=button name="Search" clickable'
            ),
            "previous_actions": ['fill("272", "mouth night guard")'],
            "previous_errors": [],
            "last_action_error": None,
        }

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("277")')
        self.assertIsNone(decision.parse_error)


if __name__ == "__main__":
    unittest.main()
