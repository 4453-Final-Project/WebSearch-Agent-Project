"""Unit tests for Task 3 shared types and compatibility shims."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
import tempfile
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.compat import load_policy_config, normalize_observation
from src.agent.types import AgentDecision, NormalizedObservation, OpenTab, PolicyConfig
from src.utils import config as config_module
from src.utils.config import get_env_var, get_model_path, get_workspace_root, resolve_model_path


class NormalizeObservationTests(unittest.TestCase):
    """Tests for observation alias handling and tab normalization."""

    def test_normalize_observation_supports_aliases(self) -> None:
        raw = {
            "task_goal": "Find the support page",
            "url": "https://example.com",
            "tabs": [{"title": "Home", "url": "https://example.com"}],
            "page_summary": "Example homepage",
            "ax_tree": "button: Support",
            "action_history": ["open homepage"],
            "error_history": ["timeout"],
        }

        observation = normalize_observation(raw)

        self.assertEqual(observation.goal, "Find the support page")
        self.assertEqual(observation.current_url, "https://example.com")
        self.assertEqual(observation.visible_page_summary, "Example homepage")
        self.assertEqual(observation.dom_or_ax_snippet, "button: Support")
        self.assertEqual(observation.previous_actions, ["open homepage"])
        self.assertEqual(observation.previous_errors, ["timeout"])
        self.assertIsNone(observation.last_action_error)

    def test_open_tabs_normalization_supports_mixed_inputs(self) -> None:
        class TabObject:
            def __init__(self, title: str, url: str) -> None:
                self.title = title
                self.url = url

        raw = {
            "goal": "Check tabs",
            "current_url": "https://current.example",
            "open_tabs": [
                {"title": "Docs", "url": "https://docs.example"},
                {"name": "About", "href": "https://about.example"},
                TabObject("Blog", "https://blog.example"),
                OpenTab(title="Support", url="https://support.example"),
            ],
            "visible_page_summary": "Summary",
            "dom_or_ax_snippet": "DOM",
            "previous_actions": [],
            "previous_errors": [],
            "last_action_error": "failed click",
        }

        observation = normalize_observation(raw)

        self.assertEqual(
            observation.open_tabs,
            [
                OpenTab(title="Docs", url="https://docs.example"),
                OpenTab(title="About", url="https://about.example"),
                OpenTab(title="Blog", url="https://blog.example"),
                OpenTab(title="Support", url="https://support.example"),
            ],
        )
        self.assertEqual(observation.last_action_error, "failed click")

    def test_normalize_observation_defaults_missing_values(self) -> None:
        observation = normalize_observation({})

        self.assertEqual(observation.goal, "")
        self.assertEqual(observation.current_url, "")
        self.assertEqual(observation.open_tabs, [])
        self.assertEqual(observation.visible_page_summary, "")
        self.assertEqual(observation.dom_or_ax_snippet, "")
        self.assertEqual(observation.previous_actions, [])
        self.assertEqual(observation.previous_errors, [])
        self.assertIsNone(observation.last_action_error)


class LoadPolicyConfigTests(unittest.TestCase):
    """Tests for policy config fallback behavior."""

    def test_load_policy_config_uses_safe_defaults(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_policy_config()

        self.assertEqual(
            config,
            PolicyConfig(
                model_path=str(get_model_path()),
                max_new_tokens=128,
                temperature=0.0,
                device=None,
            ),
        )

    def test_load_policy_config_reads_environment_overrides(self) -> None:
        env = {
            "TASK3_MODEL_PATH": "models/task3",
            "TASK3_MAX_NEW_TOKENS": "256",
            "TASK3_TEMPERATURE": "0.2",
            "TASK3_DEVICE": "cpu",
            "TASK3_QUANTIZATION_MODE": "bnb_4bit",
            "TASK3_QUANT_COMPUTE_DTYPE": "float16",
            "TASK3_QUANT_TYPE": "fp4",
            "TASK3_QUANT_USE_DOUBLE_QUANT": "false",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_policy_config()

        self.assertEqual(config.model_path, "models/task3")
        self.assertEqual(config.max_new_tokens, 256)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.quantization_mode, "bnb_4bit")
        self.assertEqual(config.quant_compute_dtype, "float16")
        self.assertEqual(config.quant_type, "fp4")
        self.assertFalse(config.quant_use_double_quant)


class ModelPathResolutionTests(unittest.TestCase):
    """Tests for local model path resolution helpers."""

    def test_resolve_model_path_prefers_explicit_path(self) -> None:
        resolved = resolve_model_path(model_path="~/models/test-model")
        self.assertEqual(resolved, Path("~/models/test-model").expanduser())

    def test_resolve_model_path_uses_model_dir_name_under_workspace_models(self) -> None:
        resolved = resolve_model_path(model_dir_name="AnotherModel")
        self.assertEqual(resolved, get_workspace_root() / "models" / "AnotherModel")


class RepoEnvLoadingTests(unittest.TestCase):
    """Tests for repo-local .env loading."""

    def test_get_env_var_loads_missing_values_from_repo_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / ".env").write_text(
                'export WA_SHOPPING="http://example.test:7770/"\nWA_MAP=http://example.test:3000\n',
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(config_module, "get_repo_root", return_value=repo_root):
                    config_module._REPO_ENV_LOADED = False
                    self.assertEqual(get_env_var("WA_SHOPPING"), "http://example.test:7770/")
                    self.assertEqual(get_env_var("WA_MAP"), "http://example.test:3000")
                    self.assertEqual(os.environ["WA_SHOPPING"], "http://example.test:7770/")

    def test_get_env_var_does_not_override_existing_shell_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / ".env").write_text('WA_SHOPPING=http://from-file.test:7770\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {"WA_SHOPPING": "http://from-shell.test:7770"}, clear=True):
                with mock.patch.object(config_module, "get_repo_root", return_value=repo_root):
                    config_module._REPO_ENV_LOADED = False
                    self.assertEqual(get_env_var("WA_SHOPPING"), "http://from-shell.test:7770")


class DataclassConstructionTests(unittest.TestCase):
    """Tests for the core Task 3 dataclass contracts."""

    def test_agent_decision_and_normalized_observation_construct_cleanly(self) -> None:
        decision = AgentDecision(
            raw_text="CLICK search",
            action_text="CLICK search",
            parse_error=None,
            should_retry=False,
        )
        observation = NormalizedObservation(
            goal="Find result",
            current_url="https://example.com",
            open_tabs=[OpenTab(title="Example", url="https://example.com")],
            visible_page_summary="A page",
            dom_or_ax_snippet="button Search",
            previous_actions=["open page"],
            previous_errors=[],
        )

        self.assertEqual(decision.action_text, "CLICK search")
        self.assertFalse(decision.should_retry)
        self.assertEqual(observation.open_tabs[0].title, "Example")


if __name__ == "__main__":
    unittest.main()
