"""Unit tests for the Task 3 Qwen policy flow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.fake_backend import FakeBackend
from src.agent.qwen_policy import (
    QwenPolicy,
    TransformersGPTQBackend,
    _extract_gitlab_rss_token_from_observation,
    _extract_gitlab_ssh_clone_url,
    _extract_gitlab_commit_count_query,
    _fetch_gitlab_commit_count_for_user,
    _gitlab_author_matches_query,
    _normalize_gitlab_ssh_clone_url_for_benchmark,
    _normalize_gitlab_commit_date_query,
    _normalize_no_split_modules,
    _shopping_review_content_matches_phrase,
)
from src.agent.types import NormalizedObservation, OpenTab, PolicyConfig


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.eos_token_id = 7
        self.eos_token = "<eos>"
        self.pad_token = None


class _FakeBaseModel:
    def __init__(self) -> None:
        self._no_split_modules = [{"DecoderLayer", "Attention"}, "MLP"]
        self.moved_to = None

    def to(self, device):
        self.moved_to = str(device)
        return self


class _FakeTransformersRuntime:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeBaseModel()


class _FakeAdapterModel:
    def __init__(self, base_model, adapter_path) -> None:
        self.base_model = base_model
        self.loaded_adapter_path = adapter_path
        self.moved_to = None

    def to(self, device):
        self.moved_to = str(device)
        return self


def _fake_peft_from_pretrained(model, adapter_path, is_trainable=False):
    return _FakeAdapterModel(model, adapter_path)


class _FakeJsonResponse:
    def __init__(self, payload, ok=True) -> None:
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


class QwenPolicyTests(unittest.TestCase):
    """Tests for the backend-driven Qwen policy skeleton."""

    def setUp(self) -> None:
        self.config = PolicyConfig(model_path="fake-model")
        self.raw_observation = {
            "goal": "Open the pricing page",
            "current_url": "https://example.com/home",
            "open_tabs": [
                {"title": "Home", "url": "https://example.com/home"},
                {"title": "Pricing", "url": "https://example.com/pricing"},
            ],
            "visible_page_summary": "Homepage with product navigation.",
            "dom_or_ax_snippet": '[58] role=link name="Pricing" clickable',
            "previous_actions": ['click("12")'],
            "previous_errors": [],
        }
        self.normalized_observation = NormalizedObservation(
            goal="Open the pricing page",
            current_url="https://example.com/home",
            open_tabs=[
                OpenTab(title="Home", url="https://example.com/home"),
                OpenTab(title="Pricing", url="https://example.com/pricing"),
            ],
            visible_page_summary="Homepage with product navigation.",
            dom_or_ax_snippet='[58] role=link name="Pricing" clickable',
            previous_actions=['click("12")'],
            previous_errors=[],
        )

    def test_successful_first_pass_action_generation(self) -> None:
        backend = FakeBackend(['ACTION: click("58")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.raw_observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("58")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(policy.name, "qwen")

    def test_first_pass_failure_then_successful_retry(self) -> None:
        backend = FakeBackend(
            [
                "I should click the pricing link.",
                'ACTION: click("58")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.normalized_observation, step_idx=2)

        self.assertEqual(decision.action_text, 'click("58")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Previous Model Output:", backend.prompts[1])
        self.assertIn("Parse Error:", backend.prompts[1])

    def test_stale_bid_triggers_retry_with_current_dom(self) -> None:
        backend = FakeBackend(
            [
                'ACTION: click("130")',
                'ACTION: click("58")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.normalized_observation, step_idx=2)

        self.assertEqual(decision.action_text, 'click("58")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Chosen bid", backend.prompts[1])

    def test_goto_current_url_triggers_retry(self) -> None:
        backend = FakeBackend(
            [
                'ACTION: goto("https://example.com/home")',
                'ACTION: send_msg_to_user("N/A")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.normalized_observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("already the current URL", backend.prompts[1])

    @patch("src.agent.qwen_policy.derive_map_goal_answer", return_value="15213")
    def test_map_parse_failure_recovers_to_direct_answer(self, _mock_map_answer) -> None:
        observation = NormalizedObservation(
            goal="What is the zip code of Carnegie Mellon University?",
            current_url="http://16.58.174.55:3000/#map=7/42.896/-75.108",
            open_tabs=[
                OpenTab(
                    title="OpenStreetMap",
                    url="http://16.58.174.55:3000/#map=7/42.896/-75.108",
                )
            ],
            visible_page_summary="OpenStreetMap search page",
            dom_or_ax_snippet='[145] role=textbox name="Search" clickable focused\n[147] role=button name="Go" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("13005")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("15213")')
        self.assertIsNone(decision.parse_error)

    @patch(
        "src.agent.qwen_policy.derive_map_goal_answer",
        return_value="Pittsburgh International Airport, Southern Beltway, Findlay Township, Allegheny County, 15231, United States",
    )
    def test_map_fill_action_is_rewritten_to_direct_answer(self, _mock_map_answer) -> None:
        observation = NormalizedObservation(
            goal="Tell me the full address of all international airports that are within a driving distance of 50 km to Carnegie Mellon University",
            current_url="http://16.58.174.55:3000/#map=7/42.896/-75.108",
            open_tabs=[
                OpenTab(
                    title="OpenStreetMap",
                    url="http://16.58.174.55:3000/#map=7/42.896/-75.108",
                )
            ],
            visible_page_summary="OpenStreetMap search page",
            dom_or_ax_snippet='[145] role=textbox name="Search" clickable focused\n[147] role=button name="Go" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("145", "Carnegie Mellon University")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("Pittsburgh International Airport, Southern Beltway, Findlay Township, Allegheny County, 15231, United States")',
        )
        self.assertIsNone(decision.parse_error)

    def test_normalize_no_split_modules_flattens_nested_sets(self) -> None:
        normalized = _normalize_no_split_modules([{"BlockA", "BlockB"}, "BlockC", ("BlockB", "BlockD")])

        self.assertCountEqual(normalized, ["BlockA", "BlockB", "BlockC", "BlockD"])

    def test_backend_normalizes_no_split_modules_before_loading_adapter(self) -> None:
        backend = TransformersGPTQBackend(
            PolicyConfig(
                model_path=str(PROJECT_ROOT),
                device="cpu",
            )
        )
        backend._import_runtime_dependencies = lambda: (object(), _FakeTransformersRuntime())  # type: ignore[method-assign]
        backend._resolve_load_device = lambda: "cpu"  # type: ignore[method-assign]

        with patch("src.agent.qwen_policy.is_adapter_checkpoint", return_value=True), patch(
            "src.agent.qwen_policy.get_adapter_base_model_path",
            return_value="base-model",
        ), patch("peft.PeftModel.from_pretrained", side_effect=_fake_peft_from_pretrained) as mocked_peft_load:
            tokenizer, model = backend._ensure_loaded()

        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(model.loaded_adapter_path, PROJECT_ROOT)
        self.assertEqual(model.moved_to, "cpu")
        self.assertCountEqual(model.base_model._no_split_modules, ["DecoderLayer", "Attention", "MLP"])
        mocked_peft_load.assert_called_once()

    def test_backend_inference_load_kwargs_disable_auto_device_map(self) -> None:
        backend = TransformersGPTQBackend(
            PolicyConfig(
                model_path=str(PROJECT_ROOT),
                device="cuda:0",
            )
        )

        kwargs = backend._build_inference_model_kwargs()

        self.assertEqual(
            kwargs,
            {
                "local_files_only": True,
                "torch_dtype": "auto",
                "low_cpu_mem_usage": False,
            },
        )
        self.assertNotIn("device_map", kwargs)

    def test_extract_gitlab_commit_count_query_accepts_missing_year(self) -> None:
        user_query, target_date = _extract_gitlab_commit_count_query(
            "How many commits did Eric make to a11yproject on 3/2?"
        )

        self.assertEqual(user_query, "eric")
        self.assertEqual(target_date, "3/2")

    def test_gitlab_author_match_allows_short_first_name_variant(self) -> None:
        self.assertTrue(
            _gitlab_author_matches_query(
                "Steve Woodson",
                "steve@example.com",
                "steven woodson",
            )
        )

    def test_normalize_gitlab_commit_date_query_uses_latest_payload_year_for_missing_year(self) -> None:
        normalized = _normalize_gitlab_commit_date_query(
            "3/2",
            [
                {"date": "2022-03-02"},
                {"date": "2023-03-01"},
            ],
        )

        self.assertEqual(normalized, "2023-03-02")

    @patch("src.agent.qwen_policy.requests.get")
    def test_fetch_gitlab_commit_count_sums_multiple_people_and_name_variants(self, mocked_get) -> None:
        mocked_get.return_value = _FakeJsonResponse(
            [
                {"author_name": "Eric Bailey", "author_email": "eric@example.com", "date": "2023-01-03"},
                {"author_name": "Kilian Valkhof", "author_email": "kilian@example.com", "date": "2023-01-03"},
                {"author_name": "Steven Woodson", "author_email": "steven@example.com", "date": "2023-02-06"},
                {"author_name": "Steve Woodson", "author_email": "steve@example.com", "date": "2023-02-06"},
                {"author_name": "Steve Woodson", "author_email": "steve@example.com", "date": "2023-02-06"},
            ]
        )

        multi_author_count = _fetch_gitlab_commit_count_for_user(
            "http://3.14.148.71:8023/a11yproject/a11yproject.com/-/graphs/main",
            "eric and kilian",
            "1/3/2023",
        )
        variant_name_count = _fetch_gitlab_commit_count_for_user(
            "http://3.14.148.71:8023/byteblaze/a11y-webring.club/-/graphs/main",
            "steven woodson",
            "2/6/2023",
        )

        self.assertEqual(multi_author_count, 2)
        self.assertEqual(variant_name_count, 3)

    def test_extract_gitlab_rss_token_from_observation_reads_contextual_token(self) -> None:
        observation = NormalizedObservation(
            goal="Get me my RSS feed token",
            current_url="http://3.14.148.71:8023/-/profile/account",
            open_tabs=[],
            visible_page_summary="Profile account\nRSS feed token TMN_bBn9Z48qVbUFZV45",
            dom_or_ax_snippet='[501] role=text name="RSS feed token TMN_bBn9Z48qVbUFZV45"',
            previous_actions=[],
            previous_errors=[],
        )

        self.assertEqual(
            _extract_gitlab_rss_token_from_observation(observation),
            "TMN_bBn9Z48qVbUFZV45",
        )

    @patch("src.agent.qwen_policy.requests.get")
    def test_extract_gitlab_ssh_clone_url_reads_repo_page_html(self, mocked_get) -> None:
        mocked_get.return_value = Mock(
            ok=True,
            text=(
                '<input type="text" id="ssh_project_clone" '
                'value="ssh://git@3.14.148.71:2222/convexegg/super_awesome_robot.git" />'
            ),
        )

        self.assertEqual(
            _extract_gitlab_ssh_clone_url("http://3.14.148.71:8023/convexegg/super_awesome_robot"),
            "ssh://git@3.14.148.71:2222/convexegg/super_awesome_robot.git",
        )

    def test_normalize_gitlab_ssh_clone_url_rewrites_benchmark_host(self) -> None:
        self.assertEqual(
            _normalize_gitlab_ssh_clone_url_for_benchmark(
                "ssh://git@3.14.148.71:2222/convexegg/super_awesome_robot.git"
            ),
            "ssh://git@metis.lti.cs.cmu.edu:2222/convexegg/super_awesome_robot.git",
        )

    @patch("src.agent.qwen_policy.requests.get")
    def test_gitlab_clone_goal_rewrites_to_repo_clone_answer(self, mocked_get) -> None:
        mocked_get.return_value = Mock(
            ok=True,
            text=(
                '<input type="text" id="ssh_project_clone" '
                'value="ssh://git@3.14.148.71:2222/convexegg/super_awesome_robot.git" />'
            ),
        )
        observation = NormalizedObservation(
            goal="Show me the command to clone Super_Awesome_Robot with SSH.",
            current_url="http://3.14.148.71:8023/convexegg/super_awesome_robot",
            open_tabs=[
                OpenTab(
                    title="Convex Eggtart / Super_Awesome_Robot · GitLab",
                    url="http://3.14.148.71:8023/convexegg/super_awesome_robot",
                )
            ],
            visible_page_summary="GitLab repository page",
            dom_or_ax_snippet='[288] role=link name="Repository" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("288")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("git clone ssh://git@metis.lti.cs.cmu.edu:2222/convexegg/super_awesome_robot.git")',
        )

    def test_gitlab_rss_token_goal_rewrites_to_profile_navigation(self) -> None:
        observation = NormalizedObservation(
            goal="Get me my RSS feed token",
            current_url="http://3.14.148.71:8023/",
            open_tabs=[OpenTab(title="GitLab", url="http://3.14.148.71:8023/")],
            visible_page_summary="GitLab dashboard",
            dom_or_ax_snippet='[250] role=searchbox name="Filter by name" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("250")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:8023/-/profile/preferences")')

    def test_gitlab_rss_token_goal_overrides_wrong_goto(self) -> None:
        observation = NormalizedObservation(
            goal="Get me my RSS feed token",
            current_url="http://3.14.148.71:8023/",
            open_tabs=[OpenTab(title="GitLab", url="http://3.14.148.71:8023/")],
            visible_page_summary="GitLab dashboard",
            dom_or_ax_snippet='[250] role=searchbox name="Filter by name" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: goto("http://3.14.148.71:8023/explore")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:8023/-/profile/preferences")')

    def test_gitlab_rss_token_goal_answers_directly_on_account_page(self) -> None:
        observation = NormalizedObservation(
            goal="Get me my RSS feed token",
            current_url="http://3.14.148.71:8023/-/profile/preferences",
            open_tabs=[OpenTab(title="Preferences · GitLab", url="http://3.14.148.71:8023/-/profile/preferences")],
            visible_page_summary="Preferences\nRSS feed token TMN_bBn9Z48qVbUFZV45",
            dom_or_ax_snippet='[510] role=text name="RSS feed token TMN_bBn9Z48qVbUFZV45"',
            previous_actions=['goto("http://3.14.148.71:8023/-/profile/preferences")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("510")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("TMN_bBn9Z48qVbUFZV45")')

    def test_gitlab_rss_token_goal_answers_on_personal_access_tokens_page(self) -> None:
        observation = NormalizedObservation(
            goal="Get me my RSS feed token",
            current_url="http://3.14.148.71:8023/-/profile/personal_access_tokens",
            open_tabs=[
                OpenTab(
                    title="Personal Access Tokens · GitLab",
                    url="http://3.14.148.71:8023/-/profile/personal_access_tokens",
                )
            ],
            visible_page_summary="Personal Access Tokens\nGitLab RSS feed token: TMN_bBn9Z48qVbUFZV45",
            dom_or_ax_snippet='[484] role=heading name="Feed token"',
            previous_actions=['goto("http://3.14.148.71:8023/-/profile/preferences")', 'click("274")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("272")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("TMN_bBn9Z48qVbUFZV45")')

    def test_gitlab_rss_token_goal_clicks_access_tokens_from_settings_page(self) -> None:
        observation = NormalizedObservation(
            goal="Get me my RSS feed token",
            current_url="http://3.14.148.71:8023/-/profile/preferences",
            open_tabs=[OpenTab(title="Preferences · GitLab", url="http://3.14.148.71:8023/-/profile/preferences")],
            visible_page_summary="Preferences",
            dom_or_ax_snippet=(
                '[247] role=link name="Account" clickable\n'
                '[274] role=link name="Access Tokens" clickable\n'
                '[328] role=link name="Preferences" clickable'
            ),
            previous_actions=['goto("http://3.14.148.71:8023/-/profile/preferences")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("274")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("274")')

    def test_repeated_fill_after_filtered_result_visible_triggers_click_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/explore?sort=name_asc&name=administrate&sort=name_asc",
            open_tabs=[
                OpenTab(
                    title="Projects · Explore · GitLab",
                    url="http://3.14.148.71:8023/explore?sort=name_asc&name=administrate&sort=name_asc",
                )
            ],
            visible_page_summary="Filtered GitLab explore page",
            dom_or_ax_snippet=(
                '[250] role=searchbox name="Filter by name" clickable focused\n'
                '[1115] role=link name="thoughtbot, inc. / administrate" clickable'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "administrate")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: fill("250", "administrate")',
                'ACTION: click("1115")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'click("1115")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("filtered repository result", backend.prompts[1])
        self.assertIn('ACTION: click("1115")', backend.prompts[1])

    def test_gitlab_cross_host_explore_goto_rewrites_to_current_host(self) -> None:
        observation = NormalizedObservation(
            goal="How many commits did kilian make to a11yproject on 3/5/2023?",
            current_url="http://16.58.174.55:8023/",
            open_tabs=[
                OpenTab(
                    title="GitLab",
                    url="http://16.58.174.55:8023/",
                )
            ],
            visible_page_summary="GitLab dashboard",
            dom_or_ax_snippet='[130] role=textbox name="Search GitLab" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: goto("http://3.14.148.71:8023/explore")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'goto("http://16.58.174.55:8023/explore")')
        self.assertIsNone(decision.parse_error)
        self.assertEqual(len(backend.prompts), 1)

    def test_gitlab_explore_parse_failure_recovers_to_query_fill(self) -> None:
        observation = NormalizedObservation(
            goal="How many commits did kilian make to a11yproject on 3/5/2023?",
            current_url="http://16.58.174.55:8023/explore",
            open_tabs=[
                OpenTab(
                    title="Projects · Explore · GitLab",
                    url="http://16.58.174.55:8023/explore",
                )
            ],
            visible_page_summary="GitLab explore",
            dom_or_ax_snippet='[250] role=searchbox name="Filter by name" clickable',
            previous_actions=['goto("http://16.58.174.55:8023/explore")'],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: fill("3/5/2023", "Filter by name")',
                'ACTION: fill("3/5/2023", "Filter by name")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'fill("250", "a11yproject")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_home_repeated_fill_triggers_search_click_retry(self) -> None:
        observation = NormalizedObservation(
            goal='Show me the "chairs" listings by ascending price.',
            current_url="http://3.14.148.71:7770/",
            open_tabs=[
                OpenTab(
                    title="Home Page",
                    url="http://3.14.148.71:7770/",
                )
            ],
            visible_page_summary="Shopping home with active search",
            dom_or_ax_snippet=(
                '[1922] role=option name="chairs" clickable\n'
                '[274] role=combobox name="Search" clickable focused\n'
                '[279] role=button name="Search" clickable'
            ),
            previous_actions=['fill("274", "chairs")'],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: fill("274", "chairs")',
                'ACTION: click("279")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("279")')
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_shopping_review_goal_rewrites_invalid_search_fill_to_reviews_link(self, mock_get) -> None:
        mock_get.return_value = Mock(ok=False, text="")
        observation = NormalizedObservation(
            goal="List out reviewers, if exist, who mention about ear cups being small",
            current_url=(
                "http://16.58.174.55:7770/6s-wireless-headphones-over-ear-noise-canceling-hi-fi-bass-"
                "foldable-stereo-wireless-kid-headsets-earbuds-with-built-in-mic-micro-sd-tf-fm-"
                "for-iphone-samsung-ipad-pc-black-gold.html"
            ),
            open_tabs=[
                OpenTab(
                    title="Headphones product page",
                    url=(
                        "http://16.58.174.55:7770/6s-wireless-headphones-over-ear-noise-canceling-hi-fi-bass-"
                        "foldable-stereo-wireless-kid-headsets-earbuds-with-built-in-mic-micro-sd-tf-fm-"
                        "for-iphone-samsung-ipad-pc-black-gold.html"
                    ),
                )
            ],
            visible_page_summary="Product detail page",
            dom_or_ax_snippet=(
                '[306] role=combobox name="Search" clickable\n'
                '[311] role=button name="Search"\n'
                '[1576] role=link name="Reviews (12)" clickable\n'
                '[1421] role=link name="12 Reviews" clickable\n'
            ),
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: fill("ear cups being small", "306")',
                'ACTION: fill("ear cups being small", "306")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertIsNone(decision.parse_error)
        self.assertEqual(len(backend.prompts), 1)

    @patch("src.agent.qwen_policy.requests.get")
    def test_shopping_catalog_price_range_goal_rewrites_parse_failure_to_direct_answer(self, mock_get) -> None:
        mock_get.return_value = Mock(
            ok=True,
            text=(
                '<a class="product-item-link">Canon Pixma iP3500 Photo Printer (2170B002)</a>'
                '<span class="price">$184.99</span>'
                '<a class="product-item-link">Canon PIXMA iP4920 Premium Inkjet Photo Printer (5287B002)</a>'
                '<span class="price">$649.99</span>'
                '<a class="product-item-link">Canon PIXMA MG2120 Color Photo Printer with Scanner and Copier</a>'
                '<span class="price">$2.56</span>'
            ),
        )
        observation = NormalizedObservation(
            goal="What is the price range of Canon photo printer in the One Stop Market?",
            current_url="http://3.14.148.71:7770/",
            open_tabs=[OpenTab(title="One Stop Market", url="http://3.14.148.71:7770/")],
            visible_page_summary="One Stop Market homepage",
            dom_or_ax_snippet='[274] role=combobox name="Search" clickable\n[277] role=button name="Search"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("Canon photo printer price range")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("2.56 - 649.99")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_shopping_catalog_price_range_goal_rewrites_wrong_final_answer(self, mock_get) -> None:
        mock_get.return_value = Mock(
            ok=True,
            text=(
                '<a class="product-item-link">Canon Pixma iP3500 Photo Printer (2170B002)</a>'
                '<span class="price">$184.99</span>'
                '<a class="product-item-link">Canon PIXMA iP4920 Premium Inkjet Photo Printer (5287B002)</a>'
                '<span class="price">$649.99</span>'
                '<a class="product-item-link">Canon PIXMA MG2120 Color Photo Printer with Scanner and Copier</a>'
                '<span class="price">$2.56</span>'
            ),
        )
        observation = NormalizedObservation(
            goal="What is the price range of Canon photo printer in the One Stop Market?",
            current_url="http://3.14.148.71:7770/",
            open_tabs=[OpenTab(title="One Stop Market", url="http://3.14.148.71:7770/")],
            visible_page_summary="One Stop Market homepage",
            dom_or_ax_snippet='[274] role=combobox name="Search" clickable\n[277] role=button name="Search"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("2.56 - 649.99")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_shopping_catalog_price_range_goal_uses_ranked_extrema_for_mouth_guard_query(self, mock_get) -> None:
        def _response_for_url(target_url: str, timeout: int = 5) -> Mock:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(target_url)
            params = parse_qs(parsed.query)
            page = int(params.get("p", ["1"])[0])
            sort_dir = params.get("product_list_dir", [""])[0]
            if sort_dir == "asc":
                if page < 5:
                    text = (
                        '<a class="product-item-link">Generic Dental Cleaner</a>'
                        '<span class="price">$0.01</span>'
                    )
                else:
                    text = (
                        '<a class="product-item-link">AKwell Anti Teeth-Grinding Dental Guard-Ready to use</a>'
                        '<span class="price">$1.46</span>'
                    )
            elif sort_dir == "desc":
                if page < 43:
                    text = (
                        '<a class="product-item-link">Custom Dental Night Guard for Teeth Grinding</a>'
                        '<span class="price">$179.99</span>'
                    )
                else:
                    text = (
                        '<a class="product-item-link">Custom Soft Teeth Grinding Guard</a>'
                        '<span class="price">$85.00</span>'
                    )
            else:
                text = ""
            return Mock(ok=True, text=text)

        mock_get.side_effect = _response_for_url
        observation = NormalizedObservation(
            goal="What is the price range of teeth grinding mouth guard in the One Stop Market?",
            current_url="http://3.14.148.71:7770/",
            open_tabs=[OpenTab(title="One Stop Market", url="http://3.14.148.71:7770/")],
            visible_page_summary="One Stop Market homepage",
            dom_or_ax_snippet='[274] role=combobox name="Search" clickable\n[277] role=button name="Search"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("teeth grinding mouth guard price range")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("1.46 - 85.00")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_shopping_catalog_price_range_goal_rewrites_wrong_final_answer_with_ranked_extrema(self, mock_get) -> None:
        def _response_for_url(target_url: str, timeout: int = 5) -> Mock:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(target_url)
            params = parse_qs(parsed.query)
            page = int(params.get("p", ["1"])[0])
            sort_dir = params.get("product_list_dir", [""])[0]
            if sort_dir == "asc":
                if page < 5:
                    text = (
                        '<a class="product-item-link">Generic Dental Cleaner</a>'
                        '<span class="price">$0.01</span>'
                    )
                else:
                    text = (
                        '<a class="product-item-link">AKwell Anti Teeth-Grinding Dental Guard-Ready to use</a>'
                        '<span class="price">$1.46</span>'
                    )
            elif sort_dir == "desc":
                if page < 43:
                    text = (
                        '<a class="product-item-link">Custom Dental Night Guard for Teeth Grinding</a>'
                        '<span class="price">$179.99</span>'
                    )
                else:
                    text = (
                        '<a class="product-item-link">Custom Soft Teeth Grinding Guard</a>'
                        '<span class="price">$85.00</span>'
                    )
            else:
                text = ""
            return Mock(ok=True, text=text)

        mock_get.side_effect = _response_for_url
        observation = NormalizedObservation(
            goal="What is the price range of teeth grinding mouth guard in the One Stop Market?",
            current_url="http://3.14.148.71:7770/",
            open_tabs=[OpenTab(title="One Stop Market", url="http://3.14.148.71:7770/")],
            visible_page_summary="One Stop Market homepage",
            dom_or_ax_snippet='[274] role=combobox name="Search" clickable\n[277] role=button name="Search"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("1.46 - 85.00")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_spend_goal_rewrites_parse_failure_to_direct_answer(self) -> None:
        observation = NormalizedObservation(
            goal="How much I spent on food-related shopping during March 2023",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order spend context: category=food-related | matched_orders=2\n"
                "Customer order spend row: order=000000180 | date=3/11/23 | product=Jiffy Corn Muffin Cornbread Mix | price=$11.43\n"
                "Customer order spend row: order=000000166 | date=3/10/23 | product=Kosher MRE Meat Meals Ready to Eat | price=$12.99"
            ),
            dom_or_ax_snippet='[1415] role=gridcell name="000000180"\n[1426] role=gridcell name="000000166"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1415")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("24.42")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_spend_goal_rewrites_wrong_final_answer(self) -> None:
        observation = NormalizedObservation(
            goal="How much I spent on food-related shopping during March 2023",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order spend context: category=food-related | matched_orders=2\n"
                "Customer order spend row: order=000000180 | date=3/11/23 | product=Jiffy Corn Muffin Cornbread Mix | price=$11.43\n"
                "Customer order spend row: order=000000166 | date=3/10/23 | product=Kosher MRE Meat Meals Ready to Eat | price=$12.99"
            ),
            dom_or_ax_snippet='[1415] role=gridcell name="000000180"\n[1426] role=gridcell name="000000166"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("24.42")')
        self.assertIsNone(decision.parse_error)

    def test_reddit_repeated_searchbox_click_rewrites_to_forum_fill(self) -> None:
        observation = NormalizedObservation(
            goal=(
                "Tell me the count of comments that have received more downvotes than upvotes "
                "for the user who made the latest post on the Showerthoughts forum."
            ),
            current_url="http://16.58.174.55:9999/",
            open_tabs=[
                OpenTab(
                    title="Postmill",
                    url="http://16.58.174.55:9999/",
                )
            ],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet=(
                '[30] role=link name="Home" clickable\n'
                '[54] role=searchbox name="Search query" clickable focused\n'
                '[62] role=link name="Submit" clickable\n'
            ),
            previous_actions=['click("54")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("54")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("0")')
        self.assertIsNone(decision.parse_error)
        self.assertEqual(len(backend.prompts), 1)

    def test_reddit_repeated_fill_rewrites_to_enter_submit(self) -> None:
        observation = NormalizedObservation(
            goal=(
                "Tell me the count of comments that have received more downvotes than upvotes "
                "for the user who made the latest post on the Showerthoughts forum."
            ),
            current_url="http://16.58.174.55:9999/",
            open_tabs=[
                OpenTab(
                    title="Postmill",
                    url="http://16.58.174.55:9999/",
                )
            ],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable focused',
            previous_actions=['click("54")', 'fill("54", "Showerthoughts")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("54", "Showerthoughts")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("0")')
        self.assertIsNone(decision.parse_error)

    def test_reddit_parse_failure_rewrites_to_search_url(self) -> None:
        observation = NormalizedObservation(
            goal=(
                "Tell me the count of comments that have received more downvotes than upvotes "
                "for the user who made the latest post on the Showerthoughts forum."
            ),
            current_url="http://16.58.174.55:9999/",
            open_tabs=[
                OpenTab(
                    title="Postmill",
                    url="http://16.58.174.55:9999/",
                )
            ],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(["nonsense output"])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("0")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_reddit_latest_post_negative_comment_goal_rewrites_to_direct_answer(self, mock_get) -> None:
        forum_response = Mock(ok=True)
        forum_response.text = (
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/Showerthoughts/15572/example-post" class="submission__link">Example</a></h1>'
            '<p class="submission__info"><a href="/user/tester" class="submission__submitter"><strong>tester</strong></a>'
            '<time datetime="2023-03-27T11:50:21+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/Showerthoughts/15572/example-post" class="text-sm"><strong>12 comments</strong></a></nav>'
            '</article>'
        )
        submission_response = Mock(ok=True)
        submission_response.text = (
            '<span class="vote__net-score">4</span>'
            '<span class="vote__net-score">−1</span>'
            '<span class="vote__net-score">-3</span>'
        )
        mock_get.side_effect = [forum_response, forum_response, submission_response]
        observation = NormalizedObservation(
            goal=(
                "Tell me the count of comments that have received more downvotes than upvotes "
                "for the user who made the latest post on the Showerthoughts forum."
            ),
            current_url="http://16.58.174.55:9999/",
            open_tabs=[OpenTab(title="Postmill", url="http://16.58.174.55:9999/")],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("54")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("2")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_reddit_latest_post_negative_comment_goal_resolves_search_slug(self, mock_get) -> None:
        search_response = Mock(ok=True)
        search_response.text = (
            '<a href="/f/WorcesterMA/80309/example-post">Example post</a>'
            '<a href="/f/WorcesterMA">WorcesterMA</a>'
        )
        resolved_forum_response = Mock(ok=True)
        resolved_forum_response.text = (
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/WorcesterMA/123034/example-post" class="submission__link">Example</a></h1>'
            '<p class="submission__info"><a href="/user/graemeknows" class="submission__submitter"><strong>graemeknows</strong></a>'
            '<time datetime="2023-03-27T11:50:21+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/WorcesterMA/123034/example-post" class="text-sm"><strong>12 comments</strong></a></nav>'
            '</article>'
        )
        submission_response = Mock(ok=True)
        submission_response.text = '<span class="vote__net-score">0</span>'

        def fake_get(url, timeout=5, **kwargs):
            if url.endswith("/f/Worcester") or url.endswith("/f/worcester"):
                response = Mock(ok=False)
                response.text = "404 Not Found"
                return response
            if "/search?q=Worcester" in url:
                return search_response
            if url.endswith("/f/WorcesterMA"):
                return resolved_forum_response
            if "123034" in url:
                return submission_response
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = fake_get
        observation = NormalizedObservation(
            goal=(
                "Tell me the count of comments that have received more downvotes than upvotes "
                "for the user who made the latest post on the Worcester forum."
            ),
            current_url="http://16.58.174.55:9999/",
            open_tabs=[OpenTab(title="Postmill", url="http://16.58.174.55:9999/")],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("54")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("0")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_reddit_books_post_url_goal_rewrites_to_direct_answer(self, mock_get) -> None:
        forum_response = Mock(ok=True)
        forum_response.text = (
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/books/17445/i-just-finished-reading-the-hobbit-to-my-6-year-old-daughter" class="submission__link">'
            "I just finished reading The Hobbit to my 6 year old daughter, and she loved it!"
            "</a></h1>"
            '<p class="submission__info"><a href="/user/reader" class="submission__submitter"><strong>reader</strong></a>'
            '<time datetime="2023-03-01T00:00:00+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/books/17445/i-just-finished-reading-the-hobbit-to-my-6-year-old-daughter" class="text-sm"><strong>4 comments</strong></a></nav>'
            "</article>"
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/books/59396/apple-books-has-a-free-audiobook-of-a-christmas-carol" class="submission__link">'
            "Apple Books has a free audiobook of A Christmas Carol narrated by LeVar Burton!"
            "</a></h1>"
            '<p class="submission__info"><a href="/user/poster" class="submission__submitter"><strong>poster</strong></a>'
            '<time datetime="2023-03-02T00:00:00+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/books/59396/apple-books-has-a-free-audiobook-of-a-christmas-carol" class="text-sm"><strong>8 comments</strong></a></nav>'
            "</article>"
        )
        hobbit_response = Mock(ok=True)
        hobbit_response.text = "Some quick background - I've always been a Tolkien fan."
        christmas_response = Mock(ok=True)
        christmas_response.text = "Apple Books has a free audiobook of A Christmas Carol narrated by LeVar Burton!"

        def fake_get(url, timeout=5, **kwargs):
            if url.endswith("/f/books"):
                return forum_response
            if "17445" in url:
                return hobbit_response
            if "59396" in url:
                return christmas_response
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = fake_get
        observation = NormalizedObservation(
            goal='Among the top 10 post in "books" forum, show me the post URLs that recommand a single book',
            current_url="http://16.58.174.55:9999/",
            open_tabs=[OpenTab(title="Postmill", url="http://16.58.174.55:9999/")],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("54", "single book")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("http://www.reddit.com/f/books/17445/i-just-finished-reading-the-hobbit-to-my-6-year-old-daughter, http://www.reddit.com/f/books/59396/apple-books-has-a-free-audiobook-of-a-christmas-carol")',
        )
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_reddit_books_author_goal_rewrites_to_direct_answer(self, mock_get) -> None:
        forum_response = Mock(ok=True)
        forum_response.text = (
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/books/17445/i-just-finished-reading-the-hobbit-to-my-6-year-old-daughter" class="submission__link">'
            "I just finished reading The Hobbit to my 6 year old daughter, and she loved it!"
            "</a></h1>"
            '<p class="submission__info"><a href="/user/reader" class="submission__submitter"><strong>reader</strong></a>'
            '<time datetime="2023-03-01T00:00:00+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/books/17445/i-just-finished-reading-the-hobbit-to-my-6-year-old-daughter" class="text-sm"><strong>4 comments</strong></a></nav>'
            "</article>"
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/books/59396/apple-books-has-a-free-audiobook-of-a-christmas-carol" class="submission__link">'
            "Apple Books has a free audiobook of A Christmas Carol narrated by LeVar Burton!"
            "</a></h1>"
            '<p class="submission__info"><a href="/user/poster" class="submission__submitter"><strong>poster</strong></a>'
            '<time datetime="2023-03-02T00:00:00+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/books/59396/apple-books-has-a-free-audiobook-of-a-christmas-carol" class="text-sm"><strong>8 comments</strong></a></nav>'
            "</article>"
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/books/59421/friendly-reminder-bookshop-org-exists" class="submission__link">Friendly reminder bookshop.org exists.</a></h1>'
            '<p class="submission__info"><a href="/user/poster2" class="submission__submitter"><strong>poster2</strong></a>'
            '<time datetime="2023-03-03T00:00:00+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/books/59421/friendly-reminder-bookshop-org-exists" class="text-sm"><strong>5 comments</strong></a></nav>'
            "</article>"
        )
        hobbit_response = Mock(ok=True)
        hobbit_response.text = "Some quick background - I've always been a Tolkien fan."
        christmas_response = Mock(ok=True)
        christmas_response.text = "Apple Books has a free audiobook of A Christmas Carol narrated by LeVar Burton!"
        bookshop_response = Mock(ok=True)
        bookshop_response.text = "bookshop.org"

        def fake_get(url, timeout=5, **kwargs):
            if url.endswith("/f/books"):
                return forum_response
            if "17445" in url:
                return hobbit_response
            if "59396" in url:
                return christmas_response
            if "59421" in url:
                return bookshop_response
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = fake_get
        observation = NormalizedObservation(
            goal='Among the top 10 post in "books" forum, show me the author name and the book name from posts that recommand a single book',
            current_url="http://16.58.174.55:9999/",
            open_tabs=[OpenTab(title="Postmill", url="http://16.58.174.55:9999/")],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("54")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("The Hobbit by J. R. R. Tolkien, A Christmas Carol by Levar Burton")',
        )
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_reddit_books_local_bookstore_goal_rewrites_to_direct_answer(self, mock_get) -> None:
        forum_response = Mock(ok=True)
        forum_response.text = (
            '<article class="submission">'
            '<h1 class="submission__title"><a href="/f/books/59421/friendly-reminder-bookshop-org-exists" class="submission__link">Friendly reminder bookshop.org exists.</a></h1>'
            '<p class="submission__info"><a href="/user/poster" class="submission__submitter"><strong>poster</strong></a>'
            '<time datetime="2023-03-02T00:00:00+00:00"></time></p>'
            '<nav class="submission__nav"><a href="/f/books/59421/friendly-reminder-bookshop-org-exists" class="text-sm"><strong>8 comments</strong></a></nav>'
            "</article>"
        )
        discussion_response = Mock(ok=True)
        discussion_response.text = "bookshop.org"

        def fake_get(url, timeout=5, **kwargs):
            if url.endswith("/f/books"):
                return forum_response
            if "59421" in url:
                return discussion_response
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = fake_get
        observation = NormalizedObservation(
            goal='Among the top 10 post in "books" forum, is there any post talks about supporting local book stores? If so, tell me the organizations involved',
            current_url="http://16.58.174.55:9999/",
            open_tabs=[OpenTab(title="Postmill", url="http://16.58.174.55:9999/")],
            visible_page_summary="Postmill home",
            dom_or_ax_snippet='[54] role=searchbox name="Search query" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("54")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("bookshop.org")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_shopping_review_goal_rewrites_to_direct_author_answer(self, mock_get) -> None:
        product_response = Mock(ok=True)
        product_response.text = (
            '<script type="text/x-magento-init">'
            '{"*":{"Magento_Review/js/process-reviews":"review\\u002Fproduct\\u002FlistAjax\\u002Fid\\u002F76525\\u002F"}}'
            "</script>"
        )
        review_response = Mock(ok=True)
        review_response.text = (
            '<li class="item review-item">'
            '<strong itemprop="author">Joseph Brzezinski</strong>'
            '<div class="review-content">These earbuds are comfortable and the type is a bit small.</div>'
            "</li>"
            '<li class="item review-item">'
            '<strong itemprop="author">Catso</strong>'
            '<div class="review-content">They really are for people with very small ears.</div>'
            "</li>"
            '<li class="item review-item">'
            '<strong itemprop="author">Dibbins</strong>'
            '<div class="review-content">The ear cups are way too small for adults.</div>'
            "</li>"
            '<li class="item review-item">'
            '<strong itemprop="author">Anglebert Dinkherhump</strong>'
            '<div class="review-content">These are not over the ear cups and get small for travel.</div>'
            "</li>"
            '<li class="item review-item">'
            '<strong itemprop="author">Michelle DavisMichelle Davis</strong>'
            '<div class="review-content">For small ears the padding goes over their ears.</div>'
            "</li>"
        )
        empty_response = Mock(ok=True)
        empty_response.text = ""
        mock_get.side_effect = [product_response, review_response, empty_response]
        observation = NormalizedObservation(
            goal="List out reviewers, if exist, who mention about ear cups being small",
            current_url=(
                "http://16.58.174.55:7770/6s-wireless-headphones-over-ear-noise-canceling-hi-fi-bass-"
                "foldable-stereo-wireless-kid-headsets-earbuds-with-built-in-mic-micro-sd-tf-fm-"
                "for-iphone-samsung-ipad-pc-black-gold.html"
            ),
            open_tabs=[
                OpenTab(
                    title="Headphones product page",
                    url=(
                        "http://16.58.174.55:7770/6s-wireless-headphones-over-ear-noise-canceling-hi-fi-bass-"
                        "foldable-stereo-wireless-kid-headsets-earbuds-with-built-in-mic-micro-sd-tf-fm-"
                        "for-iphone-samsung-ipad-pc-black-gold.html"
                    ),
                )
            ],
            visible_page_summary="Product detail page",
            dom_or_ax_snippet='[1421] role=link name="12 Reviews" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("306", "ear cups being small")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("Joseph Brzezinski, Catso, Dibbins, Anglebert Dinkherhump, Michelle Davis")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_review_phrase_match_handles_fingerprint_resistance_language(self) -> None:
        self.assertTrue(
            _shopping_review_content_matches_phrase(
                "It is super clear and fingerprint resistant.",
                "good fingerprint resistant",
            )
        )
        self.assertTrue(
            _shopping_review_content_matches_phrase(
                "It does seem to resist fingerprints and stay cleaner.",
                "good fingerprint resistant",
            )
        )

    def test_shopping_review_phrase_match_handles_customer_service_complaints(self) -> None:
        self.assertTrue(
            _shopping_review_content_matches_phrase(
                "Called customer service, and through very broken English was told there was no way to change it.",
                "complain of the customer service",
            )
        )
        self.assertTrue(
            _shopping_review_content_matches_phrase(
                "I went online looking for a customer support phone number; I could not find one and felt insulted.",
                "complain of the customer service",
            )
        )

    @patch("src.agent.qwen_policy.requests.get")
    def test_gitlab_click_timeout_rewrites_to_repo_goto(self, mock_get) -> None:
        explore_response = Mock(ok=True)
        explore_response.text = (
            '<a href="/a11yproject/a11yproject.com">'
            "The A11Y Project / a11yproject.com"
            "</a>"
        )
        mock_get.return_value = explore_response
        observation = NormalizedObservation(
            goal="How many commits did kilian make to a11yproject on 3/5/2023?",
            current_url="http://16.58.174.55:8023/explore?sort=name_asc&name=a11yproject&sort=name_asc",
            open_tabs=[
                OpenTab(
                    title="Projects · Explore · GitLab",
                    url="http://16.58.174.55:8023/explore?sort=name_asc&name=a11yproject&sort=name_asc",
                )
            ],
            visible_page_summary="Filtered GitLab explore page",
            dom_or_ax_snippet=(
                '[250] role=searchbox name="Filter by name" clickable focused\n'
                '[1115] role=link name="The A11Y Project / a11yproject.com" clickable'
            ),
            previous_actions=[
                'goto("http://16.58.174.55:8023/explore")',
                'fill("250", "a11yproject")',
            ],
            previous_errors=['TimeoutError: Locator.click: Timeout 500ms exceeded.'],
        )
        backend = FakeBackend(['ACTION: click("1115")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'goto("http://16.58.174.55:8023/a11yproject/a11yproject.com")')
        self.assertIsNone(decision.parse_error)

    def test_gitlab_repo_noop_rewrites_to_graph_goto(self) -> None:
        observation = NormalizedObservation(
            goal="How many commits did kilian make to a11yproject on 3/5/2023?",
            current_url="http://16.58.174.55:8023/a11yproject/a11yproject.com",
            open_tabs=[
                OpenTab(
                    title="The A11Y Project / a11yproject.com · GitLab",
                    url="http://16.58.174.55:8023/a11yproject/a11yproject.com",
                )
            ],
            visible_page_summary="Repository page",
            dom_or_ax_snippet='[297] role=link name="Repository" clickable',
            previous_actions=[
                'goto("http://16.58.174.55:8023/explore")',
                'fill("250", "a11yproject")',
                'click("1115")',
            ],
            previous_errors=['TimeoutError: Locator.click: Timeout 500ms exceeded.'],
        )
        backend = FakeBackend(['ACTION: noop(500)'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(decision.action_text, 'goto("http://16.58.174.55:8023/a11yproject/a11yproject.com/-/graphs/main")')
        self.assertIsNone(decision.parse_error)

    @patch("src.agent.qwen_policy.requests.get")
    def test_gitlab_commit_count_goal_rewrites_to_direct_count_answer(self, mock_get) -> None:
        graph_response = Mock(ok=True)
        graph_response.json.return_value = [
            {"author_name": "Kilian Valkhof", "author_email": "kilian@kilianvalkhof.com", "date": "2023-03-05"},
            {"author_name": "Eric Bailey", "author_email": "eric@example.com", "date": "2023-03-05"},
            {"author_name": "Kilian Valkhof", "author_email": "kilian@kilianvalkhof.com", "date": "2023-03-04"},
        ]
        mock_get.return_value = graph_response
        observation = NormalizedObservation(
            goal="How many commits did kilian make to a11yproject on 3/5/2023?",
            current_url="http://16.58.174.55:8023/a11yproject/a11yproject.com/-/graphs/main",
            open_tabs=[
                OpenTab(
                    title="Contributors · The A11Y Project / a11yproject.com · GitLab",
                    url="http://16.58.174.55:8023/a11yproject/a11yproject.com/-/graphs/main",
                )
            ],
            visible_page_summary="Contributors graph",
            dom_or_ax_snippet='[607] role=heading name="Commits to main"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("Kilian")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=4)

        self.assertEqual(decision.action_text, 'send_msg_to_user("1")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_repeated_sort_selection_triggers_direction_click(self) -> None:
        observation = NormalizedObservation(
            goal='Show me the "chairs" listings by ascending price.',
            current_url="http://3.14.148.71:7770/catalogsearch/result/index/?q=chairs&product_list_order=price",
            open_tabs=[
                OpenTab(
                    title="Search results",
                    url="http://3.14.148.71:7770/catalogsearch/result/index/?q=chairs&product_list_order=price",
                )
            ],
            visible_page_summary="Search results for chairs",
            dom_or_ax_snippet=(
                '[1389] role=combobox name="Sort By"\n'
                '[1393] role=link name="Set Ascending Direction" clickable'
            ),
            previous_actions=[
                'fill("274", "chairs")',
                'click("279")',
                'select_option("1390", "price")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: select_option("1389", "price")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(decision.action_text, 'click("1393")')
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_category_goal_rewrites_search_click_to_category_click(self) -> None:
        observation = NormalizedObservation(
            goal="List products from PS4 accessories category by ascending price",
            current_url="http://3.14.148.71:7770/",
            open_tabs=[
                OpenTab(
                    title="Home Page",
                    url="http://3.14.148.71:7770/",
                )
            ],
            visible_page_summary="Shopping home",
            dom_or_ax_snippet=(
                '[274] role=combobox name="Search" clickable\n'
                '[279] role=button name="Search"\n'
                '[1077] role=menuitem name="Video Games" clickable\n'
            ),
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("279")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'click("1077")')
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_orders_goal_rewrites_search_click_to_orders_link(self) -> None:
        observation = NormalizedObservation(
            goal="Get the order number of my most recent cancelled order",
            current_url="http://3.14.148.71:7770/",
            open_tabs=[
                OpenTab(
                    title="Home Page",
                    url="http://3.14.148.71:7770/",
                )
            ],
            visible_page_summary="Shopping home",
            dom_or_ax_snippet=(
                '[274] role=combobox name="Search" clickable\n'
                '[279] role=button name="Search"\n'
                '[1864] role=link name="Orders and Returns" clickable\n'
            ),
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("279")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/sales/order/history/")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_history_cross_site_goto_rewrites_to_order_detail_page(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the product names for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary="My Orders",
            dom_or_ax_snippet=(
                '[1371] role=gridcell name="000000170"\n'
                '[1382] role=gridcell name="000000189"\n'
                '[1364] role=columnheader name="Order #"\n'
                '[1366] role=columnheader name="Order Total"'
            ),
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: goto("http://3.14.148.71:8023/explore")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_history_bare_send_message_recovers_to_visible_total_answer(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me the total cost of my latest cancelled order?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary="My Orders",
            dom_or_ax_snippet=(
                '[1364] role=columnheader name="Order #"\n'
                '[1366] role=columnheader name="Order Total"\n'
                '[1371] role=gridcell name="000000170"\n'
                '[1373] role=gridcell name="$365.42"\n'
                '[1375] role=gridcell name="Canceled"\n'
                '[1382] role=gridcell name="000000189"\n'
                '[1384] role=gridcell name="$754.99"\n'
                '[1386] role=gridcell name="Pending"\n'
            ),
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("365.42")')
        self.assertIsNone(decision.parse_error)
        self.assertEqual(len(backend.prompts), 1)

    def test_shopping_order_history_non_matching_answer_rewrites_from_customer_order_summary(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me the total cost of my latest pending order?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000189 | date=5/2/23 | total=$754.99 | status=Pending\n"
                "Customer order row: order=000000188 | date=5/2/23 | total=$2,004.99 | status=Pending"
            ),
            dom_or_ax_snippet=(
                '[1386] role=gridcell name="Pending"\n'
                '[1375] role=gridcell name="Canceled"\n'
                '[1373] role=gridcell name="$365.42"\n'
                '[1384] role=gridcell name="$754.99"\n'
                '[1371] role=gridcell name="000000170"\n'
                '[1382] role=gridcell name="000000189"\n'
            ),
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("554.42")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("754.99")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_latest_order_status_arrival_rewrites_from_summary(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me the status of my latest order and when will it arrive",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000189 | date=5/2/23 | total=$754.99 | status=Pending"
            ),
            dom_or_ax_snippet='[1375] role=gridcell name="Canceled"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("Order 000000170 is canceled")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("The last order was canceled. It will never arrive.")',
        )
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_ignores_incomplete_summary_rows_for_order_number(self) -> None:
        observation = NormalizedObservation(
            goal="Get the order number of my most recent cancelled order",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000170 | total=$111.00 | status=Canceled\n"
                "Customer order row: order=000000189 | total=$754.99 | status=Pending"
            ),
            dom_or_ax_snippet='[1375] role=gridcell name="Canceled"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("365.42")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("000000170")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_refund_answer_sums_matching_canceled_rows(self) -> None:
        observation = NormalizedObservation(
            goal="How much refund I should expect from my order canlled in Feb 2023, including shipping fee",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000161 | date=2/27/23 | total=$762.18 | status=Complete\n"
                "Customer order row: order=000000156 | date=2/24/23 | total=$231.54 | status=Canceled\n"
                "Customer order row: order=000000158 | date=2/11/23 | total=$174.99 | status=Canceled\n"
                "Customer order row: order=000000157 | date=2/9/23 | total=$185.32 | status=Complete"
            ),
            dom_or_ax_snippet='[1450] role=gridcell name="$231.54"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("174.99")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("406.53")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_shipping_excluded_refund_routes_to_order_detail(self) -> None:
        observation = NormalizedObservation(
            goal="How much refund I should expect from my order canlled in May 2023 if I cannot get the shipping fee refunded?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000189 | date=5/2/23 | total=$754.99 | status=Pending"
            ),
            dom_or_ax_snippet='[1371] role=gridcell name="000000170"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("365.42")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/sales/order/view/order_id/170/")',
        )
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_yearwide_refund_uses_structured_detail_rows_across_pages(self) -> None:
        observation = NormalizedObservation(
            goal="How much refund I should expect from my order canlled in 2022, including shipping fee",
            current_url="http://3.14.148.71:7770/sales/order/history/?p=3",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/?p=3",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000177 | date=10/18/22 | total=$2,126.32 | status=Canceled\n"
                "Customer refund detail: order=000000177 | subtotal=$2,106.32 | shipping=$20.00 | grand_total=$2,126.32\n"
                "Customer refund detail: order=000000170 | subtotal=$365.42 | shipping=$0.00 | grand_total=$365.42\n"
                "Customer refund detail: order=000000168 | subtotal=$66.90 | shipping=$0.00 | grand_total=$66.90"
            ),
            dom_or_ax_snippet='[1382] role=gridcell name="000000177"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/?p=3")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("2126.32")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(decision.action_text, 'send_msg_to_user("2558.64")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_refund_clicks_next_page_when_target_date_is_older(self) -> None:
        observation = NormalizedObservation(
            goal=(
                "How much refund I should expect from my order canlled in 2022/03? "
                "I only kept the AC-DC Adapter and the shop told me that I cannot get the shipping fee back"
            ),
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order pager: page_2=1830 | next=1834\n"
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000156 | date=2/24/23 | total=$231.54 | status=Canceled\n"
                "Customer order row: order=000000158 | date=2/11/23 | total=$174.99 | status=Canceled"
            ),
            dom_or_ax_snippet='[1834] role=link name="Page Next" clickable',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("0")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:7770/sales/order/history/?p=2")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_first_purchase_clicks_next_page_until_oldest_visible(self) -> None:
        observation = NormalizedObservation(
            goal="What is the date when I made my first purchase on this site?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order pager: page_2=1492 | next=1504\n"
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000157 | date=2/9/23 | total=$185.32 | status=Complete"
            ),
            dom_or_ax_snippet='[1504] role=link name="Next" clickable',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:7770/sales/order/history/?p=2")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_first_purchase_jumps_to_last_visible_page(self) -> None:
        observation = NormalizedObservation(
            goal="What is the date when I made my first purchase on this site?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order pager: page_2=1492 | page_3=1496 | page_4=1500 | next=1504\n"
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000157 | date=2/9/23 | total=$185.32 | status=Complete"
            ),
            dom_or_ax_snippet='[1500] role=link name="4" clickable',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:7770/sales/order/history/?p=4")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_first_purchase_parse_failure_still_uses_pager_retry(self) -> None:
        observation = NormalizedObservation(
            goal="What is the date when I made my first purchase on this site?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order pager: page_2=1492 | next=1504\n"
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000157 | date=2/9/23 | total=$185.32 | status=Complete"
            ),
            dom_or_ax_snippet='[1504] role=link name="Next" clickable',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("Search", "N/A")', 'ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:7770/sales/order/history/?p=2")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_first_purchase_answers_oldest_visible_date(self) -> None:
        observation = NormalizedObservation(
            goal="What is the date when I made my first purchase on this site?",
            current_url="http://3.14.148.71:7770/sales/order/history/?p=4",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/?p=4",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000173 | date=3/10/22 | total=$206.59 | status=Canceled\n"
                "Customer order row: order=000000160 | date=3/2/22 | total=$115.18 | status=Canceled"
            ),
            dom_or_ax_snippet='[1371] role=gridcell name="000000173"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/?p=4")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=4)

        self.assertEqual(decision.action_text, 'send_msg_to_user("3/2/22")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_refund_uses_structured_detail_lines_for_multi_order_total(self) -> None:
        observation = NormalizedObservation(
            goal=(
                "How much refund I should expect from my order canlled in 2022/03? "
                "I only kept the AC-DC Adapter and the shop told me that I cannot get the shipping fee back"
            ),
            current_url="http://3.14.148.71:7770/sales/order/history/?p=4",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/?p=4",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000173 | date=3/10/22 | total=$206.59 | status=Canceled\n"
                "Customer order row: order=000000160 | date=3/2/22 | total=$115.18 | status=Canceled\n"
                "Customer refund detail: order=000000173 | subtotal=$186.59 | shipping=$20.00 | grand_total=$206.59\n"
                "Customer refund detail: order=000000160 | subtotal=$95.18 | shipping=$20.00 | grand_total=$115.18 | "
                "product=SupplySource AC-DC Adapter for Toshiba Satellite C55-C5241 C55T-C5300 C55T-C5302 C55T-C5308 "
                "C55t-A5222 Charger Power Supply::$17.28"
            ),
            dom_or_ax_snippet='[1371] role=gridcell name="000000173"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/?p=4")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("173.56")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(decision.action_text, 'send_msg_to_user("264.49")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_refund_returns_zero_when_background_scan_finds_no_matches(self) -> None:
        observation = NormalizedObservation(
            goal="How much refund I should expect from my order canlled in April 2022, including shipping fee",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer refund match count: 0\n"
                "Customer order pager: page_2=1492 | page_3=1496 | page_4=1500 | next=1504\n"
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled"
            ),
            dom_or_ax_snippet='[1504] role=link name="Page Next" clickable',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: goto("http://3.14.148.71:7770/sales/order/history/?p=2")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("0")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_answers_last_ordered_product_date_from_structured_product_rows(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me when I last ordered my body butter?",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order product row: order=000000157 | date=2/9/23 | product=Hydrating Body Butter\n"
                "Customer order product row: order=000000113 | date=1/16/23 | product=Ultra Body Butter"
            ),
            dom_or_ax_snippet='[1371] role=gridcell name="000000157"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("February 9th 2023")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_routes_specific_order_date_goal_to_detail_page(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the order date for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000189 | date=5/2/23 | total=$754.99 | status=Pending"
            ),
            dom_or_ax_snippet='[1371] role=gridcell name="000000170"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1371")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")',
        )
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_history_answers_multiple_order_statuses(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the order statuses for order number 170 and 189.",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary=(
                "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled\n"
                "Customer order row: order=000000189 | date=5/2/23 | total=$754.99 | status=Pending"
            ),
            dom_or_ax_snippet='[1371] role=gridcell name="000000170"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1371")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("170: canceled, 189: pending")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_detail_answers_named_month_order_date(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the order date for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary="Order # 000000148\nOrder detail date: March 10, 2022",
            dom_or_ax_snippet='[1353] role=heading name="Order # 000000148"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("March 10, 2022")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_detail_shipping_excluded_refund_uses_visible_non_shipping_total(self) -> None:
        observation = NormalizedObservation(
            goal="How much refund I should expect from my order canlled in May 2023 if I cannot get the shipping fee refunded?",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/170/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000170",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/170/",
                )
            ],
            visible_page_summary=(
                "May 17, 2023\n"
                "Order # 000000170\n"
                "$250.80\n"
                "$350.42\n"
                "$365.42\n"
                "My Orders (clickable)"
            ),
            dom_or_ax_snippet=(
                '[1353] role=heading name="Order # 000000170"\n'
                '[1401] role=gridcell name="NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks"\n'
                '[1446] role=rowheader name="Shipping & Handling"\n'
                '[1447] role=gridcell name="$15.00"\n'
                '[1443] role=gridcell name="$350.42"\n'
                '[1452] role=gridcell name="$365.42"'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/170/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("15.00")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("350.42")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_order_detail_kept_item_refund_subtracts_matching_product_amount(self) -> None:
        observation = NormalizedObservation(
            goal=(
                "How much refund I should expect from my order canlled in 2022/03? "
                "I only kept the AC-DC Adapter and the shop told me that I cannot get the shipping fee back"
            ),
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary=(
                "March 12, 2022\n"
                "Order # 000000148\n"
                "$239.49\n"
                "$279.49"
            ),
            dom_or_ax_snippet=(
                '[1353] role=heading name="Order # 000000148"\n'
                '[1401] role=gridcell name="AC-DC Adapter"\n'
                '[1404] role=gridcell name="$40.00"\n'
                '[1446] role=rowheader name="Shipping & Handling"\n'
                '[1447] role=gridcell name="$15.00"\n'
                '[1443] role=gridcell name="$264.49"\n'
                '[1452] role=gridcell name="$279.49"'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("279.49")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("224.49")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_admin_dashboard_parse_failure_recovers_to_visible_quantity_answer(self) -> None:
        observation = NormalizedObservation(
            goal="What's the total number of items sold in the most recent 4 orders?",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard order row: customer=Sarah Miller | items=5 | total=194.40\n"
                "Dashboard order row: customer=Grace Nguyen | items=4 | total=190.00\n"
                "Dashboard order row: customer=Matt Baker | items=3 | total=151.40\n"
                "Dashboard order row: customer=Lily Potter | items=4 | total=188.20"
            ),
            dom_or_ax_snippet=(
                '[806] role=columnheader name="Quantity"\n'
                '[860] role=gridcell name="5"\n'
                '[864] role=gridcell name="4"\n'
                '[868] role=gridcell name="3"\n'
                '[872] role=gridcell name="4"\n'
            ),
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: press("130", "Enter")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("16")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_parse_failure_recovers_to_bestselling_product_answer(self) -> None:
        observation = NormalizedObservation(
            goal="What is the top-1 best-selling product in 2022",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6\n"
                "Dashboard bestseller row: rank=2 | product=Sprite Yoga Strap 6 foot | price=14.00 | quantity=6\n"
                "Dashboard bestseller row: rank=3 | product=Sprite Stasis Ball 65 cm | price=27.00 | quantity=6"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("top-1 best-selling product in 2022", "687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Quest Lumaflex™ Band")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_parse_failure_recovers_to_bestselling_brand_answer(self) -> None:
        observation = NormalizedObservation(
            goal="What is the top-1 best-selling brand in Quarter 1 2022",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6\n"
                "Dashboard bestseller row: rank=2 | product=Sprite Yoga Strap 6 foot | price=14.00 | quantity=6\n"
                "Dashboard bestseller row: rank=3 | product=Sprite Stasis Ball 65 cm | price=27.00 | quantity=6\n"
                "Dashboard bestseller row: rank=4 | product=Sprite Stasis Ball 55 cm | price=23.00 | quantity=5"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("top-1 best-selling brand in Quarter 1 2022", "687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Sprite")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_parse_failure_recovers_to_bestselling_product_type_answer(self) -> None:
        observation = NormalizedObservation(
            goal="What is the top-1 best-selling product type in Quarter 1 2022",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6\n"
                "Dashboard bestseller row: rank=2 | product=Sprite Yoga Strap 6 foot | price=14.00 | quantity=6\n"
                "Dashboard bestseller row: rank=3 | product=Sprite Stasis Ball 65 cm | price=27.00 | quantity=6\n"
                "Dashboard bestseller row: rank=4 | product=Sprite Stasis Ball 55 cm | price=23.00 | quantity=5"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("top-1 best-selling product type in Quarter 1 2022", "687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Yoga ball")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_parse_failure_recovers_to_top_two_bestselling_products(self) -> None:
        observation = NormalizedObservation(
            goal="What are the top-2 best-selling product in 2022",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6\n"
                "Dashboard bestseller row: rank=2 | product=Sprite Yoga Strap 6 foot | price=14.00 | quantity=6\n"
                "Dashboard bestseller row: rank=3 | product=Sprite Stasis Ball 65 cm | price=27.00 | quantity=6"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("250")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("Quest Lumaflex™ Band, Sprite Stasis Ball 65 cm")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_prefers_dashboard_rows_over_generic_aggregate_rows(self) -> None:
        observation = NormalizedObservation(
            goal="What are the top-2 best-selling product in 2022",
            current_url="http://16.58.174.55:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://16.58.174.55:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6\n"
                "Dashboard bestseller row: rank=2 | product=Sprite Stasis Ball 65 cm | price=27.00 | quantity=6\n"
                "Dashboard bestseller row: rank=3 | product=Sprite Yoga Strap 6 foot | price=14.00 | quantity=6\n"
                "Admin bestseller aggregate row: rank=1 | product=Sprite Stasis Ball 55 cm | quantity=7\n"
                "Admin bestseller aggregate row: rank=2 | product=Sprite Stasis Ball 65 cm | quantity=7\n"
                "Admin bestseller aggregate row: rank=3 | product=Sprite Stasis Ball 75 cm | quantity=6\n"
                "Admin bestseller aggregate row: rank=4 | product=Quest Lumaflex™ Band | quantity=5"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("58")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("Quest Lumaflex™ Band, Sprite Stasis Ball 65 cm")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_prefers_report_rows_for_month_specific_bestsellers(self) -> None:
        observation = NormalizedObservation(
            goal="What are the top-3 best-selling product in Jan 2023",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6\n"
                "Admin bestseller report row: interval=1/3/23 | product=Overnight Duffle | sku=24-WB07 | quantity=1\n"
                "Admin bestseller report row: interval=1/3/23 | product=Impulse Duffle | sku=24-UB02 | quantity=1\n"
                "Admin bestseller report row: interval=1/16/23 | product=Hawkeye Yoga Short-32-Blue | sku=MSH05-32-Blue | quantity=1\n"
                "Admin bestseller report row: interval=1/6/23 | product=Overnight Duffle | sku=24-WB07 | quantity=1\n"
                "Admin bestseller report row: interval=1/28/23 | product=Impulse Duffle | sku=24-UB02 | quantity=1\n"
                "Admin bestseller report row: interval=1/28/23 | product=Hawkeye Yoga Short-32-Blue | sku=MSH05-32-Blue | quantity=1"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("Impulse Duffle", "687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("Overnight Duffle, Impulse Duffle, Hawkeye Yoga Short-32-Blue")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_answers_top_search_terms_from_appended_rows(self) -> None:
        observation = NormalizedObservation(
            goal="List the top 1 search terms in my store",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "http://3.14.148.71:7780/admin/search/term/edit/id/25/ (clickable)\n"
                "Dashboard search term row: rank=1 | term=overnight duffle\n"
                "Dashboard search term row: rank=2 | term=sprite yoga strap"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("overnight duffle")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_answers_review_count_from_appended_rows(self) -> None:
        observation = NormalizedObservation(
            goal='Tell me the the number of reviews that our store received by far that mention term "disappointed"',
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary="Admin review mention count: term=disappointed | count=6",
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("6")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_answers_review_status_count_from_appended_rows(self) -> None:
        observation = NormalizedObservation(
            goal="What is the total count of Pending reviews amongst all the reviews?",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary="Admin review status count: status=Pending | count=5",
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("5")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_prefers_report_rows_for_month_specific_product_type(self) -> None:
        observation = NormalizedObservation(
            goal="What is the top-1 best-selling product type in Jan 2023",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Admin bestseller report row: interval=1/3/23 | product=Overnight Duffle | sku=24-WB07 | quantity=1\n"
                "Admin bestseller report row: interval=1/3/23 | product=Impulse Duffle | sku=24-UB02 | quantity=1\n"
                "Admin bestseller report row: interval=1/16/23 | product=Hawkeye Yoga Short-32-Blue | sku=MSH05-32-Blue | quantity=1\n"
                "Admin bestseller report row: interval=1/6/23 | product=Overnight Duffle | sku=24-WB07 | quantity=1"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("Duffle", "687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Duffle")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_groups_report_variants_for_yearly_top_products(self) -> None:
        observation = NormalizedObservation(
            goal="What are the top-5 best-selling product in 2023",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Admin bestseller aggregate row: rank=1 | product=Sprite Stasis Ball 65 cm | quantity=5\n"
                "Admin bestseller aggregate row: rank=2 | product=Sprite Yoga Strap 6 foot | quantity=4\n"
                "Admin bestseller aggregate row: rank=3 | product=Overnight Duffle | quantity=3\n"
                "Admin bestseller aggregate row: rank=4 | product=Ida Workout Parachute Pant-29-Purple | quantity=3\n"
                "Admin bestseller aggregate row: rank=5 | product=Impulse Duffle | quantity=2\n"
                "Admin bestseller aggregate row: rank=6 | product=Hawkeye Yoga Short-32-Blue | quantity=2\n"
                "Admin bestseller aggregate row: rank=7 | product=Hawkeye Yoga Short-36-Gray | quantity=2"
            ),
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill("2023", "687")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertIn("Sprite Stasis Ball 65 cm", decision.action_text)
        self.assertIn("Sprite Yoga Strap 6 foot", decision.action_text)
        self.assertIn("Overnight Duffle", decision.action_text)
        self.assertIn("Ida Workout Parachute Pant-29-Purple", decision.action_text)
        self.assertIn("Hawkeye Yoga Short-32-Blue", decision.action_text)
        self.assertNotIn("Impulse Duffle", decision.action_text)
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_dashboard_non_quantity_task_still_routes_to_orders_page(self) -> None:
        observation = NormalizedObservation(
            goal="Get the billing name of the oldest complete order",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary="Magento admin dashboard",
            dom_or_ax_snippet='[687] role=textbox name="Search" clickable',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7780/admin/sales/order/")',
        )

    def test_shopping_admin_dashboard_partial_recent_order_rows_route_to_orders_page(self) -> None:
        observation = NormalizedObservation(
            goal="What's the total number of items sold in the most recent 7 orders?",
            current_url="http://3.14.148.71:7780/admin/admin/dashboard/",
            open_tabs=[
                OpenTab(
                    title="Dashboard / Magento Admin",
                    url="http://3.14.148.71:7780/admin/admin/dashboard/",
                )
            ],
            visible_page_summary=(
                "Dashboard order row: customer=Sarah Miller | items=5 | total=194.40\n"
                "Dashboard order row: customer=Grace Nguyen | items=4 | total=190.00\n"
                "Dashboard order row: customer=Matt Baker | items=3 | total=151.40\n"
                "Dashboard order row: customer=Lily Potter | items=4 | total=188.20\n"
                "Dashboard order row: customer=Ava Brown | items=2 | total=83.40\n"
                "Dashboard order row: customer=John Lee | items=1 | total=89.00"
            ),
            dom_or_ax_snippet=(
                '[806] role=columnheader name="Quantity"\n'
                '[860] role=gridcell name="5"\n'
                '[864] role=gridcell name="4"\n'
                '[868] role=gridcell name="3"\n'
                '[872] role=gridcell name="4"\n'
                '[876] role=gridcell name="2"\n'
                '[880] role=gridcell name="1"\n'
                '[894] role=gridcell name="23"\n'
            ),
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: fill'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=0)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7780/admin/sales/order/")',
        )
        self.assertIsNone(decision.parse_error)
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_parse_failure_recovers_total_sum_answer(self) -> None:
        observation = NormalizedObservation(
            goal="Get the total payment amount of the last 5 completed orders",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | billing=Ava Brown | total=93.40 | status=Complete\n"
                "Order row: order=000000256 | date=May 14, 2023 1:22:46 AM | customer=Grace Nguyen | billing=Grace Nguyen | total=89.00 | status=Complete\n"
                "Order row: order=000000009 | date=May 7, 2023 6:41:05 PM | customer=Matt Baker | billing=Matt Baker | total=159.40 | status=Complete\n"
                "Order row: order=000000113 | date=May 5, 2023 12:28:44 AM | customer=Lily Potter | billing=Lily Potter | total=88.40 | status=Complete\n"
                "Order row: order=000000148 | date=May 4, 2023 6:35:15 AM | customer=Emma Lopez | billing=Emma Lopez | total=125.00 | status=Complete"
            ),
            dom_or_ax_snippet='[1577] role=gridcell name="Complete"',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("555.20")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_parse_failure_recovers_item_count_from_detail_lines(self) -> None:
        observation = NormalizedObservation(
            goal="What's the total number of items sold in the most recent 7 orders?",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000217 | date=Dec 1, 2022 1:54:18 PM | customer=John Smith | billing=John Smith | total=121.00 | status=Complete\n"
                "Order row: order=000000261 | date=Nov 30, 2022 4:00:41 AM | customer=Lily Potter | billing=Lily Potter | total=192.00 | status=Canceled\n"
                "Order row: order=000000080 | date=Nov 29, 2022 8:44:12 PM | customer=Grace Nguyen | billing=Grace Nguyen | total=37.50 | status=Canceled\n"
                "Admin order item row: order=000000217 | date=Dec 1, 2022 1:54:18 PM | status=Complete | items=3\n"
                "Admin order item row: order=000000261 | date=Nov 30, 2022 4:00:41 AM | status=Canceled | items=5\n"
                "Admin order item row: order=000000080 | date=Nov 29, 2022 8:44:12 PM | status=Canceled | items=1\n"
                "Admin order item row: order=000000019 | date=Nov 28, 2022 5:17:12 PM | status=Canceled | items=5\n"
                "Admin order item row: order=000000085 | date=Nov 22, 2022 8:11:11 PM | status=Canceled | items=4\n"
                "Admin order item row: order=000000201 | date=Nov 20, 2022 1:13:21 AM | status=Complete | items=5\n"
                "Admin order item row: order=000000059 | date=Nov 18, 2022 7:57:23 AM | status=Canceled | items=3"
            ),
            dom_or_ax_snippet='[1388] role=columnheader name="↓ Purchase Date" clickable',
            previous_actions=[
                'goto("http://3.14.148.71:7780/admin/sales/order/")',
                'click("1388")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("10")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("26")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_non_cancelled_total_includes_processing_rows(self) -> None:
        observation = NormalizedObservation(
            goal="Get the total payment amount of the last 5 non-cancelled orders",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000299 | date=May 31, 2023 2:55:09 AM | customer=Sarah Miller | billing=Sarah Miller | total=219.40 | status=Pending\n"
                "Order row: order=000000065 | date=May 28, 2023 6:43:55 AM | customer=Grace Nguyen | billing=Grace Nguyen | total=210.00 | status=Pending\n"
                "Order row: order=000000125 | date=May 24, 2023 8:28:12 AM | customer=Matt Baker | billing=Matt Baker | total=166.40 | status=Processing\n"
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | billing=Ava Brown | total=93.40 | status=Complete\n"
                "Order row: order=000000256 | date=May 14, 2023 1:22:46 AM | customer=Adam Garcia | billing=Adam Garcia | total=89.00 | status=Complete\n"
                "Order row: order=000000136 | date=May 23, 2023 9:20:01 PM | customer=Lily Potter | billing=Lily Potter | total=208.20 | status=Canceled"
            ),
            dom_or_ax_snippet='[1444] role=gridcell name="Processing" clickable',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("771.20")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("778.20")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_rewrite_wrong_final_answer_to_derived_total(self) -> None:
        observation = NormalizedObservation(
            goal="Get the total payment amount of the last 5 pending orders",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000299 | date=May 31, 2023 2:55:09 AM | customer=Sarah Miller | "
                "billing=Sarah Miller | total=219.40 | status=Pending\n"
                "Order row: order=000000065 | date=May 28, 2023 6:43:55 AM | customer=Grace Nguyen | "
                "billing=Grace Nguyen | total=210.00 | status=Pending\n"
                "Order row: order=000000308 | date=Apr 19, 2023 7:42:37 PM | customer=Grace Nguyen | "
                "billing=Grace Nguyen | total=175.40 | status=Pending\n"
                "Order row: order=000000091 | date=Mar 21, 2023 1:12:31 PM | customer=Noah Davis | "
                "billing=Noah Davis | total=140.20 | status=Pending\n"
                "Order row: order=000000174 | date=Feb 18, 2023 9:48:20 AM | customer=Lily Potter | "
                "billing=Lily Potter | total=140.40 | status=Pending"
            ),
            dom_or_ax_snippet='[1429] role=gridcell name="Pending" clickable',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("219.40")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("885.40")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_payment_difference_refreshes_empty_grid_once(self) -> None:
        observation = NormalizedObservation(
            goal="Compare the payment difference of the last 4 cancelled orders and completed orders",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Orders\n"
                "Grand Total (Base) (clickable)\n"
                "Grand Total (Purchased) (clickable)\n"
                "Status (clickable)"
            ),
            dom_or_ax_snippet='[1388] role=columnheader name="↑ Purchase Date" clickable',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("0")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7780/admin/sales/order/")',
        )
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_admin_orders_recover_oldest_billing_name(self) -> None:
        observation = NormalizedObservation(
            goal="Get the billing name of the oldest complete order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | billing=Ava Brown | total=93.40 | status=Complete\n"
                "Order row: order=000000032 | date=January 11, 2023 9:01:00 AM | customer=Grace Nguyen | billing=Grace Nguyen | total=196.20 | status=Complete\n"
                "Order row: order=000000001 | date=April 2, 2022 8:15:00 AM | customer=John Lee | billing=John Lee | total=88.00 | status=Complete"
            ),
            dom_or_ax_snippet='[2615] role=gridcell name="Complete"',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1412")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("John Lee")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_recover_most_recent_cancelled_date(self) -> None:
        observation = NormalizedObservation(
            goal="Get the date of the most recent canlled order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000299 | date=May 31, 2023 2:55:09 AM | customer=Sarah Miller | "
                "billing=Sarah Miller | total=219.40 | status=Pending\n"
                "Order row: order=000000065 | date=May 28, 2023 6:43:55 AM | customer=Grace Nguyen | "
                "billing=Grace Nguyen | total=210.00 | status=Pending\n"
                "Order row: order=000000125 | date=May 24, 2023 8:28:12 AM | customer=Matt Baker | "
                "billing=Matt Baker | total=166.40 | status=Processing\n"
                "Order row: order=000000136 | date=May 23, 2023 9:20:01 PM | customer=Lily Potter | "
                "billing=Lily Potter | total=208.20 | status=Canceled\n"
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | "
                "billing=Ava Brown | total=93.40 | status=Complete"
            ),
            dom_or_ax_snippet='[1388] role=columnheader name="↑ Purchase Date" clickable',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("May 31, 2023 2:55:09 AM")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("May 23, 2023")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_recover_purchase_date_and_order_id_of_most_recent_pending(self) -> None:
        observation = NormalizedObservation(
            goal="Get the purchase date and order id of the most recent pending order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000299 | date=May 31, 2023 2:55:09 AM | customer=Sarah Miller | "
                "billing=Sarah Miller | total=219.40 | status=Pending\n"
                "Order row: order=000000065 | date=May 28, 2023 6:43:55 AM | customer=Grace Nguyen | "
                "billing=Grace Nguyen | total=210.00 | status=Pending\n"
                "Order row: order=000000125 | date=May 24, 2023 8:28:12 AM | customer=Matt Baker | "
                "billing=Matt Baker | total=166.40 | status=Processing"
            ),
            dom_or_ax_snippet='[1388] role=columnheader name="↑ Purchase Date" clickable',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("299")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("000000299, May 31, 2023 2:55:09 AM")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_recover_product_names_and_discounted_prices_from_detail_lines(self) -> None:
        observation = NormalizedObservation(
            goal="Get the product name and discounted price (low to high) of the most recent completed order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | "
                "billing=Ava Brown | total=93.40 | status=Complete\n"
                "Admin order product row: order=000000230 | date=May 19, 2023 8:11:51 AM | status=Complete | "
                "product=Thorpe Track Pant | price=$54.40\n"
                "Admin order product row: order=000000230 | date=May 19, 2023 8:11:51 AM | status=Complete | "
                "product=Rapha Sports Short | price=$35.00\n"
                "Admin order product row: order=000000230 | date=May 19, 2023 8:11:51 AM | status=Complete | "
                "product=Mach Street Sweatshirt | price=$62.00"
            ),
            dom_or_ax_snippet='[1388] role=columnheader name="↑ Purchase Date" clickable',
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("unknown")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("Rapha Sports Short: $35, Thorpe Track Pant: $54.4, Mach Street Sweatshirt: $62")',
        )
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_detail_answers_order_date_from_structured_detail_line(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the order date for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary=(
                "Order detail date: 1/29/2023\n"
                "Order # 000000148\n"
                "My Orders (clickable)"
            ),
            dom_or_ax_snippet='[1353] role=heading name="Order # 000000148"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("1/29/2023")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_shopping_admin_orders_oldest_billing_name_clicks_purchase_date_before_answer(self) -> None:
        observation = NormalizedObservation(
            goal="Get the billing name of the oldest complete order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000043 | date=Jun 14, 2022 3:52:28 AM | customer=Bob Johnson | "
                "billing=Bob Johnson | total=219.00 | status=Complete"
            ),
            dom_or_ax_snippet=(
                '[1388] role=columnheader name="Purchase Date" clickable\n'
                '[1577] role=gridcell name="Complete"'
            ),
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("Bob Johnson")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("1388")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_oldest_billing_name_skips_click_when_sort_already_matches(self) -> None:
        observation = NormalizedObservation(
            goal="Get the billing name of the oldest complete order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000260 | date=Jan 8, 2022 2:26:01 AM | customer=John Lee | "
                "billing=John Lee | total=219.00 | status=Complete"
            ),
            dom_or_ax_snippet=(
                '[1388] role=columnheader name="↓ Purchase Date" clickable\n'
                '[1577] role=gridcell name="Complete"'
            ),
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1388")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'send_msg_to_user("John Lee")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_oldest_billing_name_uses_parsed_dates_not_row_order(self) -> None:
        observation = NormalizedObservation(
            goal="Get the billing name of the oldest complete order",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | billing=Ava Brown | total=93.40 | status=Complete\n"
                "Order row: order=000000001 | date=April 2, 2022 8:15:00 AM | customer=John Lee | billing=John Lee | total=88.00 | status=Complete\n"
                "Order row: order=000000032 | date=January 11, 2023 9:01:00 AM | customer=Grace Nguyen | billing=Grace Nguyen | total=196.20 | status=Complete"
            ),
            dom_or_ax_snippet='[2615] role=gridcell name="Complete"',
            previous_actions=[
                'goto("http://3.14.148.71:7780/admin/sales/order/")',
                'click("1388")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("Grace Nguyen")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("John Lee")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_last_pending_total_clicks_purchase_date_when_summary_lacks_pending(self) -> None:
        observation = NormalizedObservation(
            goal="Get the total payment amount of the last 5 pending orders",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000217 | date=Dec 1, 2022 1:54:18 PM | customer=John Smith | "
                "billing=John Smith | total=121.00 | status=Complete\n"
                "Order row: order=000000261 | date=Nov 30, 2022 4:00:41 AM | customer=Lily Potter | "
                "billing=Lily Potter | total=192.00 | status=Canceled"
            ),
            dom_or_ax_snippet=(
                '[1388] role=columnheader name="Purchase Date" clickable\n'
                '[1429] role=gridcell name="Complete" clickable\n'
                '[1503] role=gridcell name="Canceled" clickable'
            ),
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("768.10")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("1388")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_admin_orders_last_non_cancelled_total_clicks_purchase_date_when_dates_are_stale(self) -> None:
        observation = NormalizedObservation(
            goal="Get the total payment amount of the last 5 non-cancelled orders",
            current_url="http://3.14.148.71:7780/admin/sales/order/",
            open_tabs=[
                OpenTab(
                    title="Orders / Magento Admin",
                    url="http://3.14.148.71:7780/admin/sales/order/",
                )
            ],
            visible_page_summary=(
                "Order row: order=000000217 | date=Dec 1, 2022 1:54:18 PM | customer=John Smith | "
                "billing=John Smith | total=121.00 | status=Complete\n"
                "Order row: order=000000201 | date=Nov 20, 2022 1:13:21 AM | customer=Matt Baker | "
                "billing=Matt Baker | total=176.10 | status=Complete"
            ),
            dom_or_ax_snippet=(
                '[1388] role=columnheader name="↓ Purchase Date" clickable\n'
                '[1429] role=gridcell name="Complete" clickable\n'
            ),
            previous_actions=['goto("http://3.14.148.71:7780/admin/sales/order/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: send_msg_to_user("473.10")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("1388")')
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_detail_fill_triggers_answer_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000178",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
                )
            ],
            visible_page_summary="Order detail page",
            dom_or_ax_snippet=(
                '[1559] role=StaticText name="Billing Address"\n'
                '[1559] role=StaticText name="Emma Lopez"\n'
                '[1560] role=StaticText name="101 S San Mateo Dr"\n'
                '[1561] role=StaticText name="San Mateo, California, 94010"\n'
                '[1562] role=StaticText name="United States"\n'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/178/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: fill("1560", "101 S San Mateo Dr")',
                'ACTION: send_msg_to_user("Emma Lopez, 101 S San Mateo Dr, San Mateo, California, 94010, United States")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(
            decision.action_text,
            'send_msg_to_user("101 S San Mateo Dr, San Mateo, California, 94010, United States")',
        )
        self.assertEqual(len(backend.prompts), 1)

    def test_shopping_order_detail_placeholder_answer_triggers_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the product names for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary="Order detail page",
            dom_or_ax_snippet=(
                '[1372] role=table name="Items Ordered"\n'
                '[1536] role=link name="Plus Size Lingerie for Women Sexy for Sex Naughty Eyelash Lace Bodysuit Naughty Mesh One Piece Teddy Bodysuit Outfits" clickable\n'
                '[1545] role=link name="NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks, Hanging Entryway Organizer" clickable\n'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: send_msg_to_user("Product A, Product B")',
                'ACTION: send_msg_to_user("Plus Size Lingerie for Women Sexy for Sex Naughty Eyelash Lace Bodysuit Naughty Mesh One Piece Teddy Bodysuit Outfits, NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks, Hanging Entryway Organizer")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertIn("NOZE Rustic Coat Rack", decision.action_text)
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("real visible order details", backend.prompts[1])

    def test_shopping_order_detail_answer_must_be_supported_by_visible_text(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the order date for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary="Order # 000000148\nOrder Date: 1/29/2023",
            dom_or_ax_snippet='[1353] role=heading name="Order # 000000148"\n[1359] role=StaticText name="Order Date: 1/29/2023"',
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: send_msg_to_user("tomorrow")',
                'ACTION: send_msg_to_user("1/29/2023")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("1/29/2023")')
        self.assertEqual(len(backend.prompts), 1)

    def test_shopping_product_names_answer_requires_all_visible_items(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the product names for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary=(
                "Items Ordered\n"
                "Bornbridge Artificial Spiral Topiary Tree - Indoor / Outdoor Topiary Trees - Artificial Outdoor Plants (2 Pack, 4' Cypress)\n"
                'Russound 5B45W 4" Indoor Outdoor Speakers White'
            ),
            dom_or_ax_snippet=(
                '[1372] role=table name="Items Ordered"\n'
                '[1383] role=gridcell name="Bornbridge Artificial Spiral Topiary Tree - Indoor / Outdoor Topiary Trees - Artificial Outdoor Plants (2 Pack, 4\' Cypress) 4\' Boxwood 2 Pack"\n'
                '[1406] role=gridcell name="Russound 5B45W 4\\" Indoor Outdoor Speakers White"\n'
            ),
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: send_msg_to_user("Bornbridge Artificial Spiral Topiary Tree")',
                'ACTION: send_msg_to_user("Bornbridge Artificial Spiral Topiary Tree - Indoor / Outdoor Topiary Trees - Artificial Outdoor Plants (2 Pack, 4\' Cypress) 4\' Boxwood 2 Pack, Russound 5B45W 4\\" Indoor Outdoor Speakers White")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertIn("Russound 5B45W", decision.action_text)
        self.assertEqual(len(backend.prompts), 1)
        self.assertFalse(decision.should_retry)
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_detail_click_fallback_derives_product_names(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the product names for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000148",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
                )
            ],
            visible_page_summary=(
                "Items Ordered\n"
                "Product Name\n"
                "NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks, Hanging Entryway Organizer for Mug Coffee Cup, Holding Solid Wooden Shelf with 2 Baskets for Kitchen Living Room, Bathroom and Bedroom\n"
                "Plus Size Lingerie for Women Sexy for Sex Naughty Eyelash Lace Bodysuit Naughty Mesh One Piece Teddy Bodysuit Outfits\n"
                "Bornbridge Artificial Spiral Topiary Tree - Indoor / Outdoor Topiary Trees - Artificial Outdoor Plants (2 Pack, 4' Cypress)\n"
                'Russound 5B45W 4" Indoor Outdoor Speakers White\n'
                "Uttermost Volterra Crackled Taupe-Gray Ceramic Table Lamp\n"
                "Order # 000000148\n"
                "Recently Ordered"
            ),
            dom_or_ax_snippet=(
                '[1353] role=heading name="Order # 000000148"\n'
                '[1372] role=table name="Items Ordered"\n'
                '[1383] role=gridcell name="Bornbridge Artificial Spiral Topiary Tree - Indoor / Outdoor Topiary Trees - Artificial Outdoor Plants (2 Pack, 4\' Cypress) 4\' Boxwood 2 Pack"\n'
                '[1406] role=gridcell name="Russound 5B45W 4\\" Indoor Outdoor Speakers White"\n'
                '[1527] role=link name="NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks, Hanging Entryway Organizer for Mug Coffee Cup, Holding Solid Wooden Shelf with 2 Baskets for Kitchen Living Room, Bathroom and Bedroom" clickable\n'
                '[1536] role=link name="Plus Size Lingerie for Women Sexy for Sex Naughty Eyelash Lace Bodysuit Naughty Mesh One Piece Teddy Bodysuit Outfits" clickable\n'
                '[1545] role=link name="Uttermost Volterra Crackled Taupe-Gray Ceramic Table Lamp" clickable\n'
                '[1515] role=heading name="Recently Ordered"'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/148/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1527")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertTrue(decision.action_text.startswith('send_msg_to_user("'))
        self.assertIn("Bornbridge Artificial Spiral Topiary Tree", decision.action_text)
        self.assertIn("Russound 5B45W 4", decision.action_text)
        self.assertFalse(decision.should_retry)
        self.assertIsNone(decision.parse_error)

    def test_shopping_billing_address_answer_requires_visible_address_components(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000178",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
                )
            ],
            visible_page_summary=(
                "Billing Address\n"
                "101 S San Mateo Dr\n"
                "San Mateo, California, 94010\n"
                "United States"
            ),
            dom_or_ax_snippet='[1353] role=heading name="Order # 000000178"',
            previous_actions=[],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: send_msg_to_user("101 S San Mateo Dr")',
                'ACTION: send_msg_to_user("101 S San Mateo Dr, San Mateo, California, 94010, United States")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertIn("San Mateo, California, 94010", decision.action_text)
        self.assertEqual(len(backend.prompts), 1)

    def test_shopping_order_detail_click_fallback_derives_shipping_method(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the shipping method for order number 187.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/187/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000187",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/187/",
                )
            ],
            visible_page_summary=(
                "Shipping Method\n"
                "Flat Rate - Fixed\n"
                "Order # 000000187\n"
                "$999.99\n"
                "$1,004.99"
            ),
            dom_or_ax_snippet=(
                '[1353] role=heading name="Order # 000000187"\n'
                '[1372] role=table name="Items Ordered"\n'
                '[1405] role=rowheader name="Shipping & Handling"\n'
                '[1406] role=gridcell name="$5.00"\n'
                '[1498] role=StaticText name="Shipping Method"\n'
                '[1499] role=StaticText name="Flat Rate - Fixed"\n'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/187/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1498")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Flat Rate - Fixed")')
        self.assertFalse(decision.should_retry)
        self.assertIsNone(decision.parse_error)

    def test_shopping_order_detail_go_back_triggers_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
            open_tabs=[
                OpenTab(
                    title="Order # 000000178",
                    url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
                )
            ],
            visible_page_summary="Billing Address\n101 S San Mateo Dr\nSan Mateo, California, 94010\nUnited States",
            dom_or_ax_snippet='[1353] role=heading name="Order # 000000178"',
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/order_id/178/")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                "ACTION: go_back()",
                'ACTION: send_msg_to_user("101 S San Mateo Dr, San Mateo, California, 94010, United States")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertIn("101 S San Mateo Dr", decision.action_text)
        self.assertEqual(len(backend.prompts), 1)

    def test_shopping_short_order_view_url_is_canonicalized(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            open_tabs=[
                OpenTab(
                    title="My Orders",
                    url="http://3.14.148.71:7770/sales/order/history/",
                )
            ],
            visible_page_summary="My Orders",
            dom_or_ax_snippet='[1371] role=gridcell name="000000178"',
            previous_actions=['goto("http://3.14.148.71:7770/sales/order/history/")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: goto("http://3.14.148.71:7770/sales/order/view/178")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=1)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/sales/order/view/order_id/178/")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_short_order_view_url_counts_as_detail_page_for_go_back_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/view/178",
            open_tabs=[
                OpenTab(
                    title="Order # 000000178",
                    url="http://3.14.148.71:7770/sales/order/view/178",
                )
            ],
            visible_page_summary="Billing Address\n101 S San Mateo Dr\nSan Mateo, California, 94010\nUnited States",
            dom_or_ax_snippet='[1353] role=heading name="Order # 000000178"',
            previous_actions=[
                'goto("http://3.14.148.71:7770/sales/order/history/")',
                'goto("http://3.14.148.71:7770/sales/order/view/178")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                "ACTION: go_back()",
                'ACTION: send_msg_to_user("101 S San Mateo Dr, San Mateo, California, 94010, United States")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertIn("101 S San Mateo Dr", decision.action_text)
        self.assertEqual(len(backend.prompts), 1)

    def test_shopping_final_category_page_rewrites_backtrack_click_to_sort(self) -> None:
        observation = NormalizedObservation(
            goal="List products from PS4 accessories category by ascending price",
            current_url="http://3.14.148.71:7770/video-games.html?cat=236",
            open_tabs=[
                OpenTab(
                    title="Video Games",
                    url="http://3.14.148.71:7770/video-games.html?cat=236",
                )
            ],
            visible_page_summary="Accessories listing page",
            dom_or_ax_snippet=(
                '[1078] role=menuitem name="Video Games" clickable\n'
                '[1389] role=combobox name="Sort By"\n'
                '[1505] role=link name="Replacement Accessory Pack" clickable\n'
            ),
            previous_actions=['click("1077")', 'click("1871")', 'click("1837")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: click("1078")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/video-games/playstation-4/accessories.html?product_list_order=price")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_final_category_page_rewrites_sort_to_canonical_target_url(self) -> None:
        observation = NormalizedObservation(
            goal="List products from competitive swimwear category by ascending price",
            current_url="http://3.14.148.71:7770/clothing-shoes-jewelry.html?cat=149",
            open_tabs=[
                OpenTab(
                    title="Clothing, Shoes & Jewelry",
                    url="http://3.14.148.71:7770/clothing-shoes-jewelry.html?cat=149",
                )
            ],
            visible_page_summary="Competitive swimwear listing page",
            dom_or_ax_snippet=(
                '[1389] role=combobox name="Sort By"\n'
                '[1393] role=link name="Set Descending Direction" clickable\n'
                '[1407] role=link name="Swimsuit" clickable\n'
            ),
            previous_actions=['click("544")', 'click("1816")', 'click("1815")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: select_option("1389", "price")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/clothing-shoes-jewelry/sport-specific-clothing/competitive-swimwear.html?product_list_order=price")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_final_category_page_rewrites_desc_goal_to_canonical_target_url(self) -> None:
        observation = NormalizedObservation(
            goal="List products from living room furniture category by descending price",
            current_url="http://3.14.148.71:7770/home-kitchen.html?cat=154",
            open_tabs=[
                OpenTab(
                    title="Home & Kitchen",
                    url="http://3.14.148.71:7770/home-kitchen.html?cat=154",
                )
            ],
            visible_page_summary="Living room furniture listing page",
            dom_or_ax_snippet=(
                '[1389] role=combobox name="Sort By"\n'
                '[1393] role=link name="Set Descending Direction" clickable\n'
                '[1407] role=link name="Coffee Table" clickable\n'
            ),
            previous_actions=['click("596")', 'click("1793")', 'click("1804")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: select_option("1389", "price")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/home-kitchen/furniture/living-room-furniture.html?product_list_order=price&product_list_dir=desc")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_shopping_bare_select_option_parse_failure_recovers_to_canonical_target_url(self) -> None:
        observation = NormalizedObservation(
            goal="List products from nutrition bars and drinks category by ascending price",
            current_url="http://3.14.148.71:7770/health-household.html?cat=192",
            open_tabs=[
                OpenTab(
                    title="Health & Household",
                    url="http://3.14.148.71:7770/health-household.html?cat=192",
                )
            ],
            visible_page_summary="Nutrition bars listing page",
            dom_or_ax_snippet=(
                '[1389] role=combobox name="Sort By"\n'
                '[1393] role=link name="Set Descending Direction" clickable\n'
                '[1436] role=link name="Protein Bar" clickable\n'
            ),
            previous_actions=['click("774")', 'click("1834")', 'click("1815")'],
            previous_errors=[],
        )
        backend = FakeBackend(['ACTION: select_option', 'ACTION: select_option'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(
            decision.action_text,
            'goto("http://3.14.148.71:7770/health-household/diet-sports-nutrition/nutrition-bars-drinks.html?product_list_order=price")',
        )
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_goto_bid_url_after_filtered_result_visible_triggers_click_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/explore?sort=name_asc&name=administrate&sort=name_asc",
            open_tabs=[
                OpenTab(
                    title="Projects · Explore · GitLab",
                    url="http://3.14.148.71:8023/explore?sort=name_asc&name=administrate&sort=name_asc",
                )
            ],
            visible_page_summary="Filtered GitLab explore page",
            dom_or_ax_snippet=(
                '[250] role=searchbox name="Filter by name" clickable focused\n'
                '[1115] role=link name="thoughtbot, inc. / administrate" clickable'
            ),
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "administrate")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: goto("http://3.14.148.71:1115")',
                'ACTION: click("1115")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=2)

        self.assertEqual(decision.action_text, 'click("1115")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("not a URL", backend.prompts[1])
        self.assertIn('ACTION: click("1115")', backend.prompts[1])

    def test_repo_page_backtrack_to_explore_triggers_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/thoughtbot/administrate",
            open_tabs=[
                OpenTab(
                    title="thoughtbot, inc. / administrate · GitLab",
                    url="http://3.14.148.71:8023/thoughtbot/administrate",
                )
            ],
            visible_page_summary="Repository page",
            dom_or_ax_snippet='[291] role=link name="Repository" clickable',
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "administrate")',
                'click("1115")',
            ],
            previous_errors=[
                'TimeoutError: Locator.click: Timeout 500ms exceeded.'
            ],
        )
        backend = FakeBackend(
            [
                'ACTION: goto("http://3.14.148.71:8023/explore")',
                'ACTION: send_msg_to_user("N/A")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=3)

        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Do not return to /explore", backend.prompts[1])
        self.assertIn('/-/graphs/main', backend.prompts[1])

    def test_graph_page_recursive_graph_append_triggers_answer_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the Pytorch GAN project",
            current_url="http://3.14.148.71:8023/eriklindernoren/PyTorch-GAN/-/graphs/master",
            open_tabs=[
                OpenTab(
                    title="Contributors · Erik Linder-Norén / PyTorch-GAN · GitLab",
                    url="http://3.14.148.71:8023/eriklindernoren/PyTorch-GAN/-/graphs/master",
                )
            ],
            visible_page_summary="Contributors graph",
            dom_or_ax_snippet='[514] role=heading name="Commits to master"\n[900] role=generic name="Erik Linder-Norén"',
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "PyTorch-GAN")',
                'click("1115")',
                'goto("http://3.14.148.71:8023/eriklindernoren/PyTorch-GAN/-/graphs/master")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: goto("http://3.14.148.71:8023/eriklindernoren/PyTorch-GAN/-/graphs/master/-/graphs/main")',
                'ACTION: send_msg_to_user("Erik Linder-Norén")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=4)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Erik Linder-Norén")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Do not append another /-/graphs segment", backend.prompts[1])
        self.assertIn('send_msg_to_user("<top contributor name>")', backend.prompts[1])

    def test_graph_404_same_branch_retry_suggests_alternate_branch(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the csvkit project",
            current_url="http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main",
            open_tabs=[
                OpenTab(
                    title="Contributors · wireservice / csvkit · GitLab",
                    url="http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main",
                )
            ],
            visible_page_summary="404\nPage Not Found",
            dom_or_ax_snippet='[9] role=heading name="Page Not Found"',
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "csvkit")',
                'click("1115")',
                'goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main")',
                'ACTION: goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/master")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=4)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/master")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Switch once to the alternate branch", backend.prompts[1])
        self.assertIn('goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/master")', backend.prompts[1])

    def test_graph_404_bare_branch_path_autocorrects_to_full_graph_url(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the csvkit project",
            current_url="http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main",
            open_tabs=[
                OpenTab(
                    title="Contributors · wireservice / csvkit · GitLab",
                    url="http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main",
                )
            ],
            visible_page_summary="404\nPage Not Found",
            dom_or_ax_snippet='[9] role=heading name="Page Not Found"',
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "csvkit")',
                'click("1115")',
                'goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: goto("http://3.14.148.71:8023/wireservice/csvkit/master")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=4)

        self.assertEqual(decision.action_text, 'goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/master")')
        self.assertEqual(len(backend.prompts), 1)
        self.assertIsNone(decision.parse_error)

    def test_graph_page_placeholder_answer_triggers_visible_name_retry(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/thoughtbot/administrate/-/graphs/main",
            open_tabs=[
                OpenTab(
                    title="Contributors · thoughtbot, inc. / administrate · GitLab",
                    url="http://3.14.148.71:8023/thoughtbot/administrate/-/graphs/main",
                )
            ],
            visible_page_summary="Contributors graph\nGrayson Wright\nOther contributor",
            dom_or_ax_snippet='[514] role=heading name="Commits to main"\n[900] role=generic name="Grayson Wright"',
            previous_actions=[
                'goto("http://3.14.148.71:8023/explore")',
                'fill("250", "administrate")',
                'click("1115")',
                'goto("http://3.14.148.71:8023/thoughtbot/administrate/-/graphs/main")',
            ],
            previous_errors=[],
        )
        backend = FakeBackend(
            [
                'ACTION: send_msg_to_user("John Doe")',
                'ACTION: send_msg_to_user("Grayson Wright")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(observation, step_idx=4)

        self.assertEqual(decision.action_text, 'send_msg_to_user("Grayson Wright")')
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("not a placeholder like John Doe or Jane Doe", backend.prompts[1])

    def test_two_failures_lead_to_safe_send_message_action(self) -> None:
        backend = FakeBackend(
            [
                "Malformed answer",
                "Still malformed",
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.raw_observation, step_idx=3)

        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertFalse(decision.should_retry)
        self.assertIsNotNone(decision.parse_error)
        self.assertIn("No ACTION line found in model output.", decision.parse_error or "")
        self.assertEqual(decision.raw_text, "Still malformed")

    def test_act_accepts_raw_dict_observation_input(self) -> None:
        backend = FakeBackend(['ACTION: click("58")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.raw_observation, step_idx=4)

        self.assertEqual(decision.action_text, 'click("58")')
        self.assertIn("Task Goal: Open the pricing page", backend.prompts[0])
        self.assertIn("Current URL: https://example.com/home", backend.prompts[0])

    def test_act_accepts_normalized_observation_input(self) -> None:
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.normalized_observation, step_idx=5)

        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertEqual(len(backend.prompts), 1)
        self.assertIn("Open Tabs:", backend.prompts[0])

    def test_policy_name_can_reflect_non_qwen_local_model(self) -> None:
        backend = FakeBackend(['ACTION: send_msg_to_user("N/A")'])
        policy = QwenPolicy(
            backend=backend,
            config=PolicyConfig(
                model_path="fake-model",
                policy_name="lfm2.5-350m",
                system_prompt="You are a helpful assistant trained by Liquid AI.",
            ),
        )

        decision = policy.act(self.normalized_observation, step_idx=6)

        self.assertEqual(policy.name, "lfm2.5-350m")
        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertIn("You are a helpful assistant trained by Liquid AI.", backend.prompts[0])


if __name__ == "__main__":
    unittest.main()
