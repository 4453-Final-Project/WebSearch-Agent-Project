from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import browsergym.webarena.instance as webarena_instance
from playwright.sync_api import _generated as sync_generated


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env import webarena_runner
from src.utils.config import sync_webarena_env_aliases


sync_webarena_env_aliases()
import webarena.evaluation_harness.helper_functions as helper_functions


class WebArenaRunnerTests(unittest.TestCase):
    def test_default_browser_timeout_used_when_env_missing(self) -> None:
        original = os.environ.pop("BROWSERGYM_TIMEOUT_MS", None)
        try:
            self.assertEqual(
                webarena_runner._resolve_browser_timeout_ms(),
                webarena_runner.DEFAULT_BROWSER_TIMEOUT_MS,
            )
        finally:
            if original is not None:
                os.environ["BROWSERGYM_TIMEOUT_MS"] = original

    def test_browser_timeout_env_override_is_respected(self) -> None:
        original = os.environ.get("BROWSERGYM_TIMEOUT_MS")
        os.environ["BROWSERGYM_TIMEOUT_MS"] = "45000"
        try:
            self.assertEqual(webarena_runner._resolve_browser_timeout_ms(), 45000)
        finally:
            if original is None:
                os.environ.pop("BROWSERGYM_TIMEOUT_MS", None)
            else:
                os.environ["BROWSERGYM_TIMEOUT_MS"] = original

    def test_webarena_env_aliases_are_synced_from_wa_vars(self) -> None:
        original_reddit = os.environ.pop("REDDIT", None)
        original_wa_reddit = os.environ.get("WA_REDDIT")
        os.environ["WA_REDDIT"] = "http://3.14.148.71:9999"
        try:
            webarena_runner.sync_webarena_env_aliases()
            self.assertEqual(os.environ["REDDIT"], "http://3.14.148.71:9999")
        finally:
            if original_reddit is None:
                os.environ.pop("REDDIT", None)
            else:
                os.environ["REDDIT"] = original_reddit
            if original_wa_reddit is None:
                os.environ.pop("WA_REDDIT", None)
            else:
                os.environ["WA_REDDIT"] = original_wa_reddit

    def test_shopping_login_patch_is_idempotent(self) -> None:
        webarena_runner._patch_webarena_shopping_login()
        first = webarena_instance.WebArenaInstance.ui_login
        webarena_runner._patch_webarena_shopping_login()
        second = webarena_instance.WebArenaInstance.ui_login

        self.assertIs(first, second)
        self.assertTrue(getattr(first, "_patched_for_shopping_domcontentloaded", False))

    def test_shopping_navigation_kwargs_use_domcontentloaded(self) -> None:
        normalized = webarena_runner._normalize_shopping_navigation_kwargs(
            "http://3.14.148.71:7770/customer/account/login/",
            {},
        )
        self.assertEqual(normalized["wait_until"], "domcontentloaded")

        explicit = webarena_runner._normalize_shopping_navigation_kwargs(
            "http://3.14.148.71:7770/customer/account/login/",
            {"wait_until": "load"},
        )
        self.assertEqual(explicit["wait_until"], "domcontentloaded")

        preserved = webarena_runner._normalize_shopping_navigation_kwargs(
            "http://3.14.148.71:7770/customer/account/login/",
            {"wait_until": "domcontentloaded"},
        )
        self.assertEqual(preserved["wait_until"], "domcontentloaded")

        other_domain = webarena_runner._normalize_shopping_navigation_kwargs(
            "http://3.14.148.71:8023/explore",
            {},
        )
        self.assertNotIn("wait_until", other_domain)

    def test_playwright_shopping_navigation_patch_is_idempotent(self) -> None:
        webarena_runner._patch_playwright_shopping_navigation()
        first = sync_generated.Page.goto
        webarena_runner._patch_playwright_shopping_navigation()
        second = sync_generated.Page.goto

        self.assertIs(first, second)
        self.assertTrue(getattr(first, "_patched_for_shopping_domcontentloaded", False))

    def test_openai_judge_patch_is_idempotent(self) -> None:
        webarena_runner._patch_webarena_openai_judges()
        first = helper_functions.llm_fuzzy_match
        webarena_runner._patch_webarena_openai_judges()
        second = helper_functions.llm_fuzzy_match

        self.assertIs(first, second)
        self.assertTrue(getattr(first, "_patched_for_configurable_openai_judge", False))

    def test_openai_judge_uses_configured_model_override(self) -> None:
        original_model = os.environ.get("OPENAI_JUDGE_MODEL")
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_JUDGE_MODEL"] = "gpt-5.4-nano"
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            webarena_runner._patch_webarena_openai_judges()
            captured: dict[str, object] = {}

            def fake_generate(model, messages):
                captured["model"] = model
                captured["messages"] = messages
                return "correct"

            with patch.object(
                webarena_runner,
                "_call_openai_judge_model",
                side_effect=fake_generate,
            ):
                score = helper_functions.llm_fuzzy_match(
                    pred="000000170",
                    reference="000000170",
                    question="What is the latest canceled order number?",
                )

            self.assertEqual(score, 1.0)
            self.assertEqual(captured["model"], "gpt-5.4-nano")
            self.assertIsInstance(captured["messages"], list)
        finally:
            if original_model is None:
                os.environ.pop("OPENAI_JUDGE_MODEL", None)
            else:
                os.environ["OPENAI_JUDGE_MODEL"] = original_model
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_fuzzy_judge_message_mentions_semantic_equivalence(self) -> None:
        messages = webarena_runner._build_webarena_fuzzy_judge_messages(
            pred="365.42",
            reference="365.42",
            question="What was the total for the latest canceled order?",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("semantically equivalent", messages[1]["content"])
        self.assertIn("student answer: 365.42", messages[1]["content"])

    def test_shopping_task_url_normalization_fixes_double_slash_and_query_padding(self) -> None:
        normalized = webarena_runner._normalize_shopping_task_url(
            "http://3.14.148.71:7770//catalogsearch/result/index/?product_list_order=price&q=%20iphone%2012%20phone%20case%20"
        )
        self.assertEqual(
            normalized,
            "http://3.14.148.71:7770/catalogsearch/result/index/?product_list_order=price&q=iphone+12+phone+case",
        )

    def test_shopping_task_config_normalization_only_applies_to_shopping_tasks(self) -> None:
        shopping_config = {
            "sites": ["shopping"],
            "start_url": "http://3.14.148.71:7770//",
            "eval": {
                "reference_url": "http://3.14.148.71:7770//catalogsearch/result/?q=%20chairs%20",
            },
        }
        normalized = webarena_runner._normalize_shopping_task_config(shopping_config)
        self.assertEqual(normalized["start_url"], "http://3.14.148.71:7770/")
        self.assertEqual(
            normalized["eval"]["reference_url"],
            "http://3.14.148.71:7770/catalogsearch/result/?q=chairs",
        )

        gitlab_config = {
            "sites": ["gitlab"],
            "start_url": "http://3.14.148.71:8023//explore",
            "eval": {
                "reference_url": "http://3.14.148.71:8023//explore",
            },
        }
        untouched = webarena_runner._normalize_shopping_task_config(gitlab_config)
        self.assertEqual(untouched, gitlab_config)

    def test_shopping_goal_keywords_drop_common_instruction_words(self) -> None:
        keywords = webarena_runner._shopping_goal_keywords(
            'List products from PS4 accessories category by ascending price'
        )
        self.assertIn("ps4", keywords)
        self.assertIn("accessories", keywords)
        self.assertNotIn("products", keywords)
        self.assertNotIn("ascending", keywords)
        self.assertNotIn("price", keywords)

    def test_shopping_priority_prefers_goal_relevant_items_over_footer_links(self) -> None:
        footer_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Privacy and Cookie Policy",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"ps4", "accessories"},
            "",
        )
        relevant_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Skinit Decal Gaming Skin for PS4 Console",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"ps4", "accessories"},
            "",
        )

        self.assertGreater(relevant_score, footer_score)

    def test_shopping_priority_prefers_next_category_label_over_product_hits(self) -> None:
        product_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Skinit Decal Gaming Skin for PS4 Console",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"ps4", "accessories"},
            "playstation 4",
        )
        category_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "PlayStation 4",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"ps4", "accessories"},
            "playstation 4",
        )

        self.assertGreater(category_score, product_score)

    def test_shopping_order_priority_prefers_order_number_and_status_over_view_order(self) -> None:
        generic_action_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "View Order",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"order", "number", "cancelled", "recent"},
            "",
        )
        order_number_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "000000170",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"order", "number", "cancelled", "recent", "170"},
            "",
        )
        status_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Canceled",
                "role": "cell",
                "clickable": False,
                "focused": False,
            },
            {"order", "number", "cancelled", "recent"},
            "",
        )

        self.assertGreater(order_number_score, generic_action_score)
        self.assertGreaterEqual(status_score, generic_action_score)

    def test_shopping_order_detail_priority_prefers_billing_address_over_prices(self) -> None:
        price_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "$62.18",
                "role": "gridcell",
                "clickable": False,
                "focused": False,
            },
            {"billing", "address", "order", "178"},
            "",
        )
        address_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Billing Address",
                "role": "StaticText",
                "clickable": False,
                "focused": False,
            },
            {"billing", "address", "order", "178"},
            "",
        )

        self.assertGreater(address_score, price_score)

    def test_shopping_order_detail_priority_prefers_product_titles_over_prices(self) -> None:
        price_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "$169.95",
                "role": "gridcell",
                "clickable": False,
                "focused": False,
            },
            {"order", "148", "names"},
            "",
        )
        product_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks, Hanging Entryway Organizer",
                "role": "link",
                "clickable": True,
                "focused": False,
            },
            {"order", "148", "names"},
            "",
        )

        self.assertGreater(product_score, price_score)

    def test_bboxless_billing_address_text_is_kept_for_order_goals(self) -> None:
        self.assertTrue(
            webarena_runner._allow_bboxless_shopping_order_text(
                "101 S San Mateo Dr",
                "StaticText",
                {"billing", "address", "order", "178"},
            )
        )
        self.assertFalse(
            webarena_runner._allow_bboxless_shopping_order_text(
                "Privacy and Cookie Policy",
                "StaticText",
                {"billing", "address", "order", "178"},
            )
        )

    def test_bboxless_billing_address_text_is_added_to_visible_summary(self) -> None:
        obs = {
            "goal": "Show me the billing address for order number 00178.",
            "url": "http://3.14.148.71:7770/sales/order/view/order_id/178/",
            "axtree_object": {
                "nodes": [
                    {
                        "browsergym_id": "1560",
                        "role": {"value": "link"},
                        "name": {"value": "Address Book"},
                    },
                    {
                        "role": {"value": "StaticText"},
                        "name": {"value": "Billing Address"},
                    },
                    {
                        "role": {"value": "StaticText"},
                        "name": {"value": "101 S San Mateo Dr"},
                    },
                    {
                        "role": {"value": "StaticText"},
                        "name": {"value": "San Mateo, California, 94010"},
                    },
                    {
                        "role": {"value": "StaticText"},
                        "name": {"value": "United States"},
                    },
                ]
            },
            "extra_element_properties": {
                "1560": {"clickable": True, "bbox": [0, 0, 10, 10]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Billing Address", summary)
        self.assertIn("101 S San Mateo Dr", summary)
        self.assertIn("United States", summary)

    def test_shopping_order_goal_keywords_detected(self) -> None:
        self.assertTrue(
            webarena_runner._is_shopping_order_goal_keywords({"order", "cancelled", "recent"})
        )
        self.assertFalse(
            webarena_runner._is_shopping_order_goal_keywords({"ps4", "accessories", "price"})
        )

    def test_shopping_category_score_ignores_long_product_titles_for_leaf_category(self) -> None:
        product_match = webarena_runner._shopping_category_match_score(
            "5Pcs HDMI Port Socket Connector for Sony Playstation 4 Console Replacement Accessory",
            "accessories",
        )
        category_match = webarena_runner._shopping_category_match_score(
            "Accessories 233 item",
            "accessories",
        )

        self.assertEqual(product_match, 0)
        self.assertGreater(category_match, 0)

    def test_shopping_next_category_target_prefers_deepest_visible_clickable_label(self) -> None:
        items = [
            {"name": "Video Games", "role": "menuitem"},
            {"name": "PlayStation 4 (233 item)", "role": "link"},
            {"name": "Accessories (228 item)", "role": "link"},
        ]

        target = webarena_runner._shopping_next_category_target(
            ["Video Games", "PlayStation 4", "Accessories"],
            "http://3.14.148.71:7770/video-games.html?cat=67",
            items,
        )

        self.assertEqual(target, "accessories")


if __name__ == "__main__":
    unittest.main()
