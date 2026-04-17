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

    def test_shopping_admin_item_count_priority_prefers_quantity_signals_over_prices(self) -> None:
        price_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "$219.40",
                "role": "gridcell",
                "clickable": True,
                "focused": False,
            },
            {"orders", "items", "sold", "recent"},
            "",
        )
        quantity_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Purchased",
                "role": "columnheader",
                "clickable": False,
                "focused": False,
            },
            {"orders", "items", "sold", "recent"},
            "",
        )

        self.assertGreater(quantity_score, price_score)

    def test_shopping_name_priority_prefers_person_names_for_billing_goals(self) -> None:
        date_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Apr 1, 2023 11:59:23 AM",
                "role": "gridcell",
                "clickable": True,
                "focused": False,
            },
            {"billing", "name", "oldest", "complete", "order"},
            "",
        )
        person_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "john lee",
                "role": "gridcell",
                "clickable": True,
                "focused": False,
            },
            {"billing", "name", "oldest", "complete", "order"},
            "",
        )

        self.assertGreater(person_score, date_score)

    def test_shopping_admin_recency_priority_prefers_purchase_date_header(self) -> None:
        status_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Complete",
                "role": "gridcell",
                "clickable": True,
                "focused": False,
            },
            {"total", "payment", "amount", "last", "pending", "orders"},
            "",
        )
        header_score = webarena_runner._shopping_item_priority_score(
            {
                "name": "Purchase Date",
                "role": "columnheader",
                "clickable": True,
                "focused": False,
            },
            {"total", "payment", "amount", "last", "pending", "orders"},
            "",
        )

        self.assertGreater(header_score, status_score)

    def test_admin_order_recency_snippet_forces_purchase_date_header(self) -> None:
        nodes = [
            {"browsergym_id": "1388", "role": {"value": "columnheader"}, "name": {"value": "Purchase Date"}},
        ]
        extra_props = {"1388": {"bbox": [100, 100, 10, 10], "clickable": True}}
        for index in range(60):
            bid = str(2000 + index)
            nodes.append(
                {
                    "browsergym_id": bid,
                    "role": {"value": "gridcell"},
                    "name": {"value": f"Person Name {index}"},
                }
            )
            extra_props[bid] = {"bbox": [200 + index, 120 + index, 10, 10], "clickable": True}

        obs = {
            "goal": "Get the billing name of the oldest complete order",
            "url": "http://3.14.148.71:7780/admin/sales/order/",
            "axtree_object": {"nodes": nodes},
            "extra_element_properties": extra_props,
        }

        snippet = webarena_runner._build_dom_or_ax_snippet(obs)

        self.assertIn('[1388] role=columnheader name="Purchase Date" clickable', snippet)

    def test_bboxless_billing_address_text_is_kept_for_order_goals(self) -> None:
        self.assertTrue(
            webarena_runner._allow_bboxless_shopping_order_text(
                "101 S San Mateo Dr",
                "StaticText",
                {"billing", "address", "order", "178"},
            )
        )

    def test_bboxless_shipping_method_value_is_kept_for_order_goals(self) -> None:
        self.assertTrue(
            webarena_runner._allow_bboxless_shopping_order_text(
                "Flat Rate - Fixed",
                "StaticText",
                {"shipping", "method", "order", "187"},
            )
        )

    def test_bboxless_person_name_is_kept_for_billing_name_goals(self) -> None:
        self.assertTrue(
            webarena_runner._allow_bboxless_shopping_order_text(
                "john lee",
                "StaticText",
                {"billing", "name", "oldest", "complete", "order"},
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

    def test_bboxless_shipping_method_value_is_added_to_visible_summary(self) -> None:
        obs = {
            "goal": "Show me the shipping method for order number 187.",
            "url": "http://3.14.148.71:7770/sales/order/view/order_id/187/",
            "axtree_object": {
                "nodes": [
                    {
                        "role": {"value": "StaticText"},
                        "name": {"value": "Shipping Method"},
                    },
                    {
                        "role": {"value": "StaticText"},
                        "name": {"value": "Flat Rate - Fixed"},
                    },
                    {
                        "browsergym_id": "1405",
                        "role": {"value": "rowheader"},
                        "name": {"value": "Shipping & Handling"},
                    },
                ]
            },
            "extra_element_properties": {
                "1405": {"bbox": [0, 0, 10, 10]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Shipping Method", summary)
        self.assertIn("Flat Rate - Fixed", summary)

    def test_customer_order_rows_are_added_to_visible_summary_for_history_goals(self) -> None:
        obs = {
            "goal": "Tell me the total cost of my latest pending order?",
            "url": "http://3.14.148.71:7770/sales/order/history/",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "1830", "role": {"value": "link"}, "name": {"value": "Page 2"}},
                    {"browsergym_id": "1834", "role": {"value": "link"}, "name": {"value": "Page Next"}},
                    {"browsergym_id": "1371", "role": {"value": "gridcell"}, "name": {"value": "000000170"}},
                    {"browsergym_id": "1372", "role": {"value": "gridcell"}, "name": {"value": "5/17/23"}},
                    {"browsergym_id": "1373", "role": {"value": "gridcell"}, "name": {"value": "$365.42"}},
                    {"browsergym_id": "1375", "role": {"value": "gridcell"}, "name": {"value": "Canceled"}},
                    {"browsergym_id": "1382", "role": {"value": "gridcell"}, "name": {"value": "000000189"}},
                    {"browsergym_id": "1383", "role": {"value": "gridcell"}, "name": {"value": "5/2/23"}},
                    {"browsergym_id": "1384", "role": {"value": "gridcell"}, "name": {"value": "$754.99"}},
                    {"browsergym_id": "1386", "role": {"value": "gridcell"}, "name": {"value": "Pending"}},
                ]
            },
            "extra_element_properties": {
                "1371": {"bbox": [100, 100, 10, 10]},
                "1372": {"bbox": [200, 100, 10, 10]},
                "1373": {"bbox": [300, 100, 10, 10]},
                "1375": {"bbox": [400, 100, 10, 10]},
                "1382": {"bbox": [100, 140, 10, 10]},
                "1383": {"bbox": [200, 140, 10, 10]},
                "1384": {"bbox": [300, 140, 10, 10]},
                "1386": {"bbox": [400, 140, 10, 10]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Customer order pager: page_2=1830 | next=1834", summary)
        self.assertIn("Customer order row: order=000000170", summary)
        self.assertIn("status=Canceled", summary)
        self.assertIn("Customer order row: order=000000189", summary)
        self.assertIn("total=$754.99", summary)

    def test_refund_detail_lines_are_appended_for_matching_history_rows(self) -> None:
        class FakeLocator:
            def __init__(self, values):
                self._values = values

            def all_inner_texts(self):
                return list(self._values)

        class FakeDetailPage:
            def __init__(self):
                self.goto_calls = []
                self.closed = False

            def goto(self, url, wait_until=None):
                self.goto_calls.append((url, wait_until))

            def locator(self, selector):
                if selector == "table.data.table.table-order-items tbody tr":
                    return FakeLocator(
                        [
                            "SupplySource AC-DC Adapter for Toshiba Satellite C55-C5241 Charger Power Supply B01ABCDEF $17.28",
                            "Pirate's Booty Aged White Cheddar Rice And Corn Puffs, 0.5 Ounce 12 Count $49.00",
                        ]
                    )
                if selector == "tfoot tr":
                    return FakeLocator(
                        [
                            "Subtotal $95.18",
                            "Shipping & Handling $20.00",
                            "Grand Total $115.18",
                        ]
                    )
                return FakeLocator([])

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.detail_pages = []

            def new_page(self):
                page = FakeDetailPage()
                self.detail_pages.append(page)
                return page

        class FakePage:
            def __init__(self):
                self.context = FakeContext()

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        obs = {
            "url": "http://3.14.148.71:7770/sales/order/history/?p=4",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "1371", "role": {"value": "gridcell"}, "name": {"value": "000000160"}},
                    {"browsergym_id": "1372", "role": {"value": "gridcell"}, "name": {"value": "3/2/22"}},
                    {"browsergym_id": "1373", "role": {"value": "gridcell"}, "name": {"value": "$115.18"}},
                    {"browsergym_id": "1375", "role": {"value": "gridcell"}, "name": {"value": "Canceled"}},
                ]
            },
            "extra_element_properties": {
                "1371": {"bbox": [100, 100, 10, 10]},
                "1372": {"bbox": [200, 100, 10, 10]},
                "1373": {"bbox": [300, 100, 10, 10]},
                "1375": {"bbox": [400, 100, 10, 10]},
            },
        }
        serialized_observation = {"visible_page_summary": "Customer order row: order=000000160 | date=3/2/22 | total=$115.18 | status=Canceled"}
        env = FakeEnv()

        webarena_runner._append_shopping_refund_detail_lines(
            serialized_observation,
            raw_observation=obs,
            goal=(
                "How much refund I should expect from my order canlled in 2022/03? "
                "I only kept the AC-DC Adapter and the shop told me that I cannot get the shipping fee back"
            ),
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Customer refund detail: order=000000160", summary)
        self.assertIn("subtotal=$95.18", summary)
        self.assertIn("shipping=$20.00", summary)
        self.assertIn("grand_total=$115.18", summary)
        self.assertIn(
            "product=SupplySource AC-DC Adapter for Toshiba Satellite C55-C5241 Charger Power Supply B01ABCDEF::$17.28",
            summary,
        )
        self.assertEqual(len(env.page.context.detail_pages), 2)
        self.assertTrue(all(page.closed for page in env.page.context.detail_pages))

    def test_refund_detail_lines_append_zero_match_marker_when_background_history_has_no_match(self) -> None:
        class FakeLocator:
            def __init__(self, values):
                self._values = values

            def all_inner_texts(self):
                return list(self._values)

        class FakeHistoryPage:
            def __init__(self):
                self.goto_calls = []
                self.closed = False
                self._current_url = ""

            def goto(self, url, wait_until=None):
                self.goto_calls.append((url, wait_until))
                self._current_url = url

            def locator(self, selector):
                if selector == "tbody tr":
                    if "p=1" in self._current_url:
                        return FakeLocator(["000000170 5/17/23 $365.42 Canceled"])
                    if "p=2" in self._current_url:
                        return FakeLocator(["000000156 2/24/23 $231.54 Canceled"])
                    return FakeLocator([])
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.pages = []

            def new_page(self):
                page = FakeHistoryPage()
                self.pages.append(page)
                return page

        class FakePage:
            def __init__(self):
                self.context = FakeContext()

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        obs = {
            "url": "http://3.14.148.71:7770/sales/order/history/",
            "axtree_object": {"nodes": []},
            "extra_element_properties": {},
        }
        serialized_observation = {
            "visible_page_summary": "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled"
        }
        env = FakeEnv()

        webarena_runner._append_shopping_refund_detail_lines(
            serialized_observation,
            raw_observation=obs,
            goal="How much refund I should expect from my order canlled in April 2022, including shipping fee",
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Customer refund match count: 0", summary)
        self.assertEqual(len(env.page.context.pages), 1)
        self.assertTrue(env.page.context.pages[0].closed)

    def test_order_product_lines_are_appended_for_last_ordered_product_goal(self) -> None:
        class FakeLocator:
            def __init__(self, values):
                self._values = values

            def all_inner_texts(self):
                return list(self._values)

        class FakeDetailPage:
            def __init__(self, product_rows_by_url):
                self._product_rows_by_url = product_rows_by_url
                self._current_url = ""
                self.closed = False

            def goto(self, url, wait_until=None):
                self._current_url = url

            def locator(self, selector):
                if selector == "table.data.table.table-order-items tbody tr":
                    return FakeLocator(self._product_rows_by_url.get(self._current_url, []))
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeHistoryPage:
            def __init__(self):
                self._current_url = ""
                self.closed = False

            def goto(self, url, wait_until=None):
                self._current_url = url

            def locator(self, selector):
                if selector == "tbody tr":
                    if "p=1" in self._current_url:
                        return FakeLocator(["000000170 5/17/23 $365.42 Canceled"])
                    if "p=2" in self._current_url:
                        return FakeLocator(["000000113 1/16/23 $231.54 Complete"])
                    return FakeLocator([])
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self, product_rows_by_url):
                self._product_rows_by_url = product_rows_by_url
                self.detail_pages = []
                self.history_pages = []

            def new_page(self):
                if not self.history_pages:
                    page = FakeHistoryPage()
                    self.history_pages.append(page)
                    return page
                page = FakeDetailPage(self._product_rows_by_url)
                self.detail_pages.append(page)
                return page

        class FakePage:
            def __init__(self, product_rows_by_url):
                self.context = FakeContext(product_rows_by_url)

        class FakeEnv:
            def __init__(self, product_rows_by_url):
                self.unwrapped = self
                self.page = FakePage(product_rows_by_url)

        base = "http://3.14.148.71:7770/sales/order/view/order_id"
        env = FakeEnv(
            {
                f"{base}/170/": ["Random Product $12.00"],
                f"{base}/113/": ["Ultra Body Butter $21.00"],
            }
        )
        obs = {
            "url": "http://3.14.148.71:7770/sales/order/history/",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "1371", "role": {"value": "gridcell"}, "name": {"value": "000000170"}},
                    {"browsergym_id": "1372", "role": {"value": "gridcell"}, "name": {"value": "5/17/23"}},
                    {"browsergym_id": "1373", "role": {"value": "gridcell"}, "name": {"value": "$365.42"}},
                    {"browsergym_id": "1375", "role": {"value": "gridcell"}, "name": {"value": "Canceled"}},
                ]
            },
            "extra_element_properties": {
                "1371": {"bbox": [100, 100, 10, 10]},
                "1372": {"bbox": [200, 100, 10, 10]},
                "1373": {"bbox": [300, 100, 10, 10]},
                "1375": {"bbox": [400, 100, 10, 10]},
            },
        }
        serialized_observation = {
            "visible_page_summary": "Customer order row: order=000000170 | date=5/17/23 | total=$365.42 | status=Canceled"
        }

        webarena_runner._append_shopping_order_product_lines(
            serialized_observation,
            raw_observation=obs,
            goal="Tell me when I last ordered my body butter?",
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Customer order product row: order=000000113 | date=1/16/23 | product=Ultra Body Butter", summary)
        self.assertEqual(len(env.page.context.detail_pages), 2)
        self.assertTrue(all(page.closed for page in env.page.context.detail_pages))

    def test_order_detail_date_line_is_appended_for_order_date_goal(self) -> None:
        class FakeBody:
            def inner_text(self):
                return "Order Date: 1/29/2023 Order # 000000148"

        class FakePage:
            def locator(self, selector):
                if selector == "body":
                    return FakeBody()
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "Order # 000000148"}
        obs = {"url": "http://3.14.148.71:7770/sales/order/view/order_id/148/"}

        webarena_runner._append_shopping_order_detail_date_line(
            serialized_observation,
            raw_observation=obs,
            goal="Show me the order date for order number 148.",
            env=FakeEnv(),
        )

        self.assertIn("Order detail date: 1/29/2023", serialized_observation["visible_page_summary"])

    def test_order_detail_named_month_date_line_is_appended_for_order_date_goal(self) -> None:
        class FakeBody:
            def inner_text(self):
                return "Order Date: March 10, 2022 Order # 000000148"

        class FakePage:
            def locator(self, selector):
                if selector == "body":
                    return FakeBody()
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "Order # 000000148"}
        obs = {"url": "http://3.14.148.71:7770/sales/order/view/order_id/148/"}

        webarena_runner._append_shopping_order_detail_date_line(
            serialized_observation,
            raw_observation=obs,
            goal="Show me the order date for order number 148.",
            env=FakeEnv(),
        )

        self.assertIn("Order detail date: March 10, 2022", serialized_observation["visible_page_summary"])

    def test_order_detail_date_line_falls_back_to_page_html(self) -> None:
        class FakeBody:
            def inner_text(self):
                return "Order Date:"

        class FakePage:
            def locator(self, selector):
                if selector == "body":
                    return FakeBody()
                raise AssertionError(f"Unexpected selector: {selector}")

            def content(self):
                return "<div>Order Date: <span>January 29, 2023</span></div>"

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "Order # 000000148"}
        obs = {"url": "http://3.14.148.71:7770/sales/order/view/order_id/148/"}

        webarena_runner._append_shopping_order_detail_date_line(
            serialized_observation,
            raw_observation=obs,
            goal="Show me the order date for order number 148.",
            env=FakeEnv(),
        )

        self.assertIn("Order detail date: January 29, 2023", serialized_observation["visible_page_summary"])

    def test_combined_fuzzy_reference_prefers_raw_annotation(self) -> None:
        combined = webarena_runner._resolve_combined_fuzzy_reference(
            {"reference_answer_raw_annotation": "170: cancelled, 189: pending"},
            ["170: cancelled", "189: pending"],
        )

        self.assertEqual(combined, "170: cancelled, 189: pending")

    def test_combined_fuzzy_reference_joins_multiple_references_without_annotation(self) -> None:
        combined = webarena_runner._resolve_combined_fuzzy_reference(
            {},
            ["170: cancelled", "189: pending"],
        )

        self.assertEqual(combined, "170: cancelled, 189: pending")

    def test_admin_order_item_detail_lines_are_appended_for_recent_item_goals(self) -> None:
        class FakeCellLocator:
            def __init__(self, values):
                self._values = list(values)

            def count(self):
                return len(self._values)

            def nth(self, index):
                return FakeTextNode(self._values[index])

        class FakeRowLocator:
            def __init__(self, values):
                self._values = list(values)

            def count(self):
                return len(self._values)

            def nth(self, index):
                return FakeRow(self._values[index])

        class FakeTextNode:
            def __init__(self, value):
                self._value = value

            def inner_text(self):
                return self._value

        class FakeRow:
            def __init__(self, cells):
                self._cells = list(cells)

            def locator(self, selector):
                if selector == "td":
                    return FakeCellLocator(self._cells)
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeLink:
            def __init__(self, href, row):
                self._href = href
                self._row = row

            def get_attribute(self, name):
                if name == "href":
                    return self._href
                return None

            def locator(self, selector):
                if selector == "xpath=ancestor::tr[1]":
                    return self._row
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeLinks:
            def __init__(self, links):
                self._links = list(links)

            def count(self):
                return len(self._links)

            def nth(self, index):
                return self._links[index]

        class FakeTable:
            def __init__(self, text, rows):
                self._text = text
                self._rows = rows

            def inner_text(self):
                return self._text

            def locator(self, selector):
                if selector == "tbody tr":
                    return FakeRowLocator(self._rows)
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeTables:
            def __init__(self, tables):
                self._tables = list(tables)

            def count(self):
                return len(self._tables)

            def nth(self, index):
                return self._tables[index]

        class FakeDetailPage:
            def __init__(self, table_by_url):
                self._table_by_url = table_by_url
                self._current_url = ""
                self.closed = False

            def goto(self, url, wait_until=None):
                self._current_url = url

            def locator(self, selector):
                if selector == "table":
                    return FakeTables([self._table_by_url[self._current_url]])
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self, table_by_url):
                self._table_by_url = table_by_url
                self.detail_pages = []

            def new_page(self):
                page = FakeDetailPage(self._table_by_url)
                self.detail_pages.append(page)
                return page

        class FakePage:
            def __init__(self, links, table_by_url):
                self._links = FakeLinks(links)
                self.context = FakeContext(table_by_url)

            def locator(self, selector):
                if selector == 'a[href*="/admin/sales/order/view/order_id/"]':
                    return self._links
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeEnv:
            def __init__(self, links, table_by_url):
                self.unwrapped = self
                self.page = FakePage(links, table_by_url)

        detail_url_one = "http://3.14.148.71:7780/admin/sales/order/view/order_id/217/"
        detail_url_two = "http://3.14.148.71:7780/admin/sales/order/view/order_id/261/"
        links = [
            FakeLink(
                detail_url_one,
                FakeRow(["", "000000217", "Main Website", "Dec 1, 2022 1:54:18 PM", "John Smith", "John Smith", "$121.00", "$121.00", "Complete", "View"]),
            ),
            FakeLink(
                detail_url_two,
                FakeRow(["", "000000261", "Main Website", "Nov 30, 2022 4:00:41 AM", "Lily Potter", "Lily Potter", "$192.00", "$192.00", "Canceled", "View"]),
            ),
        ]
        table_by_url = {
            detail_url_one: FakeTable(
                "Product Item Status Original Price Price Qty Subtotal Tax Amount Tax Percent Discount Amount Row Total",
                [
                    ["Product A", "Ordered", "$32.00", "$32.00", "Ordered\t1", "1", "$32.00", "$0.00", "0%", "$0.00", "$32.00"],
                    ["Product B", "Ordered", "$42.00", "$42.00", "Ordered\t2", "2", "$84.00", "$0.00", "0%", "$0.00", "$84.00"],
                ],
            ),
            detail_url_two: FakeTable(
                "Product Item Status Original Price Price Qty Subtotal Tax Amount Tax Percent Discount Amount Row Total",
                [
                    ["Product C", "Ordered", "$39.00", "$39.00", "Ordered\t5", "5", "$195.00", "$0.00", "0%", "$0.00", "$195.00"],
                ],
            ),
        }
        env = FakeEnv(links, table_by_url)
        serialized_observation = {
            "visible_page_summary": (
                "Order row: order=000000217 | date=Dec 1, 2022 1:54:18 PM | customer=John Smith | billing=John Smith | total=121.00 | status=Complete\n"
                "Order row: order=000000261 | date=Nov 30, 2022 4:00:41 AM | customer=Lily Potter | billing=Lily Potter | total=192.00 | status=Canceled"
            )
        }
        obs = {"url": "http://3.14.148.71:7780/admin/sales/order/"}

        webarena_runner._append_shopping_admin_order_item_count_lines(
            serialized_observation,
            raw_observation=obs,
            goal="What's the total number of items sold in the most recent 2 orders?",
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Admin order item row: order=000000217", summary)
        self.assertIn("items=3", summary)
        self.assertIn("Admin order item row: order=000000261", summary)
        self.assertIn("items=5", summary)
        self.assertEqual(len(env.page.context.detail_pages), 2)
        self.assertTrue(all(page.closed for page in env.page.context.detail_pages))

    def test_admin_order_product_price_lines_are_appended_for_recent_completed_order_goal(self) -> None:
        class FakeCellLocator:
            def __init__(self, values):
                self._values = list(values)

            def count(self):
                return len(self._values)

            def nth(self, index):
                return FakeTextNode(self._values[index])

        class FakeRowLocator:
            def __init__(self, values):
                self._values = list(values)

            def count(self):
                return len(self._values)

            def nth(self, index):
                return FakeRow(self._values[index])

        class FakeTextNode:
            def __init__(self, value):
                self._value = value

            def inner_text(self):
                return self._value

        class FakeRow:
            def __init__(self, cells):
                self._cells = list(cells)

            def locator(self, selector):
                if selector == "td":
                    return FakeCellLocator(self._cells)
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeLink:
            def __init__(self, href, row):
                self._href = href
                self._row = row

            def get_attribute(self, name):
                if name == "href":
                    return self._href
                return None

            def locator(self, selector):
                if selector == "xpath=ancestor::tr[1]":
                    return self._row
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeLinks:
            def __init__(self, links):
                self._links = list(links)

            def count(self):
                return len(self._links)

            def nth(self, index):
                return self._links[index]

        class FakeTable:
            def __init__(self, text, rows):
                self._text = text
                self._rows = rows

            def inner_text(self):
                return self._text

            def locator(self, selector):
                if selector == "tbody tr":
                    return FakeRowLocator(self._rows)
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeTables:
            def __init__(self, tables):
                self._tables = list(tables)

            def count(self):
                return len(self._tables)

            def nth(self, index):
                return self._tables[index]

        class FakeDetailPage:
            def __init__(self, table_by_url):
                self._table_by_url = table_by_url
                self._current_url = ""
                self.closed = False

            def goto(self, url, wait_until=None):
                self._current_url = url

            def locator(self, selector):
                if selector == "table":
                    return FakeTables([self._table_by_url[self._current_url]])
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self, table_by_url):
                self._table_by_url = table_by_url
                self.detail_pages = []

            def new_page(self):
                page = FakeDetailPage(self._table_by_url)
                self.detail_pages.append(page)
                return page

        class FakePage:
            def __init__(self, links, table_by_url):
                self._links = FakeLinks(links)
                self.context = FakeContext(table_by_url)

            def locator(self, selector):
                if selector == 'a[href*="/admin/sales/order/view/order_id/"]':
                    return self._links
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeEnv:
            def __init__(self, links, table_by_url):
                self.unwrapped = self
                self.page = FakePage(links, table_by_url)

        detail_url = "http://3.14.148.71:7780/admin/sales/order/view/order_id/230/"
        links = [
            FakeLink(
                detail_url,
                FakeRow(["", "000000230", "Main Website", "May 19, 2023 8:11:51 AM", "Ava Brown", "Ava Brown", "$93.40", "$93.40", "Complete", "View"]),
            ),
        ]
        table_by_url = {
            detail_url: FakeTable(
                "Product Item Status Original Price Price Qty Subtotal Tax Amount Tax Percent Discount Amount Row Total",
                [
                    ["Rapha Sports Short", "Ordered", "$50.00", "$35.00", "Ordered\t1", "$35.00", "$0.00", "0%", "$15.00", "$35.00"],
                    ["Mach Street Sweatshirt", "Ordered", "$80.00", "$62.00", "Ordered\t1", "$62.00", "$0.00", "0%", "$18.00", "$62.00"],
                    ["Thorpe Track Pant", "Ordered", "$68.00", "$54.40", "Ordered\t1", "$54.40", "$0.00", "0%", "$13.60", "$54.40"],
                ],
            ),
        }
        env = FakeEnv(links, table_by_url)
        serialized_observation = {
            "visible_page_summary": (
                "Order row: order=000000230 | date=May 19, 2023 8:11:51 AM | customer=Ava Brown | billing=Ava Brown | total=93.40 | status=Complete"
            )
        }
        obs = {"url": "http://3.14.148.71:7780/admin/sales/order/"}

        webarena_runner._append_shopping_admin_order_product_price_lines(
            serialized_observation,
            raw_observation=obs,
            goal="Get the product name and discounted price (low to high) of the most recent completed order",
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Admin order product row: order=000000230", summary)
        self.assertIn("product=Rapha Sports Short | price=$35.00", summary)
        self.assertIn("product=Thorpe Track Pant | price=$54.40", summary)
        self.assertIn("product=Mach Street Sweatshirt | price=$62.00", summary)
        self.assertEqual(len(env.page.context.detail_pages), 1)
        self.assertTrue(env.page.context.detail_pages[0].closed)

    def test_admin_dashboard_rows_are_added_to_visible_summary_for_recent_item_goals(self) -> None:
        obs = {
            "goal": "What's the total number of items sold in the most recent 4 orders?",
            "url": "http://3.14.148.71:7780/admin/admin/dashboard/",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "100", "role": {"value": "gridcell"}, "name": {"value": "Sarah Miller"}},
                    {"browsergym_id": "101", "role": {"value": "gridcell"}, "name": {"value": "5"}},
                    {"browsergym_id": "102", "role": {"value": "gridcell"}, "name": {"value": "$194.40"}},
                    {"browsergym_id": "110", "role": {"value": "gridcell"}, "name": {"value": "Grace Nguyen"}},
                    {"browsergym_id": "111", "role": {"value": "gridcell"}, "name": {"value": "4"}},
                    {"browsergym_id": "112", "role": {"value": "gridcell"}, "name": {"value": "$190.00"}},
                ]
            },
            "extra_element_properties": {
                "100": {"bbox": [100, 100, 50, 20]},
                "101": {"bbox": [320, 100, 20, 20]},
                "102": {"bbox": [420, 100, 40, 20]},
                "110": {"bbox": [100, 140, 50, 20]},
                "111": {"bbox": [320, 140, 20, 20]},
                "112": {"bbox": [420, 140, 40, 20]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Dashboard order row: customer=Sarah Miller | items=5 | total=194.40", summary)
        self.assertIn("Dashboard order row: customer=Grace Nguyen | items=4 | total=190.00", summary)

    def test_admin_dashboard_bestseller_rows_are_appended_for_bestselling_goals(self) -> None:
        class FakeBody:
            def inner_text(self):
                return (
                    "Dashboard\n"
                    "Bestsellers\n"
                    "Most Viewed Products\n"
                    "New Customers\n"
                    "Customers\n"
                    "Product\tPrice\tQuantity\n"
                    "Quest Lumaflex™ Band\t$19.00\t6\n"
                    "Sprite Yoga Strap 6 foot\t$14.00\t6\n"
                    "Sprite Stasis Ball 65 cm\t$27.00\t6\n"
                    "Lifetime Sales\n"
                )

        class FakePage:
            def locator(self, selector):
                if selector == "body":
                    return FakeBody()
                raise AssertionError(f"Unexpected selector: {selector}")

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "Magento admin dashboard"}
        obs = {"url": "http://3.14.148.71:7780/admin/admin/dashboard/"}

        webarena_runner._append_shopping_admin_dashboard_report_lines(
            serialized_observation,
            raw_observation=obs,
            goal="What is the top-1 best-selling product in 2022",
            env=FakeEnv(),
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn(
            "Dashboard bestseller row: rank=1 | product=Quest Lumaflex™ Band | price=19.00 | quantity=6",
            summary,
        )
        self.assertIn(
            "Dashboard bestseller row: rank=3 | product=Sprite Stasis Ball 65 cm | price=27.00 | quantity=6",
            summary,
        )

    def test_gitlab_rss_token_is_appended_from_personal_access_tokens_page(self) -> None:
        class FakeBody:
            def inner_text(self):
                return "Personal Access Tokens\nFeed token\nTMN_bBn9Z48qVbUFZV45\n"

        class FakePage:
            def evaluate(self, script):
                return "TMN_bBn9Z48qVbUFZV45"

            def locator(self, selector):
                if selector == "body":
                    return FakeBody()
                raise AssertionError(f"Unexpected selector: {selector}")

            def content(self):
                return (
                    '<html><body><h2>Feed token</h2>'
                    '<input value="TMN_bBn9Z48qVbUFZV45" /></body></html>'
                )

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "GitLab user settings"}
        obs = {"url": "http://3.14.148.71:8023/-/profile/personal_access_tokens"}

        webarena_runner._append_gitlab_rss_token_lines(
            serialized_observation,
            raw_observation=obs,
            goal="Get me my RSS feed token",
            env=FakeEnv(),
        )

        self.assertIn("GitLab RSS feed token: TMN_bBn9Z48qVbUFZV45", serialized_observation["visible_page_summary"])

    def test_extract_gitlab_rss_token_from_html_reads_copy_button_value(self) -> None:
        html_text = (
            '<div><h4>Feed token</h4>'
            '<button title="Copy feed token" data-clipboard-text="TMN_bBn9Z48qVbUFZV45"></button>'
            '</div>'
        )

        self.assertEqual(
            webarena_runner._extract_gitlab_rss_token_from_html(html_text),
            "TMN_bBn9Z48qVbUFZV45",
        )

    def test_admin_bestseller_report_rows_are_appended_for_period_goals(self) -> None:
        class FakeLocator:
            def __init__(self, values=None):
                self._values = list(values or [])

            def fill(self, value):
                return None

            def select_option(self, label=None):
                return None

            def all_inner_texts(self):
                return list(self._values)

        class FakeButton:
            def click(self):
                return None

        class FakeReportPage:
            def __init__(self):
                self.closed = False

            def goto(self, url, wait_until=None):
                return None

            def locator(self, selector):
                if selector in {'[name="report_from"]', '[name="report_to"]', 'select[name="report_period"]'}:
                    return FakeLocator()
                if selector == "#gridProductsSold_table tbody tr":
                    return FakeLocator(
                        [
                            "1/3/23\tOvernight Duffle\t24-WB07\t1",
                            "Impulse Duffle\t24-UB02\t1",
                            "1/16/23\tHawkeye Yoga Short\tMSH05-32-Blue\t1",
                            "1/28/23\tImpulse Duffle\t24-UB02\t1",
                        ]
                    )
                raise AssertionError(f"Unexpected selector: {selector}")

            def get_by_role(self, role, name=None):
                if role == "button" and name == "Refresh":
                    return FakeButton()
                raise AssertionError(f"Unexpected role lookup: {role=} {name=}")

            def wait_for_load_state(self, state=None):
                return None

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.pages = []

            def new_page(self):
                page = FakeReportPage()
                self.pages.append(page)
                return page

        class FakePage:
            def __init__(self):
                self.context = FakeContext()

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "Magento admin dashboard"}
        obs = {"url": "http://3.14.148.71:7780/admin/admin/dashboard/"}
        env = FakeEnv()

        webarena_runner._append_shopping_admin_bestseller_report_lines(
            serialized_observation,
            raw_observation=obs,
            goal="What are the top-3 best-selling product in Jan 2023",
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn(
            "Admin bestseller aggregate row: rank=1 | product=Impulse Duffle | quantity=2",
            summary,
        )
        self.assertIn(
            "Admin bestseller aggregate row: rank=2 | product=Overnight Duffle | quantity=1",
            summary,
        )
        self.assertEqual(len(env.page.context.pages), 1)
        self.assertTrue(env.page.context.pages[0].closed)

    def test_admin_search_term_rows_are_appended_for_search_term_goals(self) -> None:
        class FakeInput:
            def __init__(self, value):
                self._value = value

            def input_value(self):
                return self._value

        class FakeBody:
            def inner_text(self):
                return ""

        class FakeSearchPage:
            def __init__(self, value, popularity):
                self.closed = False
                self._value = value
                self._popularity = popularity

            def goto(self, url, wait_until=None):
                return None

            def locator(self, selector):
                if selector in {'[name="search_query"]', '[name="query_text"]', '[name="search_term"]', '[name="name"]', '#search_query'}:
                    if self._value is None:
                        raise RuntimeError("missing input")
                    return FakeInput(self._value)
                if selector in {'[name="popularity"]', '[name="num_uses"]', '[name="number_of_uses"]'}:
                    return FakeInput(self._popularity)
                if selector == "body":
                    return FakeBody()
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.pages = []
                self._rows = iter(
                    [
                        ("overnight duffle", "2"),
                        ("sprite yoga strap", "7"),
                    ]
                )

            def new_page(self):
                page = FakeSearchPage(*next(self._rows))
                self.pages.append(page)
                return page

        class FakePage:
            def __init__(self):
                self.context = FakeContext()

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {
            "visible_page_summary": (
                "http://3.14.148.71:7780/admin/search/term/edit/id/25/ (clickable)\n"
                "http://3.14.148.71:7780/admin/search/term/edit/id/19/ (clickable)\n"
                "Search Term"
            )
        }
        obs = {"url": "http://3.14.148.71:7780/admin/admin/dashboard/"}
        env = FakeEnv()

        webarena_runner._append_shopping_admin_search_term_lines(
            serialized_observation,
            raw_observation=obs,
            goal="List the top 1 search terms in my store",
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Dashboard search term row: rank=1 | term=sprite yoga strap | uses=7", summary)
        self.assertIn("Dashboard search term row: rank=2 | term=overnight duffle | uses=2", summary)
        self.assertEqual(len(env.page.context.pages), 2)
        self.assertTrue(all(page.closed for page in env.page.context.pages))

    def test_admin_review_count_rows_are_appended_for_review_term_goals(self) -> None:
        class FakeBody:
            def __init__(self, page):
                self._page = page

            def inner_text(self):
                count = self._page._counts.get(self._page._term, 0)
                return f"{count} records found"

        class FakeFilter:
            def __init__(self, page):
                self._page = page

            def fill(self, value):
                self._page._term = value

            def press(self, key):
                return None

        class FakeReviewPage:
            def __init__(self, counts):
                self.closed = False
                self._counts = counts
                self._term = ""

            def goto(self, url, wait_until=None):
                return None

            def wait_for_load_state(self, state):
                return None

            def locator(self, selector):
                if selector == "#reviewGrid_filter_detail":
                    return FakeFilter(self)
                if selector == "body":
                    return FakeBody(self)
                raise AssertionError(f"Unexpected selector: {selector}")

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.pages = []

            def new_page(self):
                page = FakeReviewPage({"disappointed": 6})
                self.pages.append(page)
                return page

        class FakePage:
            def __init__(self):
                self.url = "http://3.14.148.71:7780/admin/admin/dashboard/"
                self.context = FakeContext()

        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.page = FakePage()

        serialized_observation = {"visible_page_summary": "Magento admin dashboard"}
        obs = {"url": "http://3.14.148.71:7780/admin/admin/dashboard/"}
        env = FakeEnv()

        webarena_runner._append_shopping_admin_review_count_lines(
            serialized_observation,
            raw_observation=obs,
            goal='Tell me the the number of reviews that our store received by far that mention term "disappointed"',
            env=env,
        )

        summary = serialized_observation["visible_page_summary"]
        self.assertIn("Admin review mention count: term=disappointed | count=6", summary)
        self.assertEqual(len(env.page.context.pages), 1)
        self.assertTrue(env.page.context.pages[0].closed)

    def test_admin_order_rows_are_added_to_visible_summary_for_relevant_goals(self) -> None:
        obs = {
            "goal": "Get the total payment amount of the last 2 completed orders",
            "url": "http://3.14.148.71:7780/admin/sales/order/",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "100", "role": {"value": "gridcell"}, "name": {"value": "000000230"}},
                    {"browsergym_id": "101", "role": {"value": "gridcell"}, "name": {"value": "May 19, 2023 8:11:51 AM"}},
                    {"browsergym_id": "102", "role": {"value": "gridcell"}, "name": {"value": "Ava Brown"}},
                    {"browsergym_id": "103", "role": {"value": "gridcell"}, "name": {"value": "Ava Brown"}},
                    {"browsergym_id": "104", "role": {"value": "gridcell"}, "name": {"value": "$93.40"}},
                    {"browsergym_id": "105", "role": {"value": "gridcell"}, "name": {"value": "Complete"}},
                    {"browsergym_id": "110", "role": {"value": "gridcell"}, "name": {"value": "000000256"}},
                    {"browsergym_id": "111", "role": {"value": "gridcell"}, "name": {"value": "May 14, 2023 1:22:46 AM"}},
                    {"browsergym_id": "112", "role": {"value": "gridcell"}, "name": {"value": "John Lee"}},
                    {"browsergym_id": "113", "role": {"value": "gridcell"}, "name": {"value": "John Lee"}},
                    {"browsergym_id": "114", "role": {"value": "gridcell"}, "name": {"value": "$89.00"}},
                    {"browsergym_id": "115", "role": {"value": "gridcell"}, "name": {"value": "Complete"}},
                ]
            },
            "extra_element_properties": {
                "100": {"bbox": [171.5, 1032.5, 84.0, 148.0]},
                "101": {"bbox": [355.6, 1032.5, 116.1, 148.0]},
                "102": {"bbox": [471.7, 1032.5, 95.7, 148.0]},
                "103": {"bbox": [567.5, 1032.5, 95.7, 148.0]},
                "104": {"bbox": [663.2, 1032.5, 84.9, 148.0]},
                "105": {"bbox": [863.0, 1032.5, 85.0, 148.0]},
                "110": {"bbox": [171.5, 1180.9, 84.0, 148.0]},
                "111": {"bbox": [355.6, 1180.9, 116.1, 148.0]},
                "112": {"bbox": [471.7, 1180.9, 95.7, 148.0]},
                "113": {"bbox": [567.5, 1180.9, 95.7, 148.0]},
                "114": {"bbox": [663.2, 1180.9, 84.9, 148.0]},
                "115": {"bbox": [863.0, 1180.9, 85.0, 148.0]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Order row: order=000000230", summary)
        self.assertIn("billing=Ava Brown", summary)
        self.assertIn("total=93.40", summary)
        self.assertIn("Order row: order=000000256", summary)

    def test_admin_order_summary_includes_processing_rows_for_non_cancelled_goals(self) -> None:
        obs = {
            "goal": "Get the total payment amount of the last 5 non-cancelled orders",
            "url": "http://3.14.148.71:7780/admin/sales/order/",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "100", "role": {"value": "gridcell"}, "name": {"value": "000000299"}},
                    {"browsergym_id": "101", "role": {"value": "gridcell"}, "name": {"value": "May 31, 2023 2:55:09 AM"}},
                    {"browsergym_id": "102", "role": {"value": "gridcell"}, "name": {"value": "Sarah Miller"}},
                    {"browsergym_id": "103", "role": {"value": "gridcell"}, "name": {"value": "Sarah Miller"}},
                    {"browsergym_id": "104", "role": {"value": "gridcell"}, "name": {"value": "$219.40"}},
                    {"browsergym_id": "105", "role": {"value": "gridcell"}, "name": {"value": "Pending"}},
                    {"browsergym_id": "110", "role": {"value": "gridcell"}, "name": {"value": "000000125"}},
                    {"browsergym_id": "111", "role": {"value": "gridcell"}, "name": {"value": "May 24, 2023 8:28:12 AM"}},
                    {"browsergym_id": "112", "role": {"value": "gridcell"}, "name": {"value": "Matt Baker"}},
                    {"browsergym_id": "113", "role": {"value": "gridcell"}, "name": {"value": "Matt Baker"}},
                    {"browsergym_id": "114", "role": {"value": "gridcell"}, "name": {"value": "$166.40"}},
                    {"browsergym_id": "115", "role": {"value": "gridcell"}, "name": {"value": "Processing"}},
                ]
            },
            "extra_element_properties": {
                "100": {"bbox": [171.5, 1032.5, 84.0, 148.0]},
                "101": {"bbox": [355.6, 1032.5, 116.1, 148.0]},
                "102": {"bbox": [471.7, 1032.5, 95.7, 148.0]},
                "103": {"bbox": [567.5, 1032.5, 95.7, 148.0]},
                "104": {"bbox": [663.2, 1032.5, 84.9, 148.0]},
                "105": {"bbox": [863.0, 1032.5, 85.0, 148.0]},
                "110": {"bbox": [171.5, 1180.9, 84.0, 148.0]},
                "111": {"bbox": [355.6, 1180.9, 116.1, 148.0]},
                "112": {"bbox": [471.7, 1180.9, 95.7, 148.0]},
                "113": {"bbox": [567.5, 1180.9, 95.7, 148.0]},
                "114": {"bbox": [663.2, 1180.9, 84.9, 148.0]},
                "115": {"bbox": [863.0, 1180.9, 85.0, 148.0]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Order row: order=000000125", summary)
        self.assertIn("status=Processing", summary)

    def test_admin_order_summary_prefers_oldest_row_by_parsed_date(self) -> None:
        obs = {
            "goal": "Get the billing name of the oldest complete order",
            "url": "http://3.14.148.71:7780/admin/sales/order/",
            "axtree_object": {
                "nodes": [
                    {"browsergym_id": "100", "role": {"value": "gridcell"}, "name": {"value": "000000230"}},
                    {"browsergym_id": "101", "role": {"value": "gridcell"}, "name": {"value": "May 19, 2023 8:11:51 AM"}},
                    {"browsergym_id": "102", "role": {"value": "gridcell"}, "name": {"value": "Ava Brown"}},
                    {"browsergym_id": "103", "role": {"value": "gridcell"}, "name": {"value": "Ava Brown"}},
                    {"browsergym_id": "104", "role": {"value": "gridcell"}, "name": {"value": "$93.40"}},
                    {"browsergym_id": "105", "role": {"value": "gridcell"}, "name": {"value": "Complete"}},
                    {"browsergym_id": "110", "role": {"value": "gridcell"}, "name": {"value": "000000001"}},
                    {"browsergym_id": "111", "role": {"value": "gridcell"}, "name": {"value": "April 2, 2022 8:15:00 AM"}},
                    {"browsergym_id": "112", "role": {"value": "gridcell"}, "name": {"value": "John Lee"}},
                    {"browsergym_id": "113", "role": {"value": "gridcell"}, "name": {"value": "John Lee"}},
                    {"browsergym_id": "114", "role": {"value": "gridcell"}, "name": {"value": "$88.00"}},
                    {"browsergym_id": "115", "role": {"value": "gridcell"}, "name": {"value": "Complete"}},
                    {"browsergym_id": "120", "role": {"value": "gridcell"}, "name": {"value": "000000032"}},
                    {"browsergym_id": "121", "role": {"value": "gridcell"}, "name": {"value": "January 11, 2023 9:01:00 AM"}},
                    {"browsergym_id": "122", "role": {"value": "gridcell"}, "name": {"value": "Grace Nguyen"}},
                    {"browsergym_id": "123", "role": {"value": "gridcell"}, "name": {"value": "Grace Nguyen"}},
                    {"browsergym_id": "124", "role": {"value": "gridcell"}, "name": {"value": "$196.20"}},
                    {"browsergym_id": "125", "role": {"value": "gridcell"}, "name": {"value": "Complete"}},
                ]
            },
            "extra_element_properties": {
                "100": {"bbox": [171.5, 1032.5, 84.0, 148.0]},
                "101": {"bbox": [355.6, 1032.5, 116.1, 148.0]},
                "102": {"bbox": [471.7, 1032.5, 95.7, 148.0]},
                "103": {"bbox": [567.5, 1032.5, 95.7, 148.0]},
                "104": {"bbox": [663.2, 1032.5, 84.9, 148.0]},
                "105": {"bbox": [863.0, 1032.5, 85.0, 148.0]},
                "110": {"bbox": [171.5, 1180.9, 84.0, 148.0]},
                "111": {"bbox": [355.6, 1180.9, 116.1, 148.0]},
                "112": {"bbox": [471.7, 1180.9, 95.7, 148.0]},
                "113": {"bbox": [567.5, 1180.9, 95.7, 148.0]},
                "114": {"bbox": [663.2, 1180.9, 84.9, 148.0]},
                "115": {"bbox": [863.0, 1180.9, 85.0, 148.0]},
                "120": {"bbox": [171.5, 1329.3, 84.0, 148.0]},
                "121": {"bbox": [355.6, 1329.3, 116.1, 148.0]},
                "122": {"bbox": [471.7, 1329.3, 95.7, 148.0]},
                "123": {"bbox": [567.5, 1329.3, 95.7, 148.0]},
                "124": {"bbox": [663.2, 1329.3, 84.9, 148.0]},
                "125": {"bbox": [863.0, 1329.3, 85.0, 148.0]},
            },
        }

        summary = webarena_runner._build_visible_page_summary(obs)

        self.assertIn("Order row: order=000000001", summary)
        self.assertIn("billing=John Lee", summary)
        self.assertNotIn("Order row: order=000000032", summary)

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
