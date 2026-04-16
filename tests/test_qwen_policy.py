"""Unit tests for the Task 3 Qwen policy flow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.fake_backend import FakeBackend
from src.agent.qwen_policy import QwenPolicy
from src.agent.types import NormalizedObservation, OpenTab, PolicyConfig


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
            'send_msg_to_user("Emma Lopez, 101 S San Mateo Dr, San Mateo, California, 94010, United States")',
        )
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Do not use fill on Shopping order-detail pages", backend.prompts[1])

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
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("must copy the visible Order Date text", backend.prompts[1])

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
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("must include every visible product title", backend.prompts[1])
        self.assertIn("russound 5b45w 4", backend.prompts[1].lower())

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
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("must copy the visible Billing Address details", backend.prompts[1])
        self.assertIn("101 s san mateo dr", backend.prompts[1].lower())

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
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Do not leave a Shopping order-detail page", backend.prompts[1])

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
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Do not leave a Shopping order-detail page", backend.prompts[1])

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
