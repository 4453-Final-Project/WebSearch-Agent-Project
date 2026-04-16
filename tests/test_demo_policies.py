from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.demo_policies import (
    DemoEnvironment,
    ShoppingSearchSortDemoPolicy,
    _build_shopping_order_policy,
    _shopping_order_detail_requires_view_page,
)
from src.training import demo_policies


class DemoPoliciesTests(unittest.TestCase):
    def test_shopping_search_sort_policy_uses_search_then_sort(self) -> None:
        policy = ShoppingSearchSortDemoPolicy(
            policy_name="shopping-demo",
            query="chairs",
            sort_value="price",
            direction="asc",
        )
        root_observation = {
            "current_url": "http://3.14.148.71:7770/",
            "dom_or_ax_snippet": (
                '[274] role=combobox name="Search" clickable\n'
                '[279] role=button name="Search" clickable\n'
            ),
        }
        results_observation = {
            "current_url": "http://3.14.148.71:7770/catalogsearch/result/?q=chairs",
            "dom_or_ax_snippet": (
                '[1390] role=combobox name="Sort By"\n'
                '[1394] role=link name="Set Ascending Direction" clickable\n'
            ),
        }
        sorted_observation = {
            "current_url": "http://3.14.148.71:7770/catalogsearch/result/index/?q=chairs&product_list_order=price",
            "dom_or_ax_snippet": '[1394] role=link name="Set Ascending Direction" clickable\n',
        }

        self.assertEqual(policy.act(root_observation, 0)["action_text"], 'fill("274", "chairs")')
        self.assertEqual(policy.act(root_observation, 1)["action_text"], 'click("279")')
        self.assertEqual(policy.act(results_observation, 2)["action_text"], 'select_option("1390", "price")')
        self.assertEqual(policy.act(sorted_observation, 3)["action_text"], 'click("1394")')

    def test_shopping_search_sort_policy_stops_after_relevance_search(self) -> None:
        policy = ShoppingSearchSortDemoPolicy(
            policy_name="shopping-demo",
            query="Canon photo printer",
            sort_value=None,
            direction=None,
        )
        results_observation = {
            "current_url": "http://3.14.148.71:7770/catalogsearch/result/?q=Canon+photo+printer",
            "dom_or_ax_snippet": '[1390] role=combobox name="Sort By"\n',
        }

        self.assertEqual(policy.act(results_observation, 2)["action_text"], 'send_msg_to_user("done")')

    def test_order_detail_task_detection_prefers_view_page(self) -> None:
        self.assertTrue(_shopping_order_detail_requires_view_page("Show me the shipping method for order number 187."))
        self.assertTrue(_shopping_order_detail_requires_view_page("Show me the billing address for order number 00178."))
        self.assertFalse(_shopping_order_detail_requires_view_page("Get the order number of my most recent cancelled order"))
        self.assertFalse(_shopping_order_detail_requires_view_page("Show me the order statuses for order number 170 and 189."))

    def test_expanded_shopping_order_tasks_are_scriptable(self) -> None:
        for task_id in (96, 117, 128, 193, 202, 204, 319, 334):
            self.assertIn(task_id, demo_policies.SHOPPING_ORDER_TASK_IDS)
            self.assertIn(task_id, demo_policies._EXPLICIT_POLICY_BUILDERS)

    def test_admin_order_tasks_route_to_admin_orders_grid(self) -> None:
        env = DemoEnvironment(
            shopping_url="http://example-shopping",
            shopping_admin_url="http://example-admin/admin",
            reddit_url="http://example-reddit",
            gitlab_url="http://example-gitlab",
            wikipedia_url="http://example-wiki",
            map_url="http://example-map",
            homepage_url="http://example-home",
        )

        policy = _build_shopping_order_policy(128, env)

        self.assertEqual(policy.actions[0], 'goto("http://example-admin/admin/sales/order/")')


if __name__ == "__main__":
    unittest.main()
