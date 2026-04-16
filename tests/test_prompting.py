from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.prompting import build_user_prompt
from src.agent.types import NormalizedObservation


class PromptingTests(unittest.TestCase):
    def test_shopping_home_prompt_surfaces_search_actions(self) -> None:
        observation = NormalizedObservation(
            goal='Show me the "chairs" listings by ascending price.',
            current_url="http://3.14.148.71:7770/",
            visible_page_summary="Shopping home page",
            dom_or_ax_snippet=(
                '[274] role=combobox name="Search" clickable\n'
                '[279] role=button name="Search" clickable\n'
                '[1078] role=menuitem name="Video Games" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=0)

        self.assertIn("Shopping home page", prompt)
        self.assertIn('ACTION: fill("274", "chairs")', prompt)
        self.assertIn('ACTION: click("279")', prompt)
        self.assertIn("Do not click products or category links before searching", prompt)

    def test_shopping_home_prompt_tells_model_to_submit_after_query_is_visible(self) -> None:
        observation = NormalizedObservation(
            goal='Show me the "chairs" listings by ascending price.',
            current_url="http://3.14.148.71:7770/",
            visible_page_summary="Shopping home page with active search",
            dom_or_ax_snippet=(
                '[1922] role=option name="chairs" clickable\n'
                '[274] role=combobox name="Search" clickable focused\n'
                '[279] role=button name="Search" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=1)

        self.assertIn("do not fill it again", prompt)
        self.assertIn('click("279")', prompt)

    def test_shopping_category_home_prompt_prefers_category_navigation(self) -> None:
        observation = NormalizedObservation(
            goal="List products from PS4 accessories category by ascending price",
            current_url="http://3.14.148.71:7770/",
            visible_page_summary="Shopping home page",
            dom_or_ax_snippet=(
                '[274] role=combobox name="Search" clickable\n'
                '[279] role=button name="Search"\n'
                '[1077] role=menuitem name="Video Games" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=0)

        self.assertIn("category listing page, not a keyword search", prompt)
        self.assertIn("visible category hierarchy", prompt)
        self.assertIn('ACTION: click("1077")', prompt)
        self.assertNotIn("Do not click products or category links before searching", prompt)

    def test_shopping_home_prompt_prefers_orders_and_returns_for_account_goal(self) -> None:
        observation = NormalizedObservation(
            goal="Get the order number of my most recent cancelled order",
            current_url="http://3.14.148.71:7770/",
            visible_page_summary="Shopping home page",
            dom_or_ax_snippet=(
                '[274] role=combobox name="Search" clickable\n'
                '[279] role=button name="Search"\n'
                '[1864] role=link name="Orders and Returns" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=0)

        self.assertIn("account history, orders, refunds, or purchases", prompt)
        self.assertIn("authenticated order history page", prompt)
        self.assertIn('ACTION: goto("http://3.14.148.71:7770/sales/order/history/")', prompt)
        self.assertNotIn('ACTION: click("279")', prompt)

    def test_shopping_order_history_prompt_answers_latest_cancelled_order_number(self) -> None:
        observation = NormalizedObservation(
            goal="Get the order number of my most recent cancelled order",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            visible_page_summary="My Orders",
            dom_or_ax_snippet=(
                '[1401] role=link name="000000170" clickable\n'
                '[1402] role=link name="View Order" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=1)

        self.assertIn("do not use fill", prompt.lower())
        self.assertIn("table is sorted newest first", prompt)
        self.assertIn('ACTION: send_msg_to_user("<order number>")', prompt)
        self.assertIn('ACTION: send_msg_to_user("000000170")', prompt)

    def test_shopping_order_history_prompt_prefers_click_for_specific_order_detail_goal(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the shipping method for order number 170.",
            current_url="http://3.14.148.71:7770/sales/order/history/",
            visible_page_summary="My Orders",
            dom_or_ax_snippet=(
                '[1401] role=link name="000000170" clickable\n'
                '[1402] role=link name="View Order" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=1)

        self.assertIn("do not use fill", prompt.lower())
        self.assertIn("open its details", prompt)
        self.assertIn('ACTION: click("1401")', prompt)

    def test_shopping_order_detail_prompt_answers_from_items_ordered(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the product names for order number 148.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/148/",
            visible_page_summary="Order detail page",
            dom_or_ax_snippet=(
                '[1353] role=heading name="Order # 000000148"\n'
                '[1372] role=table name="Items Ordered"\n'
                '[1527] role=link name="NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks" clickable\n'
                '[1545] role=link name="Plus Size Lingerie for Women Sexy for Sex Naughty Eyelash Lace Bodysuit" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=2)

        self.assertIn("do not use fill", prompt.lower())
        self.assertIn("Items Ordered", prompt)
        self.assertIn("join the real visible product names with a comma", prompt)
        self.assertIn("NOZE Rustic Coat Rack Wall Mounted Shelf", prompt)

    def test_shopping_order_detail_prompt_answers_from_billing_address(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/view/order_id/178/",
            visible_page_summary="Order detail page",
            dom_or_ax_snippet=(
                '[1559] role=StaticText name="Billing Address"\n'
                '[1560] role=StaticText name="101 S San Mateo Dr"\n'
                '[1561] role=StaticText name="San Mateo, California, 94010"\n'
                '[1562] role=StaticText name="United States"\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=2)

        self.assertIn("Billing Address", prompt)
        self.assertIn("street, city, state, postal code, and country", prompt)
        self.assertIn("send_msg_to_user using the real text from this page", prompt)
        self.assertIn("101 S San Mateo Dr", prompt)

    def test_shopping_order_detail_prompt_accepts_short_order_view_url(self) -> None:
        observation = NormalizedObservation(
            goal="Show me the billing address for order number 00178.",
            current_url="http://3.14.148.71:7770/sales/order/view/178",
            visible_page_summary="Order detail page",
            dom_or_ax_snippet=(
                '[1559] role=StaticText name="Billing Address"\n'
                '[1560] role=StaticText name="101 S San Mateo Dr"\n'
                '[1561] role=StaticText name="San Mateo, California, 94010"\n'
                '[1562] role=StaticText name="United States"\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=2)

        self.assertIn("Billing Address", prompt)
        self.assertIn("do not leave this order page", prompt)

    def test_shopping_results_prompt_surfaces_sort_controls(self) -> None:
        observation = NormalizedObservation(
            goal='Show me the "iphone 12 phone case" listings by name alphabetically.',
            current_url="http://3.14.148.71:7770/catalogsearch/result/?q=iphone+12+phone+case",
            visible_page_summary="Search results",
            dom_or_ax_snippet=(
                '[1390] role=combobox name="Sort By"\n'
                '[1394] role=link name="Set Ascending Direction" clickable\n'
                '[1418] role=link name="Battery Case for iPhone 12/12 Pro" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=1)

        self.assertIn("Shopping listing pages", prompt)
        self.assertIn('ACTION: select_option("1390", "name")', prompt)
        self.assertIn("prefer sorting controls over clicking product links", prompt)

    def test_shopping_category_listing_prompt_surfaces_sort_controls(self) -> None:
        observation = NormalizedObservation(
            goal="List products from PS4 accessories category by ascending price",
            current_url="http://3.14.148.71:7770/video-games/playstation-4/accessories.html",
            visible_page_summary="Category listing",
            dom_or_ax_snippet=(
                '[1390] role=combobox name="Sort By"\n'
                '[1393] role=link name="Set Ascending Direction" clickable\n'
                '[1501] role=link name="PS4 Controller Charger" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=2)

        self.assertIn("Shopping listing pages", prompt)
        self.assertIn('ACTION: select_option("1390", "price")', prompt)
        self.assertIn('ACTION: click("1393")', prompt)

    def test_gitlab_root_prompt_highlights_explore_and_editable_controls(self) -> None:
        observation = NormalizedObservation(
            goal="Find the default branch name for the csvkit repository.",
            current_url="http://3.14.148.71:8023/",
            visible_page_summary="GitLab dashboard",
            dom_or_ax_snippet=(
                '[269] role=searchbox name="Filter by name" clickable\n'
                '[130] role=textbox name="Search GitLab" clickable\n'
                '[200] role=generic name="Projects"\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=0)

        self.assertIn("Editable Controls In Current DOM:", prompt)
        self.assertIn('[130] role=textbox name="Search GitLab" clickable', prompt)
        self.assertNotIn('[200] role=generic name="Projects"', prompt.split("Editable Controls In Current DOM:")[1])
        self.assertIn('goto("http://3.14.148.71:8023/explore")', prompt)
        self.assertIn("Only use fill on editable controls", prompt)
        self.assertIn("For fill and send_msg_to_user, use the real task-specific text", prompt)
        self.assertIn("never copy the field label itself", prompt)
        self.assertIn('fill("250", "Filter by name")', prompt)
        self.assertNotIn("gaming laptop", prompt)
        self.assertNotIn("<text to type>", prompt)

    def test_gitlab_contribution_goal_surfaces_query_hint(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/explore",
            visible_page_summary="GitLab explore",
            dom_or_ax_snippet='[250] role=searchbox name="Filter by name" clickable',
        )

        prompt = build_user_prompt(observation, step_idx=1)

        self.assertIn("repository search query should be close to 'administrate'", prompt)
        self.assertIn('ACTION: fill("250", "administrate")', prompt)
        self.assertIn("Do not swap the bid and the query text.", prompt)

    def test_gitlab_filtered_result_prompt_surfaces_click_hint(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/explore?sort=name_asc&name=administrate&sort=name_asc",
            visible_page_summary="GitLab explore filtered results",
            dom_or_ax_snippet=(
                '[250] role=searchbox name="Filter by name" clickable focused\n'
                '[1115] role=link name="thoughtbot, inc. / administrate" clickable\n'
            ),
        )

        prompt = build_user_prompt(observation, step_idx=2)

        self.assertIn("The exact repository result is already visible in the current DOM.", prompt)
        self.assertIn('ACTION: click("1115")', prompt)
        self.assertIn("Click it now instead of filling again.", prompt)

    def test_gitlab_repo_page_prompt_tells_model_not_to_return_to_explore(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the thoughtbot/administrate project",
            current_url="http://3.14.148.71:8023/thoughtbot/administrate",
            visible_page_summary="GitLab repository page",
            dom_or_ax_snippet='[291] role=link name="Repository" clickable',
            previous_errors=[
                'TimeoutError: Locator.click: Timeout 500ms exceeded.'
            ],
        )

        prompt = build_user_prompt(observation, step_idx=3)

        self.assertIn("already on the correct GitLab repository page", prompt)
        self.assertIn("Do not return to /explore", prompt)
        self.assertIn("treat the click as successful", prompt)
        self.assertIn('/-/graphs/main', prompt)
        self.assertIn('/-/graphs/master', prompt)

    def test_gitlab_graph_page_prompt_tells_model_to_answer_directly(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the Pytorch GAN project",
            current_url="http://3.14.148.71:8023/eriklindernoren/PyTorch-GAN/-/graphs/master",
            visible_page_summary="Contributors graph",
            dom_or_ax_snippet='[514] role=heading name="Commits to master"',
        )

        prompt = build_user_prompt(observation, step_idx=4)

        self.assertIn("already on the GitLab contributors graph page", prompt)
        self.assertIn('ACTION: send_msg_to_user("<top contributor name>")', prompt)
        self.assertIn("not a placeholder like John Doe", prompt)
        self.assertIn("Do not navigate back or append another /-/graphs path.", prompt)

    def test_gitlab_graph_404_prompt_tells_model_to_switch_branch_once(self) -> None:
        observation = NormalizedObservation(
            goal="Tell me who has made the most contributions, in terms of number of commits, to the csvkit project",
            current_url="http://3.14.148.71:8023/wireservice/csvkit/-/graphs/main",
            visible_page_summary="404\nPage Not Found",
            dom_or_ax_snippet='[9] role=heading name="Page Not Found"',
        )

        prompt = build_user_prompt(observation, step_idx=4)

        self.assertIn("returned Page Not Found", prompt)
        self.assertIn("Do not append another /-/graphs segment", prompt)
        self.assertIn('goto("http://3.14.148.71:8023/wireservice/csvkit/-/graphs/master")', prompt)

    def test_retry_hint_appears_after_invalid_fill_target(self) -> None:
        observation = NormalizedObservation(
            goal="Find the default branch name for the administrate repository.",
            current_url="http://3.14.148.71:8023/explore",
            visible_page_summary="Explore GitLab",
            dom_or_ax_snippet='[250] role=searchbox name="Filter by name" clickable',
            last_action_error='Error: Locator.fill: Error: Element is not an <input>, <textarea> or [contenteditable] element',
        )

        prompt = build_user_prompt(observation, step_idx=1)

        self.assertIn("Retry Hint:", prompt)
        self.assertIn("different currently visible editable bid", prompt)


if __name__ == "__main__":
    unittest.main()
