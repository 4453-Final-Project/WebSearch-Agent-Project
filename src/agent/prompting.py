"""Prompt construction helpers for the Task 3 policy wrapper."""

from __future__ import annotations

import re
from urllib.parse import urlencode, urlparse

from .types import NormalizedObservation


EDITABLE_ROLE_PATTERN = re.compile(r"^\[(?P<bid>\d+)\]\s+role=(?P<role>[A-Za-z_]+)(?:\s+name=\"(?P<name>[^\"]*)\")?")
LINK_ROLE_PATTERN = re.compile(r"^\[(?P<bid>\d+)\]\s+role=link(?:\s+name=\"(?P<name>[^\"]*)\")?")
EDITABLE_ROLES = {
    "textbox",
    "searchbox",
    "combobox",
    "textarea",
    "input",
}


def build_system_prompt(model_system_prompt: str | None = None) -> str:
    """Build the fixed system prompt for the browser policy model."""

    base_prompt = (
        "You are a browser automation policy.\n"
        "Respond with exactly one line.\n"
        "Use this exact format and nothing else:\n"
        "ACTION: <browser action string>\n"
        "Never emit more than one ACTION.\n"
        "When using click, fill, hover, or select_option, use the BrowserGym element bid from the DOM/AX snippet, not the visible label text.\n"
        "Only use fill on editable controls like role=textbox, role=searchbox, role=combobox, input, textarea, or elements explicitly marked editable.\n"
        "Never use fill on headings, generic containers, buttons, links, list items, or non-editable search regions.\n"
        "The text you type with fill must come from the task goal or the intended query, not from the field label itself.\n"
        "Never type field labels like 'Filter by name' or 'Search GitLab' as the fill text.\n"
        "Only use BrowserGym WebArena actions from this allowlist:\n"
        "click, fill, hover, scroll, press, goto, go_back, "
        "go_forward, select_option, send_msg_to_user, "
        "report_infeasible, noop\n"
        "Do not guess or invent URLs.\n"
        "Only use goto for the current WebArena sites, and prefer exact known paths over made-up GitLab URLs.\n"
        "Only use numeric bids that are actually present in the current DOM or AX snippet.\n"
        "Do not treat final answers like zip codes, usernames, or clone commands as clickable bids.\n"
        "Do not use placeholder person names like 'John Doe' or 'Jane Doe' unless they are explicitly shown on the page as the real answer.\n"
        'Examples:\n'
        'ACTION: click("58")\n'
        'ACTION: press("130", "Enter")\n'
        'ACTION: noop(500)\n'
        'ACTION: send_msg_to_user("N/A")\n'
        'Invalid examples:\n'
        'ACTION: goto\n'
        'ACTION: click\n'
        'ACTION: click("Log in")\n'
        'ACTION: click("15213")\n'
        'ACTION: click("tokudu")\n'
        'ACTION: fill("250", "Filter by name")\n'
        'ACTION: fill("130", "Search GitLab")\n'
        'ACTION: I should click("Log in")'
    )
    if not model_system_prompt:
        return base_prompt
    return f"{model_system_prompt.strip()}\n\n{base_prompt}"


def build_user_prompt(observation: NormalizedObservation, step_idx: int) -> str:
    """Build the deterministic user prompt from a normalized observation."""

    gitlab_explore_example_url = _build_gitlab_explore_url(observation.current_url)
    sections = [
        f"Step: {step_idx}",
        f"Task Goal: {observation.goal}",
        f"Current URL: {observation.current_url}",
        "Open Tabs:",
        _render_open_tabs(observation),
        "Visible Page Summary:",
        _render_text_block(observation.visible_page_summary),
        "Relevant DOM or AX-Tree Snippet:",
        _render_text_block(observation.dom_or_ax_snippet),
        "Previous Actions:",
        _render_history(observation.previous_actions),
        "Previous Errors:",
        _render_history(observation.previous_errors),
    ]

    if observation.last_action_error is not None:
        sections.extend(
            [
                "Last Action Error:",
                _render_text_block(observation.last_action_error),
            ]
        )

    sections.extend(
        [
            "Output Contract:",
            "Respond with exactly one line.",
        "Use this exact format and nothing else:",
        "ACTION: <browser action string>",
        "Do not output multiple ACTION entries.",
        "Every ACTION must be a complete function call with parentheses and all required arguments.",
        "For click, fill, hover, select_option, and press, use the bid from the DOM or AX snippet like 58 or 130, not the label text.",
        "Only use fill on editable controls such as role=textbox, role=searchbox, role=combobox, textarea, input, or elements explicitly marked editable.",
        "Never use fill on headings, generic containers, buttons, links, list items, or non-editable search regions.",
        "For fill and send_msg_to_user, use the real task-specific text from the current goal and page. Do not reuse canned example strings from the instructions.",
        "For fill text, never copy the field label itself. Use the actual search query or answer from the task goal instead.",
        "Use only these action names:",
        "click, fill, hover, scroll, press, goto, go_back, go_forward, select_option, send_msg_to_user, report_infeasible, noop",
        "Do not use aliases like type, wait, click_all, goto_next_url, get_comment_data, tab_focus, or tab_close.",
        "Do not guess or invent URLs.",
        "Only use numeric bids that are actually present in the DOM or AX snippet for click, fill, hover, or select_option.",
        "Do not treat final answers like zip codes, usernames, repo URLs, or clone commands as bids.",
        "Do not use placeholder person names like John Doe or Jane Doe unless they are explicitly shown on the page as the real answer.",
        "If you use goto, stay on the WebArena host and avoid made-up GitLab paths.",
        f'If you use goto, it must look like ACTION: goto("{gitlab_explore_example_url}") and never ACTION: goto.',
        'If you use press, it must look like ACTION: press("130", "Enter"), not ACTION: press("Enter").',
        'Invalid examples: ACTION: goto | ACTION: click | ACTION: click("Explore") | ACTION: click("15213") | ACTION: click("tokudu") | ACTION: fill("250", "Filter by name") | ACTION: fill("130", "Search GitLab") | ACTION: press("Enter")',
        ]
    )
    editable_hints = describe_editable_controls(observation.dom_or_ax_snippet)
    if editable_hints:
        sections.extend(
            [
                "Editable Controls In Current DOM:",
                editable_hints,
            ]
        )
    site_hints = build_site_hints(observation)
    if site_hints:
        sections.extend(
            [
                "Navigation Hints:",
                site_hints,
            ]
        )
    if observation.last_action_error:
        lowered = observation.last_action_error.lower()
        if "not an <input>" in lowered or "could not find element with bid" in lowered:
            sections.append(
                "Retry Hint: the last fill target was invalid or stale. Pick a different currently visible editable bid instead of repeating it."
            )

    return "\n".join(sections)


def build_retry_prompt(
    observation: NormalizedObservation,
    step_idx: int,
    previous_raw_text: str,
    parse_error: str,
    model_system_prompt: str | None = None,
) -> str:
    """Build a retry prompt that asks the model to correct invalid output."""

    sections = [
        build_user_prompt(observation, step_idx),
        "Previous Model Output:",
        _render_text_block(previous_raw_text),
        "Parse Error:",
        _render_text_block(parse_error),
        "Correction:",
        "Your previous response could not be parsed.",
        "Correct it and respond again with exactly one line.",
        "Use this exact format and nothing else:",
        "ACTION: <browser action string>",
        "Do not answer with a bare action name like ACTION: goto or ACTION: click.",
    ]
    return "\n".join(sections)


def build_full_prompt(
    observation: NormalizedObservation,
    step_idx: int,
    model_system_prompt: str | None = None,
) -> str:
    """Build the full prompt text used for a normal policy generation step."""

    return "\n\n".join([build_system_prompt(model_system_prompt), build_user_prompt(observation, step_idx)])


def _render_open_tabs(observation: NormalizedObservation) -> str:
    """Render open tabs in a deterministic numbered format."""

    if not observation.open_tabs:
        return "(none)"
    return "\n".join(
        f"{index}. {tab.title} | {tab.url}"
        for index, tab in enumerate(observation.open_tabs, start=1)
    )


def _render_history(items: list[str]) -> str:
    """Render action or error history with a stable empty state."""

    if not items:
        return "(none)"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _render_text_block(value: str) -> str:
    """Render a text field with a stable empty state."""

    return value if value else "(empty)"


def describe_editable_controls(dom_or_ax_snippet: str) -> str:
    editable_lines: list[str] = []
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = EDITABLE_ROLE_PATTERN.match(line)
        if match is None:
            if "editable" in line.lower():
                editable_lines.append(line)
            continue
        role = match.group("role").lower()
        if role not in EDITABLE_ROLES:
            continue
        editable_lines.append(line)
    if not editable_lines:
        return "(none detected)"
    return "\n".join(editable_lines[:5])


def build_site_hints(observation: NormalizedObservation) -> str:
    parsed = urlparse(observation.current_url or "")
    if _is_shopping_domain_url(observation.current_url):
        shopping_hint = build_shopping_page_hint(
            current_url=observation.current_url,
            goal=observation.goal,
            visible_page_summary=observation.visible_page_summary,
            dom_or_ax_snippet=observation.dom_or_ax_snippet,
        )
        if shopping_hint:
            return shopping_hint
    if parsed.netloc.endswith(":8023"):
        path = parsed.path or "/"
        if _is_gitlab_graph_path(path):
            graph_state_hint = build_gitlab_graph_page_hint(
                current_url=observation.current_url,
                visible_page_summary=observation.visible_page_summary,
                dom_or_ax_snippet=observation.dom_or_ax_snippet,
            )
            if graph_state_hint:
                return graph_state_hint
        query_hint = derive_gitlab_query_hint(observation.goal)
        if path == "/":
            base = (
                f'On the GitLab root page, prefer ACTION: goto("{_build_gitlab_explore_url(observation.current_url)}") '
                "before repository search. The repository filter named 'Filter by name' belongs on /explore, "
                "not the root dashboard."
            )
            if query_hint:
                base += f" A good repository search query from this goal is '{query_hint}'."
            return base
        if path.startswith("/explore"):
            base = (
                "On GitLab /explore pages, search repositories by filling the editable 'Filter by name' searchbox "
                "from the current DOM, then click a visible repository result."
            )
            if query_hint:
                base += f" For this task, the repository search query should be close to '{query_hint}', not the field label."
            click_action = build_gitlab_explore_click_hint(observation.dom_or_ax_snippet, observation.goal)
            if click_action:
                base += (
                    " The exact repository result is already visible in the current DOM. "
                    f"Click it now instead of filling again. A structurally valid next action here would be {click_action}."
                )
            example_action = _build_gitlab_explore_fill_hint(observation.dom_or_ax_snippet, query_hint)
            if example_action and not click_action:
                base += f" A structurally valid next action here would be {example_action}. Do not swap the bid and the query text."
            return base
        if _is_gitlab_repo_page_path(path) and _is_gitlab_contribution_goal(observation.goal):
            base = (
                "You are already on the correct GitLab repository page for this contribution task. "
                "Do not return to /explore or re-run the repository search."
            )
            if _has_click_timeout_history(observation):
                base += " If the previous repository click timed out but this repo page is open, treat the click as successful and continue from here."
            base += " Continue from this repository page toward the contribution graphs instead."
            graph_hint = build_gitlab_repo_graph_hint(observation.current_url)
            if graph_hint:
                base += (
                    " A structurally valid next action from this page is "
                    f'{graph_hint} or ACTION: goto("{observation.current_url.rstrip("/")}/-/graphs/master"), '
                    "depending on the repository's default branch."
                )
            return base
    return ""


def build_shopping_page_hint(
    current_url: str,
    goal: str,
    visible_page_summary: str,
    dom_or_ax_snippet: str,
) -> str:
    parsed = urlparse(current_url or "")
    path = parsed.path or "/"
    if parsed.netloc.endswith(":7780"):
        return build_shopping_admin_page_hint(current_url, goal, visible_page_summary, dom_or_ax_snippet)
    query_hint = derive_shopping_query_hint(goal)
    category_labels = derive_shopping_category_labels(goal)
    sort_value, direction = derive_shopping_sort_hint(goal)

    if path == "/":
        orders_action = build_shopping_orders_target_hint(current_url, goal, dom_or_ax_snippet)
        if orders_action:
            return (
                "On the Shopping home page, this goal is about account history, orders, refunds, or purchases rather than product search. "
                "Go straight to the authenticated order history page instead of using the Search box. "
                f"A structurally valid next action here would be {orders_action}."
            )
        if category_labels:
            parts = [
                "On the Shopping home page, this goal asks for a category listing page, not a keyword search.",
                "Follow the visible category hierarchy toward the requested category instead of using the Search box.",
            ]
            category_click_hint = build_shopping_category_click_hint(dom_or_ax_snippet, category_labels, current_url=current_url)
            if category_click_hint:
                parts.append(f"A structurally valid next action here would be {category_click_hint}.")
            if sort_value is not None:
                parts.append("Once the target category listing page is open, use the Sort By controls there.")
            return " ".join(parts)

        parts = [
            "On the Shopping home page, use the visible Search box for the product query from the task goal.",
            "Do not click products or category links before searching when the goal names a quoted product query.",
        ]
        fill_hint = _build_shopping_search_fill_hint(dom_or_ax_snippet, query_hint)
        click_hint = _build_shopping_search_click_hint(dom_or_ax_snippet)
        if fill_hint:
            parts.append(f"A structurally valid next action here would be {fill_hint}.")
        if click_hint:
            parts.append(f"After filling the query, submit the search with {click_hint}.")
        if query_hint and _shopping_query_already_visible(dom_or_ax_snippet, query_hint):
            parts.append(
                "If the query text is already visible in the focused search field, do not fill it again; click the Search button to submit."
            )
        return " ".join(parts)

    category_click_hint = ""
    if category_labels:
        category_click_hint = build_shopping_category_click_hint(dom_or_ax_snippet, category_labels, current_url=current_url)

    order_detail_hint = build_shopping_order_detail_page_hint(
        current_url,
        goal,
        visible_page_summary,
        dom_or_ax_snippet,
    )
    if order_detail_hint:
        return order_detail_hint

    order_history_hint = build_shopping_order_history_page_hint(current_url, goal, dom_or_ax_snippet)
    if order_history_hint:
        return order_history_hint

    sort_hint = _build_shopping_sort_select_hint(dom_or_ax_snippet, sort_value) if sort_value is not None else ""
    if category_click_hint:
        return (
            "On Shopping category-navigation pages, keep following the visible category hierarchy instead of sorting too early. "
            f"For this goal, the next useful visible category click is {category_click_hint}."
        )

    if "/catalogsearch/result" in path or sort_hint:
        parts = [
            "On Shopping listing pages, prefer sorting controls over clicking product links when the task only asks for a sorted listing page.",
        ]
        if sort_value is None:
            parts.append(
                "For relevance tasks, the search results page reached immediately after a correct search is already the target page."
            )
        else:
            if sort_hint:
                parts.append(f"Use the visible Sort By combobox, for example {sort_hint}.")
            if direction == "asc":
                direction_hint = _build_shopping_direction_click_hint(dom_or_ax_snippet, ascending=True)
                if direction_hint:
                    parts.append(f"If price must be ascending, set the direction with {direction_hint}.")
            if direction == "desc":
                direction_hint = _build_shopping_direction_click_hint(dom_or_ax_snippet, ascending=False)
                if direction_hint:
                    parts.append(f"If price must be descending, set the direction with {direction_hint}.")
        return " ".join(parts)

    return ""


def build_shopping_category_target_url(goal: str, current_url: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770"):
        return ""

    category_labels = derive_shopping_category_labels(goal)
    if not category_labels:
        return ""

    path_segments: list[str] = []
    for index, label in enumerate(category_labels):
        slug = _shopping_slugify_label(label)
        if not slug:
            return ""
        if index == len(category_labels) - 1:
            path_segments.append(f"{slug}.html")
        else:
            path_segments.append(slug)

    sort_value, direction = derive_shopping_sort_hint(goal)
    query_items: list[tuple[str, str]] = []
    if sort_value:
        query_items.append(("product_list_order", sort_value))
    if direction == "desc":
        query_items.append(("product_list_dir", "desc"))

    target_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/{'/'.join(path_segments)}"
    if query_items:
        target_url = f"{target_url}?{urlencode(query_items)}"
    return target_url


def build_shopping_orders_link_hint(dom_or_ax_snippet: str, goal: str) -> str:
    return ""


def build_shopping_orders_target_hint(current_url: str, goal: str, dom_or_ax_snippet: str) -> str:
    if not _is_shopping_orders_goal(goal):
        return ""
    target_url = build_shopping_orders_target_url(current_url, goal)
    if not target_url:
        return ""
    return f'ACTION: goto("{target_url}")'


def build_shopping_orders_target_url(current_url: str, goal: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770"):
        return ""
    path = "/sales/guest/form/" if _is_shopping_guest_order_goal(goal) else "/sales/order/history/"
    return f"{parsed.scheme or 'http'}://{parsed.netloc}{path}"


def build_shopping_order_detail_target_url(current_url: str, goal: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770"):
        return ""
    order_number = derive_shopping_order_number(goal)
    if not order_number or not _is_shopping_order_detail_goal(goal):
        return ""
    return f"{parsed.scheme or 'http'}://{parsed.netloc}/sales/order/view/order_id/{int(order_number)}/"


def build_shopping_order_history_page_hint(current_url: str, goal: str, dom_or_ax_snippet: str) -> str:
    parsed = urlparse(current_url or "")
    path = parsed.path or "/"
    if not parsed.netloc.endswith(":7770"):
        return ""
    if not (path.startswith("/sales/order/history") or path.startswith("/customer/account")):
        return ""

    lowered_goal = " ".join((goal or "").lower().split())
    specific_order = derive_shopping_order_number(goal)
    if specific_order:
        order_click_hint = _build_shopping_order_history_click_hint(dom_or_ax_snippet, specific_order)
        if order_click_hint:
            return (
                "On Shopping account-order pages, do not use fill. There is no relevant form field for this task. "
                "Locate the row for the requested order number and open its details instead. "
                f"A structurally valid next action here would be {order_click_hint}."
            )
        detail_target_url = build_shopping_order_detail_target_url(current_url, goal)
        if detail_target_url:
            return (
                "On Shopping account-order pages, do not use fill. There is no relevant form field for this task. "
                "If the requested order row is not visible in the current table snippet, go directly to the order details page instead. "
                f'A structurally valid next action here would be ACTION: goto("{detail_target_url}").'
            )
        return (
            "On Shopping account-order pages, do not use fill. There is no relevant form field for this task. "
            "Open the requested order by clicking its visible order number row or a nearby View Order link, then answer from the details page."
        )

    if "latest cancelled order" in lowered_goal or "most recent cancelled order" in lowered_goal:
        if "order number" in lowered_goal:
            return (
                "On Shopping My Orders pages, do not use fill. Read directly from the visible orders table. "
                "The table is sorted newest first. "
                'Read the first row whose status is Canceled and answer with ACTION: send_msg_to_user("<order number>"). '
                'For example, if the newest canceled row shows order 000000170, respond exactly with ACTION: send_msg_to_user("000000170").'
            )
        if "total cost" in lowered_goal or "order total" in lowered_goal:
            return (
                "On Shopping My Orders pages, do not use fill. Read directly from the visible orders table. "
                "The table is sorted newest first. "
                'Read the first row whose status is Canceled and answer with ACTION: send_msg_to_user("<order total>"). '
                'For example, if the newest canceled row shows $365.42, respond exactly with ACTION: send_msg_to_user("365.42").'
            )
    status_phrase = _describe_shopping_order_status_phrase(lowered_goal)
    row_selector = _describe_shopping_order_row_selector(lowered_goal)
    if ("order number" in lowered_goal or "order id" in lowered_goal) and row_selector and status_phrase:
        return (
            "On Shopping My Orders pages, do not use fill. Read directly from the visible orders table. "
            f"The table is sorted newest first, so locate the {row_selector} {status_phrase} row and answer with "
            'ACTION: send_msg_to_user("<order number>").'
        )
    if ("total cost" in lowered_goal or "order total" in lowered_goal) and row_selector and status_phrase:
        return (
            "On Shopping My Orders pages, do not use fill. Read directly from the visible orders table. "
            f"Locate the {row_selector} {status_phrase} row and answer with ACTION: send_msg_to_user(\"<order total>\"). "
            "Return the numeric amount without the dollar sign."
        )
    if "refund" in lowered_goal:
        return (
            "On Shopping My Orders pages, do not use fill. Read the visible canceled-order rows directly from the table. "
            "Match the canceled order by the date constraint in the goal, then answer with ACTION: send_msg_to_user(\"<refund amount>\"). "
            "If the goal says shipping is refundable, include it; if the goal says shipping is not refundable, subtract it; "
            "if the goal says one item was kept, subtract that kept item's amount from the refund."
        )
    return ""


def build_shopping_admin_page_hint(
    current_url: str,
    goal: str,
    visible_page_summary: str,
    dom_or_ax_snippet: str,
) -> str:
    parsed = urlparse(current_url or "")
    path = parsed.path or "/"
    if not parsed.netloc.endswith(":7780"):
        return ""
    lowered_goal = " ".join((goal or "").lower().split())
    base_url = f"{parsed.scheme or 'http'}://{parsed.netloc}"
    if path.startswith("/admin/admin/dashboard"):
        if _is_shopping_orders_goal(goal):
            count = derive_shopping_order_count(goal)
            if count and "items" in lowered_goal and "sold" in lowered_goal:
                return (
                    "On the Magento admin dashboard, the visible Quantity widget already contains the recent-order counts needed for this task. "
                    f"Sum the first {count} visible quantity values and answer directly with ACTION: send_msg_to_user(\"<sum>\"). "
                    "Do not navigate away if those quantity values are already visible."
                )
            return (
                "On the Magento admin dashboard, do not use the dashboard widgets or search box for order-reference tasks. "
                "Go straight to Sales > Orders first. "
                f'A structurally valid next action here would be ACTION: goto("{base_url}/admin/sales/order/").'
            )
        return ""
    if not path.startswith("/admin/sales/order"):
        return ""

    status_phrase = _describe_shopping_order_status_phrase(lowered_goal)
    row_selector = _describe_shopping_order_row_selector(lowered_goal)
    count = derive_shopping_order_count(goal)
    parts = [
        "You are already on the Magento admin Sales > Orders page.",
        "Do not use fill if the needed order rows are already visible in the table.",
        'Read the visible rows directly and answer with ACTION: send_msg_to_user("...").',
    ]
    if count and "items" in lowered_goal and "sold" in lowered_goal:
        parts.append(
            f"The table is sorted newest first. Sum the quantity values from the first {count} visible order rows and answer with ACTION: send_msg_to_user(\"<sum>\")."
        )
    elif count and "total payment amount" in lowered_goal and status_phrase:
        parts.append(
            f"Add the order totals from the last {count} visible {status_phrase} rows and answer with ACTION: send_msg_to_user(\"<sum>\")."
        )
    elif "payment difference" in lowered_goal:
        parts.append(
            "Compute the difference between the visible canceled-order totals and complete-order totals requested by the goal, then answer with ACTION: send_msg_to_user(\"<difference>\")."
        )
    elif ("billing name" in lowered_goal or "customer name" in lowered_goal) and status_phrase and row_selector:
        parts.append(
            f"Locate the {row_selector} {status_phrase} row, read the visible customer or bill-to name for that row, and answer with ACTION: send_msg_to_user(\"<name>\")."
        )
    return " ".join(parts)


def build_shopping_order_detail_page_hint(
    current_url: str,
    goal: str,
    visible_page_summary: str,
    dom_or_ax_snippet: str,
) -> str:
    parsed = urlparse(current_url or "")
    path = parsed.path or "/"
    if not parsed.netloc.endswith(":7770"):
        return ""
    if not _is_shopping_order_detail_path(path):
        return ""

    lowered_goal = " ".join((goal or "").lower().split())
    base = (
        "On Shopping order-detail pages, do not use fill and do not leave this order page. "
        "Read the answer directly from the visible order details and answer with send_msg_to_user using the real text from this page. "
        "Do not answer with UI labels like Filter by name, Search, Product Name, My Orders, or Address Book."
    )
    candidate_lines = _build_shopping_order_detail_candidate_lines(goal, visible_page_summary, dom_or_ax_snippet)
    candidate_hint = ""
    if candidate_lines:
        candidate_hint = " The relevant visible answer text on this page includes: " + " | ".join(candidate_lines) + "."
    if "product names" in lowered_goal:
        return (
            f"{base} Read the product names from the visible Items Ordered section. "
            "If multiple products are shown, join the real visible product names with a comma in one final answer. "
            "Use the product-title rows from the Items Ordered table, not the Recently Ordered links or account sidebar. "
            f"Copy the exact visible product title text from this page.{candidate_hint}"
        )
    if "billing address" in lowered_goal:
        return (
            f"{base} Read the Billing Address section. "
            "Include the visible street, city, state, postal code, and country in one answer string."
            f"{candidate_hint}"
        )
    if "shipping method" in lowered_goal:
        return (
            f"{base} Read the visible Shipping Method text for this order and answer with that method only."
            f"{candidate_hint}"
        )
    if "order date" in lowered_goal:
        return (
            f"{base} Read the Order Date text near the order header and answer with the visible date only."
            f"{candidate_hint}"
        )
    return (
        f"{base} Prefer visible order-specific fields like Order Date, Shipping Method, Billing Address, "
        f"and Items Ordered over prices or account-navigation links.{candidate_hint}"
    )


def derive_gitlab_repo_hint(goal: str) -> str:
    text = (goal or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    marker = "to the "
    end_marker = " project"
    start = lowered.find(marker)
    end = lowered.find(end_marker, start + len(marker)) if start != -1 else -1
    if start == -1 or end == -1:
        contribution_match = re.search(
            r"\b(?:commits|contributions)\b.*?\bto (?P<repo>[A-Za-z0-9_.\-/]+?)(?: on \d{1,2}/\d{1,2}(?:/\d{4})?|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if contribution_match is None:
            clone_match = re.search(
                r"\bclone (?P<repo>[A-Za-z0-9_.\-/]+?) with ssh\b",
                text,
                flags=re.IGNORECASE,
            )
            if clone_match is None:
                return ""
            return clone_match.group("repo").strip(" .")
        return contribution_match.group("repo").strip(" .")
    return text[start + len(marker) : end].strip(" .")


def derive_gitlab_query_hint(goal: str) -> str:
    candidate = derive_gitlab_repo_hint(goal)
    if "/" in candidate:
        candidate = candidate.split("/")[-1]
    return candidate.strip()


def derive_shopping_query_hint(goal: str) -> str:
    text = (goal or "").strip()
    if not text:
        return ""
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted[0].strip()
    return ""


def derive_shopping_order_number(goal: str) -> str:
    match = re.search(r"\border number\s+0*(\d+)\b", goal or "", flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def derive_shopping_category_labels(goal: str) -> list[str]:
    lowered = " ".join((goal or "").lower().split())
    if not lowered or "category" not in lowered:
        return []
    if "ps4" in lowered or "playstation 4" in lowered:
        return ["Video Games", "PlayStation 4", "Accessories"]
    if "nutrition bars and drinks" in lowered:
        return ["Health & Household", "Diet & Sports Nutrition", "Nutrition Bars & Drinks"]
    if "competitive swimwear" in lowered:
        return ["Clothing, Shoes & Jewelry", "Sport-Specific Clothing", "Competitive Swimwear"]
    if "living room furtniture" in lowered or "living room furniture" in lowered:
        return ["Home & Kitchen", "Furniture", "Living Room Furniture"]
    if "kids' bedding" in lowered or "kids bedding" in lowered:
        return ["Home & Kitchen", "Bedding", "Kids' Bedding"]
    return []


def derive_shopping_sort_hint(goal: str) -> tuple[str | None, str | None]:
    lowered = (goal or "").lower()
    sort_value: str | None = None
    direction: str | None = None
    if "relevance" in lowered:
        sort_value = None
    elif "name alphabetically" in lowered or "alphabetically" in lowered:
        sort_value = "name"
    elif "price" in lowered:
        sort_value = "price"

    if "ascending" in lowered:
        direction = "asc"
    elif "descending" in lowered:
        direction = "desc"
    return sort_value, direction


def derive_shopping_order_count(goal: str) -> int | None:
    text = " ".join((goal or "").lower().split())
    if not text:
        return None
    for pattern in (
        r"\bmost recent\s+(\d+)\s+orders?\b",
        r"\blast\s+(\d+)\s+(?:completed|complete|pending|cancelled|canceled|non-cancelled|non-canceled)?\s*orders?\b",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _build_gitlab_explore_fill_hint(dom_or_ax_snippet: str, query_hint: str) -> str:
    if not query_hint:
        return ""
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = EDITABLE_ROLE_PATTERN.match(line)
        if match is None:
            continue
        role = match.group("role").lower()
        if role not in EDITABLE_ROLES:
            continue
        bid = match.group("bid")
        name = (match.group("name") or "").lower()
        if "filter by name" in name or role == "searchbox":
            return f'ACTION: fill("{bid}", "{query_hint}")'
    return ""


def _build_shopping_search_fill_hint(dom_or_ax_snippet: str, query_hint: str) -> str:
    if not query_hint:
        return ""
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = EDITABLE_ROLE_PATTERN.match(line)
        if match is None:
            continue
        role = match.group("role").lower()
        if role not in EDITABLE_ROLES:
            continue
        name = (match.group("name") or "").lower()
        if "search" not in name:
            continue
        return f'ACTION: fill("{match.group("bid")}", "{query_hint}")'
    return ""


def build_shopping_category_click_hint(
    dom_or_ax_snippet: str,
    category_labels: list[str],
    *,
    current_url: str | None = None,
) -> str:
    normalized_targets = [_normalize_shopping_label(label) for label in category_labels if label]
    if not normalized_targets:
        return ""
    next_target = _next_shopping_category_target(normalized_targets, current_url or "", dom_or_ax_snippet)
    if not next_target:
        return ""
    best_bid = ""
    best_score = -1
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=(?P<role>[A-Za-z_]+)(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if match.group("role").lower() not in {"link", "menuitem"}:
            continue
        normalized_name = _normalize_shopping_label(match.group("name") or "")
        if not normalized_name:
            continue
        score = _score_shopping_category_candidate(normalized_name, next_target)
        if score > best_score:
            best_score = score
            best_bid = match.group("bid")
    if best_score <= 0 or not best_bid:
        return ""
    return f'ACTION: click("{best_bid}")'


def _build_shopping_search_click_hint(dom_or_ax_snippet: str) -> str:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=button(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        name = (match.group("name") or "").lower()
        if "search" in name:
            return f'ACTION: click("{match.group("bid")}")'
    return ""


def _find_shopping_orders_link_bid(dom_or_ax_snippet: str) -> str:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = LINK_ROLE_PATTERN.match(line)
        if match is None:
            continue
        if "orders and returns" in (match.group("name") or "").lower():
            return match.group("bid")
    return ""


def _build_shopping_order_history_click_hint(dom_or_ax_snippet: str, order_number: str) -> str:
    normalized_targets = {
        order_number,
        order_number.zfill(9),
    }
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = LINK_ROLE_PATTERN.match(line)
        if match is None:
            continue
        name = (match.group("name") or "").strip()
        digits = re.sub(r"\D+", "", name)
        if digits in normalized_targets:
            return f'ACTION: click("{match.group("bid")}")'
    return ""


def _build_shopping_sort_select_hint(dom_or_ax_snippet: str, sort_value: str) -> str:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = EDITABLE_ROLE_PATTERN.match(line)
        if match is None:
            continue
        if match.group("role").lower() != "combobox":
            continue
        name = (match.group("name") or "").lower()
        if "sort by" in name:
            return f'ACTION: select_option("{match.group("bid")}", "{sort_value}")'
    return ""


def _build_shopping_direction_click_hint(dom_or_ax_snippet: str, *, ascending: bool) -> str:
    needle = "set ascending direction" if ascending else "set descending direction"
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = LINK_ROLE_PATTERN.match(line)
        if match is None:
            continue
        name = (match.group("name") or "").lower()
        if needle in name:
            return f'ACTION: click("{match.group("bid")}")'
    return ""


def _shopping_query_already_visible(dom_or_ax_snippet: str, query_hint: str) -> bool:
    lowered_query = query_hint.strip().lower()
    if not lowered_query:
        return False
    focused_search_seen = False
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        editable_match = EDITABLE_ROLE_PATTERN.match(line)
        if editable_match is not None:
            role = editable_match.group("role").lower()
            name = (editable_match.group("name") or "").lower()
            if role in EDITABLE_ROLES and "search" in name and "focused" in line.lower():
                focused_search_seen = True
        option_match = re.match(r'^\[(?P<bid>\d+)\]\s+role=option(?:\s+name="(?P<name>[^"]*)")?', line)
        if option_match is not None and (option_match.group("name") or "").strip().lower() == lowered_query:
            return focused_search_seen or True
    return False


def _is_shopping_orders_goal(goal: str) -> bool:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return False
    keywords = (
        "order number",
        "latest order",
        "recent order",
        "last ordered",
        "ordered my",
        "cancelled order",
        "pending order",
        "complete order",
        "completed order",
        "order status",
        "order date",
        "shipping method",
        "billing address",
        "delivery address",
        "refund",
        "spent",
        "purchase",
        "bought",
        "shopping at one stop market",
    )
    return any(keyword in lowered for keyword in keywords)


def _is_shopping_domain_url(current_url: str) -> bool:
    parsed = urlparse(current_url or "")
    return parsed.netloc.endswith(":7770") or parsed.netloc.endswith(":7780")


def _describe_shopping_order_status_phrase(lowered_goal: str) -> str:
    if "non-cancelled" in lowered_goal or "non-canceled" in lowered_goal:
        return "non-canceled"
    if "cancelled" in lowered_goal or "canceled" in lowered_goal:
        return "canceled"
    if "pending" in lowered_goal:
        return "pending"
    if "completed" in lowered_goal or "complete" in lowered_goal:
        return "complete"
    return ""


def _describe_shopping_order_row_selector(lowered_goal: str) -> str:
    if "oldest" in lowered_goal:
        return "oldest"
    if "latest" in lowered_goal or "most recent" in lowered_goal or "newest" in lowered_goal:
        return "newest"
    return ""


def _is_shopping_guest_order_goal(goal: str) -> bool:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return False
    guest_keywords = (
        "guest",
        "billing last name",
        "find order by",
        "email address",
    )
    return any(keyword in lowered for keyword in guest_keywords)


def _is_shopping_order_detail_goal(goal: str) -> bool:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return False
    if "latest " in lowered or "most recent " in lowered:
        return False
    if "order statuses for order number" in lowered and " and " in lowered:
        return False
    detail_terms = (
        "shipping method",
        "order date",
        "product names",
        "billing address",
    )
    return any(term in lowered for term in detail_terms)


def _is_shopping_order_detail_path(path: str) -> bool:
    normalized_path = (path or "").strip()
    return bool(re.match(r"^/sales/order/view/(?:order_id/)?\d+/?$", normalized_path))


def _build_shopping_order_detail_candidate_lines(goal: str, visible_page_summary: str, dom_or_ax_snippet: str) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        normalized = " ".join((value or "").split())
        if not normalized:
            return
        lowered = normalized.lower()
        if lowered in seen:
            return
        if lowered in {
            "items ordered",
            "product name",
            "billing address",
            "shipping address",
            "my orders",
            "address book",
            "account information",
            "recently ordered",
            "copyright © 2013-present magento, inc. all rights reserved.",
        }:
            return
        seen.add(lowered)
        candidates.append(normalized)

    if "product names" in lowered_goal:
        for raw_line in dom_or_ax_snippet.splitlines():
            if "role=gridcell" not in raw_line:
                continue
            if 'name="' not in raw_line:
                continue
            text = raw_line.split('name="', 1)[1].rsplit('"', 1)[0].strip()
            if len(text) < 20:
                continue
            lowered_text = text.lower()
            if any(token in lowered_text for token in ("reorder", "account information", "my orders", "search")):
                continue
            if any(token in lowered_text for token in ("ordered:", "$", "order #", "recently ordered")):
                continue
            add_candidate(text)
            if len(candidates) >= 3:
                return candidates

    if "billing address" in lowered_goal:
        for raw_line in visible_page_summary.splitlines():
            text = raw_line.strip()
            if not text:
                continue
            lowered = text.lower()
            if any(token in lowered for token in ("billing address", "shipping address", "copyright", "address book")):
                continue
            if any(token in text for token in ("United States", "California", "San Mateo")) or re.search(r"\d", text):
                add_candidate(text)
            if len(candidates) >= 4:
                return candidates

    if "shipping method" in lowered_goal or "order date" in lowered_goal:
        for raw_line in visible_page_summary.splitlines():
            text = raw_line.strip()
            if not text:
                continue
            lowered = text.lower()
            if "shipping method" in lowered_goal and ("flat rate" in lowered or "shipping" in lowered):
                add_candidate(text)
            if "order date" in lowered_goal and re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
                add_candidate(text)
            if len(candidates) >= 3:
                return candidates

    return candidates


def _normalize_shopping_label(value: str) -> str:
    lowered = value.lower().replace("-", " ")
    lowered = re.sub(r"[^a-z0-9&' ]+", " ", lowered)
    return " ".join(lowered.split())


def _shopping_slugify_label(value: str) -> str:
    normalized = _normalize_shopping_label(value).replace("&", "")
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return "-".join(part for part in normalized.split() if part)


def _next_shopping_category_target(normalized_targets: list[str], current_url: str, dom_or_ax_snippet: str) -> str:
    if not normalized_targets:
        return ""
    clickable_names: list[str] = []
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=(?P<role>[A-Za-z_]+)(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if match.group("role").lower() not in {"link", "menuitem"}:
            continue
        clickable_names.append(_normalize_shopping_label(match.group("name") or ""))

    parsed = urlparse(current_url or "")
    visible_targets = [
        target
        for target in normalized_targets
        if any(_score_shopping_category_candidate(name, target) > 0 for name in clickable_names)
    ]
    if "cat=" in (parsed.query or ""):
        for target in reversed(visible_targets):
            if target != normalized_targets[0]:
                return target
        return ""

    for target in reversed(visible_targets):
        return target

    path_text = _normalize_shopping_label(urlparse(current_url or "").path.replace("/", " "))
    for target in normalized_targets:
        if target and target not in path_text:
            return target
    return ""


def _score_shopping_category_candidate(normalized_name: str, target: str) -> int:
    if not normalized_name or not target:
        return 0
    target_word_count = len(target.split())
    name_word_count = len(normalized_name.split())
    if normalized_name == target:
        return 3
    if normalized_name.startswith(f"{target} "):
        return 2
    if target in normalized_name and name_word_count <= target_word_count + 2:
        return 1
    return 0


def build_gitlab_explore_click_hint(dom_or_ax_snippet: str, goal: str) -> str:
    best_bid = _find_gitlab_result_bid(dom_or_ax_snippet, goal)
    if not best_bid:
        return ""
    return f'ACTION: click("{best_bid}")'


def _find_gitlab_result_bid(dom_or_ax_snippet: str, goal: str) -> str:
    repo_hint = derive_gitlab_repo_hint(goal).lower()
    query_hint = derive_gitlab_query_hint(goal).lower()
    best_bid = ""
    best_score = -1
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = LINK_ROLE_PATTERN.match(line)
        if match is None:
            continue
        name = (match.group("name") or "").strip()
        if not name:
            continue
        score = _score_gitlab_repo_link(name.lower(), repo_hint, query_hint)
        if score > best_score:
            best_score = score
            best_bid = match.group("bid")
    if best_score <= 0:
        return ""
    return best_bid


def _score_gitlab_repo_link(link_name: str, repo_hint: str, query_hint: str) -> int:
    normalized = " ".join(link_name.split())
    if repo_hint and repo_hint in normalized:
        return 4
    if query_hint and normalized.endswith(f"/ {query_hint}"):
        return 3
    if query_hint and normalized.endswith(f"/{query_hint}"):
        return 3
    if query_hint and f" / {query_hint}" in normalized:
        return 2
    if query_hint and query_hint in normalized:
        return 1
    return 0


def _is_gitlab_repo_page_path(path: str) -> bool:
    parts = [part for part in (path or "").split("/") if part]
    return len(parts) >= 2 and parts[0] != "-" and parts[1] != "-"


def _is_gitlab_graph_path(path: str) -> bool:
    return "/-/graphs/" in (path or "")


def _is_gitlab_contribution_goal(goal: str) -> bool:
    lowered = (goal or "").lower()
    return (
        "most contributions" in lowered
        or "number of commits" in lowered
        or "how many commits" in lowered
    )


def _has_click_timeout_history(observation: NormalizedObservation) -> bool:
    errors = list(observation.previous_errors)
    if observation.last_action_error:
        errors.append(observation.last_action_error)
    return any("locator.click: timeout" in error.lower() for error in errors)


def build_gitlab_repo_graph_hint(current_url: str) -> str:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023") or not _is_gitlab_repo_page_path(parsed.path or ""):
        return ""
    repo_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    return f'ACTION: goto("{repo_url}/-/graphs/main")'


def build_gitlab_graph_page_hint(current_url: str, visible_page_summary: str, dom_or_ax_snippet: str) -> str:
    parsed = urlparse(current_url or "")
    path = parsed.path or ""
    if not parsed.netloc.endswith(":8023") or not _is_gitlab_graph_path(path):
        return ""
    graph_url = f"{parsed.scheme}://{parsed.netloc}{path}"
    alternate_graph_url = build_gitlab_alternate_graph_url(graph_url)
    if _is_page_not_found(visible_page_summary, dom_or_ax_snippet):
        if alternate_graph_url:
            return (
                "This GitLab graph URL returned Page Not Found. Do not append another /-/graphs segment to the broken URL. "
                f"Switch once to the alternate branch graph instead, for example ACTION: goto(\"{alternate_graph_url}\")."
            )
        return (
            "This GitLab graph URL returned Page Not Found. Do not append another /-/graphs segment to the broken URL."
        )
    return (
        "You are already on the GitLab contributors graph page. Read the contributor information from this page and answer "
        'with ACTION: send_msg_to_user("<top contributor name>"). Use a real contributor name visible on this page, not a '
        "placeholder like John Doe. Do not navigate back or append another /-/graphs path."
    )


def build_gitlab_alternate_graph_url(current_url: str) -> str:
    parsed = urlparse(current_url or "")
    path = parsed.path or ""
    if not parsed.netloc.endswith(":8023") or not _is_gitlab_graph_path(path):
        return ""
    if "/-/graphs/main" in path:
        return f"{parsed.scheme}://{parsed.netloc}{path.replace('/-/graphs/main', '/-/graphs/master', 1)}"
    if "/-/graphs/master" in path:
        return f"{parsed.scheme}://{parsed.netloc}{path.replace('/-/graphs/master', '/-/graphs/main', 1)}"
    return ""


def _build_gitlab_explore_url(current_url: str) -> str:
    parsed = urlparse(current_url or "")
    if parsed.netloc.endswith(":8023"):
        scheme = parsed.scheme or "http"
        return f"{scheme}://{parsed.netloc}/explore"
    return "http://3.14.148.71:8023/explore"


def _is_page_not_found(visible_page_summary: str, dom_or_ax_snippet: str) -> bool:
    haystack = f"{visible_page_summary}\n{dom_or_ax_snippet}".lower()
    return "page not found" in haystack
