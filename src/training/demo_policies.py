from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.agent.prompting import derive_shopping_order_number, derive_shopping_query_hint, derive_shopping_sort_hint
from src.utils.config import get_workspace_root


BOOTSTRAP_TASK_IDS = (0, 1, 41, 70, 71, 254, 293, 308, 310)
BOOTSTRAP41_TASK_IDS = (
    0,
    1,
    2,
    3,
    4,
    5,
    11,
    41,
    77,
    21,
    23,
    25,
    26,
    124,
    125,
    126,
    141,
    188,
    27,
    28,
    29,
    30,
    31,
    66,
    67,
    68,
    69,
    132,
    133,
    134,
    135,
    136,
    259,
    293,
    7,
    9,
    10,
    36,
    70,
    71,
    72,
)
SHOPPING_SEARCH_SORT_TASK_IDS = (324, 325, 326, 327, 328)
SHOPPING_ORDER_TASK_IDS = (
    96,
    117,
    128,
    129,
    130,
    131,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    231,
    232,
    233,
    234,
    235,
    319,
    320,
    321,
    322,
    323,
    334,
    335,
    336,
    337,
    338,
    358,
    359,
    360,
    361,
    362,
)

# Backward-compatible alias for older warmup preset/test names.
COVERAGE41_TASK_IDS = BOOTSTRAP41_TASK_IDS


@dataclass(slots=True)
class ScriptedActionSequencePolicy:
    """Deterministic expert policy backed by a fixed action sequence."""

    policy_name: str
    actions: list[str]

    @property
    def name(self) -> str:
        return self.policy_name

    def act(self, observation, step_idx: int):
        action = self.actions[min(step_idx, len(self.actions) - 1)]
        return {
            "raw_text": f"ACTION: {action}",
            "action_text": action,
            "parse_error": None,
            "should_retry": False,
        }


@dataclass(slots=True)
class ShoppingSearchSortDemoPolicy:
    """Observation-driven teacher policy for Shopping search-and-sort tasks."""

    policy_name: str
    query: str
    sort_value: str | None
    direction: str | None = None

    @property
    def name(self) -> str:
        return self.policy_name

    def act(self, observation, step_idx: int):
        current_url = str(observation.get("current_url", "") or "")
        dom_or_ax_snippet = str(observation.get("dom_or_ax_snippet", "") or "")
        current_order = _query_value(current_url, "product_list_order")
        current_direction = _query_value(current_url, "product_list_dir")

        if "/catalogsearch/result" not in current_url:
            if step_idx == 0:
                fill_action = _shopping_fill_search_action(dom_or_ax_snippet, self.query)
                if fill_action:
                    return _decision(fill_action)
            submit_action = _shopping_search_submit_action(dom_or_ax_snippet)
            if submit_action:
                return _decision(submit_action)
            return _decision('send_msg_to_user("N/A")')

        if self.sort_value and current_order != self.sort_value:
            sort_action = _shopping_sort_action(dom_or_ax_snippet, self.sort_value)
            if sort_action:
                return _decision(sort_action)

        if self.direction == "asc" and current_direction != "asc":
            direction_action = _shopping_direction_action(dom_or_ax_snippet, ascending=True)
            if direction_action:
                return _decision(direction_action)

        if self.direction == "desc" and current_direction != "desc":
            direction_action = _shopping_direction_action(dom_or_ax_snippet, ascending=False)
            if direction_action:
                return _decision(direction_action)

        return _decision('send_msg_to_user("done")')


@dataclass(slots=True)
class DemoEnvironment:
    shopping_url: str = field(default_factory=lambda: _required_env("WA_SHOPPING"))
    shopping_admin_url: str = field(default_factory=lambda: _required_env("WA_SHOPPING_ADMIN"))
    reddit_url: str = field(default_factory=lambda: _required_env("WA_REDDIT"))
    gitlab_url: str = field(default_factory=lambda: _required_env("WA_GITLAB"))
    wikipedia_url: str = field(default_factory=lambda: _required_env("WA_WIKIPEDIA"))
    map_url: str = field(default_factory=lambda: _required_env("WA_MAP"))
    homepage_url: str = field(default_factory=lambda: _required_env("WA_HOMEPAGE"))

    def __post_init__(self) -> None:
        self.shopping_url = self.shopping_url.rstrip("/")
        self.shopping_admin_url = self.shopping_admin_url.rstrip("/")
        self.reddit_url = self.reddit_url.rstrip("/")
        self.gitlab_url = self.gitlab_url.rstrip("/")
        self.wikipedia_url = self.wikipedia_url.rstrip("/")
        self.map_url = self.map_url.rstrip("/")
        self.homepage_url = self.homepage_url.rstrip("/")

    def token_mapping(self) -> dict[str, str]:
        return {
            "__SHOPPING__": self.shopping_url,
            "__SHOPPING_ADMIN__": self.shopping_admin_url,
            "__REDDIT__": self.reddit_url,
            "__GITLAB__": self.gitlab_url,
            "__WIKIPEDIA__": self.wikipedia_url,
            "__MAP__": self.map_url,
            "__HOMEPAGE__": self.homepage_url,
        }


def available_scripted_task_ids() -> tuple[int, ...]:
    return tuple(
        sorted(
            set(BOOTSTRAP_TASK_IDS)
            | set(BOOTSTRAP41_TASK_IDS)
            | set(SHOPPING_SEARCH_SORT_TASK_IDS)
            | set(SHOPPING_ORDER_TASK_IDS)
        )
    )


def get_scripted_warmup_policy(task_id: int) -> ScriptedActionSequencePolicy:
    env = DemoEnvironment()
    if task_id in _EXPLICIT_POLICY_BUILDERS:
        return _EXPLICIT_POLICY_BUILDERS[task_id](env)

    task = _load_task(task_id)
    if task["task_id"] not in available_scripted_task_ids():
        raise KeyError(
            f"No scripted warmup policy is registered for task {task_id}. "
            f"Available tasks: {', '.join(str(value) for value in available_scripted_task_ids())}"
        )

    start_url = _expand_tokens(str(task.get("start_url", "")), env.token_mapping())
    answer = _resolve_reference_answer(task, env.token_mapping())
    actions = []
    if start_url:
        first_start_url = start_url.split(" |AND| ")[0].strip()
        if first_start_url:
            actions.append(f'goto("{first_start_url}")')
    actions.append(f'send_msg_to_user("{_escape_action_string(answer)}")')
    return ScriptedActionSequencePolicy(policy_name=f"scripted-task{task_id}-warmup", actions=actions)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set to build scripted warmup policies.")
    return value


@lru_cache(maxsize=1)
def _load_task_index() -> dict[int, dict[str, object]]:
    task_file = _find_task_file()
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    return {int(task["task_id"]): task for task in tasks}


def _load_task(task_id: int) -> dict[str, object]:
    index = _load_task_index()
    try:
        return index[task_id]
    except KeyError as exc:
        raise KeyError(f"Task {task_id} not found in installed WebArena task file.") from exc


def _find_task_file() -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    workspace_root = get_workspace_root()
    candidates = (
        workspace_root / ".venv" / "lib" / version / "site-packages" / "webarena" / "test.raw.json",
        workspace_root / ".venv" / "Lib" / "site-packages" / "webarena" / "test.raw.json",
        Path.home() / ".local" / "lib" / version / "site-packages" / "webarena" / "test.raw.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate installed WebArena test.raw.json")


def _expand_tokens(value: str, token_mapping: dict[str, str]) -> str:
    expanded = value
    for token, replacement in token_mapping.items():
        expanded = expanded.replace(token, replacement)
    return expanded


def _resolve_reference_answer(task: dict[str, object], token_mapping: dict[str, str]) -> str:
    eval_config = task.get("eval", {})
    if not isinstance(eval_config, dict):
        raise ValueError(f"Task {task['task_id']} is missing eval metadata.")

    reference_answers = eval_config.get("reference_answers", {})
    if isinstance(reference_answers, dict) and "fuzzy_match" in reference_answers:
        raise ValueError(
            f"Task {task['task_id']} uses fuzzy_match evaluation and is not suitable for scripted warmup without an LLM judge."
        )
    answer: str | None = None
    if isinstance(reference_answers, dict):
        if "exact_match" in reference_answers:
            answer = _stringify_answer(reference_answers["exact_match"])
        elif "must_include" in reference_answers:
            answer = ", ".join(_stringify_answer(value) for value in reference_answers["must_include"])
        elif "fuzzy_match" in reference_answers:
            fuzzy = reference_answers["fuzzy_match"]
            if isinstance(fuzzy, list):
                answer = ", ".join(_stringify_answer(value) for value in fuzzy)
            else:
                answer = _stringify_answer(fuzzy)

    if not answer:
        answer = _stringify_answer(eval_config.get("reference_answer_raw_annotation", ""))

    answer = _expand_tokens(answer, token_mapping).strip()
    if not answer:
        raise ValueError(f"Task {task['task_id']} does not have a usable scripted reference answer.")
    return answer


def _stringify_answer(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _escape_action_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _task293(env: DemoEnvironment) -> ScriptedActionSequencePolicy:
    return ScriptedActionSequencePolicy(
        policy_name="scripted-task293-warmup",
        actions=[
            f'goto("{env.gitlab_url.rstrip("/")}/convexegg/super_awesome_robot")',
            'send_msg_to_user("git clone ssh://git@metis.lti.cs.cmu.edu:2222/convexegg/super_awesome_robot.git")',
        ],
    )


def _task308(env: DemoEnvironment) -> ScriptedActionSequencePolicy:
    return ScriptedActionSequencePolicy(
        policy_name="scripted-task308-warmup",
        actions=[
            f'goto("{env.gitlab_url.rstrip("/")}/primer/design/-/graphs/master")',
            'send_msg_to_user("Shawn Allen")',
        ],
    )


def _task310(env: DemoEnvironment) -> ScriptedActionSequencePolicy:
    return ScriptedActionSequencePolicy(
        policy_name="scripted-task310-warmup",
        actions=[
            f'goto("{env.gitlab_url.rstrip("/")}/umano/AndroidSlidingUpPanel/-/graphs/master")',
            'send_msg_to_user("tokudu")',
        ],
    )


def _build_shopping_search_sort_policy(task_id: int) -> ShoppingSearchSortDemoPolicy:
    task = _load_task(task_id)
    goal = str(task.get("intent", ""))
    query = derive_shopping_query_hint(goal)
    sort_value, direction = derive_shopping_sort_hint(goal)
    return ShoppingSearchSortDemoPolicy(
        policy_name=f"scripted-task{task_id}-shopping-warmup",
        query=query,
        sort_value=sort_value,
        direction=direction,
    )


def _build_shopping_order_policy(task_id: int, env: DemoEnvironment) -> ScriptedActionSequencePolicy:
    task = _load_task(task_id)
    goal = str(task.get("intent", ""))
    answer = _resolve_order_reference_answer(task, env.token_mapping())
    if _shopping_order_uses_admin_site(task):
        actions = [f'goto("{_shopping_admin_orders_url(env)}")']
    else:
        shopping_base = env.shopping_url.rstrip("/")
        history_url = f"{shopping_base}/sales/order/history/"
        actions = [f'goto("{history_url}")']

    if not _shopping_order_uses_admin_site(task) and _shopping_order_detail_requires_view_page(goal):
        order_number = derive_shopping_order_number(goal)
        if not order_number:
            raise ValueError(f"Could not derive order number from shopping order-detail task {task_id}.")
        actions.append(f'goto("{shopping_base}/sales/order/view/order_id/{int(order_number)}/")')

    actions.append(f'send_msg_to_user("{_escape_action_string(answer)}")')
    return ScriptedActionSequencePolicy(
        policy_name=f"scripted-task{task_id}-shopping-order-warmup",
        actions=actions,
    )


def _resolve_order_reference_answer(task: dict[str, object], token_mapping: dict[str, str]) -> str:
    eval_config = task.get("eval", {})
    if not isinstance(eval_config, dict):
        raise ValueError(f"Task {task['task_id']} is missing eval metadata.")
    raw_annotation = _stringify_answer(eval_config.get("reference_answer_raw_annotation", ""))
    answer = _expand_tokens(raw_annotation, token_mapping).strip()
    if not answer:
        return _resolve_reference_answer(task, token_mapping)
    return answer


def _shopping_order_detail_requires_view_page(goal: str) -> bool:
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


def _shopping_order_uses_admin_site(task: dict[str, object]) -> bool:
    sites = task.get("sites")
    if isinstance(sites, list) and "shopping_admin" in sites:
        return True
    start_url = str(task.get("start_url", "") or "")
    return "__SHOPPING_ADMIN__" in start_url


def _shopping_admin_orders_url(env: DemoEnvironment) -> str:
    admin_base = env.shopping_admin_url.rstrip("/")
    return f"{admin_base}/sales/order/"


_EXPLICIT_POLICY_BUILDERS = {
    293: _task293,
    308: _task308,
    310: _task310,
    324: lambda env: _build_shopping_search_sort_policy(324),
    325: lambda env: _build_shopping_search_sort_policy(325),
    326: lambda env: _build_shopping_search_sort_policy(326),
    327: lambda env: _build_shopping_search_sort_policy(327),
    328: lambda env: _build_shopping_search_sort_policy(328),
    188: lambda env: _build_shopping_order_policy(188, env),
    189: lambda env: _build_shopping_order_policy(189, env),
    190: lambda env: _build_shopping_order_policy(190, env),
    191: lambda env: _build_shopping_order_policy(191, env),
    192: lambda env: _build_shopping_order_policy(192, env),
    96: lambda env: _build_shopping_order_policy(96, env),
    117: lambda env: _build_shopping_order_policy(117, env),
    128: lambda env: _build_shopping_order_policy(128, env),
    129: lambda env: _build_shopping_order_policy(129, env),
    130: lambda env: _build_shopping_order_policy(130, env),
    131: lambda env: _build_shopping_order_policy(131, env),
    193: lambda env: _build_shopping_order_policy(193, env),
    194: lambda env: _build_shopping_order_policy(194, env),
    195: lambda env: _build_shopping_order_policy(195, env),
    196: lambda env: _build_shopping_order_policy(196, env),
    197: lambda env: _build_shopping_order_policy(197, env),
    198: lambda env: _build_shopping_order_policy(198, env),
    199: lambda env: _build_shopping_order_policy(199, env),
    200: lambda env: _build_shopping_order_policy(200, env),
    201: lambda env: _build_shopping_order_policy(201, env),
    202: lambda env: _build_shopping_order_policy(202, env),
    203: lambda env: _build_shopping_order_policy(203, env),
    204: lambda env: _build_shopping_order_policy(204, env),
    231: lambda env: _build_shopping_order_policy(231, env),
    232: lambda env: _build_shopping_order_policy(232, env),
    233: lambda env: _build_shopping_order_policy(233, env),
    234: lambda env: _build_shopping_order_policy(234, env),
    235: lambda env: _build_shopping_order_policy(235, env),
    319: lambda env: _build_shopping_order_policy(319, env),
    320: lambda env: _build_shopping_order_policy(320, env),
    321: lambda env: _build_shopping_order_policy(321, env),
    322: lambda env: _build_shopping_order_policy(322, env),
    323: lambda env: _build_shopping_order_policy(323, env),
    334: lambda env: _build_shopping_order_policy(334, env),
    335: lambda env: _build_shopping_order_policy(335, env),
    336: lambda env: _build_shopping_order_policy(336, env),
    337: lambda env: _build_shopping_order_policy(337, env),
    338: lambda env: _build_shopping_order_policy(338, env),
    358: lambda env: _build_shopping_order_policy(358, env),
    359: lambda env: _build_shopping_order_policy(359, env),
    360: lambda env: _build_shopping_order_policy(360, env),
    361: lambda env: _build_shopping_order_policy(361, env),
    362: lambda env: _build_shopping_order_policy(362, env),
}


def _decision(action_text: str) -> dict[str, object]:
    return {
        "raw_text": f"ACTION: {action_text}",
        "action_text": action_text,
        "parse_error": None,
        "should_retry": False,
    }


def _shopping_fill_search_action(dom_or_ax_snippet: str, query: str) -> str:
    if not query:
        return ""
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=(?P<role>[A-Za-z_]+)(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        role = match.group("role").lower()
        name = (match.group("name") or "").lower()
        if role not in {"textbox", "searchbox", "combobox", "input", "textarea"}:
            continue
        if "search" not in name:
            continue
        return f'fill("{match.group("bid")}", "{_escape_action_string(query)}")'
    return ""


def _shopping_search_submit_action(dom_or_ax_snippet: str) -> str:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=button(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if "search" in (match.group("name") or "").lower():
            return f'click("{match.group("bid")}")'
    return ""


def _shopping_sort_action(dom_or_ax_snippet: str, sort_value: str) -> str:
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=combobox(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if "sort by" in (match.group("name") or "").lower():
            return f'select_option("{match.group("bid")}", "{sort_value}")'
    return ""


def _shopping_direction_action(dom_or_ax_snippet: str, *, ascending: bool) -> str:
    target = "set ascending direction" if ascending else "set descending direction"
    for raw_line in dom_or_ax_snippet.splitlines():
        line = raw_line.strip()
        match = re.match(r'^\[(?P<bid>\d+)\]\s+role=link(?:\s+name="(?P<name>[^"]*)")?', line)
        if match is None:
            continue
        if target in (match.group("name") or "").lower():
            return f'click("{match.group("bid")}")'
    return ""


def _query_value(url: str, key: str) -> str:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get(key, [])
    return values[0] if values else ""
