from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .demo_policies import (
    BOOTSTRAP41_TASK_IDS,
    SHOPPING_ORDER_EXACT_TASK_IDS,
    SHOPPING_ORDER_TASK_IDS,
    SHOPPING_SEARCH_SORT_TASK_IDS,
)


SHOPPING_ORDER_FUZZY_TASK_IDS = tuple(sorted(set(SHOPPING_ORDER_TASK_IDS) - set(SHOPPING_ORDER_EXACT_TASK_IDS)))
SHOPPING_EXACT_TASK_IDS = tuple(sorted(set(SHOPPING_ORDER_EXACT_TASK_IDS) | set(SHOPPING_SEARCH_SORT_TASK_IDS)))
SHOPPING_FULL_TASK_IDS = tuple(sorted(set(SHOPPING_ORDER_TASK_IDS) | set(SHOPPING_SEARCH_SORT_TASK_IDS)))
SHOPPING_FULL_FUZZY_TASK_IDS = tuple(sorted(set(SHOPPING_FULL_TASK_IDS) - set(SHOPPING_EXACT_TASK_IDS)))
BOOTSTRAP41_SHOPPING_ADMIN_TASK_IDS = (0, 1, 2, 3, 4, 5, 11, 41, 77)
BOOTSTRAP41_SHOPPING_TASK_IDS = (21, 23, 25, 26, 124, 125, 126, 141, 188)
BOOTSTRAP41_REDDIT_TASK_IDS = (27, 28, 29, 30, 31, 66, 67, 68, 69)
BOOTSTRAP41_GITLAB_TASK_IDS = (132, 133, 134, 135, 136, 259, 293)
BOOTSTRAP41_MAP_TASK_IDS = (7, 9, 10, 36, 70, 71, 72)
BOOTSTRAP44_SHOPPING_ADMIN_TASK_IDS = tuple(sorted(BOOTSTRAP41_SHOPPING_ADMIN_TASK_IDS + (12, 13)))
BOOTSTRAP44_SHOPPING_TASK_IDS = tuple(sorted(BOOTSTRAP41_SHOPPING_TASK_IDS + (144,)))
BOOTSTRAP44_TASK_IDS = tuple(
    sorted(
        set(BOOTSTRAP44_SHOPPING_ADMIN_TASK_IDS)
        | set(BOOTSTRAP44_SHOPPING_TASK_IDS)
        | set(BOOTSTRAP41_REDDIT_TASK_IDS)
        | set(BOOTSTRAP41_GITLAB_TASK_IDS)
        | set(BOOTSTRAP41_MAP_TASK_IDS)
    )
)
WEB_MIX88_TASK_IDS = tuple(sorted(set(SHOPPING_FULL_TASK_IDS) | set(BOOTSTRAP41_TASK_IDS)))
WEB_MIX91_TASK_IDS = tuple(sorted(set(SHOPPING_FULL_TASK_IDS) | set(BOOTSTRAP44_TASK_IDS)))


@dataclass(frozen=True, slots=True)
class TaskFamilySpec:
    name: str
    description: str
    task_ids: tuple[int, ...]
    recommended_warmup_count: int
    recommended_holdout_count: int
    requires_openai_judge: bool = False


@dataclass(frozen=True, slots=True)
class TaskSplit:
    family_name: str
    task_ids: tuple[int, ...]
    warmup_task_ids: tuple[int, ...]
    grpo_task_ids: tuple[int, ...]
    holdout_task_ids: tuple[int, ...]
    eval_task_ids: tuple[int, ...]
    split_seed: int

    @property
    def training_task_ids(self) -> tuple[int, ...]:
        return self.warmup_task_ids + self.grpo_task_ids

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["training_task_ids"] = list(self.training_task_ids)
        payload["training_task_count"] = len(self.training_task_ids)
        payload["holdout_task_count"] = len(self.holdout_task_ids)
        payload["eval_task_count"] = len(self.eval_task_ids)
        return payload


TASK_FAMILY_SPECS = {
    "shopping_search_sort": TaskFamilySpec(
        name="shopping_search_sort",
        description="Five exact-match shopping search and sort tasks.",
        task_ids=tuple(sorted(SHOPPING_SEARCH_SORT_TASK_IDS)),
        recommended_warmup_count=2,
        recommended_holdout_count=1,
    ),
    "shopping_order": TaskFamilySpec(
        name="shopping_order",
        description="Twenty-seven exact-match shopping order-history and order-detail tasks.",
        task_ids=tuple(sorted(SHOPPING_ORDER_EXACT_TASK_IDS)),
        recommended_warmup_count=8,
        recommended_holdout_count=5,
    ),
    "shopping_order_full": TaskFamilySpec(
        name="shopping_order_full",
        description="Forty-three scripted shopping order-history and order-detail tasks, including fuzzy-judged items.",
        task_ids=tuple(sorted(SHOPPING_ORDER_TASK_IDS)),
        recommended_warmup_count=14,
        recommended_holdout_count=8,
        requires_openai_judge=True,
    ),
    "shopping_exact": TaskFamilySpec(
        name="shopping_exact",
        description="Combined exact-match shopping order plus search/sort family.",
        task_ids=SHOPPING_EXACT_TASK_IDS,
        recommended_warmup_count=14,
        recommended_holdout_count=6,
    ),
    "shopping_full": TaskFamilySpec(
        name="shopping_full",
        description="Combined scripted shopping family with all order tasks plus search/sort, including fuzzy-judged items.",
        task_ids=SHOPPING_FULL_TASK_IDS,
        recommended_warmup_count=16,
        recommended_holdout_count=8,
        requires_openai_judge=True,
    ),
    "bootstrap41": TaskFamilySpec(
        name="bootstrap41",
        description="Cross-site scripted WebArena family spanning shopping, admin, reddit, gitlab, and map tasks.",
        task_ids=tuple(sorted(BOOTSTRAP41_TASK_IDS)),
        recommended_warmup_count=15,
        recommended_holdout_count=8,
    ),
    "bootstrap44": TaskFamilySpec(
        name="bootstrap44",
        description="Expanded cross-site scripted WebArena family adding validated admin review-count and storefront spend tasks to bootstrap41.",
        task_ids=BOOTSTRAP44_TASK_IDS,
        recommended_warmup_count=16,
        recommended_holdout_count=8,
    ),
    "web_mix88": TaskFamilySpec(
        name="web_mix88",
        description="Mixed shopping-plus-cross-site scripted family spanning shopping, admin, reddit, gitlab, and map tasks.",
        task_ids=WEB_MIX88_TASK_IDS,
        recommended_warmup_count=30,
        recommended_holdout_count=16,
        requires_openai_judge=True,
    ),
    "web_mix91": TaskFamilySpec(
        name="web_mix91",
        description="Mixed shopping-plus-cross-site scripted family spanning shopping, admin, reddit, gitlab, and map tasks, plus validated admin/spend expansions.",
        task_ids=WEB_MIX91_TASK_IDS,
        recommended_warmup_count=31,
        recommended_holdout_count=16,
        requires_openai_judge=True,
    ),
}

KNOWN_BENCHMARK_BLOCKERS = {
    "shopping_exact": (131,),
    "shopping_full": (204,),
    "bootstrap41": (124, 133, 141),
    "bootstrap44": (124, 133, 141),
    "web_mix88": (124, 133, 141, 204),
    "web_mix91": (124, 133, 141, 204),
}


def list_task_family_names() -> list[str]:
    return sorted(TASK_FAMILY_SPECS)


def get_task_family_spec(name: str) -> TaskFamilySpec:
    try:
        return TASK_FAMILY_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown task family {name!r}. Valid families: {', '.join(list_task_family_names())}") from exc


def family_requires_openai_judge(name: str) -> bool:
    return get_task_family_spec(name).requires_openai_judge


def get_family_known_benchmark_blockers(name: str) -> tuple[int, ...]:
    get_task_family_spec(name)
    return KNOWN_BENCHMARK_BLOCKERS.get(name, ())


def get_family_task_groups(name: str) -> dict[str, tuple[int, ...]]:
    spec = get_task_family_spec(name)
    if name == "shopping_order_full":
        return {
            "judge_free": tuple(sorted(SHOPPING_ORDER_EXACT_TASK_IDS)),
            "judge_gated": SHOPPING_ORDER_FUZZY_TASK_IDS,
        }
    if name == "shopping_full":
        return {
            "judge_free": SHOPPING_EXACT_TASK_IDS,
            "judge_gated": SHOPPING_FULL_FUZZY_TASK_IDS,
        }
    if name == "bootstrap41":
        return {
            "judge_free": tuple(sorted(BOOTSTRAP41_TASK_IDS)),
            "judge_gated": (),
            "site_shopping_admin": BOOTSTRAP41_SHOPPING_ADMIN_TASK_IDS,
            "site_shopping": BOOTSTRAP41_SHOPPING_TASK_IDS,
            "site_reddit": BOOTSTRAP41_REDDIT_TASK_IDS,
            "site_gitlab": BOOTSTRAP41_GITLAB_TASK_IDS,
            "site_map": BOOTSTRAP41_MAP_TASK_IDS,
        }
    if name == "bootstrap44":
        return {
            "judge_free": BOOTSTRAP44_TASK_IDS,
            "judge_gated": (),
            "site_shopping_admin": BOOTSTRAP44_SHOPPING_ADMIN_TASK_IDS,
            "site_shopping": BOOTSTRAP44_SHOPPING_TASK_IDS,
            "site_reddit": BOOTSTRAP41_REDDIT_TASK_IDS,
            "site_gitlab": BOOTSTRAP41_GITLAB_TASK_IDS,
            "site_map": BOOTSTRAP41_MAP_TASK_IDS,
        }
    if name == "web_mix88":
        return {
            "judge_free": tuple(sorted(set(WEB_MIX88_TASK_IDS) - set(SHOPPING_FULL_FUZZY_TASK_IDS))),
            "judge_gated": SHOPPING_FULL_FUZZY_TASK_IDS,
            "site_shopping_full": SHOPPING_FULL_TASK_IDS,
            "site_shopping_admin": BOOTSTRAP41_SHOPPING_ADMIN_TASK_IDS,
            "site_shopping": BOOTSTRAP41_SHOPPING_TASK_IDS,
            "site_reddit": BOOTSTRAP41_REDDIT_TASK_IDS,
            "site_gitlab": BOOTSTRAP41_GITLAB_TASK_IDS,
            "site_map": BOOTSTRAP41_MAP_TASK_IDS,
        }
    if name == "web_mix91":
        return {
            "judge_free": tuple(sorted(set(WEB_MIX91_TASK_IDS) - set(SHOPPING_FULL_FUZZY_TASK_IDS))),
            "judge_gated": SHOPPING_FULL_FUZZY_TASK_IDS,
            "site_shopping_full": SHOPPING_FULL_TASK_IDS,
            "site_shopping_admin": BOOTSTRAP44_SHOPPING_ADMIN_TASK_IDS,
            "site_shopping": BOOTSTRAP44_SHOPPING_TASK_IDS,
            "site_reddit": BOOTSTRAP41_REDDIT_TASK_IDS,
            "site_gitlab": BOOTSTRAP41_GITLAB_TASK_IDS,
            "site_map": BOOTSTRAP41_MAP_TASK_IDS,
        }
    return {
        "judge_free": spec.task_ids,
        "judge_gated": (),
    }


def build_task_split(
    task_ids: tuple[int, ...] | list[int],
    *,
    family_name: str,
    warmup_count: int,
    holdout_count: int,
    split_seed: int,
) -> TaskSplit:
    deduped = tuple(sorted(dict.fromkeys(task_ids)))
    if len(deduped) < 3:
        raise ValueError("Task splits require at least three tasks.")
    if warmup_count <= 0:
        raise ValueError("warmup_count must be positive.")
    if holdout_count <= 0:
        raise ValueError("holdout_count must be positive.")
    if warmup_count + holdout_count >= len(deduped):
        raise ValueError("warmup_count + holdout_count must leave at least one GRPO task.")

    shuffled = list(deduped)
    random.Random(split_seed).shuffle(shuffled)

    holdout_task_ids = tuple(sorted(shuffled[:holdout_count]))
    remaining = shuffled[holdout_count:]
    warmup_task_ids = tuple(sorted(remaining[:warmup_count]))
    grpo_task_ids = tuple(sorted(remaining[warmup_count:]))
    return TaskSplit(
        family_name=family_name,
        task_ids=deduped,
        warmup_task_ids=warmup_task_ids,
        grpo_task_ids=grpo_task_ids,
        holdout_task_ids=holdout_task_ids,
        eval_task_ids=deduped,
        split_seed=split_seed,
    )


def recommend_task_split(family_name: str, *, split_seed: int = 42) -> TaskSplit:
    spec = get_task_family_spec(family_name)
    if family_name == "shopping_search_sort":
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=(325, 326),
            grpo_task_ids=(327, 328),
            holdout_task_ids=(324,),
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "shopping_order_full":
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=(188, 190, 191, 194, 197, 198, 199, 201, 231, 319, 320, 322, 334, 335),
            grpo_task_ids=(128, 129, 130, 131, 192, 193, 196, 202, 203, 204, 232, 233, 234, 235, 321, 323, 336, 337, 359, 361, 362),
            holdout_task_ids=(96, 117, 189, 195, 200, 338, 358, 360),
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "shopping_exact":
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=(188, 190, 194, 197, 198, 199, 231, 319, 320, 322, 325, 326, 327, 328),
            grpo_task_ids=(128, 129, 130, 131, 192, 193, 196, 232, 233, 321, 323, 362),
            holdout_task_ids=(189, 195, 200, 324, 358, 360),
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "shopping_full":
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=(188, 190, 191, 194, 197, 198, 199, 201, 231, 319, 325, 326, 327, 328, 334, 359),
            grpo_task_ids=(128, 129, 130, 131, 192, 193, 196, 202, 203, 204, 232, 233, 234, 235, 320, 321, 322, 323, 335, 336, 337, 338, 361, 362),
            holdout_task_ids=(96, 117, 189, 195, 200, 324, 358, 360),
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "bootstrap41":
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=(0, 1, 11, 21, 23, 124, 27, 28, 66, 132, 133, 293, 7, 9, 70),
            grpo_task_ids=(2, 3, 4, 5, 25, 26, 125, 126, 29, 30, 31, 67, 134, 135, 136, 10, 36, 72),
            holdout_task_ids=(41, 77, 141, 188, 68, 69, 259, 71),
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "bootstrap44":
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=(0, 1, 11, 12, 21, 23, 124, 27, 28, 66, 132, 133, 293, 7, 9, 70),
            grpo_task_ids=(2, 3, 4, 5, 13, 25, 26, 125, 126, 144, 29, 30, 31, 67, 134, 135, 136, 10, 36, 72),
            holdout_task_ids=(41, 77, 141, 188, 68, 69, 259, 71),
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "web_mix88":
        shopping_full_split = recommend_task_split("shopping_full", split_seed=split_seed)
        bootstrap41_split = recommend_task_split("bootstrap41", split_seed=split_seed)
        holdout_task_ids = tuple(
            sorted(set(shopping_full_split.holdout_task_ids) | set(bootstrap41_split.holdout_task_ids))
        )
        warmup_task_ids = tuple(
            sorted(
                task_id
                for task_id in (set(shopping_full_split.warmup_task_ids) | set(bootstrap41_split.warmup_task_ids))
                if task_id not in holdout_task_ids
            )
        )
        grpo_task_ids = tuple(
            sorted(
                task_id
                for task_id in spec.task_ids
                if task_id not in warmup_task_ids and task_id not in holdout_task_ids
            )
        )
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=warmup_task_ids,
            grpo_task_ids=grpo_task_ids,
            holdout_task_ids=holdout_task_ids,
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    if family_name == "web_mix91":
        shopping_full_split = recommend_task_split("shopping_full", split_seed=split_seed)
        bootstrap44_split = recommend_task_split("bootstrap44", split_seed=split_seed)
        holdout_task_ids = tuple(
            sorted(set(shopping_full_split.holdout_task_ids) | set(bootstrap44_split.holdout_task_ids))
        )
        warmup_task_ids = tuple(
            sorted(
                task_id
                for task_id in (set(shopping_full_split.warmup_task_ids) | set(bootstrap44_split.warmup_task_ids))
                if task_id not in holdout_task_ids
            )
        )
        grpo_task_ids = tuple(
            sorted(
                task_id
                for task_id in spec.task_ids
                if task_id not in warmup_task_ids and task_id not in holdout_task_ids
            )
        )
        return TaskSplit(
            family_name=family_name,
            task_ids=spec.task_ids,
            warmup_task_ids=warmup_task_ids,
            grpo_task_ids=grpo_task_ids,
            holdout_task_ids=holdout_task_ids,
            eval_task_ids=spec.task_ids,
            split_seed=split_seed,
        )
    return build_task_split(
        spec.task_ids,
        family_name=spec.name,
        warmup_count=spec.recommended_warmup_count,
        holdout_count=spec.recommended_holdout_count,
        split_seed=split_seed,
    )
