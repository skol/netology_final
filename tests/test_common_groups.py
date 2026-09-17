"""Тесты разбиения пользователей на группы cold/warm/hot (common.groups)."""

import polars as pl

from common.groups import (
    GROUP_NAMES,
    categorize_by_history_size,
    split_users_by_history,
    user_history_sizes,
)

from helpers import make_interaction_df


def test_group_names_order():
    assert GROUP_NAMES == ["cold", "warm", "hot"]


def test_categorize_by_history_size_defaults():
    assert categorize_by_history_size(0) == "cold"
    assert categorize_by_history_size(30) == "cold"
    assert categorize_by_history_size(31) == "warm"
    assert categorize_by_history_size(70) == "warm"
    assert categorize_by_history_size(71) == "hot"


def test_categorize_custom_thresholds():
    assert categorize_by_history_size(10, cold_max=5, warm_max=20) == "warm"
    assert categorize_by_history_size(21, cold_max=5, warm_max=20) == "hot"
    assert categorize_by_history_size(5, cold_max=5, warm_max=20) == "cold"


def test_user_history_sizes_counts_all_labeled_actions():
    # user 1: 3 размеченных действия (2 позитивных, 1 негативное) -> размер истории 3
    train_df = make_interaction_df([
        (1, 10, 20, False, False, False, False, False, False),
        (1, 11, 1, False, False, False, False, False, False),
        (1, 12, 5, True, False, False, False, False, False),
        (2, 20, 30, False, False, False, False, False, False),
    ])
    sizes = user_history_sizes(train_df, min_timespent_pos=5)
    assert sizes == {1: 3, 2: 1}


def test_split_users_by_history_groups():
    rows = []
    # user 1: 5 действий -> cold
    rows += [(1, 100 + i, 1, False, False, False, False, False, False) for i in range(5)]
    # user 2: 40 действий -> warm
    rows += [(2, 200 + i, 1, False, False, False, False, False, False) for i in range(40)]
    # user 3: 80 действий -> hot
    rows += [(3, 300 + i, 1, False, False, False, False, False, False) for i in range(80)]
    # user 4: есть в трейне, но отсутствует в валидации
    rows.append((4, 400, 1, False, False, False, False, False, False))
    train_df = make_interaction_df(rows)

    val_df = pl.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3]})

    groups = split_users_by_history(
        val_df, train_df, min_timespent_pos=5, group_limit=100, seed=42)

    assert groups["cold"] == [1]
    assert groups["warm"] == [2]
    assert groups["hot"] == [3]
    assert 4 not in groups["cold"] + groups["warm"] + groups["hot"]


def test_split_users_group_limit_and_reproducibility():
    train_df = make_interaction_df(
        [(u, 1000 + u, 1, False, False, False, False, False, False) for u in range(1, 6)]
    )
    val_df = pl.DataFrame({"user_id": [1, 2, 3, 4, 5], "item_id": [1, 2, 3, 4, 5]})

    groups = split_users_by_history(
        val_df, train_df, min_timespent_pos=5,
        cold_max=100, warm_max=100, group_limit=2, seed=42,
    )
    assert len(groups["cold"]) == 2
    assert groups["warm"] == []
    assert groups["hot"] == []

    again = split_users_by_history(
        val_df, train_df, min_timespent_pos=5,
        cold_max=100, warm_max=100, group_limit=2, seed=42,
    )
    assert groups["cold"] == again["cold"]