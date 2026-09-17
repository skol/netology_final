"""Тесты единой разметки действий common.labels.assign_labels."""

import pytest

from common.labels import NEG_REACTION, POS_REACTIONS, assign_labels

from helpers import make_interaction_df


def test_constants():
    assert POS_REACTIONS == [
        "like",
        "share",
        "bookmark",
        "click_on_author",
        "open_comments",
    ]
    assert NEG_REACTION == "dislike"


def test_pos_by_timespent():
    df = make_interaction_df([(1, 1, 10, False, False, False, False, False, False)])
    out = assign_labels(df, min_timespent_pos=5)
    assert out["label"].to_list() == [1]


def test_neg_by_short_timespent():
    df = make_interaction_df([(1, 1, 2, False, False, False, False, False, False)])
    out = assign_labels(df, min_timespent_pos=5)
    assert out["label"].to_list() == [0]


@pytest.mark.parametrize(
    "reaction_idx",
    [3, 5, 6, 7, 8],  # like, share, bookmark, click_on_author, open_comments
)
def test_pos_by_any_reaction(reaction_idx):
    row = [1, 1, 0, False, False, False, False, False, False]
    row[reaction_idx] = True
    out = assign_labels(make_interaction_df([tuple(row)]), min_timespent_pos=5)
    assert out["label"].to_list() == [1]


def test_dislike_with_big_timespent_is_positive():
    # «позитив побеждает»: dislike не отменяет длинный просмотр
    df = make_interaction_df([(1, 1, 30, False, True, False, False, False, False)])
    out = assign_labels(df, min_timespent_pos=5)
    assert out["label"].to_list() == [1]


def test_dislike_with_short_timespent_is_negative():
    df = make_interaction_df([(1, 1, 2, False, True, False, False, False, False)])
    out = assign_labels(df, min_timespent_pos=5)
    assert out["label"].to_list() == [0]


def test_null_flags_are_false_and_null_timespent_is_zero():
    # все флаги None и timespent None -> действие негативное (0)
    df = make_interaction_df([(1, 1, None, None, None, None, None, None, None)])
    out = assign_labels(df, min_timespent_pos=5)
    assert out["label"].to_list() == [0]


def test_mixed_batch_labels():
    rows = [
        (1, 1, 10, False, False, False, False, False, False),  # pos by timespent
        (1, 2, 2, False, False, False, False, False, False),   # neg (skip)
        (1, 3, 2, True, False, False, False, False, False),    # pos by like
        (1, 4, 2, False, True, False, False, False, False),    # neg by dislike
        (1, 5, 30, False, True, False, False, False, False),   # pos: positive wins
    ]
    out = assign_labels(make_interaction_df(rows), min_timespent_pos=5)
    assert out["label"].to_list() == [1, 0, 1, 0, 1]


def test_all_rows_get_label():
    # согласно семантике assign_labels строк без метки не остаётся
    rows = [
        (1, 1, 10, False, False, False, False, False, False),
        (1, 2, 1, False, False, False, False, False, False),
        (1, 3, 1, True, False, False, False, False, False),
    ]
    out = assign_labels(make_interaction_df(rows), min_timespent_pos=5)
    assert len(out) == 3
    assert out["label"].null_count() == 0