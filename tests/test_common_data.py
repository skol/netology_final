"""Тесты общей подготовки данных common.data (вне цикла обучения)."""

import polars as pl
import pytest

from common.data import (
    build_id_maps,
    build_sequences_with_maps,
    build_user_sequences,
    build_validation,
    read_files,
    read_week,
    read_weeks,
)

from helpers import make_interaction_df, write_week_parquet


# ---------- словари и истории ----------

def test_build_id_maps_sorted():
    df = pl.DataFrame({"user_id": [3, 1, 3, 2], "item_id": [10, 20, 10, 30]})
    user_to_idx, item_to_idx = build_id_maps(df)
    assert user_to_idx == {1: 0, 2: 1, 3: 2}
    assert item_to_idx == {10: 0, 20: 1, 30: 2}


def test_build_sequences_preserves_row_order():
    df = pl.DataFrame({"user_id": [3, 1, 3, 1], "item_id": [10, 20, 30, 40]})
    user_to_idx = {1: 0, 3: 1}
    item_to_idx = {10: 0, 20: 1, 30: 2, 40: 3}
    history = build_sequences_with_maps(df, user_to_idx, item_to_idx)
    # пользователь 1 (idx 0): строки 20, 40 -> [1, 3]; пользователь 3 (idx 1): 10, 30 -> [0, 2]
    assert history == {0: [1, 3], 1: [0, 2]}


def test_build_user_sequences_roundtrip():
    df = pl.DataFrame({"user_id": [2, 2, 1], "item_id": [10, 20, 30]})
    user_to_idx, item_to_idx, history = build_user_sequences(df)
    assert user_to_idx == {1: 0, 2: 1}
    assert item_to_idx == {10: 0, 20: 1, 30: 2}
    assert history == {1: [0, 1], 0: [2]}


# ---------- валидация ----------

def test_build_validation_basic():
    train_df = pl.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 30]})
    val_df = pl.DataFrame({"user_id": [1, 2, 3], "item_id": [30, 10, 40]})
    user_to_idx, item_to_idx = build_id_maps(train_df)
    history = build_sequences_with_maps(train_df, user_to_idx, item_to_idx)

    val_user_idxs, val_gt = build_validation(
        val_df, train_df, user_to_idx, item_to_idx, history)

    # только юзеры из трейна; user1 -> item 30 (idx 2), user2 -> item 10 (idx 0)
    assert list(val_user_idxs) == [0, 1]
    assert val_gt == [[2], [0]]


def test_build_validation_skips_items_already_seen():
    # user 1 видел item 10 в трейне; в валидации у него только item 10 -> пара отбрасывается
    train_df = pl.DataFrame({"user_id": [1], "item_id": [10]})
    val_df = pl.DataFrame({"user_id": [1], "item_id": [10]})
    user_to_idx, item_to_idx = build_id_maps(train_df)
    history = build_sequences_with_maps(train_df, user_to_idx, item_to_idx)

    val_user_idxs, val_gt = build_validation(
        val_df, train_df, user_to_idx, item_to_idx, history)
    assert len(val_user_idxs) == 0
    assert val_gt == []


# ---------- чтение недель с диска ----------

def test_read_week_from_train(tmp_path):
    df = make_interaction_df([(1, 10, 20, False, False, False, False, False, False)])
    write_week_parquet(tmp_path, "train", 3, df)
    out = read_week(str(tmp_path), 3)
    assert out["item_id"].to_list() == [10]


def test_read_week_falls_back_to_validation(tmp_path):
    df = make_interaction_df([(1, 10, 20, False, False, False, False, False, False)])
    write_week_parquet(tmp_path, "validation", 25, df)
    out = read_week(str(tmp_path), 25)
    assert out["item_id"].to_list() == [10]


def test_read_week_train_takes_priority(tmp_path):
    train_df = make_interaction_df([(1, 10, 20, False, False, False, False, False, False)])
    val_df = make_interaction_df([(2, 20, 30, False, False, False, False, False, False)])
    write_week_parquet(tmp_path, "train", 5, train_df)
    write_week_parquet(tmp_path, "validation", 5, val_df)
    out = read_week(str(tmp_path), 5)
    assert out["item_id"].to_list() == [10]


def test_read_week_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_week(str(tmp_path), 99)


def test_read_files_concat(tmp_path):
    d1 = make_interaction_df([(1, 10, 20, False, False, False, False, False, False)])
    d2 = make_interaction_df([(2, 20, 30, False, False, False, False, False, False)])
    p1 = write_week_parquet(tmp_path, "train", 0, d1)
    p2 = write_week_parquet(tmp_path, "train", 1, d2)
    out = read_files([str(p1), str(p2)])
    assert out["user_id"].to_list() == [1, 2]


def test_read_weeks_concat(tmp_path):
    write_week_parquet(
        tmp_path, "train", 0,
        make_interaction_df([(1, 10, 20, False, False, False, False, False, False)]),
    )
    write_week_parquet(
        tmp_path, "train", 1,
        make_interaction_df([(2, 20, 30, False, False, False, False, False, False)]),
    )
    out = read_weeks(str(tmp_path), [0, 1])
    assert out["user_id"].to_list() == [1, 2]