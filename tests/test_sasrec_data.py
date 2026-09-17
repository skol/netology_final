"""Тесты подготовки данных SASRec (вне цикла обучения)."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from sasrec.config import SASRecConfig
from sasrec.data import (
    Sample,
    SequentialDataset,
    TrainIterableDataset,
    build_sliding_samples,
    collate_fn,
    extract_week_interactions,
    load_week_df,
    remap_sample,
    remap_samples,
    _resolve_week_path,
    _shard_target_weeks,
)

from helpers import make_interaction_df, write_week_parquet


# ---------- разметка недели ----------

def test_extract_week_interactions():
    df = make_interaction_df([
        (1, 10, 20, False, False, False, False, False, False),  # pos
        (1, 11, 1, False, False, False, False, False, False),   # neg (skip)
        (1, 11, 1, False, False, False, False, False, False),   # neg (дубль skipped)
        (2, 20, 2, True, False, False, False, False, False),    # pos (like)
        (2, 21, 1, False, True, False, False, False, False),    # neg (dislike)
    ])
    result = extract_week_interactions(df, min_timespent=5)
    assert result == {1: ([10], [11]), 2: ([20], [21])}


# ---------- слайдинг-окна ----------

def test_build_sliding_samples_basic():
    cfg = SASRecConfig(data_dir="x", history_weeks=2, max_history=3)
    weekly = {
        1: {1: ([10, 11], [90])},
        2: {1: ([12], [91]), 2: ([13], [])},
        3: {1: ([14], [92])},
    }
    samples = list(build_sliding_samples(cfg, weekly, target_week=3))
    assert samples == [(1, [10, 11, 12], 14, [90, 91, 92])]


def test_build_sliding_samples_excludes_target_from_history():
    cfg = SASRecConfig(data_dir="x", history_weeks=1, max_history=10)
    weekly = {
        2: {1: ([14, 99], [])},
        3: {1: ([14], [])},  # target 14 повторяется в истории -> исключается из input
    }
    samples = list(build_sliding_samples(cfg, weekly, target_week=3))
    assert samples == [(1, [99], 14, [])]


def test_build_sliding_samples_truncates_to_max_history():
    cfg = SASRecConfig(data_dir="x", history_weeks=1, max_history=2)
    weekly = {2: {1: ([1, 2, 3, 4], [])}, 3: {1: ([10], [])}}
    samples = list(build_sliding_samples(cfg, weekly, target_week=3))
    assert samples == [(1, [3, 4], 10, [])]


def test_build_sliding_samples_no_history_week():
    cfg = SASRecConfig(data_dir="x", history_weeks=2)
    weekly = {0: {1: ([5], [])}}
    assert list(build_sliding_samples(cfg, weekly, target_week=0)) == []


# ---------- ремаппинг ----------

def test_remap_sample():
    sample: Sample = (5, [10, 11], 12, [90])
    id_to_idx = {10: 0, 11: 1, 12: 2, 90: 3}
    assert remap_sample(sample, id_to_idx) == (5, [0, 1], 2, [3])


def test_remap_sample_none_when_target_unknown():
    assert remap_sample((5, [10], 999, []), {10: 0}) is None


def test_remap_sample_none_when_input_empty():
    assert remap_sample((5, [10], 12, []), {12: 0}) is None


def test_remap_samples_filters_unmappable():
    samples = [
        (1, [10, 11], 12, [90]),
        (2, [10], 999, []),  # target неизвестен -> отбрасывается
    ]
    id_to_idx = {10: 0, 11: 1, 12: 2, 90: 3}
    assert remap_samples(samples, id_to_idx) == [(1, [0, 1], 2, [3])]


# ---------- батчинг ----------

def test_collate_fn_right_padding():
    batch = [
        (1, [10, 11], 12, [90]),
        (2, [20], 21, [91, 92]),
    ]
    user_ids, inputs, positions, targets, skipped = collate_fn(batch, max_len=4)

    assert user_ids == [1, 2]
    assert inputs.shape == (2, 4)
    assert inputs[0].tolist() == [10, 11, 0, 0]
    assert inputs[1].tolist() == [20, 0, 0, 0]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert targets.tolist() == [12, 21]
    assert skipped == [[90], [91, 92]]


def test_collate_fn_truncates_long_sequences():
    batch = [(1, list(range(10)), 50, [])]
    user_ids, inputs, positions, targets, skipped = collate_fn(batch, max_len=4)
    assert inputs.shape == (1, 4)
    assert inputs[0].tolist() == [6, 7, 8, 9]


def test_sequential_dataset():
    samples = [(1, [10], 12, [90])]
    ds = SequentialDataset(samples)
    assert len(ds) == 1
    assert ds[0] == (1, [10], 12, [90])


# ---------- чтение недель ----------

def test_resolve_week_path_from_train(tmp_path):
    write_week_parquet(tmp_path, "train", 3,
                       make_interaction_df([(1, 10, 20, False, False, False, False, False, False)]))
    cfg = SASRecConfig(data_dir=str(tmp_path))
    assert _resolve_week_path(cfg, 3) == str(tmp_path / "train" / "week_03.parquet")


def test_resolve_week_path_falls_back_to_validation(tmp_path):
    write_week_parquet(tmp_path, "validation", 25,
                       make_interaction_df([(1, 10, 20, False, False, False, False, False, False)]))
    cfg = SASRecConfig(data_dir=str(tmp_path))
    assert _resolve_week_path(cfg, 25) == str(tmp_path / "validation" / "week_25.parquet")


def test_resolve_week_path_missing_raises(tmp_path):
    cfg = SASRecConfig(data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        _resolve_week_path(cfg, 0)


def test_load_week_df_columns(tmp_path):
    df = make_interaction_df([
        (1, 10, 20, False, False, False, False, False, False),
        (1, 11, 1, False, False, False, False, False, False),
    ])
    write_week_parquet(tmp_path, "train", 2, df)
    cfg = SASRecConfig(data_dir=str(tmp_path))

    out = load_week_df(cfg, 2)
    assert out.columns == [
        "user_id", "item_id", "timespent",
        "like", "dislike", "share", "bookmark", "click_on_author", "open_comments",
    ]
    assert out["item_id"].to_list() == [10, 11]


# ---------- TrainIterableDataset: потоковая выдача ----------

def _write_train_weeks(tmp_path, weeks_rows):
    """Записывает недели в tmp_path/train и возвращает SASRecConfig."""
    for week, rows in weeks_rows.items():
        write_week_parquet(tmp_path, "train", week, make_interaction_df(rows))
    return SASRecConfig(data_dir=str(tmp_path), history_weeks=2,
                        max_history=4, min_timespent_pos=5)


def _make_small_weeks():
    """Три недели: история (1,2) + target-неделя 3 (два позитива)."""
    pos = (20, False, False, False, False, False, False)   # timespent>=5
    neg = (1, False, False, False, False, False, False)    # короткий просмотр
    return {
        1: [(1, 100, *pos), (1, 101, *neg)],
        2: [(1, 102, *pos)],
        3: [(1, 103, *pos), (1, 104, *pos)],
    }


def test_train_iterable_is_finite_and_complete(tmp_path):
    cfg = _write_train_weeks(tmp_path, _make_small_weeks())
    id_to_idx = {100: 1, 101: 2, 102: 3, 103: 4, 104: 5}

    ds = TrainIterableDataset(cfg, id_to_idx, target_weeks=[3],
                              buffer_size=2, seed=42, verbose=False)
    samples = list(ds)

    # конечность: ровно два сэмпла (по одному на позитив недели 3), без потерь
    assert len(samples) == 2
    targets = sorted(s[2] for s in samples)
    assert targets == [4, 5]
    # у каждого сэмпла непустая история из индексов словаря
    for _, seq, tgt, skipped in samples:
        assert seq and all(x in id_to_idx.values() for x in seq)
        assert tgt in (4, 5)
        assert isinstance(skipped, list)


def test_train_iterable_loader_no_hang_and_correct_counts(tmp_path):
    cfg = _write_train_weeks(tmp_path, _make_small_weeks())
    id_to_idx = {100: 1, 101: 2, 102: 3, 103: 4, 104: 5}
    ds = TrainIterableDataset(cfg, id_to_idx, target_weeks=[3],
                              buffer_size=1, seed=42, verbose=False)
    loader = DataLoader(ds, batch_size=2, num_workers=0,
                        collate_fn=partial(collate_fn, max_len=cfg.max_history))

    batches = list(loader)
    assert batches
    targets = [x.item() for _, _, _, t, _ in batches for x in t]
    assert sorted(targets) == [4, 5]


def test_train_iterable_deterministic_with_same_seed(tmp_path):
    cfg = _write_train_weeks(tmp_path, _make_small_weeks())
    id_to_idx = {100: 1, 101: 2, 102: 3, 103: 4, 104: 5}

    def run():
        ds = TrainIterableDataset(cfg, id_to_idx, target_weeks=[3],
                                  buffer_size=2, seed=42, verbose=False)
        return list(ds)

    first, second = run(), run()
    assert first == second


def test_train_iterable_varies_order_between_epochs(tmp_path):
    """Одна и та же эпоха воспроизводима, но порядок между эпохами меняется."""
    pos = (20, False, False, False, False, False, False)
    neg = (1, False, False, False, False, False, False)
    cfg = _write_train_weeks(tmp_path, {
        1: [(1, 100, *pos), (1, 101, *neg)],
        2: [(1, 102, *pos)],
        3: [(1, 1000 + u, *pos) for u in range(1, 9)],   # 8 позитивов юзера 1 -> 8 сэмплов
    })
    id_to_idx = {100: 1, 101: 2, 102: 3}
    id_to_idx.update({1000 + u: 10 + u for u in range(1, 9)})

    ds = TrainIterableDataset(cfg, id_to_idx, target_weeks=[3],
                              buffer_size=3, seed=42, verbose=False)
    epoch1 = list(ds)
    epoch2 = list(ds)
    assert sorted(epoch1) == sorted(epoch2)          # тот же набор сэмплов
    assert len(epoch1) == 8                          # конечность
    assert epoch1 != epoch2                          # порядок между эпохами разный


def test_train_iterable_worker_shard_no_duplicates(tmp_path, monkeypatch):
    cfg = _write_train_weeks(tmp_path, _make_small_weeks())
    id_to_idx = {100: 1, 101: 2, 102: 3, 103: 4, 104: 5}

    # Воркер 0 из 2: должен обработать только target-неделю 3 (недели [3, ...::2]).
    monkeypatch.setattr(
        "sasrec.data.get_worker_info",
        lambda: SimpleNamespace(id=0, num_workers=2, seed=12345),
    )
    ds = TrainIterableDataset(cfg, id_to_idx, target_weeks=[3],
                              buffer_size=2, seed=42, verbose=False)
    samples = list(ds)
    # Полный набор, без дублей — воркер получил свои недели целиком.
    assert len(samples) == 2
    assert sorted(s[2] for s in samples) == [4, 5]


def test_shard_target_weeks():
    weeks = list(range(6))
    assert _shard_target_weeks(weeks, None) == weeks
    assert _shard_target_weeks(weeks, SimpleNamespace(id=0, num_workers=2)) == [0, 2, 4]
    assert _shard_target_weeks(weeks, SimpleNamespace(id=1, num_workers=2)) == [1, 3, 5]


# ---------- пиклируемость collate (для num_workers > 0 на Windows) ----------

def test_collate_fn_partial_is_picklable():
    import pickle

    fn = partial(collate_fn, max_len=4)
    restored = pickle.loads(pickle.dumps(fn))

    batch = [(1, [10, 11], 12, [90]), (2, [20], 21, [])]
    user_ids, inputs, positions, targets, skipped = restored(batch)
    assert inputs.shape == (2, 4)
    assert inputs[0].tolist() == [10, 11, 0, 0]
    assert targets.tolist() == [12, 21]
    assert skipped == [[90], []]