"""Тесты MostPopular: модель, подготовка данных и оценка (вне обучения)."""

import numpy as np
import pytest

from most_popular.config import MostPopularConfig
from most_popular.data import DataBundle, load_and_build
from most_popular.evaluate import evaluate
from most_popular.model import MostPopular

from helpers import make_interaction_df, write_week_parquet


# ---------- модель ----------

def _model() -> MostPopular:
    popularity = np.array([5, 3, 1, 4], dtype=np.int64)
    item_ranking = np.argsort(-popularity, kind="stable")  # [0, 3, 1, 2]
    return MostPopular(popularity, item_ranking)


def test_recommend_skips_seen_items():
    model = _model()
    assert model.recommend({0}, top_n=2) == [3, 1]


def test_recommend_respects_top_n():
    model = _model()
    assert model.recommend(set(), top_n=1) == [0]
    assert len(model.recommend(set(), top_n=10)) == 4


def test_recommend_empty_when_everything_seen():
    model = _model()
    assert model.recommend({0, 1, 2, 3}, top_n=10) == []


def test_rank_positions_single_gt():
    model = _model()
    positions = model.rank_positions({0}, gt={3})
    assert positions.tolist() == [1]


def test_rank_positions_gt_after_seen_prefix():
    model = _model()
    positions = model.rank_positions({0, 3, 1}, gt={2})
    assert positions.tolist() == [1]


def test_rank_positions_missing_gt_is_zero():
    model = _model()
    positions = model.rank_positions(set(), gt={99})
    assert positions.tolist() == [0]


def test_rank_positions_multiple_gt():
    model = _model()
    positions = model.rank_positions({0}, gt={3, 1})
    assert sorted(positions.tolist()) == [1, 2]


# ---------- загрузка данных ----------

def test_load_and_build(tmp_path):
    train_df = make_interaction_df([
        (1, 10, 20, False, False, False, False, False, False),  # pos
        (1, 20, 1, False, False, False, False, False, False),   # neg
        (2, 30, 30, False, False, False, False, False, False),  # pos
    ])
    val_df = make_interaction_df([
        (1, 30, 15, False, False, False, False, False, False),  # pos
        (2, 10, 2, False, False, False, False, False, False),   # neg
        (3, 10, 12, False, False, False, False, False, False),  # юзера нет в трейне
    ])
    # end_week=0 -> val_week=1 (не 25) -> файл ищется в train/
    write_week_parquet(tmp_path, "train", 0, train_df)
    write_week_parquet(tmp_path, "train", 1, val_df)

    cfg = MostPopularConfig(data_dir=str(tmp_path), start_week=0, end_week=0,
                            min_timespent_pos=5)
    bundle = load_and_build(cfg)

    assert bundle.user_to_idx == {1: 0, 2: 1}
    assert bundle.item_to_idx == {10: 0, 20: 1, 30: 2}
    assert bundle.n_users == 2
    assert bundle.n_items == 3

    # популярность: item10 -> 1, item20 -> 0, item30 -> 1
    np.testing.assert_array_equal(bundle.popularity, [1, 0, 1])
    assert bundle.item_ranking.tolist() == [0, 2, 1]

    # seen: user0 -> {10, 20}, user1 -> {30} (внутренние индексы)
    assert bundle.seen == {0: {0, 1}, 1: {2}}

    # валидация: только user 1 (idx 0) с верным item 30 (idx 2)
    assert bundle.val_user_idxs.tolist() == [0]
    assert bundle.val_gt == [[2]]


# ---------- оценка ----------

def _min_bundle() -> DataBundle:
    return DataBundle(
        user_to_idx={1: 0, 2: 1},
        item_to_idx={10: 0, 20: 1, 30: 2},
        n_users=2,
        n_items=3,
        popularity=np.array([3, 2, 1]),
        item_ranking=np.array([0, 1, 2]),
        seen={0: set(), 1: set()},
        val_user_idxs=np.array([0, 1]),
        val_gt=[[1], [2]],
    )


def test_evaluate_returns_metrics():
    bundle = _min_bundle()
    model = MostPopular(bundle.popularity, bundle.item_ranking)
    metrics = evaluate(model, bundle, k=2)

    assert set(metrics) == {"recall@2", "ndcg@2", "mrr@2"}
    # user0: ранг 2 (попадание), user1: ранг 3 (мимо) -> recall 1/2
    assert metrics["recall@2"] == pytest.approx(0.5)
    # mrr: user0 вклад 1/2 (ранг 2), user1 вклад 1/3 (ранг 3; mrr не обрезается по k)
    assert metrics["mrr@2"] == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_evaluate_return_per_user():
    bundle = _min_bundle()
    model = MostPopular(bundle.popularity, bundle.item_ranking)
    user_idxs, gt_positions = evaluate(model, bundle, k=2, return_per_user=True)

    assert user_idxs.tolist() == [0, 1]
    # user0 -> gt item 1 на ранге 2; user1 -> gt item 2 на ранге 3
    assert gt_positions[0].tolist() == [2]
    assert gt_positions[1].tolist() == [3]