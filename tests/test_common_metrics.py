"""Тесты метрик next-item рекомендаций common.metrics."""

import numpy as np
import pytest

from common.metrics import _ndcg_at_k, eval_metrics


def test_ndcg_first_position_normalized_by_full_k():
    # реализация нормализует на idcg по всем k позициям (не по числу попаданий),
    # поэтому ndcg@k для ранга 1 < 1 при k > 1
    expected = 1.0 / sum(1.0 / np.log2(i + 2) for i in range(5))
    assert _ndcg_at_k(np.array([1]), k=5) == pytest.approx(expected)


def test_ndcg_second_position():
    dcg = 1.0 / np.log2(3)
    idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3) + 1.0 / np.log2(4)
    assert _ndcg_at_k(np.array([2]), k=3) == pytest.approx(dcg / idcg)


def test_ndcg_zero_for_no_hits():
    assert _ndcg_at_k(np.array([0, 0]), k=5) == 0.0


def test_ndcg_ignores_ranks_outside_k():
    assert _ndcg_at_k(np.array([6]), k=5) == 0.0


def test_eval_metrics_perfect_single_user():
    metrics = eval_metrics([np.array([1])], k=10)
    assert metrics["recall@10"] == pytest.approx(1.0)
    # ndcg нормируется на idcg по всем k позициям
    expected_ndcg = 1.0 / sum(1.0 / np.log2(i + 2) for i in range(10))
    assert metrics["ndcg@10"] == pytest.approx(expected_ndcg)
    assert metrics["mrr@10"] == pytest.approx(1.0)


def test_eval_metrics_mixed_users():
    gt_positions = [np.array([1, 3]), np.array([0])]
    metrics = eval_metrics(gt_positions, k=3)
    # recall: 2 попадания из 3 верных айтемов
    assert metrics["recall@3"] == pytest.approx(2 / 3)
    # mrr: (1/1 + 0) / 2 пользователя
    assert metrics["mrr@3"] == pytest.approx(0.5)
    # ndcg: user0 dcg=1/log2(2)+1/log2(4)=1.5; idcg по k=3
    idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3) + 1.0 / np.log2(4)
    assert metrics["ndcg@3"] == pytest.approx((1.5 / idcg + 0.0) / 2)


def test_eval_metrics_keys():
    metrics = eval_metrics([np.array([1])], k=7)
    assert set(metrics) == {"recall@7", "ndcg@7", "mrr@7"}


def test_eval_metrics_empty():
    metrics = eval_metrics([], k=10)
    assert metrics["recall@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0
    assert metrics["mrr@10"] == 0.0