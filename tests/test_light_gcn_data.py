"""Тесты подготовки данных LightGCN: конфиг, граф, датасет (вне обучения)."""

import numpy as np
import pytest
import torch

from light_gcn.data import DataConfig, LightGCNDataset, build_normalized_adjacency


def test_dataconfig_builds_files():
    cfg = DataConfig(data_dir="data/raw", start_week=23, end_week=24)
    assert cfg.train_files[-1] == "data/raw/train/week_24.parquet"
    # end_week=24 -> val_week=25 -> файл ищется в validation/
    assert cfg.val_files == ["data/raw/validation/week_25.parquet"]


def test_build_normalized_adjacency_symmetric():
    n_users, n_items = 2, 2
    users = np.array([0, 0, 1])
    items = np.array([0, 1, 0])
    adj = build_normalized_adjacency(n_users, n_items, users, items)

    assert adj.shape == (4, 4)
    d = adj.to_dense()

    # ребро (0,2): deg(0)=2, deg(2)=2 -> 1/sqrt(4)=0.5
    assert d[0, 2].item() == pytest.approx(0.5)
    assert d[2, 0].item() == pytest.approx(0.5)
    # ребро (0,3): deg(0)=2, deg(3)=1 -> 1/sqrt(2)
    assert d[0, 3].item() == pytest.approx(1 / np.sqrt(2))
    assert d[3, 0].item() == pytest.approx(1 / np.sqrt(2))
    # ребро (1,2): deg(1)=1, deg(2)=2 -> 1/sqrt(2)
    assert d[1, 2].item() == pytest.approx(1 / np.sqrt(2))
    # симметричность и нулевая диагональ
    assert torch.allclose(d, d.T)
    assert torch.allclose(torch.diag(d), torch.zeros(4))


def test_build_normalized_adjacency_no_nan_for_isolated_node():
    # пользователь 1 не имеет рёбер — нормализация не должна давать NaN
    adj = build_normalized_adjacency(2, 2, np.array([0]), np.array([1])).to_dense()
    assert not torch.isnan(adj).any()
    assert adj[0, 3].item() == pytest.approx(1.0)


def test_light_gcn_dataset_len_and_explicit_negative():
    users = np.array([0, 1])
    items = np.array([2, 3])
    ds = LightGCNDataset(users, items, n_items=4, explicit_negatives={0: [1]}, seed=42)
    assert len(ds) == 2

    u, pos, neg = ds[0]
    assert int(u) == 0
    assert int(pos) == 2
    assert neg == 1  # единственный явный негатив пользователя 0


def test_light_gcn_dataset_random_negative():
    users = np.array([5])
    items = np.array([1])
    ds = LightGCNDataset(users, items, n_items=4, explicit_negatives={}, seed=7)
    u, pos, neg = ds[0]
    assert neg != int(pos)
    assert 0 <= neg < 4