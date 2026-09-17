"""Тесты подготовки данных FPMC: конфиг и тренировочный датасет (вне обучения)."""

import numpy as np

from fpmc.data import DataBundle, DataConfig, FPMCDataset


def test_dataconfig_builds_train_files():
    cfg = DataConfig(data_dir="data/raw", start_week=0, end_week=2)
    assert cfg.train_files == [
        "data/raw/train/week_00.parquet",
        "data/raw/train/week_01.parquet",
        "data/raw/train/week_02.parquet",
    ]
    # end_week=2 -> val_week=3 (не 25) -> файл ищется в train/
    assert cfg.val_files == ["data/raw/train/week_03.parquet"]


def test_dataconfig_val_file_week_25_in_validation():
    cfg = DataConfig(data_dir="data/raw", start_week=0, end_week=24)
    assert cfg.val_files == ["data/raw/validation/week_25.parquet"]


def test_dataconfig_custom_files_not_overridden():
    cfg = DataConfig(
        data_dir="data/raw", start_week=0, end_week=0,
        train_files=["a.parquet"], val_files=["b.parquet"],
    )
    assert cfg.train_files == ["a.parquet"]
    assert cfg.val_files == ["b.parquet"]


def _bundle() -> DataBundle:
    return DataBundle(
        user_to_idx={0: 0, 1: 1},
        item_to_idx={0: 0, 1: 1, 2: 2, 3: 3},
        n_users=2,
        n_items=4,
        train_u=np.array([0, 1]),
        train_last=np.array([1, 2]),
        train_next=np.array([2, 3]),
        history={0: [1, 2], 1: [2, 3]},
        last_item={0: 2, 1: 3},
        seen={0: {1, 2}, 1: {2, 3}},
        explicit_negatives={0: [0], 1: [0, 1]},
        val_user_idxs=np.array([0]),
        val_gt=[[3]],
    )


def test_fpmc_dataset_len_and_explicit_negative():
    ds = FPMCDataset(_bundle(), n_items=4, seed=42)
    assert len(ds) == 2

    # у пользователя 0 единственный явный негатив 0, и он != позитивному 2
    u, last, nxt, neg = ds[0]
    assert int(u) == 0
    assert int(last) == 1
    assert int(nxt) == 2
    assert neg == 0


def test_fpmc_dataset_negative_never_equals_positive():
    ds = FPMCDataset(_bundle(), n_items=4, seed=1)
    for i in range(len(ds)):
        _, _, nxt, neg = ds[i]
        assert neg != int(nxt)
        assert 0 <= neg < 4


def test_fpmc_dataset_empty():
    bundle = DataBundle(
        user_to_idx={},
        item_to_idx={},
        n_users=0,
        n_items=0,
        train_u=np.empty(0, dtype=np.int64),
        train_last=np.empty(0, dtype=np.int64),
        train_next=np.empty(0, dtype=np.int64),
        history={},
        last_item={},
        seen={},
        explicit_negatives={},
        val_user_idxs=np.array([]),
        val_gt=[],
    )
    ds = FPMCDataset(bundle, n_items=0, seed=42)
    assert len(ds) == 0