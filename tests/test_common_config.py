"""Тесты единой конфигурации common.config.CommonConfig."""

import pytest

from common.config import CommonConfig


def test_defaults():
    cfg = CommonConfig()
    assert cfg.start_week == 0
    assert cfg.end_week == 24
    assert cfg.epochs == 10
    assert cfg.k == 10
    assert cfg.min_history == 2
    assert cfg.max_history == 32
    assert cfg.min_timespent_pos == 5
    assert cfg.seed == 42


def test_val_week_is_end_plus_one():
    assert CommonConfig(end_week=10).val_week == 11


def test_val_week_25_when_end_24():
    # неделя 25 существует только в validation/
    assert CommonConfig(end_week=24).val_week == 25


def test_val_week_rejects_end_week_gt_24():
    cfg = CommonConfig(end_week=25)
    with pytest.raises(AssertionError):
        _ = cfg.val_week


def test_val_week_rejects_start_gt_end():
    cfg = CommonConfig(start_week=5, end_week=3)
    with pytest.raises(AssertionError):
        _ = cfg.val_week


def test_train_weeks_range():
    cfg = CommonConfig(start_week=1, end_week=3)
    assert list(cfg.train_weeks) == [1, 2, 3]
    assert 25 not in cfg.train_weeks


def test_train_weeks_single_week():
    cfg = CommonConfig(start_week=24, end_week=24)
    assert list(cfg.train_weeks) == [24]