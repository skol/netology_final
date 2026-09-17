"""Тесты общих CLI-флагов common.cli."""

import argparse
from dataclasses import dataclass

from common.cli import add_common_args, config_from_args


def test_add_common_args_defaults_are_none():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([])
    assert args.data_dir is None
    assert args.start_week is None
    assert args.end_week is None
    assert args.epochs is None
    assert args.lr is None
    assert args.k is None
    assert args.min_history is None
    assert args.max_history is None
    assert args.min_timespent_pos is None
    assert args.seed is None


def test_add_common_args_parses_values():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--data_dir", "data/x",
        "--start-week", "2",
        "--end-week", "24",
        "--lr", "0.01",
        "--k", "5",
        "--min-timespent-pos", "7",
        "--seed", "1",
    ])
    assert args.data_dir == "data/x"
    assert args.start_week == 2
    assert args.end_week == 24
    assert args.lr == 0.01
    assert args.k == 5
    assert args.min_timespent_pos == 7
    assert args.seed == 1
    assert args.epochs is None
    assert args.min_history is None
    assert args.max_history is None


@dataclass
class _DummyConfig:
    data_dir: str = "default"
    k: int = 10
    seed: int = 42
    epochs: int = 10


def test_config_from_args_skips_none_values():
    args = argparse.Namespace(data_dir=None, k=5, seed=None, epochs=10)
    cfg = config_from_args(args, _DummyConfig)
    assert cfg.data_dir == "default"
    assert cfg.k == 5
    assert cfg.seed == 42
    assert cfg.epochs == 10


def test_config_from_args_with_common_config():
    from common.config import CommonConfig

    args = argparse.Namespace(
        data_dir=None, start_week=None, end_week=None, epochs=None,
        lr=None, k=3, min_history=None, max_history=None,
        min_timespent_pos=None, seed=None,
    )
    cfg = config_from_args(args, CommonConfig)
    assert cfg.k == 3
    assert cfg.epochs == 10
    assert cfg.seed == 42
    assert cfg.end_week == 24