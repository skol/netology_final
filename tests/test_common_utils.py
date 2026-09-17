"""Тесты общих утилит common.utils (устройство и воспроизводимость)."""

import random

import numpy as np
import torch

from common.utils import get_device, seed_everything


def test_get_device_valid():
    assert get_device() in {"cpu", "cuda"}


def test_seed_everything_reproducible_random():
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_reproducible_numpy():
    seed_everything(7)
    a = np.random.rand(3)
    seed_everything(7)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_reproducible_torch():
    seed_everything(1)
    a = torch.rand(4).numpy()
    seed_everything(1)
    b = torch.rand(4).numpy()
    np.testing.assert_array_equal(a, b)