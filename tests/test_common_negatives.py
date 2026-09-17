"""Тесты единого negative sampling common.negatives.sample_negative."""

import random

from common.negatives import sample_negative


def test_prefers_explicit_negatives():
    rng = random.Random(42)
    result = sample_negative([3, 7, 11], n_items=100, rng=rng)
    assert result in {3, 7, 11}


def test_falls_back_when_all_explicit_excluded():
    rng = random.Random(42)
    # все явные негативы исключены -> случайный равномерный айтем
    result = sample_negative([3, 7], n_items=10, exclude={3, 7}, rng=rng)
    assert result not in {3, 7}
    assert 0 <= result < 10


def test_random_when_no_explicit():
    rng = random.Random(42)
    results = [sample_negative((), n_items=5, rng=rng) for _ in range(50)]
    assert all(0 <= r < 5 for r in results)
    assert len(set(results)) > 1


def test_respects_exclude_without_explicit():
    rng = random.Random(7)
    for _ in range(200):
        assert sample_negative((), n_items=10, exclude={4}, rng=rng) != 4


def test_deterministic_with_seed():
    rng1, rng2 = random.Random(123), random.Random(123)
    a = [sample_negative([1, 2, 3], n_items=50, rng=rng1) for _ in range(20)]
    b = [sample_negative([1, 2, 3], n_items=50, rng=rng2) for _ in range(20)]
    assert a == b


def test_works_without_rng():
    result = sample_negative((), n_items=3)
    assert 0 <= result < 3