"""Тесты загрузки pretrained эмбеддингов common.embeddings.load_embeddings_only."""

import numpy as np
import pytest

from common.embeddings import load_embeddings_only


def _write_embeddings(tmp_path, ids, vecs, name="item_embeddings.npz"):
    path = tmp_path / name
    np.savez(path, np.asarray(ids), np.asarray(vecs, dtype=np.float32))
    return path


def test_load_embeddings_only_basic(tmp_path):
    ids = [10, 20, 30]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    _write_embeddings(tmp_path, ids, vecs)

    vecs_np, id_to_pos = load_embeddings_only(
        str(tmp_path), "item_embeddings.npz", emb_dim=2)

    assert vecs_np.shape == (3, 2)
    assert vecs_np.dtype == np.float32
    assert id_to_pos == {10: 0, 20: 1, 30: 2}


def test_load_embeddings_truncates_to_emb_dim(tmp_path):
    ids = [1, 2]
    vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    _write_embeddings(tmp_path, ids, vecs)

    vecs_np, _ = load_embeddings_only(str(tmp_path), "item_embeddings.npz", emb_dim=2)
    assert vecs_np.shape == (2, 2)


def test_load_embeddings_raises_on_nan(tmp_path):
    ids = [1, 2]
    vecs = np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
    _write_embeddings(tmp_path, ids, vecs)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        load_embeddings_only(str(tmp_path), "item_embeddings.npz", emb_dim=2)


def test_load_embeddings_raises_on_inf(tmp_path):
    ids = [1, 2]
    vecs = np.array([[1.0, 0.0], [np.inf, 1.0]], dtype=np.float32)
    _write_embeddings(tmp_path, ids, vecs)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        load_embeddings_only(str(tmp_path), "item_embeddings.npz", emb_dim=2)