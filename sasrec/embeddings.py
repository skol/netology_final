"""Загрузка и сборка матрицы item-эмбеддингов."""

import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def load_embeddings_only(meta_root: str, emb_file: str, emb_dim: int):
    """Читает item_embeddings.npz и возвращает (vecs, id_to_pos)."""
    path = os.path.join(meta_root, emb_file)
    data = np.load(path)
    ids_np = data[data.files[0]]
    vecs_np = data[data.files[1]][:, :emb_dim].astype(np.float32)

    print("\nEmbedding diagnostics:")
    print("shape:", vecs_np.shape)
    print("NaN:", np.isnan(vecs_np).sum())
    print("Inf:", np.isinf(vecs_np).sum())
    print("finite:", np.isfinite(vecs_np).all())

    if not np.isfinite(vecs_np).all():
        bad_rows = np.where(~np.isfinite(vecs_np).all(axis=1))[0]
        print("Количество плохих embeddings:", len(bad_rows))
        raise RuntimeError("item_embeddings.npz contains NaN/Inf")

    id_to_pos = {int(_id): idx for idx, _id in enumerate(ids_np)}
    print(f"Загружено {len(ids_np):,} embeddings.")
    return vecs_np, id_to_pos


def build_weight_matrix(
    id_to_idx: Dict[int, int],
    vecs_np: np.ndarray,
    id_to_pos: Dict[int, int],
    emb_width: int,
) -> torch.Tensor:
    """Собирает весовую матрицу эмбеддингов; для отсутствующих — случайная инициализация."""
    num_items = len(id_to_idx) + 1
    weight_matrix = torch.zeros(num_items, emb_width, dtype=torch.float32)
    missing_count = 0
    stdv = 1.0 / np.sqrt(emb_width)
    for old_id, new_idx in id_to_idx.items():
        pos = id_to_pos.get(old_id)
        if pos is not None:
            vec = torch.from_numpy(vecs_np[pos])
            vec = F.normalize(vec, p=2, dim=0)
            if not torch.isfinite(vec).all():
                raise RuntimeError(f"Bad embedding for item {old_id}")
            weight_matrix[new_idx].copy_(vec)
        else:
            weight_matrix[new_idx].uniform_(-stdv, stdv)
            missing_count += 1

    print("Weight matrix NaN:", torch.isnan(weight_matrix).sum().item())
    print("Weight matrix Inf:", torch.isinf(weight_matrix).sum().item())

    if not torch.isfinite(weight_matrix).all():
        raise RuntimeError("weight_matrix contains NaN/Inf")
    print(f"Missing pretrained embeddings: {missing_count:,}")

    return weight_matrix