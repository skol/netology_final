"""Подготовка данных VK-LSVD для LightGCN.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet (недели идут train: 00..24, validation: 25).
Поэтому последовательность пользователя = его строки в глобальном порядке файлов.

LightGCN работает с двудольным графом "пользователь -- ролик". В качестве
положительных рёбер берутся все просмотренные пользователем ролики из истории
(коллаборативный сигнал), а предсказание следующего ролика сводится к ранжированию
кандидатов по близости эмбеддингов пользователя и ролика.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from common.data import build_user_sequences, build_validation, read_files

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@dataclass
class DataConfig:
    data_dir: str = "data/raw/VK-LSVD/subsamples/up0.001_ip0.001"
    n_train_weeks: int = 25   # недели 00..24
    min_user_history: int = 2  # минимум взаимодействий у юзера, чтобы был переход

    train_files: list = field(default_factory=list)
    val_files: list = field(default_factory=list)

    def __post_init__(self):
        if not self.train_files:
            self.train_files = [f"{self.data_dir}/train/week_{i:02}.parquet"
                                for i in range(self.n_train_weeks)]
        if not self.val_files:
            dir_name = 'train' if self.n_train_weeks < 25 else 'validation'
            self.val_files = [f"{self.data_dir}/{dir_name}/week_{self.n_train_weeks:02}.parquet"]


@dataclass
class DataBundle:
    """Готовые индексы и рёбра графа для обучения LightGCN."""
    user_to_idx: dict
    item_to_idx: dict
    n_users: int
    n_items: int

    # положительные рёбра графа "пользователь -> ролик" (из истории)
    train_users: np.ndarray
    train_items: np.ndarray

    # история (user_idx -> list[item_idx]) для фильтрации при инференсе
    history: dict

    # валидация: user_idx -> список верных item_idx
    val_user_idxs: np.ndarray
    val_gt: list


def load_and_build(cfg: DataConfig) -> DataBundle:
    """Читает train/validation parquet и строит граф взаимодействий.

    Каждый ролик в истории пользователя становится положительным ребром
    (u, i) — именно эти рёбра образуют матрицу смежности для LightGCN.
    """
    train_df = read_files(cfg.train_files)
    val_df = read_files(cfg.val_files)

    # ---------- словари + истории пользователей (в порядке строк) ----------
    user_to_idx, item_to_idx, history = build_user_sequences(train_df)

    # ---------- рёбра графа (каждое взаимодействие = коллаборативный сигнал) ----------
    edges_u: list[int] = []     # пользователи положительных рёбер
    edges_i: list[int] = []     # соответствующие ролики
    for u, seq in history.items():
        edges_u.extend([u] * len(seq))
        edges_i.extend(seq)

    # ---------- валидация ----------
    val_user_idxs, val_gt = build_validation(
        val_df, train_df, user_to_idx, item_to_idx, history)

    return DataBundle(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
        train_users=np.asarray(edges_u, dtype=np.int64),
        train_items=np.asarray(edges_i, dtype=np.int64),
        history=history,
        val_user_idxs=val_user_idxs,
        val_gt=val_gt,
    )


def build_normalized_adjacency(n_users: int, n_items: int,
                               users: np.ndarray, items: np.ndarray,
                               device=None) -> torch.Tensor:
    """Нормализованная матрица смежности A~ = D^{-1/2} A D^{-1/2} (разреженная).

    Граф двудольный: узлы 0..n_users-1 — пользователи,
    узлы n_users..n_users+n_items-1 — ролики. Вес ребра (a,b) = 1/sqrt(deg(a)*deg(b)),
    что соответствует симметричной нормализации LightGCN.

    Returns:
        COO-тензор формы [N, N], N = n_users + n_items.
    """
    N = n_users + n_items
    u_nodes = users
    i_nodes = items + n_users  # сдвиг индексов роликов за блок пользователей

    # оба направления (граф неориентированный)
    row = np.concatenate([u_nodes, i_nodes])
    col = np.concatenate([i_nodes, u_nodes])

    deg = np.bincount(row, minlength=N).astype(np.float64)
    # симметричная нормализация: 1/sqrt(deg(отправитель) * deg(получатель))
    val = 1.0 / np.sqrt(deg[row] * deg[col])

    indices = torch.as_tensor(np.stack([row, col]), dtype=torch.long)
    values = torch.as_tensor(val, dtype=torch.float32)
    # отключаем проверку инвариантов разреженности (у нас заведомо корректные рёбра)
    with torch.sparse.check_sparse_tensor_invariants(False):
        adj = torch.sparse_coo_tensor(indices, values, (N, N))
    if device is not None:
        adj = adj.to(device)
    return adj.coalesce()


class LightGCNDataset(Dataset):
    """Тренировочный датасет: по положительному ребру семплирует негативный ролик."""

    def __init__(self, users: np.ndarray, items: np.ndarray, n_items: int):
        self.users = torch.as_tensor(users)
        self.items = torch.as_tensor(items)
        self.n_items = n_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg = torch.randint(0, self.n_items, (1,)).item()
        # не допускаем негатив == правильный ответ
        while neg == self.items[idx].item():
            neg = torch.randint(0, self.n_items, (1,)).item()
        return self.users[idx], self.items[idx], neg