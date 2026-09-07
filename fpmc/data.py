"""Подготовка данных VK-LSVD для FPMC.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet (недели идут train: 00..24, validation: 25).
Поэтому последовательность пользователя = его строки в глобальном порядке файлов.
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
            self.train_files = [f"{self.data_dir}/train/week_{i:02}.parquet" for i in range(self.n_train_weeks)]
        if not self.val_files:
            dir_name = 'train' if self.n_train_weeks < 25 else 'validation'
            self.val_files = [f"{self.data_dir}/{dir_name}/week_{self.n_train_weeks:02}.parquet"]


@dataclass
class DataBundle:
    """Готовые индексы для обучения и валидации."""
    user_to_idx: dict
    item_to_idx: dict
    n_users: int
    n_items: int

    # тренировочные триплеты (u, last_item, next_item)
    train_u: np.ndarray
    train_last: np.ndarray
    train_next: np.ndarray

    # история (индекс -> список item_idx) для фильтрации при инференсе
    history: dict
    # последний айтем истории каждого юзера (для MC-части)
    last_item: dict

    # валидация: user_idx -> список верных item_idx
    val_user_idxs: np.ndarray
    val_gt: list


def load_and_build(cfg: DataConfig) -> DataBundle:
    train_df = read_files(cfg.train_files)
    val_df = read_files(cfg.val_files)

    # ---------- словари + истории пользователей (в порядке строк) ----------
    user_to_idx, item_to_idx, history = build_user_sequences(train_df)

    # ---------- FPMC-специфичные триплеты (u, last, next) ----------
    last_item: dict = {}        # user_idx -> last item_idx
    triplets = []               # (u, last, next)
    for u, seq in history.items():
        if len(seq) >= 2:
            last_item[u] = seq[-1]
            for t in range(1, len(seq)):
                triplets.append((u, seq[t - 1], seq[t]))

    triplets = np.asarray(triplets, dtype=np.int64)

    # ---------- валидация ----------
    val_user_idxs, val_gt = build_validation(
        val_df, train_df, user_to_idx, item_to_idx, history)

    return DataBundle(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
        train_u=triplets[:, 0],
        train_last=triplets[:, 1],
        train_next=triplets[:, 2],
        history=history,
        last_item=last_item,
        val_user_idxs=val_user_idxs,
        val_gt=val_gt,
    )


class FPMCDataset(Dataset):
    """Тренировочный датасет: по триплету семплирует негативный айтем."""

    def __init__(self, bundle: DataBundle, n_items: int):
        self.u = torch.as_tensor(bundle.train_u)
        self.last = torch.as_tensor(bundle.train_last)
        self.next = torch.as_tensor(bundle.train_next)
        self.n_items = n_items

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        neg = torch.randint(0, self.n_items, (1,)).item()
        # не допускаем негатив == правильный ответ
        while neg == self.next[idx].item():
            neg = torch.randint(0, self.n_items, (1,)).item()
        return self.u[idx], self.last[idx], self.next[idx], neg