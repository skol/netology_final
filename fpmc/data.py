"""Подготовка данных VK-LSVD для FPMC.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet (недели идут train: 00..24, validation: 25).
Поэтому последовательность пользователя = его строки в глобальном порядке файлов.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

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
            self.val_files = [f"{self.data_dir}/validation/week_{self.n_train_weeks:02}.parquet"]


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
    train_df = _read(cfg.train_files)
    val_df = _read(cfg.val_files)

    # глобальный порядок строк = порядок времени
    train_df = train_df.with_row_index("__idx")

    # ---------- словари ----------
    users_sorted = train_df["user_id"].unique().sort().to_list()
    items_sorted = train_df["item_id"].unique().sort().to_list()
    user_to_idx = {u: i for i, u in enumerate(users_sorted)}
    item_to_idx = {i: j for j, i in enumerate(items_sorted)}

    # ---------- истории пользователей (в порядке строк) ----------
    grouped = (train_df.group_by("user_id", maintain_order=True)
               .agg([pl.col("item_id").alias("items"),
                     pl.col("__idx").alias("idx")]))

    history: dict = {}          # user_idx -> list[item_idx]
    last_item: dict = {}        # user_idx -> last item_idx
    triplets = []               # (u, last, next)

    for user_id, items, idx in grouped.iter_rows():
        # сортируем по глобальному индексу строки -> получаем хронологический порядок
        order = np.argsort(idx)
        seq = [item_to_idx[items[o]] for o in order]
        u = user_to_idx[user_id]
        history[u] = seq
        if len(seq) >= 2:
            last_item[u] = seq[-1]
            for t in range(1, len(seq)):
                triplets.append((u, seq[t - 1], seq[t]))

    triplets = np.asarray(triplets, dtype=np.int64)

    # ---------- валидация ----------
    # юзеры из трейна, у которых есть целевые айтемы в валидации
    val_mask = val_df["user_id"].is_in(train_df["user_id"]) \
        & val_df["item_id"].is_in(train_df["item_id"])
    val_df = val_df.filter(val_mask).with_row_index("__vidx")

    val_g = (val_df.group_by("user_id", maintain_order=True)
             .agg([pl.col("item_id").alias("items"), pl.col("__vidx").alias("idx")]))

    val_user_idxs = []
    val_gt = []
    for user_id, items, idx in val_g.iter_rows():
        u = user_to_idx[user_id]
        # верные айтемы, которых НЕТ в истории (иначе предсказывать нечего)
        gt = [item_to_idx[i] for i in items if item_to_idx[i] not in set(history[u])]
        if gt:
            val_user_idxs.append(u)
            val_gt.append(gt)

    return DataBundle(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        n_users=len(users_sorted),
        n_items=len(items_sorted),
        train_u=triplets[:, 0],
        train_last=triplets[:, 1],
        train_next=triplets[:, 2],
        history=history,
        last_item=last_item,
        val_user_idxs=np.asarray(val_user_idxs, dtype=np.int64),
        val_gt=val_gt,
    )


def _read(files: list[str]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(f) for f in files])


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