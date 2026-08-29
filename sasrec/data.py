"""Подготовка данных для SASRec: недели -> слайдинг-окна -> сэмплы.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet.
"""

import os
from typing import Dict, List, Tuple

import duckdb
import polars as pl
import torch
from torch.utils.data import Dataset

from .config import SASRecConfig

# (input_seq, target_item, skipped_items)
Sample = Tuple[List[int], int, List[int]]


def load_week_df(cfg: SASRecConfig, week_num: int) -> pl.DataFrame:
    """Читает неделю через duckdb и оставляет нужные колонки."""
    file_path = os.path.join(cfg.weeks_root, f"week_{week_num:02}.parquet")

    with duckdb.connect() as con:
        query = f"""
            SELECT
                user_id,
                item_id,
                timespent,
                "like",
                dislike,
                share,
                bookmark,
                click_on_author,
                open_comments
            FROM read_parquet('{file_path}')
        """
        return con.execute(query).pl()


def extract_week_interactions(df: pl.DataFrame) -> Dict[int, Tuple[List[int], List[int]]]:
    """Возвращает user_id -> (positive_items, skipped_items).

    ВАЖНО: порядок positive_items определяется порядком строк в parquet.
    Если в данных есть timestamp/event_time — его нужно добавить в SELECT
    и отсортировать перед group_by.
    """
    positive = (
        df.filter(
            (pl.col("timespent") >= 5) | pl.col("like") | pl.col("share") | pl.col("bookmark") |
            pl.col("click_on_author") | pl.col("open_comments")
        )
    )

    negative = df.filter(
        ((pl.col("timespent") < 5) | (pl.col("dislike"))) & (
            (pl.col("like").is_null()) & (pl.col("share").is_null()) & (pl.col("bookmark").is_null()) &
            (pl.col("click_on_author").is_null()) & (pl.col("open_comments").is_null())
        )
    )

    pos_by_user = positive.group_by("user_id", maintain_order=True).agg(pl.col("item_id").implode().alias("items"))
    neg_by_user = negative.group_by("user_id").agg(pl.col("item_id").implode().alias("skipped"))
    joined = pos_by_user.join(neg_by_user, on="user_id", how="left")
    result = {}
    for user_id, items, skipped in joined.iter_rows():
        if skipped is None:
            skipped = []
        skipped = list(set(skipped))
        result[int(user_id)] = (list(items), skipped)
    return result


def build_sliding_samples(
    cfg: SASRecConfig,
    weekly_data: Dict[int, Dict[int, Tuple[List[int], List[int]]]],
    target_week: int,
) -> List[Sample]:
    """Строит samples вида: history weeks -> target week.

    Для каждого позитива target_week создаётся отдельный sample.
    input ограничивается cfg.max_len.
    """
    samples = []
    first_history_week = target_week - cfg.history_weeks
    if first_history_week < 0:
        return samples
    history_range = list(range(first_history_week, target_week))

    target_users = weekly_data[target_week].keys()

    for user_id in target_users:
        history_items = []
        history_skipped = []
        for week in history_range:
            user_data = weekly_data.get(week, {}).get(user_id)
            if user_data is None:
                continue
            items, skipped = user_data
            history_items.extend(items)
            history_skipped.extend(skipped)
        if not history_items:
            continue
        target_data = weekly_data[target_week].get(user_id)
        if target_data is None:
            continue
        target_items, target_skipped = target_data
        if not target_items:
            continue
        all_skipped = list(set(history_skipped + target_skipped))
        for target_item in target_items:
            input_items = [x for x in history_items if x != target_item and x != 0]
            if not input_items:
                continue
            input_items = input_items[-cfg.max_len:]
            samples.append((input_items, target_item, all_skipped))

    return samples


def remap_samples(samples: List[Sample], id_to_idx: Dict[int, int]) -> List[Sample]:
    """Переводит исходные id айтемов в компактные индексы словаря."""
    result = []
    for input_seq, target, skipped in samples:
        if target not in id_to_idx:
            continue
        mapped_input = [id_to_idx[x] for x in input_seq if x in id_to_idx]
        mapped_skipped = [id_to_idx[x] for x in skipped if x in id_to_idx]
        mapped_target = id_to_idx[target]
        if not mapped_input:
            continue
        result.append((mapped_input, mapped_target, mapped_skipped))
    return result


class SequentialDataset(Dataset):
    """Тренировочный датасет: (input_seq, target_item, skipped)."""

    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_item, skipped = self.samples[idx]
        return input_seq, target_item, skipped


def collate_fn(batch: List[Sample], max_len: int):
    """RIGHT PADDING: [A, B, C, 0, 0].

    Это существенно безопаснее для causal attention, чем left padding.
    """
    batch_size = len(batch)
    inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    positions = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    targets = torch.empty(batch_size, dtype=torch.long)
    skipped_items = []
    for i, (seq, target, skipped) in enumerate(batch):
        seq = seq[-max_len:]
        seq_len = len(seq)
        inputs[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        targets[i] = target
        skipped_items.append(skipped)
    return inputs, positions, targets, skipped_items