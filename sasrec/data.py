"""Подготовка данных для SASRec: недели -> слайдинг-окна -> сэмплы.

Файлы VK-LSVD не содержат колонки времени — порядок взаимодействий задаёт
порядок строк внутри parquet.

Тренировочные данные загружаются лениво через TrainIterableDataset: недели
читаются с диска по мере прохождения слайдинг-окон, вместо хранения всего
словаря weekly_data и списков *_samples в памяти. Словарь id_to_idx строится
отдельным быстрым проходом только по колонке item_id (read_week_item_ids).

Сэмплы выдаются ПОТОКОВО: build_sliding_samples — генератор, а __iter__ датасета
накапливает перемешиваемый буфер ограниченного размера (buffer_size). Первый
батч появляется сразу после заполнения буфера, а не после обработки всех окон,
память ограничена O(buffer_size), а не O(всех сэмплов эпохи).
"""

import os
import random
from typing import Dict, Iterator, List, Tuple

import duckdb
import polars as pl
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from common.labels import assign_labels

from .config import SASRecConfig

# (user_id, input_seq, target_item, skipped_items)
Sample = Tuple[int, List[int], int, List[int]]


def _resolve_week_path(cfg: SASRecConfig, week_num: int) -> str:
    """Ищет parquet недели сначала в train/, затем в validation/ (неделя 25)."""
    for sub in ("train", "validation"):
        path = os.path.join(cfg.data_dir, sub, f"week_{week_num:02}.parquet")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Не найдена неделя {week_num} ни в {cfg.data_dir}/train, ни в {cfg.data_dir}/validation"
    )


def load_week_df(cfg: SASRecConfig, week_num: int) -> pl.DataFrame:
    """Читает неделю через duckdb и оставляет нужные колонки."""
    file_path = _resolve_week_path(cfg, week_num)

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


def read_week_item_ids(cfg: SASRecConfig, week_num: int) -> list:
    """Читает только колонку item_id недели (быстрый проход для словаря)."""
    file_path = _resolve_week_path(cfg, week_num)
    with duckdb.connect() as con:
        query = f"SELECT item_id FROM read_parquet('{file_path}')"
        return con.execute(query).pl()["item_id"].to_list()


def extract_week_interactions(df: pl.DataFrame,
                              min_timespent: int) -> Dict[int, Tuple[List[int], List[int]]]:
    """Возвращает user_id -> (positive_items, skipped_items).

    Разметка позитив/негатив — единая для всех моделей
    (common.labels.assign_labels): строка не может попасть одновременно
    в positive и skipped («позитив побеждает»).

    ВАЖНО: порядок positive_items определяется порядком строк в parquet.
    Если в данных есть timestamp/event_time — его нужно добавить в SELECT
    и отсортировать перед group_by.
    """
    labeled = assign_labels(df, min_timespent)
    positive = labeled.filter(pl.col("label") == 1)
    negative = labeled.filter(pl.col("label") == 0)

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
) -> Iterator[Sample]:
    """Генератор samples вида: history weeks -> target week.

    Для каждого позитива target_week создаётся отдельный sample.
    input ограничивается cfg.max_history.

    Генератор (а не список) позволяет TrainIterableDataset выдавать батчи
    потоково: сэмплы окна не материализуются целиком в памяти.
    """
    first_history_week = target_week - cfg.history_weeks
    if first_history_week < 0:
        return
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
            input_items = input_items[-cfg.max_history:]
            yield (user_id, input_items, target_item, all_skipped)


def _remap_week_interactions(
    interactions: Dict[int, Tuple[List[int], List[int]]],
    id_to_idx: Dict[int, int],
) -> Dict[int, Tuple[List[int], List[int]]]:
    """Переводит id айтемов одной недели в компактные индексы словаря.

    Применяется ОДИН раз на окно (в _load_window), а не на каждый сэмпл:
    списки (positives, skipped) у пользователя одинаковы для всех его сэмплов,
    поэтому ремаппинг на уровне пользователя радикально дешевле. Айтемы, не
    попавшие в словарь, отбрасываются — семантика та же, что у remap_sample.
    """
    result: Dict[int, Tuple[List[int], List[int]]] = {}
    for user_id, (items, skipped) in interactions.items():
        mapped_items = [id_to_idx[x] for x in items if x in id_to_idx]
        mapped_skipped = [id_to_idx[x] for x in skipped if x in id_to_idx]
        result[user_id] = (mapped_items, mapped_skipped)
    return result


def remap_sample(sample: Sample, id_to_idx: Dict[int, int]) -> Sample | None:
    """Переводит один сэмпл в компактные индексы словаря; None, если маппить нечего."""
    user_id, input_seq, target, skipped = sample
    if target not in id_to_idx:
        return None
    mapped_input = [id_to_idx[x] for x in input_seq if x in id_to_idx]
    mapped_skipped = [id_to_idx[x] for x in skipped if x in id_to_idx]
    mapped_target = id_to_idx[target]
    if not mapped_input:
        return None
    return user_id, mapped_input, mapped_target, mapped_skipped


def remap_samples(samples: List[Sample], id_to_idx: Dict[int, int]) -> List[Sample]:
    """Переводит исходные id айтемов в компактные индексы словаря."""
    result = []
    for sample in samples:
        mapped = remap_sample(sample, id_to_idx)
        if mapped is not None:
            result.append(mapped)
    return result


def _shard_target_weeks(target_weeks: List[int], info) -> List[int]:
    """Распределяет target-недели между воркерами DataLoader.

    Каждый воркер получает непересекающееся подмножество недель, поэтому при
    num_workers > 0 данные не дублируются (в отличие от стандартного поведения
    DataLoader для IterableDataset, который гоняет весь датасет в каждом воркере).

    Args:
        target_weeks: полный список target-недель эпохи.
        info: результат torch.utils.data.get_worker_info() (None в main-процессе).

    Returns:
        Список недель, которые должен обработать текущий воркер.
    """
    if info is None:
        return target_weeks
    return target_weeks[info.id::info.num_workers]


class SequentialDataset(Dataset):
    """Материализованный датасет: (input_seq, target_item, skipped)."""

    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, input_seq, target_item, skipped = self.samples[idx]
        return user_id, input_seq, target_item, skipped


class TrainIterableDataset(IterableDataset):
    """Ленивый тренировочный датасет SASRec с потоковой выдачей батчей.

    Заменяет словарь weekly_data и список train_samples: недели читаются с диска
    по мере прохождения слайдинг-окон, сэмплы ремапятся и выдаются на лету.

    Перемешивание (DataLoader.shuffle не работает для IterableDataset) делается
    ограниченным буфером: как только буфер достигает buffer_size, он
    перемешивается и выдаётся наружу. Память ограничена O(buffer_size), а не
    O(всех сэмплов эпохи), и первый батч появляется сразу после заполнения
    первого чанка — без синхронного полного прохода всех окон.

    При num_workers > 0 target-недели распределяются между воркерами без
    дублирования (_shard_target_weeks); seed перемешивания берётся из
    worker_info.seed (уникален для пары «воркер × эпоха»). В основном процессе
    seed зависит от номера эпохи — запуск воспроизводим при фиксированном seed.
    """

    def __init__(self, cfg: SASRecConfig, id_to_idx: Dict[int, int],
                 target_weeks, buffer_size: int = 100_000, seed: int = 42,
                 verbose: bool = True):
        self.cfg = cfg
        self.id_to_idx = id_to_idx
        self.target_weeks = list(target_weeks)
        self.buffer_size = max(buffer_size, 1)
        self.seed = seed
        self.verbose = verbose
        self._epoch_counter = 0

    def _load_window(self, target_week: int):
        """Возвращает временный weekly_data для окна [target-history, target].

        Недели читаются с диска и СРАЗУ ремапятся в компактные индексы словаря
        (один раз на окно, а не на каждый сэмпл) — это основной горячий путь.
        """
        first_history_week = target_week - self.cfg.history_weeks
        if first_history_week < 0:
            return {}
        weekly: Dict[int, Dict[int, Tuple[List[int], List[int]]]] = {}
        for week in range(first_history_week, target_week + 1):
            df = load_week_df(self.cfg, week)
            interactions = extract_week_interactions(df, self.cfg.min_timespent_pos)
            weekly[week] = _remap_week_interactions(interactions, self.id_to_idx)
        return weekly

    def __iter__(self):
        info = get_worker_info()
        if info is not None:
            # В воркере: свои недели (без дублирования) и свой seed (воркер × эпоха).
            target_weeks = _shard_target_weeks(self.target_weeks, info)
            rng = random.Random(info.seed)
            main_process = False
        else:
            # В main-процессе: все недели; seed меняется от эпохи к эпохе.
            target_weeks = self.target_weeks
            rng = random.Random(self.seed + self._epoch_counter * 1000003)
            self._epoch_counter += 1
            main_process = True

        buffer: List[Sample] = []
        total = 0
        for target_week in target_weeks:
            weekly = self._load_window(target_week)
            if not weekly:
                continue
            window_total = 0
            # weekly уже отремаплена в _load_window — сэмплы сразу в компактных
            # индексах, per-sample ремаппинг не нужен.
            for sample in build_sliding_samples(self.cfg, weekly, target_week):
                buffer.append(sample)
                total += 1
                window_total += 1
                if len(buffer) >= self.buffer_size:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer = []
            if self.verbose and main_process:
                print(f"  [sasrec data] window -> {target_week}: "
                      f"{window_total:,} samples (total {total:,})", flush=True)
        if buffer:
            rng.shuffle(buffer)
            yield from buffer
        if self.verbose and main_process:
            print(f"  [sasrec data] epoch done: {total:,} samples", flush=True)


def collate_fn(batch: List[Sample], max_len: int):
    """RIGHT PADDING: [A, B, C, 0, 0].

    Это существенно безопаснее для causal attention, чем left padding.
    Дополнительно возвращает список user_id (нужен для групповых метрик).
    """
    batch_size = len(batch)
    inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    positions = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    targets = torch.empty(batch_size, dtype=torch.long)
    user_ids = []
    skipped_items = []
    for i, (user_id, seq, target, skipped) in enumerate(batch):
        seq = seq[-max_len:]
        seq_len = len(seq)
        inputs[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        targets[i] = target
        user_ids.append(user_id)
        skipped_items.append(skipped)
    return user_ids, inputs, positions, targets, skipped_items