"""Конфигурация SASRec для VK-LSVD.

Аналог DataConfig из fpmc — все гиперпараметры собраны в одном dataclass
вместо модульных глобалов.
"""

from dataclasses import dataclass


@dataclass
class SASRecConfig:
    # ---- данные ----
    weeks_root: str = "data/raw/VK-LSVD/subsamples/up0.001_ip0.001/train"
    meta_data_root: str = "data/raw/VK-LSVD/metadata"
    emb_file: str = "item_embeddings.npz"

    # ---- временные окна ----
    history_weeks: int = 2
    train_start_week: int = 21
    train_end_week: int = 23
    val_week: int = 24

    # ---- данные / обучение ----
    max_len: int = 32
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 5
    num_workers: int = 0

    # ---- модель ----
    emb_width: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # ---- метрики ----
    top_k: int = 10

    # ---- воспроизводимость ----
    seed: int = 42