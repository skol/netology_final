"""Вспомогательные функции для тестов: синтетические данные VK-LSVD."""

from pathlib import Path

import polars as pl

# Полный набор колонок взаимодействий VK-LSVD (см. common.labels).
REACTION_COLS = [
    "like",
    "dislike",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
]
ALL_COLS = ["user_id", "item_id", "timespent"] + REACTION_COLS


def make_interaction_df(rows) -> pl.DataFrame:
    """DataFrame взаимодействий из кортежей

    (user_id, item_id, timespent, like, dislike, share,
     bookmark, click_on_author, open_comments).
    """
    return pl.DataFrame({col: [r[i] for r in rows] for i, col in enumerate(ALL_COLS)})


def write_week_parquet(data_dir: Path, sub: str, week: int, df: pl.DataFrame) -> Path:
    """Записывает parquet недели в data_dir/<sub>/week_XX.parquet."""
    path = data_dir / sub / f"week_{week:02}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path