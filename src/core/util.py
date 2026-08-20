import os
from typing import Any

import duckdb
import numpy as np
import polars as pl
from polars import DataFrame

from src.config import WEEKS_TRAIN_DATA, META_DATA_ROOT


def load_train_week_df(week_num: int) -> DataFrame:
    """
    загружаем данные о действиях пользователей по номеру недели
    :param week_num: номер недели, для которой выполняется загрузка
    :return: polars DataFrame
    """
    if not (-1 < week_num < 25):
        raise ValueError(f"Неверный номер недели: {week_num:02}")
    file_path = os.path.join(WEEKS_TRAIN_DATA, f"week_{week_num:02}.parquet")
    con = duckdb.connect()

    view_exists = con.execute(
        f"""select count(*) from information_schema.views where table_name = 'week_{week_num:02}'"""
    ).fetchone()[0]

    if not view_exists:
        con.execute(
            f"""
            create view week_{week_num:02} as
            select *
            from read_parquet('{file_path}')
            where timespent >= 5 and (
                "like" is not null or
                dislike is not null or
                share is not null or
                bookmark is not null or
                click_on_author is not null or
                open_comments is not null
            )
            """
        )

    return con.execute(f"""select user_id, item_id, timespent from week_{week_num:02}""").pl()

def load_items_df() -> DataFrame:
    """
    читаем данные о роликах
    :return: polars DataFrame
    """
    file_path = os.path.join(META_DATA_ROOT, "items_metadata.parquet")
    return pl.read_parquet(file_path)

def load_embeddings() -> Any:
    """
    загружаем эмбединги в том же порядке, что и записи в наборе items
    :return: список векторов
    """
    file_path = os.path.join(META_DATA_ROOT, "item_embeddings.npz")
    data = np.load(file_path)
    return data[data.files[1]]

def load_users_df() -> DataFrame:
    """
    читаем данные о пользователях
    :return: polars DataFrame
    """
    file_path = os.path.join(META_DATA_ROOT, "users_metadata.parquet")
    return pl.read_parquet(file_path)