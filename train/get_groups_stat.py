import numpy as np
import polars as pl
from common.data import read_files

# быстро получаем статистику по пользователям, чтобы проще было определятся с границами групп

train_df = read_files([
    f"data/raw/VK-LSVD/subsamples/up0.001_ip0.001/train/week_{i:02}.parquet"
    for i in range(23, 25)
])

sizes = (train_df.group_by("user_id").agg(pl.len().alias("n")).sort("n"))
arr = sizes["n"].to_numpy()
print("n_users:", len(arr))
print("min:", arr.min())
print("p25:", np.percentile(arr, 25))
print("p50:", np.percentile(arr, 50))
print("p75:", np.percentile(arr, 75))
print("p90:", np.percentile(arr, 90))
print("p95:", np.percentile(arr, 95))
print("max:", arr.max())