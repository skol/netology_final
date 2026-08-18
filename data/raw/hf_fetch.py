import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

subsample_name = 'up0.001_ip0.001'
content_embedding_size = 32

train_interactions_files = [f'subsamples/{subsample_name}/train/week_{i:02}.parquet' for i in range(25)]
val_interactions_file = [f'subsamples/{subsample_name}/validation/week_25.parquet']

metadata_files = ['metadata/users_metadata.parquet', 'metadata/items_metadata.parquet', 'metadata/item_embeddings.npz']

for file in (train_interactions_files + val_interactions_file + metadata_files):
    hf_hub_download(repo_id='deepvk/VK-LSVD', repo_type='dataset', filename=file, local_dir='VK-LSVD')

train_interactions = pl.concat([pl.scan_parquet(f'VK-LSVD/{file}') for file in train_interactions_files])
train_interactions = train_interactions.collect(engine='streaming')

val_interactions = pl.read_parquet(f'VK-LSVD/{val_interactions_file[0]}')

train_users = train_interactions.select('user_id').unique()
train_items = train_interactions.select('item_id').unique()

item_ids = np.load('VK-LSVD/metadata/item_embeddings.npz')['item_id']
item_embeddings = np.load('VK-LSVD/metadata/item_embeddings.npz')['embedding']

mask = np.isin(item_ids, train_items.to_numpy())
item_ids = item_ids[mask]
item_embeddings = item_embeddings[mask]
item_embeddings = item_embeddings[:, :content_embedding_size]

users_metadata = pl.read_parquet('VK-LSVD/metadata/users_metadata.parquet')
items_metadata = pl.read_parquet('VK-LSVD/metadata/items_metadata.parquet')

users_metadata = users_metadata.join(train_users, on='user_id')
items_metadata = items_metadata.join(train_items, on='item_id')
items_metadata = items_metadata.join(pl.DataFrame({'item_id': item_ids, 'embedding': item_embeddings}), on='item_id')
