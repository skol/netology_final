import pandas as pd
df = pd.read_parquet('../data/raw/VK-LSVD/subsamples/up0.001_ip0.001/train/week_01.parquet')
print(df.describe(include='all').T)
