from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.core.util import load_items_df, load_embeddings, load_train_week_df

MIN_HISTORY_SIZE = 100
WINDOW_SIZE = 5
MAX_DEPTH = 4  # оставляем 4 для сравнения
N_ESTIMATORS = 50
N_REPEATS_PERM = 5
TOP_K = 10
MAX_USERS = 100

print("Загрузка данных...")
items_meta = load_items_df()
item_ids = items_meta["item_id"].to_numpy()
embeddings = load_embeddings()
emb_dict = dict(zip(item_ids, embeddings))

df = load_train_week_df(0)
selected_users = df.filter(pl.len().over("user_id") > MIN_HISTORY_SIZE)["user_id"].to_list()
selected_users = selected_users[:MAX_USERS]
print(f"Анализируем {len(selected_users)} пользователей")


def analyze_user(user_id, df, emb_dict):
    """Анализ одного пользователя"""
    user_data = df.filter(pl.col('user_id') == user_id).tail(MIN_HISTORY_SIZE)

    user_embs = []
    for item_id in user_data['item_id']:
        emb = emb_dict.get(item_id)
        if emb is not None:
            user_embs.append(emb)

    if len(user_embs) < WINDOW_SIZE + 5:
        return None

    user_embs = np.array(user_embs)

    scaler = StandardScaler()
    user_embs_norm = scaler.fit_transform(user_embs)

    X, y = [], []
    for i in range(len(user_embs_norm) - WINDOW_SIZE):
        X.append(user_embs_norm[i:i + WINDOW_SIZE].flatten())
        y.append(user_embs_norm[i + WINDOW_SIZE])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 5:
        return None

    forest = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    forest.fit(X, y)

    perm_result = permutation_importance(
        forest, X, y,
        n_repeats=N_REPEATS_PERM,
        random_state=42
    )

    return perm_result.importances_mean[-64:]


print("Сбор важностей...")
importance_matrix = []
for user in selected_users:
    imp = analyze_user(user, df, emb_dict)
    if imp is not None:
        importance_matrix.append(imp)

importance_matrix = np.array(importance_matrix)
print(f"Получено {len(importance_matrix)} пользователей")

top_k_per_user = []
for imp in importance_matrix:
    top_indices = np.argsort(imp)[-TOP_K:].tolist()
    top_k_per_user.append(top_indices)

counter = Counter([idx for sublist in top_k_per_user for idx in sublist])

print("Создание графиков...")

fig, ax = plt.subplots(figsize=(14, 6))
dims = sorted(counter.keys())
counts = [counter[d] for d in dims]
bars = ax.bar(dims, counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Размерность эмбеддинга')
ax.set_ylabel('Количество пользователей, у которых она в топ-10')
ax.set_title('Частота попадания размерностей в топ-10 важных')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('masks_frequency.png', dpi=150)
print("✅ masks_frequency.png")

fig, ax = plt.subplots(figsize=(16, 10))
sample_users = min(50, len(importance_matrix))
sns.heatmap(
    importance_matrix[:sample_users],
    cmap='RdBu_r',
    ax=ax,
    xticklabels=range(1, 65),
    yticklabels=[f"U{i}" for i in range(sample_users)],
    cbar_kws={'label': 'Важность'}
)
ax.set_xlabel('Размерность эмбеддинга')
ax.set_ylabel('Пользователь')
ax.set_title('Важность размерностей эмбеддинга для разных пользователей\n(Красный = более важная)')
plt.tight_layout()
plt.savefig('masks_heatmap.png', dpi=150)
print("✅ masks_heatmap.png")

print("\n" + "=" * 50)
print("СРАВНЕНИЕ СО СТАРОЙ ВЕРСИЕЙ")
print("=" * 50)
print(f"Количество пользователей: {len(importance_matrix)}")

# Считаем, сколько размерностей попали в топ-10 хотя бы 1 раз
unique_dims = len(counter)
print(f"Уникальных размерностей в топ-10: {unique_dims}/64")

# Самые частые размерности
top10 = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nТоп-10 самых частых размерностей:")
for dim, count in top10:
    pct = count / len(importance_matrix) * 100
    print(f"  Размерность {dim:2d}: {pct:5.1f}% пользователей")
