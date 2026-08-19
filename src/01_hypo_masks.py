# гипотеза с получением маски значимых атрибутов векторов пользователя
from collections import Counter

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

print("Загружаю эмбеддинги...")

items_meta = pl.read_parquet("../data/raw/VK-LSVD/metadata/items_metadata.parquet")
item_ids = items_meta["item_id"].to_numpy()

data = np.load("../data/raw/VK-LSVD/metadata/item_embeddings.npz")

embeddings = data[data.files[1]]

emb_dict = dict(zip(item_ids, embeddings))
print(f"Загружено {len(emb_dict)} эмбеддингов")

print("Загружаю взаимодействия (одна неделя)...")
df = duckdb.sql("""
                SELECT user_id, item_id, timespent
                FROM '../data/raw/VK-LSVD/subsamples/up0.001_ip0.001/train/week_00.parquet'
                """).df()

print(f"Загружено {len(df)} записей")
print(f"Уникальных пользователей: {df['user_id'].nunique()}")

# ====================
# 2. Отбор активных пользователей
# ====================

# Берем только тех, у кого >= 15 действий (чтобы было с чем работать)
user_counts = df.groupby('user_id').size()
active_users = user_counts[user_counts >= 15].index.tolist()

# Для скорости берем случайную выборку из 200 пользователей (если хочешь больше - увеличь)
sample_size = min(200, len(active_users))
np.random.seed(42)
selected_users = np.random.choice(active_users, size=sample_size, replace=False)

print(f"Выбрано {len(selected_users)} активных пользователей для анализа")


# ====================
# 3. Функция анализа важности для одного пользователя
# ====================

def get_feature_importance_for_user(user_id, df, emb_dict, window_size=5):
    """
    Возвращает массив важностей для 64 размерностей эмбеддинга
    """
    # История пользователя (сортируем по порядку, т.к. timestamp нет)
    user_data = df[df['user_id'] == user_id].head(100)

    # Собираем эмбеддинги
    user_embs = []
    for item_id in user_data['item_id']:
        emb = emb_dict.get(item_id)
        if emb is not None:
            user_embs.append(emb)

    if len(user_embs) < window_size + 1:
        return None

    user_embs = np.array(user_embs)

    # Формируем выборку
    X, y = [], []
    for i in range(len(user_embs) - window_size):
        X.append(user_embs[i:i + window_size].flatten())
        y.append(user_embs[i + window_size])

    X = np.array(X)
    y = np.array(y)

    if len(X) < 5:
        return None

    # Обучаем дерево (глубина небольшая, чтобы не переобучаться)
    tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree.fit(X, y)

    # Важности для всех признаков
    all_imp = tree.feature_importances_

    # Нас интересует только последнее окно (самый свежий контекст)
    last_window_imp = all_imp[-64:]

    return last_window_imp


# ====================
# 4. Сбор важностей для всех пользователей
# ====================

print("Собираю важности для каждого пользователя...")
importance_matrix = []
users_with_data = []

for user in selected_users:
    imp = get_feature_importance_for_user(user, df, emb_dict)
    if imp is not None:
        importance_matrix.append(imp)
        users_with_data.append(user)

importance_matrix = np.array(importance_matrix)
print(f"Удалось собрать данные для {len(importance_matrix)} пользователей")

# ====================
# 5. Проверка гипотезы: различаются ли маски?
# ====================

# Для каждого пользователя находим топ-10 размерностей
top_k_per_user = []
for imp in importance_matrix:
    top_indices = np.argsort(imp)[-10:].tolist()
    top_k_per_user.append(top_indices)

# Считаем, насколько часто каждая размерность попадает в топ-10
all_top = [idx for sublist in top_k_per_user for idx in sublist]
counter = Counter(all_top)

# Если какая-то размерность встречается более чем у 50% пользователей, то маски не персонализированы
threshold = 0.5 * len(top_k_per_user)
common_features = [dim for dim, count in counter.items() if count > threshold]

print("\n=== РЕЗУЛЬТАТ ПРОВЕРКИ ===")
print(f"Всего пользователей: {len(top_k_per_user)}")
print(f"Размерностей, которые попали в топ-10 чаще чем у 50% пользователей: {len(common_features)}")

if len(common_features) < 5:
    print("✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА: маски персонализированы, общих важных размерностей почти нет")
else:
    print("❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА: есть общие размерности, важные для большинства")
    print(f"   Самые частые: {common_features[:10]}")

# ====================
# 6. Визуализация (сохраняется в файл)
# ====================

# Тепловая карта важности для 20 случайных пользователей
fig, ax = plt.subplots(figsize=(16, 8))
sample_users_for_plot = min(20, len(importance_matrix))
sns.heatmap(importance_matrix[:sample_users_for_plot], cmap='RdBu_r', ax=ax, xticklabels=range(1, 65),
            yticklabels=[f"U{i}" for i in range(sample_users_for_plot)])
ax.set_xlabel('Dimension')
ax.set_ylabel('User')
ax.set_title('Важность размерностей эмбеддинга для разных пользователей\n(Красный = более важная)')
plt.tight_layout()
plt.savefig('masks_heatmap.png', dpi=150)
print("\nТепловая карта сохранена в masks_heatmap.png")

# Гистограмма частоты попадания размерностей в топ-10
fig, ax = plt.subplots(figsize=(14, 6))
dims = sorted(counter.keys())
counts = [counter[d] for d in dims]
ax.bar(dims, counts, color='steelblue')
ax.axhline(y=threshold, color='red', linestyle='--', label=f'50% порог ({int(threshold)})')
ax.set_xlabel('Размерность эмбеддинга')
ax.set_ylabel('Количество пользователей, у которых она в топ-10')
ax.set_title('Частота попадания размерностей в топ-10 важных')
ax.legend()
plt.tight_layout()
plt.savefig('masks_frequency.png', dpi=150)
print("Гистограмма сохранена в masks_frequency.png")

print("\n✅ Анализ завершен!")
