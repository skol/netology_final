# продолжение предыдущей гипотезы, проверяем наличие общих и индивидуальных атрибутов
import duckdb
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ====================
# 1. Загрузка данных
# ====================
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
# 2. Глобальная статистика (по всей базе)
# ====================

print("\nСчитаю глобальную статистику...")
global_mean = embeddings.mean(axis=0)
global_std = embeddings.std(axis=0) + 1e-8

# Якорные размерности (из прошлого эксперимента)
ANCHOR_DIMS = [54, 55, 57, 58, 59, 60, 61, 62, 63]


# ====================
# 3. Функция построения Z-профиля
# ====================

def build_z_profile(user_embeddings, threshold=2.0):
    """Строит профиль: якорные = глобальные, остальные = только если Z>2"""
    user_mean = user_embeddings.mean(axis=0)
    z_scores = np.abs((user_mean - global_mean) / global_std)

    profile = global_mean.copy()
    for dim in range(64):
        if dim not in ANCHOR_DIMS and z_scores[dim] > threshold:
            profile[dim] = user_mean[dim]
    return profile


# ====================
# 4. Подготовка данных для эксперимента
# ====================

print("\nГотовлю данные для эксперимента...")

# Берем пользователей с >= 30 действий
user_counts = df.groupby('user_id').size()
active_users = user_counts[user_counts >= 30].index.tolist()

# Для скорости берем 100 пользователей (если хочешь больше - увеличь)
sample_users = np.random.choice(active_users, size=min(100, len(active_users)), replace=False)

X_baseline = []  # только эмбеддинг видео
X_avg_profile = []  # эмбеддинг + усредненный профиль
X_z_profile = []  # эмбеддинг + Z-профиль
y = []  # следующий эмбеддинг

for user in tqdm(sample_users, desc="Обработка пользователей"):
    user_data = df[df['user_id'] == user].sort_index()
    user_embs = []
    for item_id in user_data['item_id']:
        emb = emb_dict.get(item_id)
        if emb is not None:
            user_embs.append(emb)
    if len(user_embs) < 10:
        continue

    user_embs = np.array(user_embs)

    # Профили
    avg_profile = user_embs.mean(axis=0)
    z_profile = build_z_profile(user_embs)

    # Для каждой пары (текущее видео -> следующее)
    for i in range(len(user_embs) - 1):
        X_baseline.append(user_embs[i])
        X_avg_profile.append(np.concatenate([user_embs[i], avg_profile]))
        X_z_profile.append(np.concatenate([user_embs[i], z_profile]))
        y.append(user_embs[i + 1])

X_baseline = np.array(X_baseline)
X_avg_profile = np.array(X_avg_profile)
X_z_profile = np.array(X_z_profile)
y = np.array(y)

print(f"\nСобрано {len(X_baseline)} примеров")


# ====================
# 5. Обучение и сравнение моделей
# ====================

def evaluate_model(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name:20} | MSE: {mse:.6f}")
    return mse


print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===\n")
mse_baseline = evaluate_model(X_baseline, y, "1. Baseline (только видео)")
mse_avg = evaluate_model(X_avg_profile, y, "2. + Усредненный профиль")
mse_z = evaluate_model(X_z_profile, y, "3. + Z-Profile (гибрид)")

print("\n=== ВЫВОДЫ ===")
print(f"Улучшение усредненного профиля над Baseline: {(mse_baseline - mse_avg) / mse_baseline * 100:.2f}%")
print(f"Улучшение Z-Profile над Baseline: {(mse_baseline - mse_z) / mse_baseline * 100:.2f}%")
print(f"Улучшение Z-Profile над усредненным: {(mse_avg - mse_z) / mse_avg * 100:.2f}%")