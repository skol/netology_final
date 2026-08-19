import duckdb
import numpy as np
import polars as pl
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
# 2. Анализ периодичности для одного пользователя
# ====================

def analyze_periodicity(user_sequence, n_dims_to_test=5):
    """
    Проверяет, есть ли периодичность в изменении эмбеддингов пользователя.
    """
    # user_sequence: список эмбеддингов (N x 64)
    sequence = np.array(user_sequence)

    results = {}
    for dim in range(min(n_dims_to_test, sequence.shape[1])):
        signal = sequence[:, dim]

        # FFT
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal))

        # Амплитуды (убираем нулевую частоту)
        amplitudes = np.abs(fft_vals)
        amplitudes[0] = 0

        # Находим пики
        peaks, _ = find_peaks(amplitudes, height=0.1 * max(amplitudes))

        if len(peaks) > 0:
            # Берем топ-3 частоты
            top_freqs = peaks[np.argsort(amplitudes[peaks])[-3:]]
            results[dim] = {
                'frequencies': top_freqs,
                'amplitudes': amplitudes[top_freqs],
                'periods': [len(signal) / f for f in top_freqs if f > 0]
            }
        else:
            results[dim] = None

    return results


# Берем одного пользователя с длинной историей
user_counts = df.groupby('user_id').size()
active_users = user_counts[user_counts >= 50].index.tolist()

if len(active_users) > 0:
    test_user = np.random.choice(active_users)
    print(f"\nАнализирую пользователя: {test_user}")

    # Собираем его последовательность
    user_data = df[df['user_id'] == test_user].sort_index()
    user_sequence = []
    for item_id in user_data['item_id']:
        emb = emb_dict.get(item_id)
        if emb is not None:
            user_sequence.append(emb)

    if len(user_sequence) >= 20:
        results = analyze_periodicity(user_sequence, n_dims_to_test=10)

        print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
        for dim, res in results.items():
            if res is not None and len(res['periods']) > 0:
                periods = [int(p) for p in res['periods'] if p > 0]
                print(f"Размерность {dim}: найдены периоды = {periods}")
            else:
                print(f"Размерность {dim}: периодичность не обнаружена")

        # Визуализация (опционально)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for idx, dim in enumerate(range(min(10, len(user_sequence[0])))):
            signal = np.array([emb[dim] for emb in user_sequence])
            axes[idx].plot(signal)
            axes[idx].set_title(f'Dim {dim}')
            axes[idx].set_xlabel('Action #')

        plt.tight_layout()
        plt.savefig('periodicity_check.png', dpi=150)
        print("\nГрафики сохранены в periodicity_check.png")
    else:
        print("Недостаточно данных для анализа")
else:
    print("Нет пользователей с достаточной историей")
