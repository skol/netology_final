from collections import Counter

import duckdb
import numpy as np
import polars as pl
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

print("Загрузка эмбеддингов и сессий...")

items_meta = pl.read_parquet("../data/raw/VK-LSVD/metadata/items_metadata.parquet")
item_ids = items_meta["item_id"].to_numpy()
data = np.load("../data/raw/VK-LSVD/metadata/item_embeddings.npz")
embeddings = data[data.files[1]]
emb_dict = dict(zip(item_ids, embeddings))

con = duckdb.connect()
con.execute(
    """
    create view week_00 as 
    select * 
    from read_parquet('../data/raw/VK-LSVD/subsamples/up0.001_ip0.001/train/week_10.parquet')
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

# Фильтруем пользователей, у которых длинная история
MIN_HISTORY = 100

query = """
select user_id, array_agg(item_id)
from week_00 
group by user_id 
having count(*) > ?
"""

user_histories = dict(con.execute(query, [MIN_HISTORY]).fetchall())

df_filtered = con.execute(query, [MIN_HISTORY]).pl()


def evaluate_user_predictability(item_list, emb_dict, n_dims=10):
    # Собираем матрицу эмбеддингов для пользователя
    seq = [emb_dict[i_id] for i_id in item_list if i_id in emb_dict]
    if len(seq) < MIN_HISTORY:
        return None
    matrix = np.array(seq)  # Формат: (N_steps, Dimensions)
    n_steps, dim_size = matrix.shape

    # Будем валидироваться на последних 3-х шагах истории пользователя
    val_steps = 3
    train_matrix = matrix[:-val_steps]
    val_matrix = matrix[-val_steps:]

    detected_any_period = False
    gains_over_naive = []
    global_periods = []

    for dim in range(min(n_dims, dim_size)):
        raw_signal = train_matrix[:, dim].astype(np.float32)

        # ПРАВКА 1: Сглаживание скользящим средним (окно 3) для удаления ВЧ-шума
        signal = pl.Series(raw_signal).rolling_mean(window_size=3, min_periods=1).to_numpy()

        # 1. Детрендинг (Выделяем линейную тенденцию)
        x_train = np.arange(len(signal))
        trend_coeffs = np.polyfit(x_train, signal, deg=1)
        detrended = signal - np.polyval(trend_coeffs, x_train)

        # 2. FFT анализ
        fft_vals = fft(detrended)
        freqs = fftfreq(len(signal))
        half = len(signal) // 2

        pos_freqs = freqs[:half]
        amplitudes = np.abs(fft_vals[:half])
        amplitudes[0] = 0  # глушим константу

        # Порог поиска пиков
        peaks, _ = find_peaks(amplitudes, height=0.2 * max(amplitudes))

        best_period = None
        if len(peaks) > 0:
            top_peak = peaks[np.argsort(amplitudes[peaks])[-1]]
            if pos_freqs[top_peak] > 0:
                best_period = 1.0 / pos_freqs[top_peak]

                # ПРАВКА 2: Фильтруем и запоминаем только макро-периоды (от 4 и выше)
                if 4.0 <= best_period < len(signal):
                    detected_any_period = True
                    global_periods.append(round(best_period, 1))

        # 3. Наш прогноз = продолжение тренда + (если нашли макро-период) зацикливание волны
        x_val = np.arange(len(signal), len(signal) + val_steps)
        smart_pred = np.polyval(trend_coeffs, x_val)

        if best_period and best_period >= 4.0:
            shift = int(round(best_period))
            if 0 < shift <= len(detrended):
                window = detrended[-shift: -shift + val_steps] if shift >= val_steps else detrended[-val_steps:]
                if window.shape == smart_pred.shape:
                    smart_pred += window

        # =====================================================================
        # ПРАВКА 3: ВАЛИДАЦИЯ НАПРАВЛЕНИЯ ТРЕНДА (ВВЕРХ ИЛИ ВНИЗ)
        # =====================================================================
        # На сколько реально изменился сигнал в будущем относительно последней известной точки
        val_signal = val_matrix[:, dim].astype(np.float32)
        real_delta = np.mean(val_signal) - raw_signal[-1]

        # На сколько наше предсказание ожидает изменение
        smart_delta = np.mean(smart_pred) - raw_signal[-1]

        # Если знаки изменений совпадают — маска угадала вектор движения (успех)
        # Наивный прогноз (delta=0) здесь автоматически проигрывает, так как ничего не угадывает
        if np.sign(real_delta) == np.sign(smart_delta) and real_delta != 0:
            gains_over_naive.append(1.0)  # Угадали
        else:
            gains_over_naive.append(0.0)  # Не угадали

    return {
        "has_period": detected_any_period,
        "avg_gain": np.mean(gains_over_naive) if gains_over_naive else 0,
        "periods": global_periods
    }


print("Запуск циклического анализа по всей когорте...")

stats = {"total_tested": 0, "users_with_periods": 0, "better_than_naive_count": 0, "all_discovered_periods": []}

for user_id, item_list in user_histories.items():
    res = evaluate_user_predictability(item_list, emb_dict, n_dims=64)

    if res is not None:
        stats["total_tested"] += 1
        if res["has_period"]:
            stats["users_with_periods"] += 1
        if res["avg_gain"] > 0:  # Ошибка уменьшилась по сравнению с "прошлым шагом"
            stats["better_than_naive_count"] += 1
        stats["all_discovered_periods"].extend(res["periods"])

print("\n=== ФИНАЛЬНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ ===")
pct_period = (stats["users_with_periods"] / stats["total_tested"]) * 100
pct_gain = (stats["better_than_naive_count"] / stats["total_tested"]) * 100

print(f"Всего валидировано пользователей: {stats['total_tested']}")
print(f"Доля пользователей с выраженной цикличностью: {pct_period:.2f}%")
print(f"Доля пользователей, где тренд+период точнее наивного прогноза: {pct_gain:.2f}%")

if len(stats["all_discovered_periods"]) > 0:
    most_common = Counter(stats["all_discovered_periods"]).most_common(64)
    print("\nТоп-64 самых частых периодов (в количестве видеороликов):")
    for p, count in most_common:
        print(f"  Период повторения: {p} действий (встретился {count} раз)")
