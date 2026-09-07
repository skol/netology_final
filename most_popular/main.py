"""Точка входа: загрузка данных -> MostPopular -> метрики на валидации.

Пример:
    python -m most_popular.main --start-week 0 --end-week 23 --k 10
"""

import argparse
import time

from .config import MostPopularConfig
from .data import load_and_build
from .evaluate import evaluate
from .model import MostPopular


def parse_args():
    p = argparse.ArgumentParser(description="MostPopular для VK-LSVD")
    p.add_argument("--data_dir", type=str, default="data/raw/VK-LSVD/subsamples/up0.001_ip0.001")
    p.add_argument("--start-week", type=int, default=0, help="Начальная неделя тренировочного периода")
    p.add_argument("--end-week", type=int, default=23, help="Конечная неделя тренировочного периода (включительно)")
    p.add_argument("--min-timespent-pos", type=int, default=5, help="Порог timespent для положительного примера (>=)")
    p.add_argument("--k", type=int, default=10, help="Глубина топ-k")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MostPopularConfig(
        data_dir=args.data_dir,
        start_week=args.start_week,
        end_week=args.end_week,
        min_timespent_pos=args.min_timespent_pos,
        k=args.k,
    )
    print(f"Тренировочные недели: {cfg.start_week}..{cfg.end_week}, контрольная: {cfg.val_week}")

    t0 = time.time()
    bundle = load_and_build(cfg)
    print(f"users={bundle.n_users} items={bundle.n_items} val_users={len(bundle.val_user_idxs)} (загрузка {time.time()-t0:.1f}s)")

    model = MostPopular(bundle.popularity, bundle.item_ranking)

    t1 = time.time()
    metrics = evaluate(model, bundle, k=cfg.k)
    print(f"метрики ({time.time()-t1:.1f}s): {metrics}")

    # пример рекомендаций для первого валидируемого юзера
    u = bundle.val_user_idxs[0]
    rec = model.recommend(bundle.seen[u], top_n=cfg.k)
    print(f"пример рекомендаций для user_idx={u}: {rec}")


if __name__ == "__main__":
    main()