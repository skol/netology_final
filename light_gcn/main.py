"""Точка входа: загрузка данных -> тренировка LightGCN -> метрики.

Пример:
    python -m light_gcn.main --epochs 10 --dim 64 --k 10
"""

import argparse
import time

import torch

from common.utils import get_device

from .data import DataConfig, load_and_build
from .model import LightGCN
from .train import evaluate, train_model


def parse_args():
    p = argparse.ArgumentParser(description="LightGCN для VK-LSVD")
    p.add_argument("--data_dir", type=str,
                   default="data/raw/VK-LSVD/subsamples/up0.001_ip0.001")
    p.add_argument("--n_train_weeks", type=int, default=25)
    p.add_argument("--min_user_history", type=int, default=2)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"device: {device}")

    t0 = time.time()
    cfg = DataConfig(data_dir=args.data_dir, n_train_weeks=args.n_train_weeks,
                     min_user_history=args.min_user_history)
    bundle = load_and_build(cfg)
    print(f"users={bundle.n_users} items={bundle.n_items} "
          f"edges={len(bundle.train_users)} val_users={len(bundle.val_user_idxs)} "
          f"({time.time()-t0:.1f}s)")

    model = LightGCN(n_users=bundle.n_users, n_items=bundle.n_items,
                     dim=args.dim, n_layers=args.n_layers)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, bundle, device=device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    t1 = time.time()
    metrics = evaluate(model, bundle, device=device, k=args.k)
    print(f"metrics ({time.time()-t1:.1f}s): {metrics}")

    # сохраняем состояние модели на диск
    torch.save(model.state_dict(), "light_gcn.pt")
    print("saved to light_gcn.pt")


if __name__ == "__main__":
    main()