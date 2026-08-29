"""Точка входа: загрузка данных -> построение словаря -> обучение SASRec -> метрики.

Пример:
    python -m sasrec.main
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import SASRecConfig
from .data import (
    SequentialDataset,
    build_sliding_samples,
    collate_fn,
    extract_week_interactions,
    load_week_df,
    remap_samples,
)
from .embeddings import build_weight_matrix, load_embeddings_only
from .model import SASRec
from .train import evaluate, get_device, seed_everything, train_epoch


def main():
    cfg = SASRecConfig()
    device = get_device()
    print(f"Device: {device}")
    print(f"Temporal setup: [{cfg.history_weeks} weeks] -> 1 week")

    # --------------------------------------------------------
    # Загрузка недель
    # --------------------------------------------------------
    all_weeks = list(range(cfg.train_start_week, cfg.val_week + 1))
    weekly_data = {}
    for week in all_weeks:
        print(f"Loading week {week}...")
        df = load_week_df(cfg, week)
        weekly_data[week] = extract_week_interactions(df)
        print(f"  users: {len(weekly_data[week]):,}")

    # --------------------------------------------------------
    # Train samples (слайдинг-окна)
    # --------------------------------------------------------
    train_samples = []
    for target_week in range(cfg.train_start_week + cfg.history_weeks, cfg.train_end_week + 1):
        samples = build_sliding_samples(cfg, weekly_data=weekly_data, target_week=target_week)
        print(f"Train window [{target_week - cfg.history_weeks}, {target_week - 1}]"
              f" -> {target_week}: {len(samples):,} samples")
        train_samples.extend(samples)

    # --------------------------------------------------------
    # Validation: [w22, w23] -> w24
    # --------------------------------------------------------
    val_samples = build_sliding_samples(cfg, weekly_data=weekly_data, target_week=cfg.val_week)
    print(f"\nValidation: [{cfg.val_week - cfg.history_weeks}, {cfg.val_week - 1}]"
          f" -> {cfg.val_week}")
    print(f"Val samples: {len(val_samples):,}")

    if not train_samples:
        raise RuntimeError("No training samples.")
    if not val_samples:
        raise RuntimeError("No validation samples.")

    # --------------------------------------------------------
    # Vocabulary строится только по train + val-target айтемам
    # --------------------------------------------------------
    train_ids = set()
    for input_seq, target, skipped in train_samples:
        train_ids.update(input_seq)
        train_ids.add(target)
        train_ids.update(skipped)

    # Для оценки добавляем target валидации в словарь.
    for _, target, _ in val_samples:
        train_ids.add(target)

    id_to_idx = {old_id: idx + 1 for idx, old_id in enumerate(sorted(train_ids))}
    num_items = len(id_to_idx) + 1
    print(f"\nVocabulary: {len(id_to_idx):,} items")

    # --------------------------------------------------------
    # Pretrained embeddings -> весовая матрица
    # --------------------------------------------------------
    vecs_np, id_to_pos = load_embeddings_only(cfg.meta_data_root, cfg.emb_file, cfg.emb_width)
    weight_matrix = build_weight_matrix(
        id_to_idx=id_to_idx, vecs_np=vecs_np, id_to_pos=id_to_pos, emb_width=cfg.emb_width
    )

    # --------------------------------------------------------
    # Remap
    # --------------------------------------------------------
    train_samples = remap_samples(train_samples, id_to_idx)
    val_samples = remap_samples(val_samples, id_to_idx)
    print(f"Train samples after remap: {len(train_samples):,}")
    print(f"Val samples after remap: {len(val_samples):,}")

    # --------------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------------
    train_dataset = SequentialDataset(train_samples)
    val_dataset = SequentialDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, cfg.max_len),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: collate_fn(batch, cfg.max_len),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = SASRec(
        num_items=num_items,
        emb_dim=cfg.emb_width,
        max_len=cfg.max_len,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    with torch.no_grad():
        model.item_embedding.weight.copy_(weight_matrix.to(device))
    # padding embedding должен оставаться нулевым.
    with torch.no_grad():
        model.item_embedding.weight[0].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_ndcg = -1.0
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        hit, ndcg = evaluate(model, val_loader, device=device, top_k=cfg.top_k)
        print(f"Epoch {epoch + 1:02d}: Loss={loss:.5f} | "
              f"Hit@{cfg.top_k}={hit:.5f} | NDCG@{cfg.top_k}={ndcg:.5f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "id_to_idx": id_to_idx,
                    "config": {
                        "emb_width": cfg.emb_width,
                        "max_len": cfg.max_len,
                        "history_weeks": cfg.history_weeks,
                        "top_k": cfg.top_k,
                    },
                    "best_ndcg": best_ndcg,
                },
                "sasrec_best.pth",
            )
            print("  -> saved best model")
    print("\nTraining finished.")


if __name__ == "__main__":
    seed_everything(42)
    main()