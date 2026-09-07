"""Тренировка SASRec с BPR-подобным loss и оценка Hit@k / NDCG@k."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.utils import get_device, seed_everything

from .model import SASRec


def train_epoch(model: SASRec, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    for input_batch, pos_batch, target_batch, skipped_batch in loader:
        optimizer.zero_grad()
        input_batch = input_batch.to(device)
        pos_batch = pos_batch.to(device)
        target_batch = target_batch.to(device)
        user_vecs = model(input_batch, pos_batch)
        user_vecs_norm = F.normalize(user_vecs, p=2, dim=1)
        pos_embs = model.item_embedding(target_batch)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=1)

        # ---- Negative sampling ----
        neg_indices = []
        num_items = model.item_embedding.num_embeddings
        for tgt, skipped in zip(target_batch.detach().cpu().tolist(), skipped_batch):
            candidates = [s for s in skipped if s != tgt and s != 0]
            if candidates:
                neg_idx = random.choice(candidates)
            else:
                neg_idx = random.randint(1, num_items - 1)
                while neg_idx == tgt:
                    neg_idx = random.randint(1, num_items - 1)
            neg_indices.append(neg_idx)
        neg_indices = torch.tensor(neg_indices, dtype=torch.long, device=device)
        neg_embs = model.item_embedding(neg_indices)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=1)

        pos_scores = (user_vecs_norm * pos_embs_norm).sum(dim=1)
        neg_scores = (user_vecs_norm * neg_embs_norm).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became NaN/Inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: SASRec, loader: DataLoader, device: str, top_k: int = 10):
    model.eval()

    hits = 0
    ndcgs = 0.0
    total = 0

    item_weights = model.item_embedding.weight
    item_weights_norm = F.normalize(item_weights, p=2, dim=1)
    if not torch.isfinite(item_weights_norm).all():
        raise RuntimeError("NaN/Inf in item embeddings during evaluation")

    for input_batch, pos_batch, target_batch, _ in loader:
        input_batch = input_batch.to(device)
        pos_batch = pos_batch.to(device)
        target_batch = target_batch.to(device)
        user_vecs = model(input_batch, pos_batch)
        user_vecs_norm = F.normalize(user_vecs, p=2, dim=1)
        if not torch.isfinite(user_vecs_norm).all():
            raise RuntimeError("NaN/Inf in user vectors during evaluation")
        scores = torch.matmul(user_vecs_norm, item_weights_norm.T)

        # Не рекомендуем padding.
        scores[:, 0] = -float("inf")

        # Не рекомендуем уже просмотренные items.
        seen = input_batch.ne(0)
        scores.scatter_(
            1, input_batch,
            torch.where(
                seen,
                torch.full_like(input_batch, -float("inf"), dtype=scores.dtype),
                scores.gather(1, input_batch),
            ),
        )
        _, topk_indices = torch.topk(scores, k=top_k, dim=1)
        for i, tgt in enumerate(target_batch):
            matches = (topk_indices[i] == tgt)
            if matches.any():
                rank = (torch.where(matches)[0][0].item() + 1)
                hits += 1
                ndcgs += (1.0 / np.log2(rank + 1))
            total += 1

    if total == 0:
        return 0.0, 0.0

    return hits / total, ndcgs / total