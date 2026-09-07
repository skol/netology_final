"""Метрики next-item рекомендаций: Recall@k, NDCG@k, MRR@k."""

from common.metrics import eval_metrics

from .data import DataBundle


def evaluate(model, bundle: DataBundle, k: int = 10) -> dict:
    """Recall@k / NDCG@k / MRR@k на юзерах из валидации."""
    gt_positions = []
    for pos, u in enumerate(bundle.val_user_idxs):
        gt = bundle.val_gt[pos]
        ranks = model.rank_positions(bundle.seen[u], gt)
        gt_positions.append(ranks)
    return eval_metrics(gt_positions, k)