"""Full-catalogue new-user evaluation, with unrated relevance left unknown."""

from __future__ import annotations

import numpy as np

from .model import fold_in


def top_items(scores: np.ndarray, excluded: np.ndarray, k: int = 5) -> np.ndarray:
    if k < 1:
        raise ValueError("Recommendation count must be positive.")
    candidates = np.setdiff1d(np.arange(len(scores)), excluded)
    # Stable tie breaking uses catalogue order (ascending MovieLens ID).
    return candidates[np.lexsort((candidates, -scores[candidates]))[:k]]


def evaluate_profiles(mean: float, bias: np.ndarray, factors: np.ndarray, profiles: dict,
                      regularization: float, bias_regularization: float,
                      fixed_scores: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    discounts = 1 / np.log2(np.arange(2, 7))
    rows, recommendations = [], set()
    for index in range(len(profiles["user_ids"])):
        seeds, ratings = profiles["support_items"][index], profiles["support_ratings"][index]
        start, end = profiles["target_offsets"][index:index + 2]
        targets = profiles["target_items"][start:end]
        truth = profiles["target_ratings"][start:end]
        scores = (fold_in(mean, bias, factors, seeds, ratings, regularization, bias_regularization)
                  if fixed_scores is None else fixed_scores)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model scores; reduce the learning rate and check input data.")
        top = top_items(scores, seeds)
        recommendations.update(top.tolist())
        positive = targets[truth >= 4]
        hits = np.isin(top, positive)
        error = np.clip(scores[targets], .5, 5) - truth if fixed_scores is None else None
        if len(positive):
            ndcg = float((hits * discounts[:len(top)]).sum() / discounts[:min(5, len(positive))].sum())
            recall = float(hits.sum() / len(positive))
            hit_rate = float(hits.any())
        else:
            ndcg = recall = hit_rate = np.nan
        rows.append([float(np.mean(error ** 2)) if error is not None else np.nan, ndcg, recall, hit_rate])
    per_user = np.asarray(rows)
    positive_users = int(np.isfinite(per_user[:, 1]).sum())
    metrics = {
        "users": len(rows), "users_with_relevant_targets": positive_users,
        "rmse_macro": float(np.sqrt(per_user[:, 0].mean())) if fixed_scores is None else None,
        "ndcg_at_5": float(np.nanmean(per_user[:, 1])) if positive_users else None,
        "recall_at_5": float(np.nanmean(per_user[:, 2])) if positive_users else None,
        "hit_rate_at_5": float(np.nanmean(per_user[:, 3])) if positive_users else None,
        "catalog_coverage_at_5": len(recommendations) / len(bias),
    }
    return metrics, per_user


def paired_comparison(model: np.ndarray, baseline: np.ndarray, seed: int) -> dict:
    differences = model[:, 1] - baseline[:, 1]
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return {"ndcg_at_5_difference": None, "bootstrap_95_percent": None}
    rng = np.random.default_rng(seed)
    # Loop keeps the confidence interval memory bounded for large evaluations.
    samples = [rng.choice(differences, len(differences), replace=True).mean() for _ in range(2000)]
    return {"ndcg_at_5_difference": float(differences.mean()),
            "bootstrap_95_percent": np.quantile(samples, [.025, .975]).tolist()}
