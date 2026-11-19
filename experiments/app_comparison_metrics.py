"""Ranking metrics for the common-catalogue CineMatch comparison.

Scores never receive target labels. Catalogue positions must already be ordered
by ascending MovieLens ID; stable ties preserve this common order. Unknown movie
relevance earns no recovery credit and is never counted as an explicit dislike.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


RANKING_METRICS = (
    "ndcg_at_5", "hit_rate_at_5", "recall_at_5", "recall_at_10",
    "precision_at_5", "top_1_known_positive", "explicit_low_rate_at_5",
)


def _integer_array(value, name, ndim):
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array")
    return array


def evaluate(scores, profiles):
    """Return aggregate metrics, named per-user arrays, and up to ten ranks.

    `profiles` supplies support_items/support_ratings (user x support), and
    target_offsets/target_items/target_ratings (ragged held-out observations).
    Target ratings remain original MovieLens stars; >=4 is known positive and
    <=2 is explicitly low. All support items are excluded, including low and
    neutral verdicts. Hit rate and precision retain users with no positive
    targets as zeros; only recall and NDCG omit those users with NaN values.

    Precision and explicit-low rate divide by five, including for short lists;
    missing slots earn no recovery credit. Short/empty-list rates are separate.
    `top5_concentration` is the fraction of top-five exposure slots occupied by
    the five most frequently recommended movies across all users. The exposure
    HHI is also reported; neither is a measure of individual film enjoyment.
    """
    scores = np.asarray(scores)
    if (scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]
            or not np.isfinite(scores).all()):
        raise ValueError("scores must be a nonempty finite user-by-catalogue matrix")
    n_users, n_movies = scores.shape
    supports = _integer_array(profiles["support_items"], "support_items", 2)
    support_ratings = np.asarray(profiles["support_ratings"], dtype=np.float64)
    offsets = _integer_array(profiles["target_offsets"], "target_offsets", 1)
    targets = _integer_array(profiles["target_items"], "target_items", 1)
    target_ratings = np.asarray(profiles["target_ratings"], dtype=np.float64)
    if (supports.shape[0] != n_users or support_ratings.shape != supports.shape
            or not np.isfinite(support_ratings).all()):
        raise ValueError("support items and finite ratings must align with score users")
    if (offsets.shape != (n_users + 1,) or offsets[0] != 0
            or offsets[-1] != len(targets) or np.any(np.diff(offsets) < 0)
            or target_ratings.shape != targets.shape
            or not np.isfinite(target_ratings).all()
            or np.any(target_ratings < .5) or np.any(target_ratings > 5)):
        raise ValueError("target offsets, items and original MovieLens ratings must align")
    if ((supports.size and (supports.min() < 0 or supports.max() >= n_movies))
            or (targets.size and (targets.min() < 0 or targets.max() >= n_movies))):
        raise ValueError("profile movie indices must be inside the catalogue")
    if "user_ids" in profiles:
        user_ids = np.asarray(profiles["user_ids"])
        if user_ids.shape != (n_users,) or len(np.unique(user_ids)) != n_users:
            raise ValueError("user_ids must uniquely identify each score row")
    else:
        user_ids = np.arange(n_users, dtype=np.int64)

    per_user = {name: np.zeros(n_users, dtype=np.float64) for name in RANKING_METRICS}
    per_user["user_ids"] = user_ids.copy()
    per_user["positive_target_count"] = np.zeros(n_users, dtype=np.int64)
    per_user["known_target_count"] = np.zeros(n_users, dtype=np.int64)
    per_user["recommended_count_at_5"] = np.zeros(n_users, dtype=np.int64)
    depth = min(10, max(n_movies - supports.shape[1], 0))
    top_indices = np.empty((n_users, depth), dtype=np.int64)
    counts = np.zeros(n_movies, dtype=np.int64)
    discounts = 1. / np.log2(np.arange(2, 7))

    for user in range(n_users):
        seen = supports[user]
        lo, hi = offsets[user:user + 2]
        truth_items = targets[lo:hi]
        truth_ratings = target_ratings[lo:hi]
        if len(np.unique(seen)) != len(seen):
            raise ValueError("a support movie may occur only once per user")
        if len(np.unique(truth_items)) != len(truth_items):
            raise ValueError("a target movie may occur only once per user")
        if np.intersect1d(seen, truth_items).size:
            raise ValueError("support movies cannot also be held-out targets")
        eligible = np.ones(n_movies, dtype=bool)
        eligible[seen] = False
        candidates = np.flatnonzero(eligible)
        order = np.argsort(-scores[user, candidates], kind="stable")
        top = candidates[order[:depth]]
        top_indices[user] = top
        top5 = top[:5]
        positive = truth_items[truth_ratings >= 4.]
        low = truth_items[truth_ratings <= 2.]
        hits5 = np.isin(top5, positive)
        hits10 = np.isin(top, positive)
        per_user["positive_target_count"][user] = len(positive)
        per_user["known_target_count"][user] = len(truth_items)
        per_user["recommended_count_at_5"][user] = len(top5)
        per_user["hit_rate_at_5"][user] = float(hits5.any())
        per_user["precision_at_5"][user] = hits5.sum() / 5.
        per_user["top_1_known_positive"][user] = float(hits5[0]) if len(hits5) else 0.
        per_user["explicit_low_rate_at_5"][user] = np.isin(top5, low).sum() / 5.
        if len(positive):
            per_user["ndcg_at_5"][user] = (
                np.dot(hits5, discounts[:len(top5)]) / discounts[:min(5, len(positive))].sum()
            )
            per_user["recall_at_5"][user] = hits5.sum() / len(positive)
            per_user["recall_at_10"][user] = hits10.sum() / len(positive)
        else:
            for name in ("ndcg_at_5", "recall_at_5", "recall_at_10"):
                per_user[name][user] = np.nan
        counts[top5] += 1

    metrics = {}
    for name in RANKING_METRICS:
        values = per_user[name]
        metrics[name] = float(np.nanmean(values)) if np.isfinite(values).any() else None
    exposure_slots = int(counts.sum())
    shares = counts / exposure_slots if exposure_slots else np.zeros(n_movies)
    metrics.update({
        "users": n_users,
        "users_with_relevant_targets": int(np.count_nonzero(per_user["positive_target_count"])),
        "users_without_relevant_targets": int(np.count_nonzero(per_user["positive_target_count"] == 0)),
        "catalogue_items": n_movies,
        "distinct_recommended_movies": int(np.count_nonzero(counts)),
        "catalog_coverage_at_5": float(np.count_nonzero(counts) / n_movies),
        "top5_exposure_slots": exposure_slots,
        "top5_concentration": float(np.sort(shares)[-5:].sum()),
        "exposure_hhi_at_5": float(np.dot(shares, shares)),
        "short_list_rate_at_5": float(np.mean(per_user["recommended_count_at_5"] < 5)),
        "empty_list_rate": float(np.mean(per_user["recommended_count_at_5"] == 0)),
    })
    return metrics, per_user, top_indices


def paired_bootstrap(candidate, reference, metric="ndcg_at_5", *, confidence=.95,
                     repetitions=5000, seed=20261119, batch_size=128):
    """Percentile CI of a paired mean difference, resampling complete users.

    Accept named per-user dictionaries from evaluate(), or equally shaped
    one-dimensional arrays. For dictionaries, user IDs must be identically
    ordered; this protects against pairing results from different populations.
    A confidence of 1-.05/3 provides a Bonferroni interval for three declared
    primary comparisons. This is not a correction for unreported model tuning.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(candidate, Mapping) or isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping):
            raise ValueError("both paired inputs must use the same representation")
        if not np.array_equal(candidate["user_ids"], reference["user_ids"]):
            raise ValueError("paired comparison requires identical user IDs and order")
        a = np.asarray(candidate[metric], dtype=np.float64)
        b = np.asarray(reference[metric], dtype=np.float64)
    else:
        a, b = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("paired arrays must be one-dimensional and equally shaped")
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(finite_a, finite_b):
        raise ValueError("paired results must omit the same non-evaluable users")
    differences = a[finite_a] - b[finite_b]
    result = {
        "metric": metric, "users": len(differences), "excluded_users": int((~finite_a).sum()),
        "confidence": float(confidence), "repetitions": int(repetitions), "seed": int(seed),
        "difference": None, "confidence_interval": None,
    }
    if not len(differences):
        return result
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        choices = rng.integers(len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[choices].mean(axis=1)
    tail = (1. - confidence) / 2.
    result["difference"] = float(differences.mean())
    result["confidence_interval"] = np.quantile(estimates, [tail, 1. - tail]).tolist()
    return result
