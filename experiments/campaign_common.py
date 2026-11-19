"""Shared campaign contracts. Ranking arithmetic delegates to the established evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from cinematch.data import digest, load_arrays
from cinematch.evaluation import evaluate_profiles, top_items, paired_comparison

SAMPLE = "mf16-sample-20261023"
FULL = "mf16-full-users-fixed-catalog-20261117"
CAMPAIGN = "cinematch-systematic-20261119"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def data_path(root, scale):
    if scale not in ("sample", "full"):
        raise ValueError("Training scale must be sample or full")
    return root / ("artifacts/data" if scale == "sample" else f"artifacts/experiments/{FULL}/data")


def benchmark_identity(root):
    old = root / "artifacts/data"
    return hashlib.sha256((digest(old / "validation.npz") + digest(old / "catalog.csv") +
                           digest(root / "src/cinematch/evaluation.py")).encode()).hexdigest()


def check_training(training, full_manifest, expected_movies):
    reserved = set(full_manifest["sampled_validation_user_ids"] + full_manifest["sampled_test_user_ids"]
                   + full_manifest["fresh_test_user_ids"])
    if reserved.intersection(training["user_ids"].tolist()):
        raise ValueError("Training users overlap reserved holdouts")
    if not np.array_equal(training["movie_ids"], expected_movies):
        raise ValueError("Training catalogue IDs/order differ")
    n, m = len(training["user_ids"]), len(expected_movies)
    for name, length in (("user_index", n), ("movie_index", m)):
        if np.any(training[name] < 0) or np.any(training[name] >= length):
            raise ValueError(f"Invalid {name}")
    if not np.array_equal(np.bincount(training["movie_index"], minlength=m), training["item_counts"]):
        raise ValueError("Training item counts differ from actual ratings")


def load_training(root, scale):
    training = load_arrays(data_path(root, scale), "train")
    full_manifest = json.loads((data_path(root, "full") / "manifest.json").read_text())
    with np.load(root / "artifacts/data/train.npz", allow_pickle=False) as old:
        check_training(training, full_manifest, old["movie_ids"])
    return training


def validation_profiles(root):
    old, full = data_path(root, "sample"), data_path(root, "full")
    if digest(old / "validation.npz") != digest(full / "validation.npz"):
        raise ValueError("Frozen validation profiles differ")
    return load_arrays(old, "validation")


def one_profile(profiles, i):
    lo, hi = profiles["target_offsets"][i:i+2]
    return {"user_ids": profiles["user_ids"][i:i+1],
            "support_items": profiles["support_items"][i:i+1],
            "support_ratings": profiles["support_ratings"][i:i+1],
            "target_offsets": np.array([0, hi-lo]),
            "target_items": profiles["target_items"][lo:hi],
            "target_ratings": profiles["target_ratings"][lo:hi]}


def evaluate_score_matrix(scores, profiles, rating_scores=False):
    """Identical original metric function per user; custom scores never receive targets."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != len(profiles["user_ids"]) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite user-by-catalogue matrix")
    m = scores.shape[1]
    rows, tops = [], []
    for i, values in enumerate(scores):
        p = one_profile(profiles, i)
        _, row = evaluate_profiles(0., np.zeros(m), np.empty((m, 0)), p, 1., 5., fixed_scores=values)
        if rating_scores:
            row[0, 0] = np.mean((np.clip(values[p["target_items"]], .5, 5.) - p["target_ratings"]) ** 2)
        rows.append(row[0])
        tops.append(top_items(values, p["support_items"][0]))
    rows, tops = np.asarray(rows), np.asarray(tops)
    eligible = int(np.isfinite(rows[:, 1]).sum())
    metrics = {"users": len(rows), "users_with_relevant_targets": eligible,
               "rmse_macro": float(np.sqrt(rows[:, 0].mean())) if rating_scores else None,
               "ndcg_at_5": float(np.nanmean(rows[:, 1])) if eligible else None,
               "recall_at_5": float(np.nanmean(rows[:, 2])) if eligible else None,
               "hit_rate_at_5": float(np.nanmean(rows[:, 3])) if eligible else None,
               "catalog_coverage_at_5": len(np.unique(tops)) / m,
               "distinct_recommended_movies": int(len(np.unique(tops)))}
    return metrics, rows, tops


def comparisons(metrics, rows, references):
    result = {}
    for name, (ref_metrics, ref_rows) in references.items():
        changes = {}
        for key in ("ndcg_at_5", "hit_rate_at_5", "recall_at_5", "catalog_coverage_at_5", "rmse_macro"):
            a, b = metrics.get(key), ref_metrics.get(key)
            changes[key] = {"absolute": None if a is None or b is None else a-b,
                            "relative_percent": None if a is None or b in (None, 0) else 100*(a-b)/b}
        result[name] = {"changes": changes, **paired_comparison(rows, ref_rows, 42)}
    return result


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    return (values-values.mean()) / std if std > 1e-12 else np.zeros_like(values)


def blend_scores(personal, counts, items, ratings, config):
    """Candidate generation is a scoring stage; every catalogue title stays evaluable."""
    alpha = float(config.get("alpha", .25))
    if not 0 <= alpha <= 1:
        raise ValueError("Blend alpha must be in [0, 1]")
    if config.get("confidence"):
        # Confidence uses onboarding only, never hidden targets.
        n = int(np.count_nonzero(np.asarray(ratings) >= 4))
        alpha *= n / (n + float(config.get("confidence_strength", 5)))
    if config.get("candidate_count"):
        k = int(config["candidate_count"])
        selected = top_items(np.asarray(counts), np.asarray(items), k)
        result = np.full(len(counts), float(np.min(personal))-1e6)
        result[selected] = personal[selected]
        return result
    return alpha*zscore(personal) + (1-alpha)*zscore(np.log1p(counts))
