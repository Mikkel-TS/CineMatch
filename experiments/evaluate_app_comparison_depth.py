"""Descriptive ranking-depth follow-up; preserves the original frozen comparison.

Only the evaluation cutoff changes. No fitting, new support opinions or model
selection. The original confirmation users are now known evaluation users.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from app_comparison_metrics import paired_bootstrap
from anton_adapter import file_sha256, load_model
from campaign_neighbors import load_neighbors
from run_app_comparison import (ROOT, RUN, VENDOR, METHODS, arrays, now,
                                score_profile, verify_frozen, write_json)


def evaluate_at_k(scores, profiles, k):
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Evaluation depth must be a positive integer")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite matrix")
    n, m = scores.shape
    support = np.asarray(profiles["support_items"])
    offsets = np.asarray(profiles["target_offsets"])
    targets = np.asarray(profiles["target_items"])
    ratings = np.asarray(profiles["target_ratings"], dtype=np.float64)
    user_ids = np.asarray(profiles["user_ids"])
    if (support.ndim != 2 or support.shape[0] != n or user_ids.shape != (n,)
            or len(np.unique(user_ids)) != n or offsets.shape != (n + 1,)
            or offsets[0] != 0 or offsets[-1] != len(targets)
            or np.any(np.diff(offsets) < 0) or ratings.shape != targets.shape
            or not np.isfinite(ratings).all() or np.any((ratings < .5) | (ratings > 5))):
        raise ValueError("Invalid profile alignment or ratings")
    for indices in (support, targets, offsets):
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Indices must be integers")
    if (support.size and np.any((support < 0) | (support >= m))) or (targets.size and np.any((targets < 0) | (targets >= m))):
        raise ValueError("Movie index outside catalogue")
    depth = min(k, m - support.shape[1])
    if depth < 0:
        raise ValueError("Support exceeds catalogue")
    names = [f"{name}_at_{k}" for name in ("ndcg", "hit_rate", "precision", "recall", "explicit_low_rate")]
    per_user = {name: np.zeros(n, dtype=np.float64) for name in names}
    per_user["user_ids"] = user_ids.copy()
    tops = np.empty((n, depth), dtype=np.int64)
    counts = np.zeros(m, dtype=np.int64)
    weights = 1. / np.log2(np.arange(2, k + 2))
    positive_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo, hi = offsets[i:i + 2]
        observed = targets[lo:hi]
        truth = ratings[lo:hi]
        if len(np.unique(support[i])) != support.shape[1] or len(np.unique(observed)) != len(observed):
            raise ValueError("Duplicate profile movie")
        if np.intersect1d(support[i], observed).size:
            raise ValueError("Disclosed movies overlap hidden targets")
        eligible = np.ones(m, dtype=bool)
        eligible[support[i]] = False
        candidates = np.flatnonzero(eligible)
        top = candidates[np.argsort(-scores[i, candidates], kind="stable")[:depth]]
        tops[i] = top
        positive = observed[truth >= 4.]
        low = observed[truth <= 2.]
        hits = np.isin(top, positive)
        positive_counts[i] = len(positive)
        per_user[f"hit_rate_at_{k}"][i] = float(hits.any())
        per_user[f"precision_at_{k}"][i] = hits.sum() / k
        per_user[f"explicit_low_rate_at_{k}"][i] = np.isin(top, low).sum() / k
        if len(positive):
            ideal = weights[:min(k, len(positive))].sum()
            per_user[f"ndcg_at_{k}"][i] = np.dot(hits, weights[:depth]) / ideal
            per_user[f"recall_at_{k}"][i] = hits.sum() / len(positive)
        else:
            per_user[f"ndcg_at_{k}"][i] = np.nan
            per_user[f"recall_at_{k}"][i] = np.nan
        counts[top] += 1
    metrics = {name: float(np.nanmean(values)) if np.isfinite(values).any() else None
               for name, values in per_user.items() if name != "user_ids"}
    metrics.update({"users": n, "users_with_relevant_targets": int((positive_counts > 0).sum()),
                    "catalogue_items": m, "evaluation_depth": k,
                    "distinct_recommended_movies": int((counts > 0).sum()),
                    "catalogue_coverage": float((counts > 0).mean()),
                    "short_list_rate": float(depth < k)})
    return metrics, per_user, tops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()
    if args.k < 10:
        raise ValueError("This extension must retain the original top-ten prefix")
    run = args.run_dir.resolve()
    verify_frozen(run)
    if not (run / "test/completed.json").exists():
        raise ValueError("Original comparison must be complete")
    output = run / f"depth{args.k}"
    if output.exists():
        raise ValueError("Follow-up output already exists; refusing to overwrite")
    plan = {"created_at_utc": now(), "evaluation_depth": args.k, "input_positive_opinions": 5,
            "profile": "positive5", "split": "test", "frozen_sha256": file_sha256(run / "frozen.json"),
            "original_results_sha256": file_sha256(run / "test/results.json"),
            "extension_source_sha256": file_sha256(Path(__file__)),
            "status": "User-requested descriptive follow-up on already examined confirmation users",
            "changes": "Evaluation cutoff only; no retraining, new opinions, tuning or ensemble",
            "metric": "Binary relevance raw rating>=4; logarithmic discount; ideal normalization at new cutoff",
            "uncertainty": "Paired user-bootstrap95% intervals; descriptive, not a new confirmatory claim"}
    write_json(output / "plan.json", plan)
    p = arrays(run / "data/test/positive5.npz")
    with threadpool_limits(limits=4):
        conditional = load_neighbors(run / "conditional")
        anton = load_model(run / "anton", vendor_dir=VENDOR)
        if not np.array_equal(conditional["movie_ids"], anton["ids"]):
            raise ValueError("Model catalogues differ")
        values, diagnostics = score_profile(p, conditional, anton)
        aggregate, per_user = {}, {}
        for method in METHODS:
            metrics, user_metrics, tops = evaluate_at_k(values[method], p, args.k)
            prior = arrays(run / "test" / f"positive5-{method}.npz")
            if not np.array_equal(prior["user_ids"], p["user_ids"]) or not np.array_equal(prior["top_indices"], tops[:, :10]):
                raise ValueError(f"Original top-ten rankings changed: {method}")
            aggregate[method], per_user[method] = metrics, user_metrics
            np.savez_compressed(output / f"{method}.npz", top_indices=tops, **user_metrics)
        differences = {f"conditional-vs-{ref}": paired_bootstrap(
            per_user["conditional"], per_user[ref], metric=f"ndcg_at_{args.k}",
            confidence=.95, repetitions=5000, seed=20261119)
            for ref in ("anton", "rating_count", "rating_mean")}
    report = {**plan, "completed_at_utc": now(), "users": len(p["user_ids"]),
              "catalogue_items": len(anton["ids"]), "metrics": aggregate,
              "differences": differences, "original_top10_prefixes_match": True,
              "scoring_diagnostics": diagnostics}
    write_json(output / "results.json", report)
    record = ROOT / "experiments/comparisons" / RUN.name / f"depth{args.k}"
    write_json(record / "plan.json", plan)
    write_json(record / "results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
