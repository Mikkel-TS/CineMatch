"""Evaluate a requested onboarding size using frozen comparison models.

Positive seeds keep their original randomized nested order. This is a
descriptive follow-up on previously examined users, not a new confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from app_comparison_metrics import evaluate, paired_bootstrap
from anton_adapter import file_sha256, load_model
from campaign_neighbors import load_neighbors
from prepare_app_comparison import make_profiles
from run_app_comparison import (ROOT, RUN, VENDOR, METHODS, arrays, now,
                                score_profile, verify_frozen, write_json)


def reconstruct_observations(profiles):
    """Recover raw observed ratings, including the previously disclosed ones."""
    ids = np.asarray(profiles["user_ids"])
    supports = np.asarray(profiles["support_items"])
    raw = np.asarray(profiles["support_original_ratings"])
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids)
            or supports.ndim != 2 or supports.shape[0] != len(ids)
            or raw.shape != supports.shape or not np.isfinite(raw).all()
            or np.any((raw < 4) | (raw > 5))):
        raise ValueError("Expected unique users and positive original support ratings")
    offsets = np.asarray(profiles["target_offsets"])
    targets = np.asarray(profiles["target_items"])
    ratings = np.asarray(profiles["target_ratings"])
    if (offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(targets)
            or np.any(np.diff(offsets) < 0) or ratings.shape != targets.shape
            or not np.isfinite(ratings).all() or np.any((ratings < .5) | (ratings > 5))):
        raise ValueError("Invalid hidden rating observations")
    users, items, values = [], [], []
    for i, uid in enumerate(ids):
        lo, hi = offsets[i:i + 2]
        movies = np.concatenate([supports[i], targets[lo:hi]])
        if len(np.unique(movies)) != len(movies):
            raise ValueError("Duplicate or overlapping disclosed/hidden movie")
        users.append(np.full(len(movies), uid, dtype=np.int64))
        items.append(movies)
        values.append(np.concatenate([raw[i], ratings[lo:hi]]))
    held = {"user_id": np.concatenate(users), "movie_index": np.concatenate(items),
            "rating": np.concatenate(values)}
    order = np.argsort(held["user_id"], kind="stable")
    return {key: value[order] for key, value in held.items()}


def build_positive_profile(profiles, size, seed=20261119):
    """Reproduce the established seed sampling, with a different prefix size."""
    if not isinstance(size, (int, np.integer)) or not 1 <= size <= profiles["support_items"].shape[1]:
        raise ValueError("Requested size must lie within the original positive seed prefix")
    held = reconstruct_observations(profiles)
    new = make_profiles(held, profiles["user_ids"], "positive", int(size), seed)
    for key in ("support_items", "support_ratings", "support_original_ratings"):
        if not np.array_equal(new[key], profiles[key][:, :size]):
            raise ValueError(f"Original randomized positive prefix changed: {key}")
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--positive-opinions", type=int, default=15)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    verify_frozen(run)
    if not (run / "test/completed.json").exists():
        raise ValueError("Original comparison must be completed")
    label = f"onboarding{args.positive_opinions}_top5"
    output = run / label
    record = ROOT / "experiments/comparisons" / run.name / label
    if output.exists() or record.exists():
        raise ValueError("Follow-up output already exists; refusing to overwrite")
    source = run / "data/test/positive20.npz"
    original = arrays(source)
    p = build_positive_profile(original, args.positive_opinions)
    original_five = arrays(run / "data/test/positive5.npz")
    if args.positive_opinions >= 5:
        for key in ("support_items", "support_ratings", "support_original_ratings"):
            if not np.array_equal(p[key][:, :5], original_five[key]):
                raise ValueError("Original five positive inputs changed")
    plan = {"created_at_utc": now(), "positive_onboarding_opinions": args.positive_opinions,
            "evaluation_depth": 5, "split": "test", "users": len(p["user_ids"]),
            "status": "User-requested descriptive follow-up on previously examined users",
            "changes": "Onboarding prefix length only; models, score functions and top-five metric unchanged",
            "target_policy": "All observed movies except newly disclosed prefix; raw ratings preserved",
            "frozen_sha256": file_sha256(run / "frozen.json"),
            "source_profile_sha256": file_sha256(source),
            "original_results_sha256": file_sha256(run / "test/results.json"),
            "followup_source_sha256": file_sha256(Path(__file__)),
            "no_retraining_or_ensemble": True,
            "caveat": "Additional positive inputs become watched exclusions and leave the hidden target set; cross-size absolute scores are not a controlled measure of onboarding benefit"}
    write_json(output / "plan.json", plan)
    np.savez_compressed(output / "profiles.npz", **p)
    with threadpool_limits(limits=4):
        conditional = load_neighbors(run / "conditional")
        anton = load_model(run / "anton", vendor_dir=VENDOR)
        if not np.array_equal(conditional["movie_ids"], anton["ids"]):
            raise ValueError("Model catalogue/order mismatch")
        values, diagnostics = score_profile(p, conditional, anton)
        metrics, per_user = {}, {}
        for method in METHODS:
            metrics[method], per_user[method], tops = evaluate(values[method], p)
            np.savez_compressed(output / f"{method}.npz", top_indices=tops, **per_user[method])
        differences = {f"conditional-vs-{ref}": paired_bootstrap(
            per_user["conditional"], per_user[ref], metric="ndcg_at_5", confidence=.95,
            repetitions=5000, seed=20261119) for ref in ("anton", "rating_count", "rating_mean")}
    report = {**plan, "completed_at_utc": now(), "catalogue_items": len(anton["ids"]),
              "metrics": metrics, "differences": differences, "scoring_diagnostics": diagnostics,
              "profile_sha256": file_sha256(output / "profiles.npz"),
              "original_seed_prefixes_match": True}
    write_json(output / "results.json", report)
    write_json(record / "plan.json", plan)
    write_json(record / "results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
