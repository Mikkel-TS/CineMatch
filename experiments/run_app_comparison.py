"""Frozen direct comparison of refitted CineMatch and Anton model methods.

Run prepare_app_comparison.py first. Train/freeze, validate, then confirm once.
No changes to production artifacts, no ensemble search, no test-based fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anton_adapter import (UPSTREAM_COMMIT, file_sha256, fit_validation_weights,
                           load_components, load_model, load_upstream,
                           score_candidates, train_components)
from app_comparison_metrics import evaluate, paired_bootstrap
from campaign_neighbors import load_neighbors, score_neighbors, train_neighbors

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "experiments/vendor/anton_6e85958"
RUN = ROOT / "artifacts/experiments/app-model-comparison-20261119-v1"
PROFILES = ("positive5", "positive10", "positive20", "mixed10", "mixed20")
METHODS = ("conditional", "anton", "rating_count", "rating_mean")
CONFIG = {"feedback": "positive", "similarity": "conditional", "top_k": 100,
          "shrink": 10., "positive_threshold": 4., "aggregation": "positive",
          "block_size": 128}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def now():
    return datetime.now(timezone.utc).isoformat()


def policy():
    return {
        "run_id": RUN.name, "upstream_commit": UPSTREAM_COMMIT,
        "objective": "Recover hidden recorded likes on a common verified catalogue",
        "primary_profile": "positive5", "primary_metric": "ndcg_at_5",
        "primary_comparisons": ["conditional-vs-anton", "conditional-vs-rating_count", "conditional-vs-rating_mean"],
        "confidence": .95, "familywise_confidence": 1 - .05 / 3,
        "bootstrap_repetitions": 5000, "bootstrap_seed": 20261119,
        "minimum_worthwhile_absolute_ndcg_gain": .01,
        "interpretation_of_margin": "One NDCG percentage point is a prespecified practical screening margin, not a validated business value",
        "profiles": list(PROFILES), "methods": list(METHODS),
        "conditional_config": CONFIG,
        "anton": {"serving": "Pinned score_candidates, including clipping and NO under-ten gate",
                  "neighbors_per_source": 50, "fit_profile": 10, "fit_targets_maximum": 30,
                  "fitting": "One least-squares fit on validation, rounded to four decimals",
                  "sources": ["content", "audience", "tags", "film_mean"],
                  "tag_policy": "Exclude every old and new heldout user before fitting vocabulary"},
        "rating_conversion": {"app_proxy": "floor(MovieLens rating), clipped to 1..5",
                              "anton_serving": "1 + 0.75 * (app verdict - 1)",
                              "anton_fit": "Native upstream 1 + (raw rating - 0.5) * 2/3",
                              "relevant_targets": "Raw MovieLens >=4; low targets <=2"},
        "exclusions": "Only disclosed opinions are watched exclusions; all remaining observed labels are hidden targets",
        "candidate_order": "Shared ascending MovieLens IDs; stable ties",
        "scope": "Refitted algorithms on shared catalogue, not unchanged production artifacts or an online A/B test",
        "secondary_profiles": "Exploratory, not additional confirmatory winner selection",
        "under_ten_policy_diagnostic": "Declared Anton fallback would equal the rating_mean baseline; primary uses actual code",
        "hybrids": "None: compare standalone models first",
    }


def source_paths():
    paths = [Path(__file__), ROOT / "experiments/anton_adapter.py",
             ROOT / "experiments/app_comparison_metrics.py",
             ROOT / "experiments/prepare_app_comparison.py",
             ROOT / "experiments/campaign_neighbors.py"]
    paths += [VENDOR / f for f in ("cinematch.py", "movielens.py", "rating_model.py")]
    return paths


def relative_hashes(paths, base):
    return {str(p.relative_to(base)): file_sha256(p) for p in paths}


def train(run):
    if (run / "frozen.json").exists():
        raise ValueError("Already frozen; use a new run directory to change a candidate")
    run.mkdir(parents=True, exist_ok=True)
    plan_path = run / "policy.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != policy():
        raise ValueError("Existing policy differs; preserve it and create a new run")
    write_json(plan_path, policy())
    contract_path = run / "training_contract.json"
    contract_inputs = [run / "data/training.npz", run / "data/cohorts.npz",
                       run / "data/catalogue.csv", run / "data/validation/fit10.npz"]
    contract = {"policy": policy(), "inputs": relative_hashes(contract_inputs, run),
                "sources": relative_hashes(source_paths(), ROOT),
                "tags_sha256": file_sha256(ROOT / "Dataset/ml-32m/tags.csv"),
                "content_index_manifest_sha256": file_sha256(ROOT / "artifacts/app-comparison/anton-content-index.manifest.json")}
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Training inputs changed; cached components cannot be reused")
    else:
        if (run / "conditional/model.npz").exists() or (run / "anton/components.json").exists():
            raise ValueError("Cannot establish provenance of existing models without a training contract")
        write_json(contract_path, contract)
    train_data = arrays(run / "data/training.npz")
    cohort = arrays(run / "data/cohorts.npz")
    excluded = cohort["excluded_tag_user_ids"]
    if np.intersect1d(train_data["user_ids"], excluded).size:
        raise ValueError("Heldout user leakage in training")
    catalogue = pd.read_csv(run / "data/catalogue.csv")
    times = {"conditional_reused": (run / "conditional/model.npz").exists(),
             "anton_components_reused": (run / "anton/components.json").exists()}
    start = time.perf_counter()
    if not (run / "conditional/model.npz").exists():
        train_neighbors(train_data, CONFIG, run / "conditional")
    times["conditional_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    if (run / "anton/components.json").exists():
        components = load_components(run / "anton", vendor_dir=VENDOR)
    else:
        components = train_components(
            train_data, catalogue, vendor_dir=VENDOR,
            data_dir=ROOT.parent / "extra_data/archive",
            index_path=ROOT / "artifacts/app-comparison/anton-content-index.joblib",
            tags_path=ROOT / "Dataset/ml-32m/tags.csv",
            excluded_tag_users=excluded, output=run / "anton")
    times["anton_components_seconds"] = time.perf_counter() - start
    fit = arrays(run / "data/validation/fit10.npz")
    if np.intersect1d(fit["user_ids"], train_data["user_ids"]).size:
        raise ValueError("Validation users leaked into training")
    upstream = load_upstream(VENDOR)
    for key in ("support_ratings", "target_ratings"):
        fit[key] = upstream.rating.to_four_star(fit[key].astype(np.float64))
    start = time.perf_counter()
    fit_validation_weights(components, fit, vendor_dir=VENDOR, output=run / "anton")
    times["anton_weight_fit_seconds"] = time.perf_counter() - start
    write_json(run / "training_times.json", times)
    inputs = [run / "policy.json", run / "training_contract.json", run / "data/manifest.json", run / "data/plan.json",
              run / "data/catalogue.csv", run / "conditional/model.npz",
              run / "anton/model.json", run / "anton/provenance.json"]
    inputs += sorted((run / "data").rglob("*.npz"))
    versions = {n: importlib.metadata.version(n) for n in
                ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    frozen = {"frozen_at_utc": now(), "source_hashes": relative_hashes(source_paths(), ROOT),
              "artifact_hashes": relative_hashes(inputs, run), "versions": versions,
              "confirmation_labels_read_for_scoring": False}
    write_json(run / "frozen.json", frozen)
    print(json.dumps({"status": "trained_and_frozen", "times": times, "versions": versions}), flush=True)


def verify_frozen(run):
    frozen = json.loads((run / "frozen.json").read_text())
    for base, key in ((ROOT, "source_hashes"), (run, "artifact_hashes")):
        for name, expected in frozen[key].items():
            if file_sha256(base / name) != expected:
                raise ValueError(f"Frozen input changed: {name}")
    return frozen


def score_profile(profiles, conditional, anton):
    n, m = len(profiles["user_ids"]), len(conditional["movie_ids"])
    values = {k: np.zeros((n, m), dtype=np.float64) for k in METHODS}
    elapsed = {k: 0. for k in ("conditional", "anton")}
    tie_counts, clipped_top = [], []
    for i, (items, verdicts) in enumerate(zip(profiles["support_items"], profiles["support_ratings"])):
        start = time.perf_counter()
        values["conditional"][i] = score_neighbors(conditional, items, verdicts, CONFIG)
        elapsed["conditional"] += time.perf_counter() - start
        start = time.perf_counter()
        cand, predictions = score_candidates(anton, items, 1. + .75 * (verdicts - 1.), vendor_dir=VENDOR)
        values["anton"][i, cand] = predictions
        elapsed["anton"] += time.perf_counter() - start
        tie_counts.append(int(np.count_nonzero(predictions == predictions.max())))
        clipped_top.append(bool(predictions.max() == 4.))
        values["rating_count"][i] = conditional["counts"]
        values["rating_mean"][i] = anton["film_dev"]
    diagnostics = {"scoring_seconds": elapsed,
                   "milliseconds_per_user": {k: 1000 * v / n for k, v in elapsed.items()},
                   "anton_mean_number_tied_for_highest_score": float(np.mean(tie_counts)),
                   "anton_fraction_users_top_score_clipped_to_four": float(np.mean(clipped_top))}
    return values, diagnostics


def assess(run, split):
    frozen = verify_frozen(run)
    folder = run / split
    sentinel = folder / "started.json"
    if sentinel.exists():
        raise ValueError(f"{split} already started; preserve its result/failure, do not rerun silently")
    write_json(sentinel, {"started_at_utc": now(), "frozen_sha256": file_sha256(run / "frozen.json")})
    conditional = load_neighbors(run / "conditional")
    anton = load_model(run / "anton", vendor_dir=VENDOR)
    if not np.array_equal(conditional["movie_ids"], anton["ids"]):
        raise ValueError("Model catalogue/order mismatch")
    report = {"split": split, "frozen_sha256": file_sha256(run / "frozen.json"),
              "upstream_commit": UPSTREAM_COMMIT, "catalogue_size": len(anton["ids"]),
              "profiles": {}, "scope": policy()["scope"]}
    try:
        for profile in PROFILES:
            p = arrays(run / "data" / split / f"{profile}.npz")
            values, diagnostics = score_profile(p, conditional, anton)
            aggregates, per_user = {}, {}
            for method, scores in values.items():
                aggregates[method], per_user[method], tops = evaluate(scores, p)
                np.savez_compressed(folder / f"{profile}-{method}.npz",
                                    top_indices=tops, **per_user[method])
            differences = {}
            references = ("anton", "rating_count", "rating_mean") if profile == "positive5" else ("anton",)
            for reference in references:
                label = f"conditional-vs-{reference}"
                differences[label] = {
                    "paired_95": paired_bootstrap(per_user["conditional"], per_user[reference],
                                                   repetitions=5000, seed=20261119),
                }
                if profile == "positive5":
                    differences[label]["familywise_95_bonferroni"] = paired_bootstrap(
                        per_user["conditional"], per_user[reference], confidence=1 - .05 / 3,
                        repetitions=5000, seed=20261119)
            report["profiles"][profile] = {"metrics": aggregates, "differences": differences,
                                           "diagnostics": diagnostics}
            write_json(folder / "results.partial.json", report)
            print(json.dumps({"profile": profile, "ndcg5": {k: v["ndcg_at_5"] for k, v in aggregates.items()},
                              "users": len(p["user_ids"])}), flush=True)
        report["completed_at_utc"] = now()
        report["under_ten_declared_fallback"] = {
            "positive5": "If the export metadata's ten-rating rule is enforced, Anton's ordering equals rating_mean here.",
            "primary": "Actual upstream recommendation function; no fallback gate."}
        write_json(folder / "results.json", report)
        write_json(folder / "completed.json", {"results_sha256": file_sha256(folder / "results.json"),
                                               "completed_at_utc": now()})
    except BaseException as error:
        write_json(folder / "failure.json", {"type": type(error).__name__, "message": str(error),
                                             "failed_at_utc": now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "confirm"))
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage == "train":
            train(args.run_dir.resolve())
        else:
            if args.stage == "confirm" and not (args.run_dir / "validation/completed.json").exists():
                raise ValueError("Validate the frozen implementation before confirmation")
            assess(args.run_dir.resolve(), "validation" if args.stage == "validate" else "test")


if __name__ == "__main__":
    main()
