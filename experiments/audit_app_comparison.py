"""Independent reconstruction of saved common-catalogue benchmark metrics.

This audit does not import the benchmark evaluator. It reconstructs metrics
with scalar arithmetic and set membership, then checks sample rankings against
the pinned native Anton scorer and an explicit conditional-edge summation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from anton_adapter import load_upstream


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/experiments/app-model-comparison-20261119-v1"
VENDOR = ROOT / "experiments/vendor/anton_6e85958"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def check_equal(actual, expected, context):
    a, b = np.asarray(actual), np.asarray(expected)
    if not np.allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True):
        raise AssertionError(f"Metric discrepancy: {context}")


def audit(run, split, output):
    report_path = run / split / "results.json"
    report = json.loads(report_path.read_text())
    checked, inputs, max_error = [], {str(report_path): digest(report_path)}, 0.
    discounts = [1. / math.log2(rank + 1.) for rank in range(1, 6)]
    for mode, mode_report in report["profiles"].items():
        profile_path = run / "data" / split / f"{mode}.npz"
        p = load(profile_path)
        inputs[str(profile_path)] = digest(profile_path)
        for method, recorded in mode_report["metrics"].items():
            saved_path = run / split / f"{mode}-{method}.npz"
            saved = load(saved_path)
            inputs[str(saved_path)] = digest(saved_path)
            if not np.array_equal(saved["user_ids"], p["user_ids"]):
                raise AssertionError(f"User alignment mismatch: {mode}/{method}")
            reconstructed = {key: [] for key in (
                "ndcg_at_5", "hit_rate_at_5", "recall_at_5", "recall_at_10",
                "precision_at_5", "top_1_known_positive", "explicit_low_rate_at_5",
            )}
            recommended = set()
            for i, top in enumerate(saved["top_indices"]):
                seen = set(map(int, p["support_items"][i]))
                if seen.intersection(map(int, top)) or len(set(map(int, top))) != len(top):
                    raise AssertionError(f"Invalid ranked recommendations: {mode}/{method}/{i}")
                lo, hi = map(int, p["target_offsets"][i:i + 2])
                observed = {
                    int(movie): float(rating) for movie, rating in
                    zip(p["target_items"][lo:hi], p["target_ratings"][lo:hi])
                }
                positives = {movie for movie, rating in observed.items() if rating >= 4.}
                low = {movie for movie, rating in observed.items() if rating <= 2.}
                first5 = list(map(int, top[:5]))
                hits5 = [movie in positives for movie in first5]
                hits10 = sum(int(movie) in positives for movie in top[:10])
                dcg = sum(discounts[rank] for rank, hit in enumerate(hits5) if hit)
                reconstructed["ndcg_at_5"].append(
                    dcg / sum(discounts[:min(5, len(positives))]) if positives else float("nan"))
                reconstructed["hit_rate_at_5"].append(float(any(hits5)))
                reconstructed["recall_at_5"].append(sum(hits5) / len(positives) if positives else float("nan"))
                reconstructed["recall_at_10"].append(hits10 / len(positives) if positives else float("nan"))
                reconstructed["precision_at_5"].append(sum(hits5) / 5.)
                reconstructed["top_1_known_positive"].append(float(hits5[0]) if hits5 else 0.)
                reconstructed["explicit_low_rate_at_5"].append(sum(movie in low for movie in first5) / 5.)
                recommended.update(first5)
            for key, values in reconstructed.items():
                values = np.asarray(values)
                check_equal(values, saved[key], f"{mode}/{method}/{key}/per-user")
                valid = values[np.isfinite(values)]
                aggregate = float(valid.sum() / len(valid)) if len(valid) else None
                if aggregate is None:
                    if recorded[key] is not None:
                        raise AssertionError(f"Missing aggregate mismatch: {mode}/{method}/{key}")
                else:
                    check_equal(aggregate, recorded[key], f"{mode}/{method}/{key}/aggregate")
                    max_error = max(max_error, abs(aggregate - recorded[key]))
            if len(recommended) != recorded["distinct_recommended_movies"]:
                raise AssertionError(f"Coverage mismatch: {mode}/{method}")
            check_equal(len(recommended) / report["catalogue_size"], recorded["catalog_coverage_at_5"],
                        f"{mode}/{method}/coverage")
            checked.append({"profile": mode, "method": method, "users": len(p["user_ids"]),
                            "metrics": list(reconstructed), "per_user_and_aggregate_match": True})

    primary = load(run / "data" / split / "positive5.npz")
    anton_saved = load(run / split / "positive5-anton.npz")["top_indices"]
    conditional_saved = load(run / split / "positive5-conditional.npz")["top_indices"]
    upstream = load_upstream(VENDOR)
    anton_path = run / "anton/model.json"
    conditional_path = run / "conditional/model.npz"
    native_model = upstream.rating.load_export(anton_path)
    conditional = load(conditional_path)
    if not np.array_equal(native_model["ids"], conditional["movie_ids"]):
        raise AssertionError("Model catalogue/order mismatch")
    native_checks, edge_checks = [], []
    for i in range(min(10, len(primary["user_ids"]))):
        seeds = primary["support_items"][i]
        verdicts = primary["support_ratings"][i]
        opinions = {
            int(native_model["ids"][seed]): 1. + .75 * (float(verdict) - 1.)
            for seed, verdict in zip(seeds, verdicts)
        }
        candidates, predictions = upstream.rating.score_candidates(native_model, opinions)
        ranked = sorted(range(len(candidates)), key=lambda j: (-float(predictions[j]), int(candidates[j])))
        native_top = np.asarray([int(candidates[j]) for j in ranked[:10]])
        if not np.array_equal(native_top, anton_saved[i]):
            raise AssertionError(f"Native Anton ranking mismatch: row {i}")
        native_checks.append(int(primary["user_ids"][i]))

        scores = [0.] * len(conditional["movie_ids"])
        for seed, verdict in zip(seeds, verdicts):
            weight = max(float(verdict) - 3., 0.)
            lo, hi = map(int, conditional["neighbor_indptr"][int(seed):int(seed) + 2])
            for edge in range(lo, hi):
                movie = int(conditional["neighbor_indices"][edge])
                scores[movie] += weight * float(conditional["neighbor_weights"][edge])
        if not any(scores):
            scores = list(map(float, conditional["counts"]))
        seen = set(map(int, seeds))
        candidates = [j for j in range(len(scores)) if j not in seen]
        manual_top = np.asarray(sorted(candidates, key=lambda j: (-scores[j], j))[:10])
        if not np.array_equal(manual_top, conditional_saved[i]):
            raise AssertionError(f"Manual conditional ranking mismatch: row {i}")
        edge_checks.append(int(primary["user_ids"][i]))
    inputs[str(anton_path)] = digest(anton_path)
    inputs[str(conditional_path)] = digest(conditional_path)
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split, "status": "passed", "independent_metric_recomputation": True,
        "method": "Scalar DCG arithmetic and set membership; benchmark metric implementation not imported",
        "users_per_profile": len(primary["user_ids"]), "profile_method_pairs": len(checked),
        "checks": checked, "maximum_aggregate_absolute_error": max_error,
        "native_anton_top10_checks": native_checks, "manual_conditional_top10_checks": edge_checks,
        "upstream_commit": report["upstream_commit"],
        "independent_confirmation": split == "test",
        "validation_caveat": "Validation users fit Anton's weights; matching validation arithmetic does not establish generalization",
        "inputs_sha256": inputs, "audit_source_sha256": digest(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "split", "status", "profile_method_pairs", "users_per_profile", "maximum_aggregate_absolute_error")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.split, args.output.resolve())


if __name__ == "__main__":
    main()
