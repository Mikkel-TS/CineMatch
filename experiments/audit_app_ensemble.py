"""Independent data, ranking and metric audit after ensemble confirmation.

This script never runs confirmation or selects a new model. It checks frozen
inputs, reconstructs the split and profiles from their source rows, recomputes
metrics using scalar arithmetic, and checks sample output ranks independently.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from anton_adapter import load_upstream
from run_app_ensemble import RUN, ROOT, VENDOR, verify


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key].copy() for key in file.files}


def equal(actual, expected, context):
    if not np.array_equal(actual, expected):
        raise AssertionError(f"Mismatch: {context}")


def close(actual, expected, context):
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise AssertionError(f"Numerical mismatch: {context}")


def audit_data(run, policy, manifest):
    groups = load(run / "data/cohorts.npz")
    previous = Path(manifest["previous_run"])
    prior = load(previous / "data/cohorts.npz")
    old = np.unique(np.concatenate([prior[key] for key in
                    ("old_excluded_user_ids", "validation_user_ids", "test_user_ids")]))
    equal(groups["old_excluded_user_ids"], old, "old holdout identities")
    names = ("training_user_ids", "anton_fit_user_ids", "blend_validation_user_ids",
             "test_user_ids", "old_excluded_user_ids")
    for a, b in itertools.combinations(names, 2):
        if np.intersect1d(groups[a], groups[b]).size:
            raise AssertionError(f"Cohort leakage: {a}/{b}")
    excluded = np.unique(np.concatenate([groups[key] for key in names[1:]]))
    equal(groups["excluded_tag_user_ids"], excluded, "all excluded tag authors")
    equal(np.sort(np.concatenate([groups["anton_fit_user_ids"], groups["blend_validation_user_ids"]])),
          groups["validation_user_ids"], "fit and blend split partition")
    provenance = json.loads((run / "anton/provenance.json").read_text())
    tag_info = provenance["tag_features"]
    assert tag_info["excluded_user_count"] == len(excluded)
    assert tag_info["excluded_user_ids_sha256"] == hashlib.sha256(excluded.astype(np.int64).tobytes()).hexdigest()
    assert digest(ROOT / "Dataset/ml-32m/tags.csv") == tag_info["source_sha256"]
    source_path = previous / "data/training.npz"
    assert digest(source_path) == manifest["source_training_sha256"]
    source, training = load(source_path), load(run / "data/training.npz")
    equal(source["movie_ids"], training["movie_ids"], "identical common catalogue")
    assert len(training["movie_ids"]) == 3788
    source_ids, source_users = source["user_ids"], source["user_index"]
    equal(source_ids, np.sort(source_ids), "ascending source users")
    equal(groups["source_user_ids"], source_ids, "source user identities")
    if np.intersect1d(source_ids, old).size:
        raise AssertionError("Previous held-out users occur in source training")
    counts = np.bincount(source_users, minlength=len(source_ids))
    positives = np.bincount(source_users[source["rating"] >= 4.], minlength=len(source_ids))
    eligible = source_ids[(counts >= policy["minimum_common_ratings"]) &
                          (positives >= policy["minimum_common_positives"])]
    permutation = np.random.default_rng(policy["seed"]).permutation(eligible)
    validation_count = policy["anton_fit_users"] + policy["blend_validation_users"]
    expected_validation = np.sort(permutation[:validation_count])
    expected_test = np.sort(permutation[validation_count:validation_count + policy["confirmation_users"]])
    equal(groups["validation_user_ids"], expected_validation, "deterministic fresh validation selection")
    equal(groups["test_user_ids"], expected_test, "deterministic fresh confirmation selection")
    validation_permutation = np.random.default_rng(policy["seed"] + 1).permutation(expected_validation)
    equal(groups["anton_fit_user_ids"], np.sort(validation_permutation[:policy["anton_fit_users"]]), "dedicated fit selection")
    equal(groups["blend_validation_user_ids"], np.sort(validation_permutation[policy["anton_fit_users"]:]), "dedicated blend selection")
    keep_user = ~np.isin(source_ids, excluded)
    equal(training["user_ids"], source_ids[keep_user], "training population after fresh exclusions")
    equal(groups["training_user_ids"], training["user_ids"], "training cohort manifest")
    keep_rows = keep_user[source_users]
    remap = np.full(len(source_ids), -1, dtype=np.int64)
    remap[keep_user] = np.arange(keep_user.sum())
    equal(training["user_index"], remap[source_users[keep_rows]], "all retained training row authors")
    equal(training["movie_index"], source["movie_index"][keep_rows], "all retained training row movies")
    equal(training["rating"], source["rating"][keep_rows], "all retained training ratings")
    equal(training["item_counts"], np.bincount(training["movie_index"], minlength=len(training["movie_ids"])), "training popularity counts")
    training_rows = len(training["rating"])
    del training, remap, keep_rows
    held_rows = ~keep_user[source_users]
    held_users = source_ids[source_users[held_rows]]
    held_items = source["movie_index"][held_rows]
    held_values = source["rating"][held_rows]
    order = np.argsort(held_users, kind="stable")
    held_users, held_items, held_values = held_users[order], held_items[order], held_values[order]
    del source, order, held_rows
    checked = {}
    for filename, key, size, stream in (
            ("fit10", "anton_fit_user_ids", 10, 2),
            ("validation15", "blend_validation_user_ids", 15, 1),
            ("test15", "test_user_ids", 15, 1)):
        p = load(run / f"data/{filename}.npz")
        equal(p["user_ids"], groups[key], f"{filename} user identities")
        assert p["support_items"].shape == (len(p["user_ids"]), size)
        minimum_positive_targets = math.inf
        for row, user in enumerate(p["user_ids"]):
            lo, hi = np.searchsorted(held_users, user, side="left"), np.searchsorted(held_users, user, side="right")
            movies, ratings = held_items[lo:hi], held_values[lo:hi]
            canonical = np.argsort(movies, kind="stable")
            movies, ratings = movies[canonical], ratings[canonical]
            assert len(np.unique(movies)) == len(movies)
            available = np.flatnonzero(ratings >= 4.) if stream == 1 else np.arange(len(movies))
            rng = np.random.default_rng(np.random.SeedSequence([policy["seed"], int(user), stream]))
            shuffle = rng.permutation(available)
            supports = shuffle[:size]
            equal(p["support_items"][row], movies[supports], f"{filename}/{user} deterministic inputs")
            equal(p["support_original_ratings"][row], ratings[supports], f"{filename}/{user} raw inputs")
            expected_ratings = np.clip(np.floor(ratings[supports]), 1, 5) if stream == 1 else ratings[supports]
            equal(p["support_ratings"][row], expected_ratings, f"{filename}/{user} disclosed verdicts")
            if stream == 1:
                keep = np.ones(len(movies), dtype=bool)
                keep[supports] = False
                targets = np.flatnonzero(keep)
            else:
                targets = shuffle[size:size + 30]
            start, stop = p["target_offsets"][row:row + 2]
            equal(p["target_items"][start:stop], movies[targets], f"{filename}/{user} hidden movies")
            equal(p["target_ratings"][start:stop], ratings[targets], f"{filename}/{user} raw hidden labels")
            minimum_positive_targets = min(minimum_positive_targets, int((ratings[targets] >= 4.).sum()))
        checked[filename] = {"users": len(p["user_ids"]), "disclosed_opinions": size,
                             "minimum_remaining_known_positives": minimum_positive_targets}
    return {"disjoint_cohorts": {name: len(groups[name]) for name in names},
            "excluded_tag_users": len(excluded), "training_rows_exactly_reproduced": training_rows,
            "fresh_cohort_selection_reproduced": True, "all_profiles_reproduced_from_source_ratings": checked}


def scalar_metrics(p, saved, recorded, movie_count):
    equal(saved["user_ids"], p["user_ids"], "prediction user alignment")
    keys = ("ndcg_at_5", "hit_rate_at_5", "recall_at_5", "recall_at_10",
            "precision_at_5", "top_1_known_positive", "explicit_low_rate_at_5")
    values = {key: [] for key in keys}
    exposures = [0] * movie_count
    discounts = [1. / math.log2(rank + 1.) for rank in range(1, 6)]
    for i, top in enumerate(saved["top_indices"]):
        seen = set(map(int, p["support_items"][i]))
        recommendations = list(map(int, top))
        assert len(recommendations) == 10 and len(set(recommendations)) == 10
        assert not seen.intersection(recommendations)
        assert all(0 <= item < movie_count for item in recommendations)
        lo, hi = p["target_offsets"][i:i + 2]
        observed = dict(zip(map(int, p["target_items"][lo:hi]), map(float, p["target_ratings"][lo:hi])))
        assert not seen.intersection(observed)
        positive = {item for item, rating in observed.items() if rating >= 4.}
        low = {item for item, rating in observed.items() if rating <= 2.}
        assert len(positive) >= 6
        first_five = recommendations[:5]
        hits = [item in positive for item in first_five]
        values["ndcg_at_5"].append(sum(discounts[j] for j, hit in enumerate(hits) if hit) /
                                     sum(discounts[:min(5, len(positive))]))
        values["hit_rate_at_5"].append(float(any(hits)))
        values["recall_at_5"].append(sum(hits) / len(positive))
        values["recall_at_10"].append(sum(item in positive for item in recommendations) / len(positive))
        values["precision_at_5"].append(sum(hits) / 5.)
        values["top_1_known_positive"].append(float(hits[0]))
        values["explicit_low_rate_at_5"].append(sum(item in low for item in first_five) / 5.)
        assert saved["positive_target_count"][i] == len(positive)
        assert saved["known_target_count"][i] == len(observed)
        assert saved["recommended_count_at_5"][i] == 5
        for item in first_five:
            exposures[item] += 1
    max_error = 0.
    for key, rows in values.items():
        close(rows, saved[key], f"{key} each user")
        aggregate = sum(rows) / len(rows)
        close(aggregate, recorded[key], f"{key} aggregate")
        max_error = max(max_error, abs(aggregate - recorded[key]))
    count = len(p["user_ids"])
    assert recorded["users"] == count and recorded["users_with_relevant_targets"] == count
    assert recorded["users_without_relevant_targets"] == 0
    distinct = sum(value > 0 for value in exposures)
    assert recorded["distinct_recommended_movies"] == distinct
    assert recorded["top5_exposure_slots"] == 5 * count
    close(recorded["catalog_coverage_at_5"], distinct / movie_count, "catalogue coverage")
    close(recorded["top5_concentration"], sum(sorted(exposures)[-5:]) / (5 * count), "top five exposure concentration")
    close(recorded["exposure_hhi_at_5"], sum((x / (5 * count)) ** 2 for x in exposures), "exposure HHI")
    assert recorded["short_list_rate_at_5"] == 0 and recorded["empty_list_rate"] == 0
    return {key: np.asarray(rows) for key, rows in values.items()}, max_error


def manual_rank_features(scores, seen, offset):
    output = np.zeros(len(scores), dtype=np.float64)
    order = sorted((j for j in range(len(scores)) if j not in seen), key=lambda j: (-scores[j], j))
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        mean_rank = ((start + 1) + stop) / 2.
        for j in order[start:stop]:
            output[j] = 1. / (offset + mean_rank)
        start = stop
    return output


def audit_rankings(run, p, saved, policy, selection):
    upstream = load_upstream(VENDOR)
    native = upstream.rating.load_export(run / "anton/model.json")
    conditional = load(run / "conditional/model.npz")
    equal(conditional["movie_ids"], native["ids"], "standalone catalogue ordering")
    n = len(native["ids"])
    sample = np.linspace(0, len(p["user_ids"]) - 1, min(20, len(p["user_ids"])), dtype=int)
    for i in sample:
        seeds, verdicts = p["support_items"][i], p["support_ratings"][i]
        opinions = {int(native["ids"][movie]): 1. + .75 * (float(verdict) - 1.) for movie, verdict in zip(seeds, verdicts)}
        candidates, predictions = upstream.rating.score_candidates(native, opinions)
        anton = np.zeros(n)
        anton[candidates] = predictions
        scores = np.zeros(n)
        for seed, verdict in zip(seeds, verdicts):
            low, high = conditional["neighbor_indptr"][seed:seed + 2]
            for edge in range(low, high):
                scores[int(conditional["neighbor_indices"][edge])] += max(float(verdict) - 3., 0.) * float(conditional["neighbor_weights"][edge])
        if not np.any(scores):
            scores = conditional["counts"].astype(float)
        seen = set(map(int, seeds))
        c_rank = manual_rank_features(scores, seen, policy["fusion"]["offset"])
        a_rank = manual_rank_features(anton, seen, policy["fusion"]["offset"])
        weight = selection["anton_weight"]
        blend = (1. - weight) * c_rank + weight * a_rank
        for method, raw in (("conditional", scores), ("anton", anton), ("blend", blend)):
            expected = sorted((j for j in range(n) if j not in seen), key=lambda j: (-raw[j], j))[:10]
            equal(expected, saved[method]["top_indices"][i], f"independent {method} top ten/user {i}")
    for name, scores in (("rating_count", conditional["counts"]), ("rating_mean", native["film_dev"])):
        base_order = sorted(range(n), key=lambda j: (-float(scores[j]), j))
        for i, seeds in enumerate(p["support_items"]):
            seen = set(map(int, seeds))
            expected = list(itertools.islice((j for j in base_order if j not in seen), 10))
            equal(expected, saved[name]["top_indices"][i], f"baseline top ten/{name}/{i}")
    return {"native_anton_manual_conditional_and_manual_fusion_users": p["user_ids"][sample].tolist(),
            "both_popularity_baselines_all_users": len(p["user_ids"]), "average_tie_ranks_recomputed_without_scipy": True}


def audit_complementarity(p, saved, independent, recorded):
    keys = [key for key in recorded if key != "interpretation"]
    values = {key: 0 for key in keys}
    overlap_sum = 0
    for i in range(len(p["user_ids"])):
        start, stop = p["target_offsets"][i:i + 2]
        positives = {int(item) for item, rating in zip(p["target_items"][start:stop],
                     p["target_ratings"][start:stop]) if rating >= 4.}
        top = {name: set(map(int, saved[name]["top_indices"][i, :5]))
               for name in ("conditional", "anton", "blend")}
        hits = {name: movies.intersection(positives) for name, movies in top.items()}
        c, a, b = hits["conditional"], hits["anton"], hits["blend"]
        overlap_sum += len(top["conditional"].intersection(top["anton"]))
        values["users_with_anton_unique_known_like"] += int(bool(a.difference(c)))
        values["anton_unique_known_like_slots"] += len(a.difference(c))
        values["conditional_unique_known_like_slots"] += len(c.difference(a))
        values["anton_rescues_zero_hit_conditional_users"] += int(bool(a) and not c)
        values["conditional_rescues_zero_hit_anton_users"] += int(bool(c) and not a)
        values["blend_rescues_zero_hit_conditional_users"] += int(bool(b) and not c)
        values["blend_loses_conditional_hit_users"] += int(bool(c) and not b)
        values["blend_known_like_slots_gained_vs_conditional"] += len(b.difference(c))
        values["blend_known_like_slots_lost_vs_conditional"] += len(c.difference(b))
    values["mean_standalone_top5_overlap"] = overlap_sum / len(p["user_ids"])
    deltas = independent["blend"]["ndcg_at_5"] - independent["conditional"]["ndcg_at_5"]
    values["blend_users_improved_ndcg"] = int((deltas > 1e-12).sum())
    values["blend_users_worsened_ndcg"] = int((deltas < -1e-12).sum())
    values["blend_users_equal_ndcg"] = int((np.abs(deltas) <= 1e-12).sum())
    for key, value in values.items():
        close(value, recorded[key], f"complementarity/{key}")
    return values


def audit(run):
    if not (run / "test/completed.json").exists():
        raise ValueError("Confirmation must finish before auditing; this script does not run confirmation")
    for name in ("data_frozen.json", "models_frozen.json", "selection_frozen.json"):
        verify(run, name)
    policy = json.loads((run / "policy.json").read_text())
    manifest = json.loads((run / "data_manifest.json").read_text())
    results = json.loads((run / "test_results.json").read_text())
    completed = json.loads((run / "test/completed.json").read_text())
    assert completed["results_sha256"] == digest(run / "test_results.json")
    selection = json.loads((run / "selection.json").read_text())
    validation = json.loads((run / "validation_results.json").read_text())["metrics_by_anton_weight"]
    assert {float(w) for w in validation} == {0., 1., *policy["fusion"]["anton_weights"]}
    expected = sorted(policy["fusion"]["anton_weights"], key=lambda w: (-validation[str(w)]["ndcg_at_5"], w))[0]
    assert selection["anton_weight"] == expected
    close(selection["conditional_weight"], 1. - expected, "complementary fusion weights")
    close(selection["validation_ndcg"], validation[str(expected)]["ndcg_at_5"], "selected validation metric")
    close(selection["validation_difference"], validation[str(expected)]["ndcg_at_5"] - validation["0.0"]["ndcg_at_5"], "selected validation gain")
    started = json.loads((run / "test/started.json").read_text())
    assert datetime.fromisoformat(selection["selected_at_utc"]) < datetime.fromisoformat(started["started_at_utc"])
    assert started["selection_sha256"] == digest(run / "selection_frozen.json")
    data = audit_data(run, policy, manifest)
    p = load(run / "data/test15.npz")
    assert len(p["user_ids"]) == policy["confirmation_users"] == results["users"]
    saved, independent, errors = {}, {}, []
    for method, metrics in results["metrics"].items():
        saved[method] = load(run / f"test/{method}.npz")
        independent[method], error = scalar_metrics(p, saved[method], metrics, results["catalogue_movies"])
        errors.append(error)
    ranks = audit_rankings(run, p, saved, policy, selection)
    complementary = audit_complementarity(p, saved, independent, results["complementarity"])
    differences = independent["blend"]["ndcg_at_5"] - independent["conditional"]["ndcg_at_5"]
    close(differences.mean(), results["primary_comparison"]["difference"], "primary paired gain")
    rng = np.random.default_rng(policy["seed"])
    bootstrap = []
    for start in range(0, policy["bootstrap_repetitions"], 64):
        count = min(64, policy["bootstrap_repetitions"] - start)
        chosen = rng.integers(len(differences), size=(count, len(differences)))
        bootstrap.extend(differences[chosen].mean(axis=1))
    close(np.quantile(bootstrap, [.025, .975]), results["primary_comparison"]["confidence_interval"], "independent primary bootstrap interval")
    production_checks = {path: digest(path) == expected for path, expected in manifest["production_hashes"].items()}
    assert all(production_checks.values())
    summary = {"status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
               "data": data, "validation_selection_replayed": True,
               "confirmation_follows_frozen_selection": True, "frozen_hashes_match": True,
               "users": len(p["user_ids"]), "methods": list(saved),
               "independent_scalar_metrics_per_user_and_aggregate": list(independent["blend"]),
               "maximum_aggregate_absolute_error": max(errors),
               "independent_primary_bootstrap_matches": True, "ranking_checks": ranks,
               "independent_complementarity_diagnostics": complementary,
               "production_artifacts_unchanged": production_checks,
               "audit_source_sha256": digest(Path(__file__)),
               "inputs_sha256": {str(path.relative_to(run)): digest(path) for path in
                   [run / "test_results.json", run / "selection.json", run / "data/test15.npz",
                    *sorted((run / "test").glob("*.npz"))]}}
    for folder in (run, ROOT / "experiments/comparisons" / run.name):
        (folder / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "users", "methods", "maximum_aggregate_absolute_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    audit(parser.parse_args().run_dir.resolve())
