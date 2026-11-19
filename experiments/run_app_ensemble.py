"""Validation-selected rank fusion, then one fresh 15-input/top-five confirmation.

Prepare -> train -> select -> confirm. Never updates the app or prior experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from anton_adapter import (file_sha256, train_components, fit_validation_weights,
                           load_model, load_upstream, UPSTREAM_COMMIT)
from app_comparison_metrics import evaluate, paired_bootstrap
from campaign_neighbors import train_neighbors, load_neighbors
from prepare_app_comparison import prepare_arrays, make_profiles, write_npz, profile_summary
from run_app_comparison import (ROOT, RUN as PREVIOUS, VENDOR, CONFIG, METHODS,
                                arrays, now, score_profile, source_paths,
                                relative_hashes, verify_frozen, write_json)

RUN = ROOT / "artifacts/experiments/app-model-ensemble-20261119-v1"
SEED = 2026111901
WEIGHTS = (.025, .05, .10, .15, .20, .30, .40, .50)


def reciprocal_rank_scores(scores, support_items, offset=60.):
    """Average ranks for tied unseen scores; seen entries have zero contribution."""
    scores, support = np.asarray(scores), np.asarray(support_items)
    if (scores.ndim != 2 or not scores.size or not np.isfinite(scores).all()
            or support.ndim != 2 or support.shape[0] != scores.shape[0]
            or not np.issubdtype(support.dtype, np.integer)
            or np.any(support < 0) or np.any(support >= scores.shape[1])
            or not np.isfinite(offset) or offset < 0):
        raise ValueError("Finite scores, aligned support indices and nonnegative offset required")
    result = np.zeros_like(scores, dtype=np.float64)
    for i, seen in enumerate(support):
        if len(np.unique(seen)) != len(seen):
            raise ValueError("Support movies must be unique")
        eligible = np.ones(scores.shape[1], dtype=bool)
        eligible[seen] = False
        result[i, eligible] = 1. / (offset + rankdata(-scores[i, eligible], method="average"))
    return result


def fuse_rank_scores(conditional_features, anton_features, anton_weight):
    a, b = np.asarray(conditional_features), np.asarray(anton_features)
    if (not np.isfinite(anton_weight) or not 0 <= anton_weight <= 1
            or a.shape != b.shape or a.ndim != 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Aligned finite rank features and weight in [0,1] required")
    return (1. - anton_weight) * a + anton_weight * b


def select_blend(metrics_by_weight):
    candidates = [(float(weight), value["ndcg_at_5"]) for weight, value in metrics_by_weight.items()
                  if 0 < float(weight) < 1]
    if not candidates or any(not np.isfinite(value) for _, value in candidates):
        raise ValueError("At least one nonzero blend with finite NDCG is required")
    return min(candidates, key=lambda item: (-item[1], item[0]))[0]


def record(run):
    return ROOT / "experiments/comparisons" / run.name


def save_record(run, name, value):
    write_json(run / name, value)
    write_json(record(run) / name, value)


def sources():
    return [Path(__file__), *source_paths()]


def freeze(run, name, artifacts):
    value = {"frozen_at_utc": now(), "source_hashes": relative_hashes(sources(), ROOT),
             "artifact_hashes": relative_hashes(artifacts, run)}
    save_record(run, name, value)
    return value


def verify(run, name):
    value = json.loads((run / name).read_text())
    for base, key in ((ROOT, "source_hashes"), (run, "artifact_hashes")):
        for path, expected in value[key].items():
            if file_sha256(base / path) != expected:
                raise ValueError(f"Frozen input changed: {path}")
    return value


def policy():
    return {
        "created_at_utc": now(), "upstream_commit": UPSTREAM_COMMIT,
        "onboarding_positive_opinions": 15, "recommendations": 5,
        "seed": SEED, "anton_fit_users": 1000, "blend_validation_users": 1000,
        "confirmation_users": 2000, "minimum_common_ratings": 35, "minimum_common_positives": 21,
        "cohort_source": "Prior comparison training only; all previous heldout users remain excluded",
        "freshness_limit": "Newly reserved users from the same historical dataset, not new real-world users; earlier algorithm development used the broader dataset",
        "fusion": {"formula": "(1-alpha)/(60+rank_conditional) + alpha/(60+rank_anton)",
                   "rank_ties": "Average ranks among unseen movies; final score ties ascending MovieLens ID",
                   "offset": 60., "anton_weights": list(WEIGHTS)},
        "selection": "Highest validation NDCG@5 among nonzero blends; ties choose less Anton. If conditional wins validation, retain challenger for the prespecified test and report validation rejection.",
        "primary_comparison": "Frozen blend minus conditional, paired-user NDCG@5",
        "confidence": .95, "bootstrap_repetitions": 5000,
        "practical_absolute_ndcg_margin": .01,
        "margin_meaning": "Screening convention from prior benchmark, not validated product value",
        "secondary": "Hit rate, known low-rating rate, catalogue coverage, exposure concentration, overlap and unique known likes; exploratory",
        "no_tradeoff_acceptance": "No quality sacrifice is automatically accepted for wider coverage or fewer dislikes",
        "conditional_config": CONFIG,
        "anton_fit": "Unchanged native ten-raw-rating profile/maximum30 targets; dedicated calibration cohort, disjoint from blend selection",
        "support": "Raw positive>=4; app floor(raw)4/5; conditional weights1/2; Anton 1+.75*(verdict-1)",
        "targets": "All remaining raw observed ratings; positive>=4, low<=2, unrated unknown earns no credit",
        "catalogue": "Unchanged verified common catalogue of3788 movies; all15 disclosed movies excluded",
        "scope": "Retrained algorithm comparison; no app integration or automatic model adoption",
    }


def prepare(run):
    if run.exists() or record(run).exists():
        raise ValueError("Use a new run directory; refusing to overwrite")
    verify_frozen(PREVIOUS)
    save_record(run, "policy.json", policy())
    prior = arrays(PREVIOUS / "data/cohorts.npz")
    old = np.unique(np.concatenate([prior[key] for key in
                    ("old_excluded_user_ids", "validation_user_ids", "test_user_ids")]))
    base = arrays(PREVIOUS / "data/training.npz")
    training, cohorts, held, counts = prepare_arrays(
        base, base["movie_ids"], old, validation_users=2000, test_users=2000, seed=SEED)
    validation = np.random.default_rng(SEED + 1).permutation(cohorts["validation_user_ids"])
    cohorts["anton_fit_user_ids"] = np.sort(validation[:1000])
    cohorts["blend_validation_user_ids"] = np.sort(validation[1000:])
    write_npz(run / "data/training.npz", training)
    write_npz(run / "data/cohorts.npz", cohorts)
    shutil.copy2(PREVIOUS / "data/catalogue.csv", run / "data/catalogue.csv")
    profiles = {}
    for name, group, mode, size in (
            ("fit10", "anton_fit_user_ids", "fit", 10),
            ("validation15", "blend_validation_user_ids", "positive", 15),
            ("test15", "test_user_ids", "positive", 15)):
        p = make_profiles(held, cohorts[group], mode, size, SEED)
        write_npz(run / f"data/{name}.npz", p)
        profiles[name] = profile_summary(p)
    production = [ROOT.parent / "peter/CineMatch/CineMatch/Core/Resources/cinematch_knn.json",
                  ROOT / "artifacts/experiments/cm-20261119-full-knn-conditional/model/model.npz"]
    manifest = {"prepared_at_utc": now(), "counts": counts, "profiles": profiles,
                "previous_run": str(PREVIOUS), "previous_frozen_sha256": file_sha256(PREVIOUS / "frozen.json"),
                "source_training_sha256": file_sha256(PREVIOUS / "data/training.npz"),
                "prior_excluded_users": len(old), "all_excluded_tag_users": len(cohorts["excluded_tag_user_ids"]),
                "production_hashes": {str(p): file_sha256(p) for p in production},
                "source_environment": json.loads((PREVIOUS / "frozen.json").read_text())["versions"],
                "test_scores_computed": False}
    save_record(run, "data_manifest.json", manifest)
    freeze(run, "data_frozen.json", [run / "policy.json", run / "data_manifest.json", *sorted((run / "data").glob("*"))])
    print(json.dumps({"status": "prepared", "counts": counts, "profiles": profiles}), flush=True)


def train(run):
    verify(run, "data_frozen.json")
    if (run / "training_started.json").exists():
        raise ValueError("Training already started; preserve this run")
    write_json(run / "training_started.json", {"started_at_utc": now()})
    training, cohorts = arrays(run / "data/training.npz"), arrays(run / "data/cohorts.npz")
    started = time.perf_counter()
    train_neighbors(training, CONFIG, run / "conditional")
    conditional_seconds = time.perf_counter() - started
    started = time.perf_counter()
    components = train_components(
        training, pd.read_csv(run / "data/catalogue.csv"), vendor_dir=VENDOR,
        data_dir=ROOT.parent / "extra_data/archive",
        index_path=ROOT / "artifacts/app-comparison/anton-content-index.joblib",
        tags_path=ROOT / "Dataset/ml-32m/tags.csv",
        excluded_tag_users=cohorts["excluded_tag_user_ids"], output=run / "anton")
    fit = arrays(run / "data/fit10.npz")
    upstream = load_upstream(VENDOR)
    for key in ("support_ratings", "target_ratings"):
        fit[key] = upstream.rating.to_four_star(fit[key].astype(np.float64))
    fit_validation_weights(components, fit, vendor_dir=VENDOR, output=run / "anton")
    save_record(run, "training_times.json", {"conditional_seconds": conditional_seconds,
                                             "anton_seconds": time.perf_counter() - started})
    for name in ("fit.json", "provenance.json"):
        write_json(record(run) / f"anton_{name}", json.loads((run / "anton" / name).read_text()))
    freeze(run, "models_frozen.json", [run / "data_frozen.json", run / "conditional/model.npz",
                                       *sorted((run / "anton").glob("*"))])
    print(json.dumps({"status": "trained", "test_scores_computed": False}), flush=True)


def score(run, profile):
    c, a = load_neighbors(run / "conditional"), load_model(run / "anton", vendor_dir=VENDOR)
    if not np.array_equal(c["movie_ids"], a["ids"]):
        raise ValueError("Model catalogues differ")
    disclosed = {key: profile[key] for key in ("user_ids", "support_items", "support_ratings")}
    values, diagnostics = score_profile(disclosed, c, a)
    features = {key: reciprocal_rank_scores(values[key], profile["support_items"])
                for key in ("conditional", "anton")}
    # Rank conversion must preserve both standalone recommendation orders exactly.
    for key in features:
        if not np.array_equal(evaluate(values[key], profile)[2], evaluate(features[key], profile)[2]):
            raise AssertionError(f"Rank conversion changed standalone order: {key}")
    return values, features, diagnostics


def select(run):
    verify(run, "data_frozen.json")
    verify(run, "models_frozen.json")
    if (run / "selection_started.json").exists():
        raise ValueError("Selection already started; preserve this run")
    write_json(run / "selection_started.json", {"started_at_utc": now()})
    p = arrays(run / "data/validation15.npz")
    values, features, diagnostics = score(run, p)
    metrics = {str(w): evaluate(fuse_rank_scores(features["conditional"], features["anton"], w), p)[0]
               for w in (0., *WEIGHTS, 1.)}
    weight = select_blend(metrics)
    selected = metrics[str(weight)]["ndcg_at_5"]
    reference = metrics["0.0"]["ndcg_at_5"]
    save_record(run, "validation_results.json", {"metrics_by_anton_weight": metrics,
                                                 "scoring_diagnostics": diagnostics})
    save_record(run, "selection.json", {
        "selected_at_utc": now(), "anton_weight": weight, "conditional_weight": 1. - weight,
        "validation_ndcg": selected, "conditional_validation_ndcg": reference,
        "validation_difference": selected - reference,
        "validation_prefers_blend": selected > reference,
        "test_predictions_computed_before_selection": False,
        "selection_rule": "Best nonzero prespecified blend; ties prefer smaller Anton weight"})
    freeze(run, "selection_frozen.json", [run / name for name in
        ("data_frozen.json", "models_frozen.json", "selection.json", "validation_results.json")])
    for path in sources():
        output = run / "frozen_source" / path.relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
    print(json.dumps({"status": "selected_and_frozen", "anton_weight": weight,
                      "validation_difference": selected - reference}), flush=True)


def complementarity(profiles, tops, per_user):
    overlaps, a_unique_hits, c_unique_hits, gained, lost = [], [], [], [], []
    a_rescues, c_rescues, blend_rescues, blend_loses = 0, 0, 0, 0
    for i in range(len(profiles["user_ids"])):
        lo, hi = profiles["target_offsets"][i:i + 2]
        positive = set(profiles["target_items"][lo:hi][profiles["target_ratings"][lo:hi] >= 4])
        c, a, b = (set(tops[key][i, :5]) for key in ("conditional", "anton", "blend"))
        overlaps.append(len(c & a))
        a_unique_hits.append(len((a - c) & positive))
        c_unique_hits.append(len((c - a) & positive))
        gained.append(len((b - c) & positive))
        lost.append(len((c - b) & positive))
        a_rescues += bool(a & positive) and not bool(c & positive)
        c_rescues += bool(c & positive) and not bool(a & positive)
        blend_rescues += bool(b & positive) and not bool(c & positive)
        blend_loses += bool(c & positive) and not bool(b & positive)
    difference = per_user["blend"]["ndcg_at_5"] - per_user["conditional"]["ndcg_at_5"]
    return {"mean_standalone_top5_overlap": float(np.mean(overlaps)),
            "users_with_anton_unique_known_like": int(np.count_nonzero(a_unique_hits)),
            "anton_unique_known_like_slots": int(sum(a_unique_hits)),
            "conditional_unique_known_like_slots": int(sum(c_unique_hits)),
            "anton_rescues_zero_hit_conditional_users": int(a_rescues),
            "conditional_rescues_zero_hit_anton_users": int(c_rescues),
            "blend_rescues_zero_hit_conditional_users": int(blend_rescues),
            "blend_loses_conditional_hit_users": int(blend_loses),
            "blend_known_like_slots_gained_vs_conditional": int(sum(gained)),
            "blend_known_like_slots_lost_vs_conditional": int(sum(lost)),
            "blend_users_improved_ndcg": int(np.count_nonzero(difference > 1e-12)),
            "blend_users_worsened_ndcg": int(np.count_nonzero(difference < -1e-12)),
            "blend_users_equal_ndcg": int(np.count_nonzero(np.abs(difference) <= 1e-12)),
            "interpretation": "Retrospective diagnostics only; hidden labels never choose output movies"}


def confirm(run):
    for name in ("data_frozen.json", "models_frozen.json", "selection_frozen.json"):
        verify(run, name)
    if (run / "test/started.json").exists():
        raise ValueError("Confirmation already started; do not rerun or tune")
    write_json(run / "test/started.json", {"started_at_utc": now(),
                                          "selection_sha256": file_sha256(run / "selection_frozen.json")})
    p = arrays(run / "data/test15.npz")
    selected = json.loads((run / "selection.json").read_text())
    values, features, diagnostics = score(run, p)
    values["blend"] = fuse_rank_scores(features["conditional"], features["anton"], selected["anton_weight"])
    metrics, per_user, tops = {}, {}, {}
    for method, value in values.items():
        metrics[method], per_user[method], tops[method] = evaluate(value, p)
        write_npz(run / f"test/{method}.npz", {"top_indices": tops[method], **per_user[method]})
    primary = paired_bootstrap(per_user["blend"], per_user["conditional"], seed=SEED)
    secondary = {f"blend-vs-{reference}/{metric}": paired_bootstrap(
                    per_user["blend"], per_user[reference], metric=metric, seed=SEED)
                 for reference, metric in (("anton", "ndcg_at_5"), ("rating_count", "ndcg_at_5"),
                    ("conditional", "hit_rate_at_5"), ("conditional", "precision_at_5"),
                    ("conditional", "explicit_low_rate_at_5"))}
    manifest = json.loads((run / "data_manifest.json").read_text())
    production_unchanged = all(file_sha256(Path(path)) == sha for path, sha in manifest["production_hashes"].items())
    if not production_unchanged:
        raise AssertionError("Production artifact changed during experiment")
    result = {"completed_at_utc": now(), "users": len(p["user_ids"]), "catalogue_movies": values["blend"].shape[1],
              "onboarding_positive_opinions": 15, "recommendations": 5, "selection": selected,
              "metrics": metrics, "primary_comparison": primary,
              "secondary_unadjusted_comparisons": secondary,
              "complementarity": complementarity(p, tops, per_user), "scoring_diagnostics": diagnostics,
              "evidence": {"positive_gain_95_interval": primary["confidence_interval"][0] > 0,
                           "point_gain_at_least_0_01": primary["difference"] >= .01,
                           "lower_bound_at_least_0_01": primary["confidence_interval"][0] >= .01},
              "production_unchanged": production_unchanged}
    save_record(run, "test_results.json", result)
    write_json(run / "test/completed.json", {"completed_at_utc": now(), "results_sha256": file_sha256(run / "test_results.json")})
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "select", "confirm"))
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[args.stage](args.run_dir.resolve())


if __name__ == "__main__":
    main()
