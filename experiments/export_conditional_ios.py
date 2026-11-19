#!/usr/bin/env python3
"""Export the frozen conditional graph only after complete live TMDB verification.

The graph remains intact. Eligibility gates public inputs and ranked outputs;
model indices, directions, stored weights, popularity counts and tie order stay
unchanged. No holdout profiles are read. The exporter also emits deterministic
synthetic-profile Python/Swift parity fixtures and a conservative legacy bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from campaign_neighbors import load_neighbors, score_neighbors
from verify_tmdb_catalogue import (
    FINAL_STATUSES, MODEL_PATH, POLICY, ROOT, SELECTED_MODEL_SHA256, SELECTED_RUN, VERIFIED_STATUSES,
    atomic_json, classify, load_catalogue, normalize_title, read_csv, reusable, sha256,
    split_title, utc_now,
)

FORMAT = "cinematch.conditional-neighbors"
SCORING = {"aggregation": "positive", "ratingCenter": 3,
           "fallback": "training_rating_count", "tieBreak": "original_catalogue_order"}


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def validate_report(report: dict, catalogue: list[dict], sources: dict,
                    max_age_days: float = 7.0) -> dict[int, dict]:
    if (report.get("schemaVersion") != 1 or report.get("policy") != POLICY
            or report.get("selectedModelRun") != SELECTED_RUN
            or report.get("complete") is not True
            or report.get("liveLookupPerformed") is not True
            or report.get("tmdbIdentityVerified") is not True):
        raise ValueError("A complete live TMDB verification report is required; refusing an unverified bundle.")
    if report.get("sources") != sources:
        raise ValueError("Verification source checksums differ from the current model or MovieLens files.")
    results = report.get("movies", [])
    if len(results) != len(catalogue):
        raise ValueError("Verification report does not cover every original model position.")
    by_id = {}
    for expected, row in zip(catalogue, results):
        for field in ("modelIndex", "movieLensId", "tmdbId", "expectedImdbId"):
            if row.get(field) != expected[field]:
                raise ValueError("Verification rows are not aligned with the frozen model and link identities.")
        if row.get("status") not in FINAL_STATUSES:
            raise ValueError("Verification has unresolved requests or unknown statuses.")
        if expected.get("preflightStatus"):
            if row["status"] != expected["preflightStatus"]:
                raise ValueError("Missing or duplicate mappings cannot be made eligible by the report.")
        else:
            # Every API-derived classification must be recent, including definitive exclusions.
            if not reusable({"kind": "movie", "fetchedAt": row.get("verifiedAt")}, max_age_days):
                raise ValueError("Verification contains stale or invalid API timestamps.")
            if row["status"] in VERIFIED_STATUSES:
                if row.get("httpStatus") != 200 or row.get("tmdbMovie", {}).get("id") != row["tmdbId"]:
                    raise ValueError("Verified identity lacks the matching successful TMDB movie record.")
            response = {"kind": "not_found" if row.get("httpStatus") == 404 else "movie",
                        "httpStatus": row.get("httpStatus"), "fetchedAt": row.get("verifiedAt"),
                        "movie": row.get("tmdbMovie", {})}
            if row.get("httpStatus") not in (200, 404) or classify(expected, response)["status"] != row["status"]:
                raise ValueError("Verification classification does not match its retained API evidence.")
        if row["movieLensId"] in by_id:
            raise ValueError("Duplicate MovieLens IDs in verification report.")
        by_id[row["movieLensId"]] = row
    eligible = [row for row in results if row["status"] in VERIFIED_STATUSES]
    ids = [row["tmdbId"] for row in eligible]
    if not eligible or len(ids) != len(set(ids)):
        raise ValueError("At least one eligible movie and unique public TMDB IDs are required.")
    if report.get("eligibleCount") != len(eligible):
        raise ValueError("Verification eligible count is inconsistent.")
    return by_id


def validate_graph(model: dict) -> None:
    n = len(model["movie_ids"])
    offsets, indices, weights = (model[key] for key in
                                  ("neighbor_indptr", "neighbor_indices", "neighbor_weights"))
    if (len(set(map(int, model["movie_ids"]))) != n or len(model["counts"]) != n
            or np.any(model["counts"] < 0) or len(offsets) != n + 1 or offsets[0] != 0
            or np.any(np.diff(offsets) < 0) or offsets[-1] != len(indices)
            or len(indices) != len(weights) or np.any(indices < 0) or np.any(indices >= n)
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Invalid or misaligned frozen neighbour arrays.")
    info = model["info"]
    if (info.get("similarity") != "conditional" or info.get("feedback") != "positive"
            or info.get("shrink") != 10 or info.get("top_k") != 100
            or info.get("config", {}).get("aggregation") != "positive"):
        raise ValueError("Export requires the selected positive conditional K100/shrink10 scoring configuration.")


def legacy_bridge(frontend_path: Path, mapping_path: Path, movies_path: Path,
                  verified: dict[int, dict]) -> tuple[list[dict], dict]:
    films = json.loads(frontend_path.read_text())["films"]
    existing = defaultdict(list)
    for row in read_csv(mapping_path):
        existing[int(row["letterboxd_id"])].append(row)
    # Same-year normalized title uniqueness is measured over the complete dataset.
    full_title_index = defaultdict(set)
    for movie in read_csv(movies_path):
        title, year = split_title(movie["title"])
        if year is not None:
            full_title_index[(normalize_title(title), year)].add(int(movie["movieId"]))
    accepted, details, outcomes = [], [], Counter()
    for film in films:
        source_id = int(film["id"])
        key = (normalize_title(film["title"]), film["year"])
        prior = existing.get(source_id, [])
        candidates = {int(row["movieId"]) for row in prior}
        reason = ""
        if prior:
            if len(candidates) != 1:
                reason = "ambiguous_existing_mapping"
            elif any(row.get("match_type") != "exact_title_and_year" for row in prior):
                reason = "existing_mapping_requires_review"
        else:
            candidates = full_title_index.get(key, set())
        if not reason and len(full_title_index.get(key, set())) > 1:
            reason = "ambiguous_full_catalogue_title_year"
        if not reason and len(candidates) != 1:
            reason = "missing_or_ambiguous_mapping"
        target = next(iter(candidates)) if len(candidates) == 1 else None
        row = verified.get(target)
        if not reason and (row is None or row["status"] not in VERIFIED_STATUSES):
            reason = "outside_verified_supported_catalogue"
        if not reason:
            movie = row["tmdbMovie"]
            names = {normalize_title(value) for value in
                     [movie.get("title"), movie.get("original_title"), *movie.get("alternative_titles", [])]
                     if isinstance(value, str) and value}
            release = movie.get("release_date") or ""
            if (key[0] not in names or not release[:4].isdigit()
                    or int(release[:4]) != film["year"]):
                reason = "legacy_title_or_year_not_confirmed_by_tmdb"
        if reason:
            outcomes[reason] += 1
            details.append({"letterboxdId": source_id, "movieLensId": target,
                            "accepted": False, "reason": reason})
        else:
            outcomes["accepted_exact_title_year_and_verified_tmdb"] += 1
            accepted.append({"letterboxdId": source_id, "tmdbId": row["tmdbId"]})
            details.append({"letterboxdId": source_id, "movieLensId": target,
                            "tmdbId": row["tmdbId"], "accepted": True,
                            "reason": "Exact legacy title/year matches the verified TMDB movie or an official alternate title."})
    if len({item["letterboxdId"] for item in accepted}) != len(accepted):
        raise ValueError("Duplicate legacy source identities.")
    return sorted(accepted, key=lambda item: item["letterboxdId"]), {
        "schemaVersion": 1, "generatedAt": utc_now(), "counts": dict(sorted(outcomes.items())),
        "frontendBundleSHA256": sha256(frontend_path), "existingMappingSHA256": sha256(mapping_path),
        "policy": "Unambiguous identity, no year-offset/fuzzy acceptance, exact live TMDB title/alternate and release year.",
        "opinionsMigrated": False, "mappings": details,
    }


def build_bundle(model: dict, catalogue: list[dict], verified: dict[int, dict],
                 source_checksum: str, report_checksum: str, legacy: list[dict]) -> dict:
    validate_graph(model)
    movies = []
    for index, original in enumerate(catalogue):
        row = verified[original["movieLensId"]]
        eligible = row["status"] in VERIFIED_STATUSES
        fallback_title, fallback_year = split_title(row["movieLensTitle"])
        metadata = row.get("tmdbMovie", {}) if eligible else {}
        release = metadata.get("release_date") or ""
        year = int(release[:4]) if release[:4].isdigit() else fallback_year or 0
        movies.append({
            "movieLensId": original["movieLensId"], "tmdbId": row["tmdbId"],
            "eligible": eligible, "title": metadata.get("title") or fallback_title,
            "year": year, "genres": metadata.get("genres") or original["movieLensGenres"],
            "trainingCount": int(model["counts"][index]),
        })
    catalogue_checksum = hashlib.sha256(canonical_bytes({"movies": movies, "legacyMappings": legacy})).hexdigest()
    return {
        "format": FORMAT, "schemaVersion": 1, "modelType": "positive_conditional_neighbors",
        "modelVersion": f"{SELECTED_RUN}+tmdb-{catalogue_checksum[:12]}",
        "sourceModelSHA256": source_checksum, "tmdbVerificationSHA256": report_checksum,
        "catalogueSHA256": catalogue_checksum, "tmdbIdentityVerified": True,
        "scoring": SCORING, "movies": movies, "legacyMappings": legacy,
        "neighborOffsets": model["neighbor_indptr"].tolist(),
        "neighborIndices": model["neighbor_indices"].tolist(),
        # tolist preserves the exact value of every stored float32 as a JSON double.
        "neighborWeights": model["neighbor_weights"].tolist(),
    }


def reference_case(model: dict, bundle: dict, name: str, ratings: list[dict],
                   excluding: list[int], limit: int) -> dict:
    index = {movie["tmdbId"]: i for i, movie in enumerate(bundle["movies"]) if movie["eligible"]}
    accepted = [rating for rating in ratings if rating["tmdbId"] in index]
    items = [index[rating["tmdbId"]] for rating in accepted]
    values = [rating["stars"] for rating in accepted]
    scores = score_neighbors(model, items, values, {"aggregation": "positive"})
    # The Swift core excludes every provided rated film, plus explicit watched exclusions.
    excluded = set(excluding) | {rating["tmdbId"] for rating in ratings}
    available = [i for i, movie in enumerate(bundle["movies"])
                 if movie["eligible"] and movie["tmdbId"] not in excluded]
    available.sort(key=lambda i: (-float(scores[i]), i))
    personalized = any(max(value - 3, 0) > 0 and
                       np.any(model["neighbor_weights"][model["neighbor_indptr"][item]:model["neighbor_indptr"][item + 1]])
                       for item, value in zip(items, values))
    return {
        "name": name, "ratings": ratings, "excluding": excluding, "limit": limit,
        "expected": {"ratingsUsed": len(accepted),
                     "positiveRatingsUsed": sum(value > 3 for value in values),
                     "personalized": bool(personalized),
                     "results": [{"tmdbId": bundle["movies"][i]["tmdbId"], "score": float(scores[i])}
                                 for i in available[:limit]]},
    }


def reference_cases(model: dict, bundle: dict) -> list[dict]:
    eligible = [movie["tmdbId"] for movie in bundle["movies"] if movie["eligible"]]
    if len(eligible) < 8:
        raise ValueError("Production fixtures require at least eight supported movies.")
    ratings = lambda values: [{"tmdbId": eligible[i], "stars": value} for i, value in enumerate(values)]
    popular = reference_case(model, bundle, "no_signal_popularity", [], [], 50)
    mixed = ratings([5, 4, 1, 2, 3, 4.5])
    cases = [popular,
             reference_case(model, bundle, "five_positive", ratings([5, 4, 5, 4, 4]), [], 50),
             reference_case(model, bundle, "mixed_verdicts", mixed, [], 50),
             reference_case(model, bundle, "negative_and_neutral_only", ratings([1, 2, 3]), [], 50),
             reference_case(model, bundle, "unknown_rating_ignored", mixed + [{"tmdbId": 2_000_000_000, "stars": 5}], [], 50),
             reference_case(model, bundle, "watched_candidates_excluded", mixed,
                            [row["tmdbId"] for row in popular["expected"]["results"][:12]], 50),
             reference_case(model, bundle, "single_seed_zero_score_tail", ratings([4]), [], len(eligible)),
             reference_case(model, bundle, "all_eligible_excluded", mixed, eligible, 50)]
    return cases


def synthetic_artifacts() -> tuple[dict, dict]:
    # Independent small graph: directionality, an ineligible high-score node, equal
    # scores, an isolated seed, and popularity ties are deliberately represented.
    ids = [10, 20, 30, 40, 50, 60, 70]
    model = {"movie_ids": np.array(ids), "counts": np.array([8, 7, 7, 5, 4, 9, 3]),
             "neighbor_indptr": np.array([0, 3, 4, 5, 5, 5, 6, 6]),
             "neighbor_indices": np.array([1, 2, 5, 2, 0, 4]),
             "neighbor_weights": np.array([.25, .25, .9, .5, .125, .75], dtype=np.float32),
             "info": {"similarity": "conditional", "feedback": "positive", "shrink": 10,
                      "top_k": 100, "config": {"aggregation": "positive"}}}
    movies = [{"movieLensId": value, "tmdbId": 100 + i, "eligible": i != 5,
               "title": f"Fixture {value}", "year": 2000, "genres": ["Drama"],
               "trainingCount": int(model["counts"][i])} for i, value in enumerate(ids)]
    bundle = {"format": FORMAT, "schemaVersion": 1, "modelType": "positive_conditional_neighbors",
              "modelVersion": "synthetic-test-fixture-only", "sourceModelSHA256": "0" * 64,
              "tmdbIdentityVerified": True, "scoring": SCORING, "movies": movies,
              "legacyMappings": [{"letterboxdId": 900100, "tmdbId": 100}],
              "neighborOffsets": model["neighbor_indptr"].tolist(),
              "neighborIndices": model["neighbor_indices"].tolist(),
              "neighborWeights": model["neighbor_weights"].tolist()}
    case = lambda name, ratings, excluding=[], limit=20: reference_case(model, bundle, name, ratings, excluding, limit)
    fixtures = {"schemaVersion": 1, "modelVersion": bundle["modelVersion"], "scoreTolerance": 1e-10,
                "fixtureOnly": True,
                "cases": [case("direction_ties_zero_tail", [{"tmdbId": 100, "stars": 5}]),
                          case("weighted_sum", [{"tmdbId": 100, "stars": 4}, {"tmdbId": 101, "stars": 5}]),
                          case("fallback_counts_tie", []),
                          case("negative_only", [{"tmdbId": 100, "stars": 1}, {"tmdbId": 103, "stars": 2}]),
                          case("isolated_positive_fallback", [{"tmdbId": 106, "stars": 5}]),
                          case("ineligible_and_unknown_ignored", [{"tmdbId": 105, "stars": 5}, {"tmdbId": 999, "stars": 4}]),
                          case("exclude_positive_hits", [{"tmdbId": 100, "stars": 5}], [101, 102]),
                          case("fractional_positive", [{"tmdbId": 100, "stars": 3.5}])]}
    return bundle, fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Dataset/ml-32m")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/integration/tmdb_verification_report.json")
    parser.add_argument("--frontend", type=Path, default=ROOT.parent / "peter/CineMatch")
    parser.add_argument("--max-verification-age-days", type=float, default=7.0)
    parser.add_argument("--synthetic-only", action="store_true", help="Write explicitly synthetic test artifacts only, never a production bundle.")
    args = parser.parse_args()
    fixture_dir = args.frontend / "ml/conditional_knn"
    if args.synthetic_only:
        synthetic, cases = synthetic_artifacts()
        atomic_json(fixture_dir / "synthetic_model.json", synthetic)
        atomic_json(fixture_dir / "synthetic_cases.json", cases)
        print("Wrote synthetic test artifacts only; production model remains unchanged.")
        return
    if sha256(args.model) != SELECTED_MODEL_SHA256:
        raise ValueError("This export requires the exact frozen selected model artifact.")
    catalogue, sources = load_catalogue(args.model, args.data_dir)
    report = json.loads(args.report.read_text())
    verified = validate_report(report, catalogue, sources, args.max_verification_age_days)
    model = load_neighbors(args.model)
    legacy, legacy_report = legacy_bridge(
        args.frontend / "CineMatch/Core/Resources/cinematch_model.json",
        ROOT.parent / "extra_data/id_mapping.csv", args.data_dir / "movies.csv", verified)
    bundle = build_bundle(model, catalogue, verified, sources["modelSHA256"], sha256(args.report), legacy)
    payload = canonical_bytes(bundle)
    checksum = hashlib.sha256(payload).hexdigest()
    fixtures = {"schemaVersion": 1, "modelVersion": bundle["modelVersion"],
                "sourceModelSHA256": sources["modelSHA256"], "bundleSHA256": checksum,
                "scoreTolerance": 1e-10, "cases": reference_cases(model, bundle)}
    output = args.frontend / "CineMatch/Core/Resources/cinematch_knn.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    atomic_json(fixture_dir / "reference_cases.json", fixtures)
    atomic_json(args.frontend / "CineMatchTests/knn_reference_cases.json", fixtures)
    synthetic, synthetic_cases = synthetic_artifacts()
    atomic_json(fixture_dir / "synthetic_model.json", synthetic)
    atomic_json(fixture_dir / "synthetic_cases.json", synthetic_cases)
    atomic_json(ROOT / "experiments/integration/legacy_mapping_verified.json", legacy_report)
    manifest = {"schemaVersion": 1, "exportedAt": utc_now(), "modelVersion": bundle["modelVersion"],
                "bundleSHA256": checksum, "sourceModelSHA256": sources["modelSHA256"],
                "verificationReportSHA256": sha256(args.report), "catalogueSHA256": bundle["catalogueSHA256"],
                "graphNodes": len(catalogue), "directedEdges": len(model["neighbor_indices"]),
                "eligibleMovies": report["eligibleCount"], "legacyMappings": len(legacy),
                "bundleBytes": len(payload), "scoreContract": SCORING,
                "verificationCounts": report["counts"], "verificationPolicy": POLICY,
                "finalHoldoutRead": False, "frozenModelModified": False}
    atomic_json(fixture_dir / "manifest.json", manifest)
    atomic_json(ROOT / "experiments/integration/ios_export_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
