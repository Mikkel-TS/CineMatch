#!/usr/bin/env python3
"""Freeze common-catalogue data for the Anton/conditional app comparison.

This command prepares labels and disjoint cohorts only. It trains no model and
computes no predictions. Existing production artifacts and old holdout labels
are never modified or read. A nonempty output directory is rejected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL_DATA = "artifacts/experiments/mf16-full-users-fixed-catalog-20261117/data"
MODES = (("positive5", "positive", 5), ("positive10", "positive", 10),
         ("positive20", "positive", 20), ("mixed10", "mixed", 10),
         ("mixed20", "mixed", 20))
BAD_LETTERBOXD = 1020663
GOOD_LETTERBOXD = 1001148
MISSING_TMDB = 768362


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_npz(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)


def app_verdicts(ratings: np.ndarray) -> np.ndarray:
    """Declared MovieLens-to-app proxy: floor, then clip to verdicts 1..5."""
    ratings = np.asarray(ratings)
    if not np.isfinite(ratings).all() or np.any((ratings < .5) | (ratings > 5)):
        raise ValueError("MovieLens ratings must be finite and within .5..5")
    return np.clip(np.floor(ratings), 1, 5).astype(np.float32)


def common_catalogue(bundle: dict, metadata: pd.DataFrame, cache: dict) -> tuple[list, list]:
    """Use verified legacy membership, correcting one evidenced identity collision."""
    if not bundle.get("tmdbIdentityVerified"):
        raise ValueError("The app catalogue must have verified TMDB identity")
    if metadata["id"].duplicated().any():
        raise ValueError("Letterboxd metadata IDs must be unique")
    meta = metadata.set_index("id")
    eligible = {}
    for i, movie in enumerate(bundle["movies"]):
        if movie["eligible"]:
            if movie["tmdbId"] in eligible:
                raise ValueError("Eligible TMDB IDs must be unique")
            eligible[movie["tmdbId"]] = (i, movie)
    accepted, rejected = {}, []
    for link in bundle["legacyMappings"]:
        lb, tmdb = int(link["letterboxdId"]), int(link["tmdbId"])
        if tmdb not in eligible:
            continue
        if lb not in meta.index:
            raise ValueError(f"Missing Letterboxd metadata for {lb}")
        if lb == BAD_LETTERBOXD and tmdb == MISSING_TMDB:
            verified = cache["responses"][str(MISSING_TMDB)]["movie"]
            good, bad = meta.loc[GOOD_LETTERBOXD], meta.loc[BAD_LETTERBOXD]
            if (verified["title"] != "Missing" or verified["runtime"] != 111
                    or good["name"] != "Missing" or int(good["minute"]) != 111
                    or bad["name"] != "The Missing" or int(bad["minute"]) != 90
                    or "June" not in str(good["description"])
                    or "mouthless" not in str(bad["description"])):
                raise ValueError("Known identity correction evidence has changed")
            rejected.append({
                "letterboxdId": lb, "tmdbId": tmdb,
                "movieLensId": int(eligible[tmdb][1]["movieLensId"]),
                "reason": "Distinct film collapsed by article-insensitive title/year matching",
                "rejected_title": str(bad["name"]), "rejected_runtime": int(bad["minute"]),
                "rejected_overview": str(bad["description"]),
                "retained_letterboxdId": GOOD_LETTERBOXD,
                "verified_tmdb_title": verified["title"], "verified_tmdb_runtime": verified["runtime"],
                "verified_tmdb_imdb_id": verified.get("imdb_id"),
                "verified_tmdb_overview": verified["overview"],
                "retained_letterboxd_overview": str(good["description"]),
            })
            continue
        if tmdb in accepted:
            raise ValueError(f"Unresolved duplicate Letterboxd mapping for TMDB {tmdb}")
        original_index, movie = eligible[tmdb]
        row = meta.loc[lb]
        if pd.isna(row["rating"]):
            raise ValueError(f"Letterboxd {lb} is outside Anton's rated metadata index")
        accepted[tmdb] = {
            "movieId": int(movie["movieLensId"]), "letterboxdId": lb, "tmdbId": tmdb,
            "title": movie["title"], "year": movie["year"],
            "original_model_index": original_index,
        }
    rows = sorted(accepted.values(), key=lambda x: x["movieId"])
    if not rows or len({x["movieId"] for x in rows}) != len(rows):
        raise ValueError("Common catalogue must contain unique MovieLens IDs")
    return rows, rejected


def prepare_arrays(base: dict, movies: np.ndarray, old_holdouts: np.ndarray,
                   validation_users: int = 1000, test_users: int = 2000,
                   seed: int = 20261119, min_ratings: int = 35,
                   min_positives: int = 21) -> tuple[dict, dict, dict, dict]:
    """Select fresh users from prior training and remove them before any fitting."""
    movies = np.asarray(movies, dtype=np.int64)
    source_ids = np.asarray(base["user_ids"], dtype=np.int64)
    source_movies = np.asarray(base["movie_ids"], dtype=np.int64)
    if not len(movies) or not np.array_equal(movies, np.unique(movies)):
        raise ValueError("Common MovieLens IDs must be unique and ascending")
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Source user IDs must be unique")
    if np.intersect1d(source_ids, old_holdouts).size:
        raise ValueError("Source training users overlap old holdouts")
    if not np.isin(movies, source_movies).all():
        raise ValueError("Common movie missing from source training catalogue")
    if min(validation_users, test_users) < 1 or min_ratings < 1 or min_positives < 1:
        raise ValueError("Cohort sizes and eligibility thresholds must be positive")
    source_user = np.asarray(base["user_index"])
    source_item = np.asarray(base["movie_index"])
    source_rating = np.asarray(base["rating"])
    if (source_user.shape != source_item.shape or source_user.shape != source_rating.shape
            or source_user.ndim != 1 or np.any(source_user < 0)
            or np.any(source_user >= len(source_ids)) or np.any(source_item < 0)
            or np.any(source_item >= len(source_movies))):
        raise ValueError("Source interactions have invalid shape or indices")
    app_verdicts(source_rating)  # Validate all source values, including excluded rows.
    movie_mask = np.isin(source_movies, movies)
    common_mask = movie_mask[source_item]
    old_to_new_movie = np.full(len(source_movies), -1, dtype=np.int64)
    old_to_new_movie[movie_mask] = np.searchsorted(movies, source_movies[movie_mask])
    users = source_user[common_mask]
    items = old_to_new_movie[source_item[common_mask]]
    ratings = source_rating[common_mask].astype(np.float32)
    counts = np.bincount(users, minlength=len(source_ids))
    positives = np.bincount(users[ratings >= 4], minlength=len(source_ids))
    eligible = np.sort(source_ids[(counts >= min_ratings) & (positives >= min_positives)])
    if len(eligible) < validation_users + test_users:
        raise ValueError("Too few eligible users for disjoint fresh cohorts")
    selected = np.random.default_rng(seed).permutation(eligible)
    validation = np.sort(selected[:validation_users])
    test = np.sort(selected[validation_users:validation_users + test_users])
    val_mask = np.isin(source_ids, validation)
    test_mask = np.isin(source_ids, test)
    train_user_mask = (counts > 0) & ~val_mask & ~test_mask
    keep = train_user_mask[users]
    training_ids = np.sort(source_ids[train_user_mask])
    remap_user = np.full(len(source_ids), -1, dtype=np.int64)
    remap_user[train_user_mask] = np.searchsorted(training_ids, source_ids[train_user_mask])
    training = {
        "user_ids": training_ids, "movie_ids": movies,
        "user_index": remap_user[users[keep]], "movie_index": items[keep],
        "rating": ratings[keep], "item_counts": np.bincount(items[keep], minlength=len(movies)),
    }
    masks = {
        "source_user_ids": source_ids, "training_user_ids": training_ids,
        "validation_user_ids": validation, "test_user_ids": test,
        "old_excluded_user_ids": np.sort(np.unique(old_holdouts)),
        "excluded_tag_user_ids": np.unique(np.concatenate([old_holdouts, validation, test])),
        "source_common_movie_mask": movie_mask, "source_common_row_mask": common_mask,
        "source_user_is_validation": val_mask, "source_user_is_test": test_mask,
        "source_user_is_training": train_user_mask,
    }
    # Only held-out interactions need sorting/grouping; do not sort all 22M rows.
    held_rows = (val_mask | test_mask)[users]
    held = {"user_id": source_ids[users[held_rows]], "movie_index": items[held_rows],
            "rating": ratings[held_rows]}
    order = np.argsort(held["user_id"], kind="stable")
    held = {k: v[order] for k, v in held.items()}
    info = {
        "source_training_users": len(source_ids), "source_training_rows": len(source_rating),
        "common_training_users_before_new_split": int((counts > 0).sum()),
        "common_training_rows_before_new_split": len(ratings),
        "common_positive_rows_before_new_split": int((ratings >= 4).sum()),
        "eligible_fresh_users": len(eligible), "training_users": len(training_ids),
        "training_rows": len(training["rating"]), "validation_users": len(validation),
        "test_users": len(test), "old_excluded_users": len(np.unique(old_holdouts)),
        "zero_training_count_movies": int((training["item_counts"] == 0).sum()),
    }
    return training, masks, held, info


def make_profiles(held: dict, user_ids: np.ndarray, mode: str, size: int,
                  seed: int = 20261119) -> dict:
    """Nested positive/mixed seeds; raw hidden labels are never transformed."""
    if mode not in ("positive", "mixed", "fit") or size < 1:
        raise ValueError("Unknown profile mode or invalid size")
    ids = np.asarray(user_ids, dtype=np.int64)
    support, support_raw, target_items, target_ratings = [], [], [], []
    offsets = [0]
    for uid in ids:
        lo = np.searchsorted(held["user_id"], uid, side="left")
        hi = np.searchsorted(held["user_id"], uid, side="right")
        items, ratings = held["movie_index"][lo:hi], held["rating"][lo:hi]
        if not len(items) or len(np.unique(items)) != len(items):
            raise ValueError("Each held-out user needs unique observed movie ratings")
        # Canonical item order makes profiles independent of source interaction order.
        canonical = np.argsort(items, kind="stable")
        items, ratings = items[canonical], ratings[canonical]
        available = np.flatnonzero(ratings >= 4) if mode == "positive" else np.arange(len(items))
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid), 1 if mode == "positive" else 2]))
        shuffled = rng.permutation(available)
        if len(shuffled) < size:
            raise ValueError("Held-out user has insufficient support ratings")
        selected = shuffled[:size]
        if mode == "fit":
            targets = shuffled[size:size + 30]
        else:
            keep = np.ones(len(items), dtype=bool)
            keep[selected] = False
            targets = np.flatnonzero(keep)
        if not len(targets):
            raise ValueError("Held-out user has no target ratings")
        support.append(items[selected])
        support_raw.append(ratings[selected])
        target_items.append(items[targets])
        target_ratings.append(ratings[targets])
        offsets.append(offsets[-1] + len(targets))
    original = np.asarray(support_raw, dtype=np.float32)
    return {
        "user_ids": ids, "support_items": np.asarray(support, dtype=np.int64),
        "support_ratings": original.copy() if mode == "fit" else app_verdicts(original),
        "support_original_ratings": original,
        "target_offsets": np.asarray(offsets, dtype=np.int64),
        "target_items": np.concatenate(target_items).astype(np.int64),
        "target_ratings": np.concatenate(target_ratings).astype(np.float32),
    }


def profile_summary(data: dict) -> dict:
    positives = np.add.reduceat((data["target_ratings"] >= 4).astype(np.int64), data["target_offsets"][:-1])
    return {"users": len(data["user_ids"]), "support_size": data["support_items"].shape[1],
            "target_ratings": len(data["target_ratings"]),
            "users_with_positive_targets": int((positives > 0).sum()),
            "min_target_count": int(np.diff(data["target_offsets"]).min()),
            "max_target_count": int(np.diff(data["target_offsets"]).max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-users", type=int, default=1000)
    parser.add_argument("--test-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("Output must be a new or empty directory; refusing to overwrite frozen data")
    sources = {
        "training": root / FULL_DATA / "train.npz",
        "training_manifest": root / FULL_DATA / "manifest.json",
        "app_bundle": root.parent / "peter/CineMatch/CineMatch/Core/Resources/cinematch_knn.json",
        "letterboxd_movies": root.parent / "extra_data/archive/movies.csv",
        "tmdb_cache": root / "experiments/integration/tmdb_verification_cache.json",
    }
    source_info = {k: {"path": str(v), "sha256": digest(v), "bytes": v.stat().st_size} for k, v in sources.items()}
    plan = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Data preparation only; no model scores or candidate selection",
        "sources": source_info, "preparation_source_sha256": digest(Path(__file__)),
        "seed": args.seed, "validation_users": args.validation_users, "test_users": args.test_users,
        "eligibility": {"minimum_common_ratings": 35, "minimum_common_positive_ratings": 21,
                        "positive_threshold": 4, "source_population": "Existing full-training users only"},
        "selection": "Permutation of ascending eligible user IDs with numpy default_rng(seed); validation first, test next; output groups sorted",
        "catalogue": "Verified eligible legacy overlap, unique TMDB/MovieLens identities, ascending MovieLens ID; reject evidenced LB1020663 alias",
        "support_proxy": "floor(raw MovieLens rating), clipped to 1..5; 4/4.5 become Great4, 5 becomes Fantastic5; 0.5 becomes1 and3.5 becomes3",
        "support_sampling": "Per-user SeedSequence([seed,userId,stream]); canonical item order; stream1 positive,stream2 mixed/fit; nested prefixes",
        "ranking_targets": "All remaining observed common-catalogue ratings, original .5..5 scale; relevant if>=4; unrated is unknown",
        "fit10": "Validation only: ten random RAW historical ratings and up to30 remaining raw targets; Anton source .5..5 to1..4 conversion belongs in adapter",
        "cohort_bias": "Users are historical MovieLens raters with>=35 common ratings and>=21 positives; random positive selections approximate, not reproduce, self-selected favourites",
        "scope": "Refitted algorithm comparison on common catalogue, not unchanged production artifacts or online app satisfaction",
        "test_scores_read": False,
    }
    write_json(output / "plan.json", plan)
    bundle = json.loads(sources["app_bundle"].read_text())
    cache = json.loads(sources["tmdb_cache"].read_text())
    metadata = pd.read_csv(sources["letterboxd_movies"], usecols=["id", "name", "date", "minute", "rating", "description"])
    catalogue, identity_rejections = common_catalogue(bundle, metadata, cache)
    if len(catalogue) != 3788:
        raise ValueError(f"Expected reviewed common catalogue of3788 movies, found{len(catalogue)}")
    pd.DataFrame(catalogue).to_csv(output / "catalogue.csv", index=False)
    original_manifest = json.loads(sources["training_manifest"].read_text())
    old_holdouts = np.unique(np.concatenate([original_manifest[k] for k in
        ("sampled_validation_user_ids", "sampled_test_user_ids", "fresh_test_user_ids")]))
    with np.load(sources["training"], allow_pickle=False) as loaded:
        base = {k: loaded[k] for k in loaded.files}
    training, cohorts, held, counts = prepare_arrays(
        base, np.asarray([m["movieId"] for m in catalogue]), old_holdouts,
        args.validation_users, args.test_users, args.seed)
    write_npz(output / "training.npz", training)
    write_npz(output / "cohorts.npz", cohorts)
    profiles = {}
    for split in ("validation", "test"):
        for name, mode, size in MODES:
            data = make_profiles(held, cohorts[f"{split}_user_ids"], mode, size, args.seed)
            path = f"{split}/{name}.npz"
            write_npz(output / path, data)
            profiles[path] = profile_summary(data)
    data = make_profiles(held, cohorts["validation_user_ids"], "fit", 10, args.seed)
    write_npz(output / "validation/fit10.npz", data)
    profiles["validation/fit10.npz"] = profile_summary(data)
    artifacts = {str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {
        **plan, "counts": {"common_movies": len(catalogue), **counts},
        "identity_rejections": identity_rejections, "profiles": profiles,
        "validation_user_ids": cohorts["validation_user_ids"].tolist(),
        "test_user_ids": cohorts["test_user_ids"].tolist(),
        "old_excluded_user_ids": cohorts["old_excluded_user_ids"].tolist(),
        "artifacts": artifacts, "status": "prepared; test predictions not computed",
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "counts": manifest["counts"], "profiles": profiles}, indent=2))


if __name__ == "__main__":
    main()
