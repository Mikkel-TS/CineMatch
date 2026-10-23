"""Reproducible user-disjoint splits; item eligibility uses training ratings only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_arrays(directory: Path, name: str) -> dict[str, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{name}.npz"
    if digest(path) != manifest["sha256"][path.name]:
        raise ValueError(f"Prepared data changed: {path}. Prepare a new dataset directory.")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def make_profiles(frame: pd.DataFrame, support_count: int, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    users, support_i, support_r, target_i, target_r = [], [], [], [], []
    offsets = [0]
    excluded = 0
    for user_id, group in frame.groupby("userId", sort=True):
        if len(group) <= support_count:
            excluded += 1
            continue
        order = rng.permutation(len(group))
        si, ti = order[:support_count], order[support_count:]
        items = group["movie_index"].to_numpy(dtype=np.int64)
        ratings = group["rating"].to_numpy(dtype=np.float32)
        users.append(int(user_id))
        support_i.append(items[si])
        support_r.append(ratings[si])
        target_i.extend(items[ti])
        target_r.extend(ratings[ti])
        offsets.append(len(target_i))
    if not users:
        raise ValueError("No evaluable users remain. Lower --min-item-ratings or use more training users.")
    arrays = {
        "user_ids": np.asarray(users, dtype=np.int64),
        "support_items": np.asarray(support_i, dtype=np.int64),
        "support_ratings": np.asarray(support_r, dtype=np.float32),
        "target_offsets": np.asarray(offsets, dtype=np.int64),
        "target_items": np.asarray(target_i, dtype=np.int64),
        "target_ratings": np.asarray(target_r, dtype=np.float32),
    }
    return arrays, {"evaluated_users": len(users), "excluded_users_too_few_ratings": excluded,
                    "support_ratings": len(users) * support_count, "target_ratings": len(target_i)}


def prepare(data_dir: Path, output: Path, train_users: int = 10000,
            validation_users: int = 500, test_users: int = 500,
            support_count: int = 5, min_item_ratings: int = 10,
            seed: int = 42, chunk_size: int = 1000000) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new directory; existing splits are preserved.")
    if train_users < 0 or min(validation_users, test_users, support_count, min_item_ratings, chunk_size) < 1:
        raise ValueError("Counts must be positive; --train-users 0 means all remaining users.")
    ratings_path = data_dir / "ratings.csv"
    unique_users = set()
    total_rows = 0
    print("Reading user IDs across the ratings file...", flush=True)
    for chunk in pd.read_csv(ratings_path, usecols=["userId"], dtype={"userId": "int64"}, chunksize=chunk_size):
        unique_users.update(chunk["userId"].unique().tolist())
        total_rows += len(chunk)
    ordered = np.asarray(sorted(unique_users), dtype=np.int64)
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(ordered)
    reserve = validation_users + test_users
    n_train = train_users or len(ordered) - reserve
    if n_train < 1 or n_train + reserve > len(ordered):
        raise ValueError(f"Requested too many users: {len(ordered)} available.")
    train_ids = ordered[:n_train]
    val_ids = ordered[n_train:n_train + validation_users]
    test_ids = ordered[n_train + validation_users:n_train + reserve]
    selected = np.concatenate([train_ids, val_ids, test_ids])
    parts = []
    print(f"Loading histories for {len(selected):,} sampled users...", flush=True)
    for chunk in pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"],
                             dtype={"userId": "int64", "movieId": "int64", "rating": "float32"},
                             chunksize=chunk_size):
        subset = chunk.loc[chunk["userId"].isin(selected)]
        if not subset.empty:
            parts.append(subset)
    frame = pd.concat(parts, ignore_index=True)
    values = frame["rating"].to_numpy()
    if not (np.isfinite(values).all() and ((values >= .5) & (values <= 5)).all()
            and np.isclose(values * 2, np.round(values * 2)).all()):
        raise ValueError("Ratings must be finite half-star values from 0.5 through 5.")
    if frame.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user/movie pairs found; resolve them before preparing data.")
    train = frame.loc[frame["userId"].isin(train_ids)].copy()
    validation_rows_before_filter = int(frame["userId"].isin(val_ids).sum())
    test_rows_before_filter = int(frame["userId"].isin(test_ids).sum())
    counts = train.groupby("movieId").size()
    movie_ids = counts[counts >= min_item_ratings].index.to_numpy(dtype=np.int64)
    if len(movie_ids) < support_count + 5:
        raise ValueError("Too few eligible movies. Lower --min-item-ratings or increase training users.")
    original_train_rows = len(train)
    frame = frame.loc[frame["movieId"].isin(movie_ids)].copy()
    frame["movie_index"] = np.searchsorted(movie_ids, frame["movieId"].to_numpy())
    train = frame.loc[frame["userId"].isin(train_ids)]
    actual_train_ids = np.sort(train["userId"].unique())
    train_arrays = {
        "user_index": np.searchsorted(actual_train_ids, train["userId"].to_numpy()).astype(np.int64),
        "movie_index": train["movie_index"].to_numpy(dtype=np.int64),
        "rating": train["rating"].to_numpy(dtype=np.float32),
        "user_ids": actual_train_ids.astype(np.int64), "movie_ids": movie_ids,
        "item_counts": counts.loc[movie_ids].to_numpy(dtype=np.int64),
    }
    val_frame = frame.loc[frame["userId"].isin(val_ids)]
    test_frame = frame.loc[frame["userId"].isin(test_ids)]
    val, val_info = make_profiles(val_frame, support_count, seed + 1)
    test, test_info = make_profiles(test_frame, support_count, seed + 2)
    val_info["excluded_users_too_few_ratings"] = validation_users - len(val["user_ids"])
    test_info["excluded_users_too_few_ratings"] = test_users - len(test["user_ids"])
    val_info.update({"rating_rows_before_catalog_filter": validation_rows_before_filter,
                     "rating_rows_after_catalog_filter": len(val_frame)})
    test_info.update({"rating_rows_before_catalog_filter": test_rows_before_filter,
                      "rating_rows_after_catalog_filter": len(test_frame)})
    movies = pd.read_csv(data_dir / "movies.csv").set_index("movieId")
    if not movies.index.is_unique or not np.isin(movie_ids, movies.index).all():
        raise ValueError("Movie metadata must have one row for every eligible movie ID.")
    catalog = movies.loc[movie_ids, ["title", "genres"]].fillna("").reset_index()
    catalog["training_rating_count"] = train_arrays["item_counts"]
    output.mkdir(parents=True)
    for name, arrays in [("train", train_arrays), ("validation", val), ("test", test)]:
        np.savez_compressed(output / f"{name}.npz", **arrays)
    catalog.to_csv(output / "catalog.csv", index=False)
    manifest = {
        "format_version": 1, "seed": seed,
        "design": "Retrospective reconstruction for disjoint new users; random onboarding ratings, not a temporal forecast.",
        "source": {"ratings_path": str(ratings_path.resolve()), "rows_scanned": total_rows},
        "support_count": support_count, "min_item_ratings": min_item_ratings,
        "candidate_movies": len(movie_ids), "sampled_training_users": n_train,
        "training_users": len(actual_train_ids), "training_ratings_before_filter": original_train_rows,
        "training_ratings": len(train), "training_mean": float(train["rating"].mean()),
        "sampled_train_user_ids": train_ids.tolist(),
        "sampled_validation_user_ids": val_ids.tolist(), "sampled_test_user_ids": test_ids.tolist(),
        "validation": val_info, "test": test_info,
        "sha256": {name: digest(output / name) for name in ["train.npz", "validation.npz", "test.npz", "catalog.csv"]},
    }
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(train):,} training ratings and {len(movie_ids):,} candidate movies.", flush=True)
    return manifest
