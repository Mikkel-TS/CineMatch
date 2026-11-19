"""Run one recorded full-user experiment against the frozen mf16 benchmark.

Run from the repository root with .venv/bin/python -B experiments/run_full_data.py.
Source and historical artifacts are preserved; each run needs a new run ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def select_users(all_users, original, fresh_count, seed):
    old_train = set(original["sampled_train_user_ids"])
    validation = set(original["sampled_validation_user_ids"])
    test = set(original["sampled_test_user_ids"])
    if old_train & validation or old_train & test or validation & test:
        raise ValueError("Reference user groups overlap.")
    if not (old_train | validation | test) <= set(all_users):
        raise ValueError("Reference users are missing from the source.")
    available = np.asarray(sorted(set(all_users) - old_train - validation - test), dtype=np.int64)
    if len(available) < fresh_count:
        raise ValueError("Not enough previously unused users for a fresh final holdout.")
    fresh = np.sort(np.random.default_rng(seed).choice(available, fresh_count, replace=False))
    train_ids = np.asarray(sorted(set(all_users) - validation - test - set(fresh)), dtype=np.int64)
    return train_ids, fresh


def prepare_full(source, reference, output, fresh_count=500, seed=20261117, chunk_size=1_000_000):
    from cinematch.data import make_profiles

    original = json.loads((reference / "manifest.json").read_text())
    for name, expected in original["sha256"].items():
        if digest(reference / name) != expected:
            raise ValueError(f"Reference data changed: {name}")
    catalog = pd.read_csv(reference / "catalog.csv")
    movies = catalog.movieId.to_numpy(dtype=np.int64)
    if len(np.unique(movies)) != len(movies) or np.any(np.diff(movies) <= 0):
        raise ValueError("Reference catalogue must contain unique sorted movie IDs.")
    ratings_path = source / "ratings.csv"
    source_hash = digest(ratings_path)
    all_users, scanned = set(), 0
    print("Scanning full data and reserving evaluation users...", flush=True)
    for chunk in pd.read_csv(ratings_path, usecols=["userId"], chunksize=chunk_size):
        all_users.update(chunk.userId.unique().tolist())
        scanned += len(chunk)
    train_ids, fresh = select_users(all_users, original, fresh_count, seed)
    # Fixed-size buffers bound preparation memory without retaining DataFrames.
    user_buffer = np.empty(scanned, dtype=np.int32)
    item_buffer = np.empty(scanned, dtype=np.int32)
    rating_buffer = np.empty(scanned, dtype=np.float32)
    used = before_filter = 0
    fresh_frames = []
    for chunk in pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"],
                             dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
                             chunksize=chunk_size):
        values = chunk.rating.to_numpy()
        if not (np.isfinite(values).all() and ((values >= .5) & (values <= 5)).all()
                and np.isclose(values * 2, np.round(values * 2)).all()):
            raise ValueError("Invalid source ratings.")
        is_train = chunk.userId.isin(train_ids)
        in_catalog = chunk.movieId.isin(movies)
        before_filter += int(is_train.sum())
        selected = chunk.loc[is_train & in_catalog]
        end = used + len(selected)
        user_buffer[used:end] = np.searchsorted(train_ids, selected.userId.to_numpy())
        item_buffer[used:end] = np.searchsorted(movies, selected.movieId.to_numpy())
        rating_buffer[used:end] = selected.rating.to_numpy()
        used = end
        reserved = chunk.loc[chunk.userId.isin(fresh) & in_catalog].copy()
        if len(reserved):
            fresh_frames.append(reserved)
    if used == 0:
        raise ValueError("No training ratings remain.")
    if digest(ratings_path) != source_hash:
        raise ValueError("Source ratings changed during preparation.")
    active = np.bincount(user_buffer[:used], minlength=len(train_ids)) > 0
    actual_users = train_ids[active]
    remap = np.cumsum(active) - 1
    arrays = {
        "user_index": remap[user_buffer[:used]].astype(np.int64),
        "movie_index": item_buffer[:used].astype(np.int64),
        "rating": rating_buffer[:used].copy(),
        "user_ids": actual_users, "movie_ids": movies,
        "item_counts": np.bincount(item_buffer[:used], minlength=len(movies)).astype(np.int64),
    }
    del user_buffer, item_buffer, rating_buffer
    pairs = arrays["user_index"] * len(movies) + arrays["movie_index"]
    if len(np.unique(pairs)) != used:
        raise ValueError("Duplicate user/movie ratings.")
    del pairs
    with np.load(reference / "train.npz", allow_pickle=False) as old:
        if not np.array_equal(movies, old["movie_ids"]):
            raise ValueError("Reference movie ordering differs.")
        if np.any(arrays["item_counts"] < old["item_counts"]):
            raise ValueError("Expanded training did not retain the original movie support.")
    fresh_frame = pd.concat(fresh_frames, ignore_index=True)
    fresh_frame["movie_index"] = np.searchsorted(movies, fresh_frame.movieId.to_numpy())
    fresh_profiles, fresh_info = make_profiles(fresh_frame, original["support_count"], seed + 1)
    fresh_info["reserved_users"] = len(fresh)
    fresh_info["excluded_users_too_few_ratings"] = len(fresh) - len(fresh_profiles["user_ids"])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "train.npz", **arrays)
    np.savez_compressed(output / "fresh_test.npz", **fresh_profiles)
    for name in ("validation.npz", "test.npz"):
        shutil.copyfile(reference / name, output / name)
    catalog["training_rating_count"] = arrays["item_counts"]
    catalog.to_csv(output / "catalog.csv", index=False)
    manifest = {
        "format_version": 1, "seed": original["seed"],
        "design": "All available training users; frozen original catalogue and validation/legacy-test profiles.",
        "source": {"ratings_path": str(ratings_path.resolve()), "rows_scanned": scanned, "sha256": source_hash},
        "reference_manifest_sha256": digest(reference / "manifest.json"),
        "support_count": original["support_count"], "min_item_ratings": original["min_item_ratings"],
        "catalogue_policy": "Freeze original movie IDs/order; recompute training counts using expanded training users.",
        "candidate_movies": len(movies), "sampled_training_users": len(train_ids),
        "training_users": len(actual_users), "training_ratings_before_filter": before_filter,
        "training_ratings": used, "training_mean": float(arrays["rating"].mean()),
        "sampled_train_user_ids": train_ids.tolist(),
        "sampled_validation_user_ids": original["sampled_validation_user_ids"],
        "sampled_test_user_ids": original["sampled_test_user_ids"],
        "fresh_test_user_ids": fresh.tolist(), "fresh_test_seed": seed,
        "validation": original["validation"], "test": original["test"], "fresh_test": fresh_info,
        "test_status": "Legacy test already inspected; fresh_test must remain unused for model selection.",
        "sha256": {name: digest(output / name) for name in
                   ("train.npz", "validation.npz", "test.npz", "fresh_test.npz", "catalog.csv")},
    }
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {used:,} ratings from {len(actual_users):,} users; catalogue fixed at {len(movies):,}.", flush=True)
    return manifest


def compare(reference_model, model_path, data):
    from cinematch.data import load_arrays
    from cinematch.evaluation import evaluate_profiles, paired_comparison
    from cinematch.training import load_checkpoint, checkpoint_parameters

    profiles = load_arrays(data, "validation")
    old, new = load_checkpoint(reference_model), load_checkpoint(model_path)
    if not torch.equal(old["movie_ids"], new["movie_ids"]):
        raise ValueError("Comparison catalogue/order differs.")
    metrics, rows = {}, {}
    for label, checkpoint in (("sample_model", old), ("full_user_model", new)):
        mean, bias, factors = checkpoint_parameters(checkpoint)
        metrics[label], rows[label] = evaluate_profiles(mean, bias, factors, profiles,
            checkpoint["fold_in_regularization"], checkpoint["bias_regularization"])
        pop = label.replace("model", "popularity")
        metrics[pop], rows[pop] = evaluate_profiles(mean, bias, factors, profiles, 1., 5.,
                                                   fixed_scores=checkpoint["item_counts"].numpy())
    result = {
        "evaluation": "Frozen validation users, support ratings, targets, catalogue and metric implementation.",
        "metrics": metrics,
        "full_vs_sample": paired_comparison(rows["full_user_model"], rows["sample_model"], 42),
        "full_vs_full_popularity": paired_comparison(rows["full_user_model"], rows["full_user_popularity"], 42),
        "full_vs_sample_popularity": paired_comparison(rows["full_user_model"], rows["sample_popularity"], 42),
        "limits": "Exploratory validation comparison from one training seed; bootstrap covers validation users only.",
        "test_evaluated": False, "fresh_test_evaluated": False,
    }
    return result, rows, profiles["user_ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="mf16-full-users-fixed-catalog-20261117")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id):
        parser.error("Use a lowercase run ID with letters, numbers, hyphens or underscores.")
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        parser.error("Run from the CineMatch repository root.")
    reference = root / "artifacts/data"
    old_model = root / "artifacts/mf16/model.pt"
    run = root / "artifacts/experiments" / args.run_id
    record = root / "experiments/records" / f"{args.run_id}.json"
    if run.exists() or record.exists():
        parser.error("Run ID already exists; choose a new --run-id.")
    original_checkpoint = torch.load(old_model, map_location="cpu", weights_only=True)
    if original_checkpoint["data_manifest_sha256"] != digest(reference / "manifest.json"):
        parser.error("Historical checkpoint and reference dataset do not match.")
    run.mkdir(parents=True)
    log = (run / "run.log").open("x", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    record.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((old_model.parent / "config.json").read_text())
    parameters = {k: config[k] for k in ("factors", "epochs", "batch_size", "learning_rate", "regularization",
                                       "fold_in_grid", "bias_regularization", "patience", "seed", "threads")}
    benchmark = hashlib.sha256((digest(reference / "validation.npz") +
        digest(reference / "catalog.csv") + digest(root / "src/cinematch/evaluation.py")).encode()).hexdigest()
    plan = {
        "run_id": args.run_id, "parent_run": "mf16-sample-20261023", "created_at": now(),
        "status": "preparing", "benchmark_id": benchmark,
        "question": "Does using all available training users improve recommendation quality on the existing catalogue?",
        "hypothesis": "More real ratings improve movie profiles and new-user ranking; this remains unproven.",
        "user_decisions": ["Try full-data training.", "Keep the current 9,021-movie catalogue for a clean comparison."],
        "implementation_decisions": [
            "Keep original validation and legacy-test profiles byte-for-byte, including onboarding and targets.",
            "Reserve 500 previously unused users for a future final test before expanding training.",
            "Keep model architecture, objective, hyperparameters and epoch-selection policy unchanged.",
            "Compare both historical and newly fitted popularity controls on the frozen validation benchmark.",
            "Do not evaluate either test group or automatically replace the existing model.",
        ],
        "changed_variable": "Training-user population and the ratings they contribute within the frozen catalogue.",
        "interpretation_limits": [
            "More ratings also mean more optimizer updates per epoch; this is not an equal-compute comparison.",
            "A fixed random seed does not guarantee identical initial item factors when user-table size changes.",
            "Historical checkpoint source revision is not embedded; the current run records its exact source snapshot.",
        ],
        "parameters": parameters, "reference_checkpoint_sha256": digest(old_model),
        "reference_manifest_sha256": digest(reference / "manifest.json"),
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain=v1"),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "artifact_directory": str(run.relative_to(root)),
    }
    shutil.copytree(root / "src", run / "source/src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copyfile(Path(__file__), run / "source/run_full_data.py")
    for name in ("pyproject.toml", "requirements.lock"):
        shutil.copyfile(root / name, run / "source" / name)
    (run / "source/git.diff").write_text(git("diff", "HEAD", "--", "src", "pyproject.toml", "requirements.lock"))
    plan["source_sha256"] = {str(p.relative_to(run / "source")): digest(p)
                              for p in sorted((run / "source").rglob("*")) if p.is_file()}
    shutil.copytree(old_model.parent, run / "reference_model")
    write_json(run / "plan.json", plan)
    write_json(record, plan)
    # Import the recorded source snapshot so concurrent source edits cannot alter this run.
    sys.path.insert(0, str(run / "source/src"))
    started = time.monotonic()
    try:
        manifest = prepare_full(root / "Dataset/ml-32m", reference, run / "data")
        plan.update(status="training", training_users=manifest["training_users"],
                    training_ratings=manifest["training_ratings"], dataset_sha256=digest(run / "data/manifest.json"))
        write_json(record, plan)
        from cinematch.training import train
        train(run / "data", run / "model", **parameters)
        comparison, rows, user_ids = compare(run / "reference_model/model.pt", run / "model/model.pt", run / "data")
        write_json(run / "comparison.json", comparison)
        np.savez_compressed(run / "validation_per_user.npz", user_ids=user_ids, **rows)
        plan.update(status="complete", finished_at=now(), elapsed_seconds=time.monotonic() - started,
                    checkpoint_sha256=digest(run / "model/model.pt"), comparison=comparison,
                    adoption_decision="Pending review; the historical model remains unchanged.")
        write_json(run / "result.json", plan)
        write_json(record, plan)
        print(json.dumps(comparison, indent=2), flush=True)
    except BaseException as error:
        plan.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}",
                    elapsed_seconds=time.monotonic() - started)
        write_json(run / "result.json", plan)
        write_json(record, plan)
        raise


if __name__ == "__main__":
    main()
