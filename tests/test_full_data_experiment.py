"""Protect benchmark identity and holdout isolation before expensive training."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cinematch.data import digest, load_arrays, prepare


spec = importlib.util.spec_from_file_location("full_data_experiment",
    Path(__file__).parents[1] / "experiments/run_full_data.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture
def sample(tmp_path):
    source, reference = tmp_path / "source", tmp_path / "reference"
    source.mkdir()
    users, movies = np.arange(40), np.arange(10, 170, 10)
    rows = [(int(u), int(m), 4.5 if u % 2 == j % 2 else 2.)
            for u in users for j, m in enumerate(movies)]
    unused = int(np.random.default_rng(42).permutation(users)[-1])
    rows.append((unused, 9999, 5.))
    pd.DataFrame(rows, columns=["userId", "movieId", "rating"]).to_csv(source / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [*movies, 9999], "title": [f"Film {m}" for m in [*movies, 9999]],
                  "genres": "Drama"}).to_csv(source / "movies.csv", index=False)
    prepare(source, reference, train_users=20, validation_users=5, test_users=5,
            min_item_ratings=1, seed=42)
    return source, reference


def test_full_training_freezes_benchmark_and_excludes_all_holdouts(sample, tmp_path):
    source, reference = sample
    output = tmp_path / "full"
    original = json.loads((reference / "manifest.json").read_text())
    manifest = experiment.prepare_full(source, reference, output, fresh_count=3, chunk_size=37)
    training, previous = load_arrays(output, "train"), load_arrays(reference, "train")
    np.testing.assert_array_equal(training["movie_ids"], previous["movie_ids"])
    assert 9999 not in training["movie_ids"]
    assert set(previous["user_ids"]) < set(training["user_ids"])
    held_out = set(original["sampled_validation_user_ids"] + original["sampled_test_user_ids"]
                   + manifest["fresh_test_user_ids"])
    assert not set(training["user_ids"]) & held_out
    assert not set(manifest["fresh_test_user_ids"]) & set(previous["user_ids"])
    assert len(training["user_ids"]) == 40 - 5 - 5 - 3
    for name in ("validation.npz", "test.npz"):
        assert digest(output / name) == digest(reference / name)
    fresh = load_arrays(output, "fresh_test")
    assert set(fresh["user_ids"]) <= set(manifest["fresh_test_user_ids"])
    raw = pd.read_csv(source / "ratings.csv")
    expected = raw.loc[raw.userId.isin(training["user_ids"]) & raw.movieId.isin(training["movie_ids"])]
    assert manifest["training_ratings"] == len(expected)
    counts = expected.groupby("movieId").size().reindex(training["movie_ids"])
    np.testing.assert_array_equal(training["item_counts"], counts)
    reconstructed = pd.DataFrame({"userId": training["user_ids"][training["user_index"]],
                                  "movieId": training["movie_ids"][training["movie_index"]],
                                  "rating": training["rating"]})
    pd.testing.assert_frame_equal(reconstructed.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)


def test_full_preparation_reproducible_across_chunk_sizes(sample, tmp_path):
    source, reference = sample
    a, b = tmp_path / "a", tmp_path / "b"
    experiment.prepare_full(source, reference, a, fresh_count=3, chunk_size=31)
    experiment.prepare_full(source, reference, b, fresh_count=3, chunk_size=103)
    for name in ("train", "validation", "test", "fresh_test"):
        x, y = load_arrays(a, name), load_arrays(b, name)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])


def test_full_preparation_rejects_duplicate_ratings(sample, tmp_path):
    source, reference = sample
    user = load_arrays(reference, "train")["user_ids"][0]
    frame = pd.read_csv(source / "ratings.csv")
    duplicate = frame.loc[frame.userId == user].iloc[:1]
    pd.concat([frame, duplicate]).to_csv(source / "ratings.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        experiment.prepare_full(source, reference, tmp_path / "full", fresh_count=3)
