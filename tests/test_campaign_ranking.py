"""Correctness of sampled ranking training and inference for unseen users."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("campaign_ranking",
    Path(__file__).parents[1] / "experiments/campaign_ranking.py")
ranking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranking)


@pytest.fixture
def data():
    users, items, ratings = [], [], []
    for user in range(12):
        for item in range(user % 2, 8, 2):
            users.append(user)
            items.append(item)
            ratings.append(4.5)
        users.append(user)
        items.append(8)
        ratings.append(1.)
    return {"user_ids": np.arange(12) + 100,
            "movie_ids": np.arange(10) * 7 + 50,
            "user_index": np.asarray(users), "movie_index": np.asarray(items),
            "rating": np.asarray(ratings, dtype=np.float32),
            "item_counts": np.bincount(items, minlength=10)}


@pytest.fixture
def validation():
    return {"user_ids": np.array([200, 201]),
            "support_items": np.array([[0, 2], [1, 3]]),
            "support_ratings": np.full((2, 2), 4.5),
            "target_offsets": np.array([0, 2, 4]),
            "target_items": np.array([4, 6, 5, 7]),
            "target_ratings": np.array([5., 4., 5., 4.])}


def test_pair_sampler_excludes_every_observation_and_self_positive(data):
    index = ranking.prepare_sampling(data)
    sampled = ranking.sample_pairs(index, np.random.default_rng(7), 5000)
    users, anchor, positive, negative = sampled
    assert np.all(anchor != positive)
    observed = set(zip(data["user_index"], data["movie_index"]))
    liked = set(zip(data["user_index"][data["rating"] >= 4],
                    data["movie_index"][data["rating"] >= 4]))
    assert all((u, a) in liked and (u, p) in liked and (u, n) not in observed
               for u, a, p, n in zip(*sampled))
    assert not np.any(negative == 8), "Known dislikes are also excluded from unrated negatives"
    repeated = ranking.sample_pairs(index, np.random.default_rng(7), 5000)
    for a, b in zip(sampled, repeated):
        np.testing.assert_array_equal(a, b)


def test_cold_user_scores_use_movie_order_and_signed_ratings():
    model = {"context_factors": np.array([[1.], [-1.], [0.]]),
             "target_factors": np.array([[1.], [-1.], [0.]]),
             "bias": np.zeros(3), "item_counts": np.array([1, 3, 2]),
             "movie_ids": np.array([80, 10, 90])}
    liked = ranking.score_ranking(model, np.array([0]), np.array([5.]), {"aggregation": "signed"})
    disliked = ranking.score_ranking(model, np.array([0]), np.array([1.]), {"aggregation": "signed"})
    np.testing.assert_allclose(liked, [1, -1, 0])
    np.testing.assert_allclose(disliked, [-1, 1, 0])
    np.testing.assert_allclose(ranking.score_ranking(model, np.array([], dtype=int), np.array([])), np.log1p([1, 3, 2]))
    with pytest.raises(ValueError):
        ranking.score_ranking(model, [3], [4.])
    with pytest.raises(ValueError):
        ranking.score_ranking(model, [0, 0], [4., 4.])


def test_training_reload_and_validation_targets_never_enter_sampler(data, validation, tmp_path, monkeypatch):
    calls = []
    original = ranking.prepare_sampling

    def capture(arrays, threshold):
        calls.append(arrays)
        assert set(arrays["user_ids"]) == set(data["user_ids"])
        return original(arrays, threshold)

    monkeypatch.setattr(ranking, "prepare_sampling", capture)
    config = {"epochs": 3, "pairs_per_epoch": 128, "batch_size": 32,
              "factors": 3, "seed": 8, "threads": 1}
    model = ranking.train_ranking(data, config, tmp_path / "a", validation)
    assert calls == [data]
    assert set(model) >= {"movie_ids", "context_factors", "target_factors", "bias"}
    assert "user_ids" not in model
    np.testing.assert_array_equal(model["movie_ids"], data["movie_ids"])
    with np.load(tmp_path / "a/model.npz", allow_pickle=False) as saved:
        loaded = {key: saved[key] for key in saved.files}
    for items, ratings in zip(validation["support_items"], validation["support_ratings"]):
        expected = ranking.score_ranking(model, items, ratings)
        np.testing.assert_array_equal(ranking.score_ranking(loaded, items, ratings), expected)
        assert expected.shape == (10,) and np.isfinite(expected).all()
    second = ranking.train_ranking(data, config, tmp_path / "b", validation)
    for key in ["context_factors", "target_factors", "bias", "selected_epoch"]:
        np.testing.assert_array_equal(second[key], model[key])
    with pytest.raises(ValueError, match="already contains"):
        ranking.train_ranking(data, config, tmp_path / "a", validation)


def test_training_rejects_overlapping_validation_users(data, validation, tmp_path):
    validation = {**validation, "user_ids": np.array([100, 201])}
    with pytest.raises(ValueError, match="identities overlap"):
        ranking.train_ranking(data, {}, tmp_path / "bad", validation)


def test_sampler_rejects_duplicate_and_unsampleable_histories(data):
    duplicated = {**data, "user_index": np.r_[data["user_index"], data["user_index"][0]],
                  "movie_index": np.r_[data["movie_index"], data["movie_index"][0]],
                  "rating": np.r_[data["rating"], data["rating"][0]]}
    with pytest.raises(ValueError, match="Duplicate"):
        ranking.prepare_sampling(duplicated)
    with pytest.raises(ValueError, match="two liked movies"):
        ranking.prepare_sampling({**data, "rating": np.ones_like(data["rating"])})
