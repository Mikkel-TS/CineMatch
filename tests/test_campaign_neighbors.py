"""Small, hand-checkable correctness cases, not ranking-quality experiments."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "campaign_neighbors", Path(__file__).parents[1] / "experiments/campaign_neighbors.py")
neighbors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neighbors)


@pytest.fixture
def data():
    # Positive sets: u0={0,1}, u1={0,1,2}, u2={0,2}, u3={1}.
    # u3 dislikes item3, so it has an observation but no positive interaction.
    return {
        "user_ids": np.arange(4), "movie_ids": np.array([40, 10, 30, 20]),
        "user_index": np.array([0, 0, 1, 1, 1, 2, 2, 3, 3]),
        "movie_index": np.array([0, 1, 0, 1, 2, 0, 2, 1, 3]),
        "rating": np.array([5., 4., 5., 4., 4.5, 4., 5., 4., 1.]),
        "item_counts": np.array([3, 3, 2, 1]),
    }


def dense(model):
    out = np.zeros((len(model["movie_ids"]), len(model["movie_ids"])))
    for row in range(len(out)):
        lo, hi = model["neighbor_indptr"][row:row + 2]
        out[row, model["neighbor_indices"][lo:hi]] = model["neighbor_weights"][lo:hi]
    return out


def test_cosine_cooccurrence_shrink_diagonal_and_catalogue_order(data, tmp_path):
    model = neighbors.train_neighbors(data, {"top_k": 3, "shrink": 2., "block_size": 1}, tmp_path)
    expected = np.array([
        [0, 2 / 3 * .5, 2 / np.sqrt(6) * .5, 0],
        [2 / 3 * .5, 0, 1 / np.sqrt(6) / 3, 0],
        [2 / np.sqrt(6) * .5, 1 / np.sqrt(6) / 3, 0, 0],
        [0, 0, 0, 0],
    ])
    np.testing.assert_allclose(dense(model), expected, rtol=1e-6)
    np.testing.assert_array_equal(model["movie_ids"], data["movie_ids"])
    np.testing.assert_array_equal(model["feedback_counts"], [3, 3, 2, 0])
    assert model["info"]["binary_interactions"] == 8


def test_conditional_direction_and_observed_feedback(data, tmp_path):
    model = neighbors.train_neighbors(data, {
        "top_k": 3, "shrink": 0, "similarity": "conditional", "feedback": "observed"
    }, tmp_path)
    weights = dense(model)
    assert weights[0, 2] == pytest.approx(2 / 3)
    assert weights[2, 0] == pytest.approx(1.)
    assert weights[1, 3] == pytest.approx(1 / 3)
    assert weights[3, 1] == pytest.approx(1.)
    assert not np.diag(weights).any()


def test_top_k_deterministic_ties_use_movie_ids_and_blocks_do_not_matter(data, tmp_path):
    data = {**data, "user_index": np.zeros(4, dtype=int), "movie_index": np.arange(4),
            "rating": np.full(4, 5.), "item_counts": np.ones(4, dtype=int)}
    first = neighbors.train_neighbors(data, {"top_k": 1, "shrink": 0, "block_size": 1}, tmp_path / "a")
    second = neighbors.train_neighbors(data, {"top_k": 1, "shrink": 0, "block_size": 3}, tmp_path / "b")
    # Equal similarities: choose lowest actual movie ID, excluding the row itself.
    np.testing.assert_array_equal(first["neighbor_indices"], [1, 3, 1, 1])
    for key in ("neighbor_indptr", "neighbor_indices", "neighbor_weights"):
        np.testing.assert_array_equal(first[key], second[key])


def test_signed_dislikes_finite_full_length_and_saved_reload(data, tmp_path):
    model = neighbors.train_neighbors(data, {"top_k": 3, "shrink": 0}, tmp_path)
    liked = neighbors.score_neighbors(model, [0], [5], {"aggregation": "signed"})
    disliked = neighbors.score_neighbors(model, [0], [1], {"aggregation": "signed"})
    np.testing.assert_allclose(disliked, -liked)
    assert liked.shape == (4,) and np.isfinite(liked).all()
    assert disliked[2] < disliked[3]
    np.testing.assert_array_equal(neighbors.score_neighbors(model, [], [], {}), data["item_counts"])
    loaded = neighbors.load_neighbors(tmp_path)
    np.testing.assert_array_equal(neighbors.score_neighbors(loaded, [0], [5], {}), liked)
    assert loaded["info"] == model["info"]
    with pytest.raises(ValueError, match="already exists"):
        neighbors.train_neighbors(data, {}, tmp_path)


def test_duplicate_training_pairs_are_binary(data, tmp_path):
    original = neighbors.train_neighbors(data, {"shrink": 0}, tmp_path / "original")
    repeated = dict(data)
    for key in ("user_index", "movie_index", "rating"):
        repeated[key] = np.concatenate([data[key], data[key][:1]])
    duplicate = neighbors.train_neighbors(repeated, {"shrink": 0}, tmp_path / "duplicate")
    np.testing.assert_array_equal(dense(original), dense(duplicate))


@pytest.mark.parametrize("items,ratings", [([4], [5]), ([0, 0], [5, 4]), ([0], [float("nan")]), ([.5], [4])])
def test_invalid_support_rejected(data, tmp_path, items, ratings):
    model = neighbors.train_neighbors(data, {}, tmp_path)
    with pytest.raises(ValueError):
        neighbors.score_neighbors(model, items, ratings, {})


def test_content_aligns_ids_normalizes_genres_and_dislikes(data, tmp_path):
    catalog = pd.DataFrame({"movieId": [10, 20, 40, 30],
                            "genres": ["Drama", "(no genres listed)", "Action", "Action|Drama"]})
    model = neighbors.train_content(catalog, data, {}, tmp_path)
    liked = neighbors.score_content(model, [0], [5], {})
    np.testing.assert_allclose(liked, [1., 0., 1 / np.sqrt(2), 0.], rtol=1e-6)
    np.testing.assert_allclose(neighbors.score_content(model, [0], [1], {}), -liked)
    assert liked.shape == (4,) and np.isfinite(liked).all()
    np.testing.assert_array_equal(neighbors.score_content(model, [3], [5], {}), data["item_counts"])
    loaded = neighbors.load_content(tmp_path / "model.npz")
    np.testing.assert_array_equal(neighbors.score_content(loaded, [0], [5], {}), liked)
    assert model["info"]["items_without_genres"] == 1


def test_content_rejects_missing_identity_and_unverified_genres(data, tmp_path):
    catalog = pd.DataFrame({"movieId": data["movie_ids"], "genres": ["Drama"] * 4})
    with pytest.raises(ValueError, match="unique metadata"):
        neighbors.train_content(catalog.iloc[:3], data, {}, tmp_path / "missing")
    catalog.loc[0, "genres"] = "Invented mood"
    with pytest.raises(ValueError, match="Unrecognised"):
        neighbors.train_content(catalog, data, {}, tmp_path / "invented")
