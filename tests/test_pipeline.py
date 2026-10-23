from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cinematch.data import digest, load_arrays, prepare
from cinematch.evaluation import evaluate_profiles, top_items
from cinematch.model import fold_in
from cinematch.recommend import Recommender
from cinematch.training import evaluate_checkpoint, train


@pytest.fixture
def raw_data(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    users = np.arange(100, 140)
    movie_ids = np.arange(10, 170, 10)
    rows = []
    for user in users:
        for index, movie in enumerate(movie_ids):
            rating = 4.5 if (index % 2 == user % 2) else 2.
            rows.append((user, movie, rating))
    # This film is known only to a validation user and must not become a candidate.
    val_user = np.random.default_rng(123).permutation(users)[20]
    rows.append((val_user, 9999, 5.))
    pd.DataFrame(rows, columns=["userId", "movieId", "rating"]).to_csv(source / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [*movie_ids, 9999],
                  "title": [f"Film {i}" for i in [*movie_ids, 9999]],
                  "genres": "Drama"}).to_csv(source / "movies.csv", index=False)
    return source


@pytest.fixture
def prepared(raw_data, tmp_path):
    output = tmp_path / "prepared"
    prepare(raw_data, output, train_users=20, validation_users=8, test_users=8,
            support_count=5, min_item_ratings=1, seed=123, chunk_size=37)
    return output


def test_splits_and_training_only_catalogue(prepared, raw_data):
    training = load_arrays(prepared, "train")
    validation = load_arrays(prepared, "validation")
    test = load_arrays(prepared, "test")
    groups = [set(x["user_ids"]) for x in [training, validation, test]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert 9999 not in training["movie_ids"]
    original = pd.read_csv(raw_data / "ratings.csv")
    expected = original[original.userId.isin(training["user_ids"])].groupby("movieId").size()
    np.testing.assert_array_equal(training["item_counts"], expected.loc[training["movie_ids"]])
    for profiles in [validation, test]:
        assert profiles["support_items"].shape[1] == 5
        for index in range(len(profiles["user_ids"])):
            lo, hi = profiles["target_offsets"][index:index + 2]
            assert not set(profiles["support_items"][index]) & set(profiles["target_items"][lo:hi])


def test_preparation_reproducible_across_chunk_sizes(raw_data, prepared, tmp_path):
    other = tmp_path / "other"
    prepare(raw_data, other, train_users=20, validation_users=8, test_users=8,
            support_count=5, min_item_ratings=1, seed=123, chunk_size=103)
    for name in ["train", "validation", "test"]:
        first, second = load_arrays(prepared, name), load_arrays(other, name)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_rejects_changed_data_and_existing_outputs(prepared, raw_data):
    with pytest.raises(ValueError, match="already exists"):
        prepare(raw_data, prepared)
    with (prepared / "train.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_arrays(prepared, "train")


def test_fold_in_changes_taste_without_changing_movies():
    factors = np.array([[1.], [-1.], [0.]])
    bias = np.zeros(3)
    original = factors.copy()
    liked = fold_in(3., bias, factors, np.array([0]), np.array([5.]), .1, 100.)
    disliked = fold_in(3., bias, factors, np.array([0]), np.array([1.]), .1, 100.)
    assert liked[0] > liked[2] > liked[1]
    assert disliked[0] < disliked[2] < disliked[1]
    np.testing.assert_array_equal(factors, original)
    np.testing.assert_allclose(fold_in(3., bias, factors, np.array([], dtype=int), np.array([])), 3.)


@pytest.mark.parametrize("items,ratings", [([0, 0], [3, 4]), ([-1], [3]), ([3], [3]), ([0], [float('nan')])])
def test_fold_in_rejects_invalid_inputs(items, ratings):
    with pytest.raises(ValueError):
        fold_in(3., np.zeros(3), np.ones((3, 2)), np.array(items), np.array(ratings))


def test_ranking_metrics_and_watched_exclusion():
    profiles = {"user_ids": np.array([1]), "support_items": np.array([[0]]),
                "support_ratings": np.array([[3.]]), "target_offsets": np.array([0, 2]),
                "target_items": np.array([2, 3]), "target_ratings": np.array([4., 2.])}
    scores = np.array([5., 4., 3., 2., 1., 0.])
    metrics, _ = evaluate_profiles(0., scores, np.empty((6, 0)), profiles, 1., 5.)
    assert metrics["ndcg_at_5"] == pytest.approx(1 / np.log2(3))
    assert metrics["recall_at_5"] == 1.
    ranking_only, _ = evaluate_profiles(0., scores, np.empty((6, 0)), profiles, 1., 5., fixed_scores=scores)
    assert ranking_only["ndcg_at_5"] == metrics["ndcg_at_5"]
    assert ranking_only["rmse_macro"] is None
    assert 0 not in top_items(scores, np.array([0]))
    np.testing.assert_array_equal(top_items(np.ones(5), np.array([0]), 2), [1, 2])


def test_training_serving_and_explicit_test_evaluation(prepared, tmp_path, monkeypatch):
    output = tmp_path / "run"
    test_digest = digest(prepared / "test.npz")
    original_load = __import__("cinematch.training", fromlist=["load_arrays"]).load_arrays

    def guarded_load(directory, name):
        assert name != "test", "Training must never load the test profiles"
        return original_load(directory, name)

    with monkeypatch.context() as context:
        context.setattr("cinematch.training.load_arrays", guarded_load)
        result = train(prepared, output, factors=4, epochs=3, batch_size=64,
                       fold_in_grid=(1., 5.), threads=1, seed=7)
    assert result["test_evaluated"] is False
    assert digest(prepared / "test.npz") == test_digest
    model = Recommender(output / "model.pt")
    seeds = [(10, 4.5), (20, 2.), (30, 4.5), (40, 2.), (50, 4.5)]
    answer = model.recommend(seeds, exclude=(60,))
    assert answer["personalized"] is True
    assert len(answer["recommendations"]) == 5
    assert not {x["movieId"] for x in answer["recommendations"]} & {10, 20, 30, 40, 50, 60}
    assert answer == Recommender(output / "model.pt").recommend(seeds, exclude=(60,))
    assert model.recommend([])["personalized"] is False
    assert model.search("Film 10")[0]["movieId"] == 10
    with pytest.raises(ValueError, match="outside"):
        model.recommend([(9999, 4.)])
    with pytest.raises(ValueError, match="more than once"):
        model.recommend([(10, 4.), (10, 3.)])
    with pytest.raises(ValueError, match="half-star"):
        model.recommend([(10, 3.7)])
    report_path = output / "test.json"
    report = evaluate_checkpoint(output / "model.pt", prepared, report_path)
    assert report["model"]["users"] == 8
    assert report["paired_comparison"]["bootstrap_95_percent"] is not None
    assert report["popularity"]["rmse_macro"] is None
    assert report["paired_vs_model_prior"]["bootstrap_95_percent"] is not None
    with pytest.raises(ValueError, match="already exists"):
        evaluate_checkpoint(output / "model.pt", prepared, report_path)
