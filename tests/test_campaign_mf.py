"""Correctness checks for isolated MF experiments, not model-quality claims."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cinematch.model import fold_in
from cinematch.evaluation import evaluate_profiles


spec = importlib.util.spec_from_file_location("campaign_mf",
    Path(__file__).parents[1] / "experiments/campaign_mf.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def model():
    return {"mean": 3.4, "bias": np.array([.4, -.2, .1, -.5]),
        "factors": np.array([[1., .1], [.2, .8], [.8, -.2], [-.5, -.8]]),
        "movie_ids": np.array([10, 20, 40, 80]), "item_counts": np.array([100, 10, 40, 20])}


@pytest.fixture
def training():
    u, i = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    r = np.where(u % 2 == i % 2, 4.5, 1.5).astype(np.float32)
    data = {"user_ids": np.arange(10, 22), "movie_ids": np.arange(100, 220, 10),
            "user_index": u.ravel(), "movie_index": i.ravel(), "rating": r.ravel(),
            "item_counts": np.full(12, 12), "holdout_user_ids": np.array([101, 102, 103, 104])}
    seeds = np.array([[0, 2, 3], [1, 2, 3]])
    target = [np.setdiff1d(np.arange(12), row) for row in seeds]
    validation = {"user_ids": np.array([101, 102]), "support_items": seeds,
        "support_ratings": np.array([[4.5, 4.5, 1.5], [4.5, 1.5, 4.5]]),
        "target_items": np.concatenate(target), "target_offsets": np.array([0, 9, 18]),
        "target_ratings": np.concatenate([np.where(target[0] % 2 == 0, 4.5, 1.5),
                                          np.where(target[1] % 2 == 1, 4.5, 1.5)])}
    return data, validation


def test_default_scorer_is_original_fold_in_and_preserves_catalogue(model):
    items, ratings = np.array([0, 3]), np.array([4.5, 1.])
    expected = fold_in(model["mean"], model["bias"], model["factors"], items, ratings, 1., 5.)
    np.testing.assert_array_equal(campaign.scorer(model, items, ratings, {}), expected)
    np.testing.assert_array_equal(campaign.scorer(model, [], [], {}), model["mean"] + model["bias"])
    np.testing.assert_array_equal(model["movie_ids"], [10, 20, 40, 80])
    bad = {**model, "movie_ids": np.array([10, 20])}
    with pytest.raises(ValueError, match="Movie IDs"):
        campaign.scorer(bad, items, ratings, {})


def test_bias_transforms_apply_to_fit_and_prediction_and_centering_stays_rating_scale(model):
    items, ratings = [0, 3], [4.5, 1.]
    cfg = {"movie_bias_scale": .5, "bias_shrinkage_tau": 20., "fold_in_regularization": .1}
    transformed = .5 * model["bias"] * model["item_counts"] / (model["item_counts"] + 20.)
    expected = fold_in(model["mean"], transformed, model["factors"], items, ratings, .1, 5.)
    np.testing.assert_allclose(campaign.scorer(model, items, ratings, cfg), expected)
    centered = campaign.scorer(model, items, ratings, {"user_center": True})
    np.testing.assert_array_equal(centered,
        fold_in(np.mean(ratings), model["bias"], model["factors"], items, ratings, 1., 5.))
    np.testing.assert_array_equal(model["bias"], [.4, -.2, .1, -.5])


def test_no_intercept_solve_has_expected_linear_system_and_no_target_input(model):
    items, ratings = np.array([0, 3]), np.array([4.5, 1.])
    scores = campaign.scorer(model, items, ratings, {"fit_user_bias": False})
    x = model["factors"][items]
    coefficient = np.linalg.solve(x.T @ x + np.eye(2),
                                  x.T @ (ratings - model["mean"] - model["bias"][items]))
    np.testing.assert_allclose(scores, model["mean"] + model["bias"] + model["factors"] @ coefficient)
    with pytest.raises(TypeError):
        campaign.scorer(model, items, ratings, {}, target_ratings=np.array([5.]))


def test_centered_validation_retains_existing_metric_and_exclusion_rules(model):
    model = {**model, "mean": 3.5}
    profiles = {"user_ids": np.array([999]), "support_items": np.array([[0, 3]]),
        "support_ratings": np.array([[4.5, 2.5]]), "target_offsets": np.array([0, 2]),
        "target_items": np.array([1, 2]), "target_ratings": np.array([2., 4.5])}
    expected, _ = evaluate_profiles(3.5, model["bias"], model["factors"], profiles, 1., 5.)
    actual = campaign._validation_metrics(model, profiles,
                                           campaign._configuration({"user_center": True}))
    assert actual == expected


def test_training_rejects_holdout_overlap_and_movie_mapping_errors(training, tmp_path):
    data, validation = training
    overlapping = {**data, "user_ids": np.array([101, *data["user_ids"][1:]])}
    with pytest.raises(ValueError, match="overlap"):
        campaign.train_mf(overlapping, {}, tmp_path / "overlap", validation)
    reserved = {**data, "holdout_user_ids": np.array([data["user_ids"][0]])}
    with pytest.raises(ValueError, match="overlap"):
        campaign.train_mf(reserved, {}, tmp_path / "reserved", validation)
    for changed in ({"movie_ids": data["movie_ids"][::-1]}, {"item_counts": np.zeros(12, dtype=int)}):
        with pytest.raises(ValueError, match="Movie IDs|Item counts"):
            campaign.train_mf({**data, **changed}, {}, tmp_path / "bad_movies", validation)
    with pytest.raises(ValueError, match="movie-ID order"):
        campaign.train_mf(data, {}, tmp_path / "bad_validation",
                          {**validation, "movie_ids": data["movie_ids"][::-1]})
    assert not (tmp_path / "overlap").exists()


def test_training_learns_and_serialized_new_user_predictions_match(training, tmp_path):
    data, validation = training
    output = tmp_path / "fit"
    model = campaign.train_mf(data, {"factors": 4, "epochs": 8, "patience": 9,
        "learning_rate": .04, "batch_size": 24, "fold_in_grid": [.1, 1.],
        "bias_count_regularization": 10., "threads": 8}, output, validation)
    history = json.loads((output / "history.json").read_text())
    assert history["epochs"][-1]["training_rmse_during_updates"] < history["epochs"][0]["training_rmse_during_updates"]
    assert not history["test_evaluated"] and not history["fresh_test_evaluated"]
    assert model["config"]["threads"] == 2
    assert model["factors"].shape == (12, 4)
    np.testing.assert_array_equal(model["movie_ids"], data["movie_ids"])
    np.testing.assert_array_equal(model["item_counts"], data["item_counts"])
    restored = dict(np.load(output / "model.npz", allow_pickle=False))
    assert not any("user" in name for name in restored)
    saved_config = json.loads(str(restored["config_json"].item()))
    assert saved_config == model["config"]
    assert saved_config == json.loads((output / "fit_config.json").read_text())
    args = (validation["support_items"][0], validation["support_ratings"][0])
    scores = campaign.scorer(model, *args)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, campaign.scorer(restored, *args))
    with pytest.raises(ValueError, match="not empty"):
        campaign.train_mf(data, {}, output, validation)


def test_hidden_targets_do_not_enter_optimizer(training, tmp_path):
    data, validation = training
    cfg = {"factors": 2, "epochs": 1, "fold_in_grid": [1.], "loss": "huber",
           "train_movie_bias": False, "train_user_bias": False}
    first = campaign.train_mf(data, cfg, tmp_path / "first", validation)
    other_targets = {**validation, "target_ratings": np.full_like(validation["target_ratings"], 4.)}
    second = campaign.train_mf(data, cfg, tmp_path / "second", other_targets)
    for key in ("mean", "bias", "factors", "movie_ids", "item_counts"):
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.any(first["bias"])
