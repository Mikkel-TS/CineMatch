"""Benchmark-parity and holdout-boundary checks, separate from model experiments."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from cinematch.evaluation import evaluate_profiles, top_items
from cinematch.model import fold_in


EXPERIMENTS = Path(__file__).parents[1] / "experiments"
spec = importlib.util.spec_from_file_location("campaign_common", EXPERIMENTS / "campaign_common.py")
common = importlib.util.module_from_spec(spec)
sys.modules["campaign_common"] = common
spec.loader.exec_module(common)
spec = importlib.util.spec_from_file_location("campaign_runner_for_tests", EXPERIMENTS / "run_campaign.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def profiles():
    # User 20 deliberately has no relevant targets; all users still count for RMSE/coverage.
    return {
        "user_ids": np.array([10, 20, 30]),
        "support_items": np.array([[0, 6], [1, 7], [2, 4]]),
        "support_ratings": np.array([[.5, 5.], [3., 3.], [5., 1.]]),
        "target_offsets": np.array([0, 4, 7, 11]),
        "target_items": np.array([1, 2, 3, 4, 0, 2, 3, 0, 1, 5, 6]),
        "target_ratings": np.array([4., 5., 1., 3., 1., 2., 3., 5., 4., 2., 4.]),
    }


def assert_metrics_equal(candidate, original):
    for name, value in original.items():
        if value is None:
            assert candidate[name] is None
        else:
            assert candidate[name] == pytest.approx(value, abs=1e-14)


def test_rating_metric_parity_including_clipping_ties_exclusions_and_no_positive_user(profiles):
    mean, bias = 3., np.array([-7., 0., 0., 0., 4., 0., 0., 0.])
    factors = np.empty((8, 0))
    scores = np.asarray([fold_in(mean, bias, factors, items, ratings, 1., 5.)
                         for items, ratings in zip(profiles["support_items"], profiles["support_ratings"])])
    expected, expected_rows = evaluate_profiles(mean, bias, factors, profiles, 1., 5.)
    actual, actual_rows, tops = common.evaluate_score_matrix(scores, profiles, rating_scores=True)
    assert_metrics_equal(actual, expected)
    np.testing.assert_allclose(actual_rows, expected_rows, equal_nan=True, atol=1e-14)
    assert actual["users"] == 3 and actual["users_with_relevant_targets"] == 2
    assert np.isnan(actual_rows[1, 1:]).all() and np.isfinite(actual_rows[1, 0])
    assert actual["distinct_recommended_movies"] == len(np.unique(tops))
    for items, row, top in zip(profiles["support_items"], scores, tops):
        assert not set(items) & set(top)
        np.testing.assert_array_equal(top, top_items(row, items))
    # Equal scores use ascending catalogue position, and high-scoring seen items are excluded.
    np.testing.assert_array_equal(tops[0], [4, 1, 2, 3, 5])
    np.testing.assert_array_equal(tops[2], [1, 3, 5, 6, 7])


def test_ranking_only_metric_parity_and_no_rating_rmse(profiles):
    values = np.array([100., 100., 80., 60., 60., 20., 10., 5.])
    expected, expected_rows = evaluate_profiles(0., np.zeros(8), np.empty((8, 0)), profiles, 1., 5.,
                                                fixed_scores=values)
    actual, actual_rows, tops = common.evaluate_score_matrix(
        np.broadcast_to(values, (3, 8)), profiles, rating_scores=False)
    assert_metrics_equal(actual, expected)
    np.testing.assert_allclose(actual_rows, expected_rows, equal_nan=True)
    assert actual["rmse_macro"] is None and np.isnan(actual_rows[:, 0]).all()
    assert tops.shape == (3, 5)


def test_entire_cohort_without_positive_targets_keeps_coverage(profiles):
    profiles["target_ratings"] = np.full(11, 2.)
    values = np.ones(8)
    expected, _ = evaluate_profiles(0., values, np.empty((8, 0)), profiles, 1., 5., fixed_scores=values)
    actual, _, _ = common.evaluate_score_matrix(np.broadcast_to(values, (3, 8)), profiles)
    assert_metrics_equal(actual, expected)
    assert actual["users_with_relevant_targets"] == 0
    assert actual["ndcg_at_5"] is None and actual["catalog_coverage_at_5"] > 0


@pytest.mark.parametrize("scores", [np.ones(8), np.ones((2, 8)), np.full((3, 8), np.nan), np.full((3, 8), np.inf)])
def test_invalid_score_matrix_rejected(profiles, scores):
    with pytest.raises(ValueError, match="finite user-by-catalogue"):
        common.evaluate_score_matrix(scores, profiles)


@pytest.fixture
def training():
    return {
        "user_ids": np.array([101, 102]), "movie_ids": np.array([10, 30, 50]),
        "user_index": np.array([0, 0, 1, 1]), "movie_index": np.array([0, 1, 1, 2]),
        "rating": np.array([4., 5., 3., 2.]), "item_counts": np.array([1, 2, 1]),
    }


@pytest.fixture
def manifest():
    return {"sampled_validation_user_ids": [201], "sampled_test_user_ids": [301], "fresh_test_user_ids": [401]}


def test_training_contract_accepts_aligned_disjoint_data(training, manifest):
    common.check_training(training, manifest, np.array([10, 30, 50]))


@pytest.mark.parametrize("heldout_user", [201, 301, 401])
def test_every_reserved_holdout_identity_is_rejected(training, manifest, heldout_user):
    training["user_ids"][0] = heldout_user
    with pytest.raises(ValueError, match="reserved holdouts"):
        common.check_training(training, manifest, np.array([10, 30, 50]))


def test_same_movie_set_in_wrong_order_is_rejected(training, manifest):
    training["movie_ids"] = training["movie_ids"][::-1]
    with pytest.raises(ValueError, match="IDs/order"):
        common.check_training(training, manifest, np.array([10, 30, 50]))


def test_stale_popularity_counts_rejected(training, manifest):
    training["item_counts"] = np.array([2, 1, 1])
    with pytest.raises(ValueError, match="item counts"):
        common.check_training(training, manifest, np.array([10, 30, 50]))


@pytest.mark.parametrize("key,value", [("user_index", -1), ("user_index", 2), ("movie_index", 3)])
def test_out_of_range_training_mapping_rejected(training, manifest, key, value):
    training[key][0] = value
    with pytest.raises(ValueError, match=f"Invalid {key}"):
        common.check_training(training, manifest, np.array([10, 30, 50]))


class SupportOnly(dict):
    def __getitem__(self, key):
        if key not in ("support_items", "support_ratings"):
            raise AssertionError(f"Inference read forbidden profile field: {key}")
        return super().__getitem__(key)


def test_inference_and_confidence_blend_never_read_targets(profiles):
    counts = np.array([15, 8, 8, 5, 4, 3, 2, 1])
    calls = []

    def score(items, ratings):
        calls.append((items.copy(), ratings.copy()))
        personal = np.arange(8, dtype=float) * float(ratings.mean())
        return common.blend_scores(personal, counts, items, ratings,
                                   {"alpha": .5, "confidence": True, "confidence_strength": 5})

    guarded = SupportOnly(profiles)
    scores = runner.scoring_matrix(score, guarded)
    assert scores.shape == (3, 8) and np.isfinite(scores).all()
    assert len(calls) == 3
    for index, (items, ratings) in enumerate(calls):
        np.testing.assert_array_equal(items, profiles["support_items"][index])
        np.testing.assert_array_equal(ratings, profiles["support_ratings"][index])
    altered = SupportOnly({**profiles, "target_items": None, "target_ratings": object(), "target_offsets": "unreadable"})
    np.testing.assert_array_equal(runner.scoring_matrix(score, altered), scores)


def test_blend_endpoints_and_constant_scores_are_finite():
    personal, counts = np.arange(8, dtype=float), np.array([8, 7, 6, 5, 4, 3, 2, 1])
    np.testing.assert_allclose(common.blend_scores(personal, counts, [], [], {"alpha": 0}), common.zscore(np.log1p(counts)))
    np.testing.assert_allclose(common.blend_scores(personal, counts, [], [], {"alpha": 1}), common.zscore(personal))
    result = common.blend_scores(np.ones(8), np.ones(8), [], [], {"alpha": .5})
    np.testing.assert_array_equal(result, np.zeros(8))
    assert np.isfinite(result).all()


def test_candidate_reranking_preserves_full_catalogue_and_excludes_support():
    personal, counts = np.arange(8, dtype=float), np.arange(8, 0, -1)
    scores = common.blend_scores(personal, counts, np.array([0]), np.array([5.]), {"candidate_count": 3})
    assert scores.shape == (8,) and np.isfinite(scores).all()
    np.testing.assert_array_equal(scores[[1, 2, 3]], personal[[1, 2, 3]])
    assert max(scores[[0, 4, 5, 6, 7]]) < min(scores[[1, 2, 3]])
    np.testing.assert_array_equal(top_items(scores, np.array([0]), 3), [3, 2, 1])


@pytest.mark.parametrize("alpha", [-.01, 1.01, float("nan")])
def test_invalid_blend_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        common.blend_scores(np.ones(8), np.ones(8), [], [], {"alpha": alpha})
