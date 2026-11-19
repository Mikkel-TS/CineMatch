"""Hand-checked ranking and inference contracts for the direct comparison."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "app_comparison_metrics", Path(__file__).parents[1] / "experiments/app_comparison_metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


@pytest.fixture
def profiles():
    return {
        "user_ids": np.array([15, 30]),
        "support_items": np.array([[0, 1], [0, 1]]),
        "support_ratings": np.array([[1., 3.], [4., 5.]]),
        "target_offsets": np.array([0, 4, 6]),
        "target_items": np.array([2, 4, 8, 3, 3, 7]),
        "target_ratings": np.array([4., 5., 4., 1., 2., 3.]),
    }


def test_hand_calculated_ranking_exclusions_unknowns_and_empty_positive_user(profiles):
    # Highest-scoring low and neutral supports must still be excluded.
    scores = np.tile(np.arange(11, -1, -1, dtype=float), (2, 1))
    result, rows, top = metrics.evaluate(scores, profiles)
    np.testing.assert_array_equal(top, np.tile(np.arange(2, 12), (2, 1)))
    discounts = 1. / np.log2([2, 3, 4])
    assert rows["ndcg_at_5"][0] == pytest.approx((1. + .5) / discounts.sum())
    assert rows["recall_at_5"][0] == pytest.approx(2/3)
    assert rows["recall_at_10"][0] == 1.
    assert rows["precision_at_5"][0] == .4
    assert rows["top_1_known_positive"][0] == 1.
    assert rows["explicit_low_rate_at_5"][0] == .2
    assert np.isnan(rows["ndcg_at_5"][1])
    assert np.isnan(rows["recall_at_5"][1])
    assert rows["hit_rate_at_5"][1] == 0.
    assert result["hit_rate_at_5"] == .5
    assert result["precision_at_5"] == .2
    assert result["explicit_low_rate_at_5"] == .2
    assert result["users_with_relevant_targets"] == 1
    assert result["users_without_relevant_targets"] == 1
    assert result["distinct_recommended_movies"] == 5
    assert result["top5_concentration"] == 1.
    assert result["exposure_hhi_at_5"] == pytest.approx(.2)


def test_ties_use_catalogue_order_and_never_hidden_labels(profiles):
    scores = np.ones((2, 12))
    before = scores.copy()
    _, _, top = metrics.evaluate(scores, profiles)
    altered = dict(profiles, target_ratings=6 - profiles["target_ratings"])
    _, _, altered_top = metrics.evaluate(scores, altered)
    np.testing.assert_array_equal(top, np.tile(np.arange(2, 12), (2, 1)))
    np.testing.assert_array_equal(top, altered_top)
    np.testing.assert_array_equal(scores, before)


def test_short_lists_earn_no_padding_credit():
    profile = {
        "support_items": np.array([[0, 1]]), "support_ratings": np.array([[1., 5.]]),
        "target_offsets": np.array([0, 1]), "target_items": np.array([2]),
        "target_ratings": np.array([5.]),
    }
    result, rows, top = metrics.evaluate(np.ones((1, 3)), profile)
    np.testing.assert_array_equal(top, [[2]])
    assert result["precision_at_5"] == .2
    assert result["recall_at_5"] == 1.
    assert result["ndcg_at_5"] == 1.
    assert result["short_list_rate_at_5"] == 1.
    assert result["empty_list_rate"] == 0.
    assert rows["recommended_count_at_5"][0] == 1


def test_empty_candidate_list_and_no_positive_targets():
    profile = {
        "support_items": np.array([[0, 1]]), "support_ratings": np.ones((1, 2)),
        "target_offsets": np.array([0, 0]), "target_items": np.array([], dtype=int),
        "target_ratings": np.array([]),
    }
    result, _, top = metrics.evaluate(np.ones((1, 2)), profile)
    assert top.shape == (1, 0)
    assert result["ndcg_at_5"] is None
    assert result["precision_at_5"] == 0.
    assert result["empty_list_rate"] == 1.
    assert result["catalog_coverage_at_5"] == 0.


@pytest.mark.parametrize("change,match", [
    ({"support_items": np.array([[0, 0], [0, 1]])}, "only once"),
    ({"target_items": np.array([0, 4, 8, 3, 3, 7])}, "held-out targets"),
    ({"target_items": np.array([2, 2, 8, 3, 3, 7])}, "only once"),
    ({"target_items": np.array([12, 4, 8, 3, 3, 7])}, "inside the catalogue"),
    ({"target_offsets": np.array([0, 7, 6])}, "offsets"),
    ({"target_ratings": np.ones(6)*6}, "MovieLens"),
])
def test_invalid_profile_contracts_rejected(profiles, change, match):
    with pytest.raises(ValueError, match=match):
        metrics.evaluate(np.ones((2, 12)), dict(profiles, **change))


@pytest.mark.parametrize("value", [np.ones(12), np.full((2, 12), np.inf), np.full((2, 12), np.nan)])
def test_invalid_scores_rejected(profiles, value):
    with pytest.raises(ValueError, match="finite user-by-catalogue"):
        metrics.evaluate(value, profiles)


def test_paired_constant_gain_has_exact_interval_and_keeps_pairs():
    reference = np.array([0., .1, .3, np.nan, .6])
    candidate = reference + .2
    result = metrics.paired_bootstrap(candidate, reference)
    assert result["users"] == 4
    assert result["excluded_users"] == 1
    assert result["difference"] == pytest.approx(.2)
    assert result["confidence_interval"] == pytest.approx([.2, .2])


def test_pairing_identity_and_eligibility_are_enforced():
    a = {"user_ids": np.array([1, 2]), "ndcg_at_5": np.array([.2, .4])}
    b = {"user_ids": np.array([2, 1]), "ndcg_at_5": np.array([.1, .2])}
    with pytest.raises(ValueError, match="identical user IDs"):
        metrics.paired_bootstrap(a, b)
    with pytest.raises(ValueError, match="same non-evaluable"):
        metrics.paired_bootstrap([.2, .4], [.1, np.nan])


def test_bootstrap_is_reproducible_and_bonferroni_interval_is_wider():
    a, b = np.array([.1, .7, .2, .8, .9]), np.array([.2, .3, .4, .1, .5])
    standard = metrics.paired_bootstrap(a, b, batch_size=1)
    chunked = metrics.paired_bootstrap(a, b, batch_size=333)
    corrected = metrics.paired_bootstrap(a, b, confidence=1-.05/3)
    assert standard == chunked
    assert corrected["confidence_interval"][0] <= standard["confidence_interval"][0]
    assert corrected["confidence_interval"][1] >= standard["confidence_interval"][1]
    assert standard["difference"] == pytest.approx(.24)


def test_bootstrap_no_evaluable_users():
    result = metrics.paired_bootstrap(np.array([np.nan]), np.array([np.nan]))
    assert result["users"] == 0
    assert result["difference"] is None and result["confidence_interval"] is None
