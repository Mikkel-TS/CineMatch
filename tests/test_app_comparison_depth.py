"""Boundary and normalization checks for the requested top-15 extension."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from evaluate_app_comparison_depth import evaluate_at_k


def profile(targets, ratings):
    return {
        "user_ids": np.array([101]),
        "support_items": np.array([[0]]),
        "support_ratings": np.array([[1.]]),
        "target_offsets": np.array([0, len(targets)]),
        "target_items": np.array(targets, dtype=np.int64),
        "target_ratings": np.array(ratings, dtype=float),
    }


def test_rank_fifteen_counts_but_rank_sixteen_does_not():
    # Excluding source movie0 means movie15 is exactly recommendation15.
    p = profile([15, 16, 7], [4., 5., 1.])
    result, rows, top = evaluate_at_k(np.arange(21, 0, -1, dtype=float)[None, :], p, 15)
    np.testing.assert_array_equal(top, [np.arange(1, 16)])
    assert result["ndcg_at_15"] == pytest.approx(.25 / (1 + 1/np.log2(3)))
    assert result["hit_rate_at_15"] == 1.
    assert result["recall_at_15"] == .5
    assert result["precision_at_15"] == pytest.approx(1/15)
    assert result["explicit_low_rate_at_15"] == pytest.approx(1/15)


def test_ideal_ranking_normalizes_to_one_with_fewer_than_fifteen_positives():
    p = profile([1, 2, 19], [4., 5., 2.])
    result, _, _ = evaluate_at_k(np.arange(21, 0, -1, dtype=float)[None, :], p, 15)
    assert result["ndcg_at_15"] == pytest.approx(1.)
    assert result["recall_at_15"] == 1.


def test_full_fifteen_positive_ideal():
    p = profile(np.arange(1, 17), np.full(16, 4.))
    result, _, _ = evaluate_at_k(np.arange(21, 0, -1, dtype=float)[None, :], p, 15)
    assert result["ndcg_at_15"] == pytest.approx(1.)
    assert result["precision_at_15"] == 1.
    assert result["recall_at_15"] == pytest.approx(15/16)


def test_ties_and_watched_exclusions_do_not_use_hidden_labels():
    p = profile([1, 2, 15], [4., 1., 5.])
    changed = dict(p, target_ratings=np.array([1., 5., 2.]))
    _, _, top = evaluate_at_k(np.ones((1, 21)), p, 15)
    _, _, other = evaluate_at_k(np.ones((1, 21)), changed, 15)
    np.testing.assert_array_equal(top, [np.arange(1, 16)])
    np.testing.assert_array_equal(top, other)


def test_zero_positive_user_has_no_ndcg_but_retains_hit_denominator():
    p = profile([1, 2], [2., 3.])
    result, rows, _ = evaluate_at_k(np.ones((1, 21)), p, 15)
    assert result["ndcg_at_15"] is None
    assert np.isnan(rows["ndcg_at_15"][0])
    assert result["hit_rate_at_15"] == 0.
    assert result["precision_at_15"] == 0.


def test_short_list_precision_keeps_requested_denominator():
    p = profile([1], [5.])
    result, _, top = evaluate_at_k(np.ones((1, 3)), p, 15)
    np.testing.assert_array_equal(top, [[1, 2]])
    assert result["ndcg_at_15"] == 1.
    assert result["precision_at_15"] == pytest.approx(1/15)

