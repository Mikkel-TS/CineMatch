"""Boundary checks for validation-selected fusion of two recommendation ranks."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from app_comparison_metrics import evaluate
from run_app_ensemble import reciprocal_rank_scores, fuse_rank_scores, select_blend


def profile():
    return {
        "user_ids": np.array([71]),
        "support_items": np.array([[0, 4]]),
        "support_ratings": np.array([[4., 5.]]),
        "target_offsets": np.array([0, 4]),
        "target_items": np.array([1, 2, 5, 7]),
        "target_ratings": np.array([4., 1., 5., 2.]),
    }


def test_ties_share_average_rank_after_watched_movies_are_removed():
    scores = np.array([[100., 9., 9., 7., 7., 1.]])
    before = scores.copy()
    actual = reciprocal_rank_scores(scores, np.array([[0, 4]]))
    expected = np.array([[0., 1/61.5, 1/61.5, 1/63., 0., 1/64.]])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(scores, before)
    assert np.isfinite(actual).all()


def test_ranks_ignore_raw_score_scale_and_strict_monotonic_transform():
    scores = np.array([[3., 2., 2., -1., 0., 4.], [0., 4., -1., 4., 2., 3.]])
    supports = np.array([[0, 4], [1, 2]])
    original = reciprocal_rank_scores(scores, supports)
    transformed = reciprocal_rank_scores(np.exp(scores), supports)
    np.testing.assert_array_equal(original, transformed)


@pytest.mark.parametrize("weight,source", [(0., "conditional"), (1., "anton")])
def test_fusion_endpoints_preserve_standalone_rankings_including_ties(weight, source):
    p = profile()
    raw = {
        "conditional": np.array([[90., 9., 9., 7., 80., 3., 2., 1.]]),
        "anton": np.array([[40., 1., 5., 3., 50., 4., 5., 2.]]),
    }
    features = {name: reciprocal_rank_scores(scores, p["support_items"])
                for name, scores in raw.items()}
    blended = fuse_rank_scores(features["conditional"], features["anton"], weight)
    _, _, expected = evaluate(raw[source], p)
    _, _, actual = evaluate(blended, p)
    np.testing.assert_array_equal(actual, expected)
    assert not np.isin(actual, p["support_items"]).any()


def test_fusion_rankings_do_not_change_when_hidden_labels_change():
    p = profile()
    c = reciprocal_rank_scores(np.array([[10., 9., 8., 7., 6., 5., 4., 3.]]), p["support_items"])
    a = reciprocal_rank_scores(np.array([[1., 3., 5., 9., 2., 7., 8., 4.]]), p["support_items"])
    fused = fuse_rank_scores(c, a, .2)
    np.testing.assert_allclose(fused, .8*c + .2*a)
    before, _, top = evaluate(fused, p)
    changed = dict(p, target_ratings=6 - p["target_ratings"])
    after, _, changed_top = evaluate(fused, changed)
    np.testing.assert_array_equal(top, changed_top)
    assert before["ndcg_at_5"] != after["ndcg_at_5"]


def test_selection_uses_best_validation_ndcg_and_excludes_standalone_endpoints():
    result = select_blend({
        "0.0": {"ndcg_at_5": .9},
        "1.0": {"ndcg_at_5": 1.},
        "0.05": {"ndcg_at_5": .4},
        "0.1": {"ndcg_at_5": .5},
        "0.2": {"ndcg_at_5": .45},
    })
    assert result == .1


def test_selection_tie_prefers_less_anton_weight_independent_of_dict_order():
    candidates = {
        "0.4": {"ndcg_at_5": .5},
        "0.2": {"ndcg_at_5": .5},
        "0.1": {"ndcg_at_5": .4},
    }
    assert select_blend(candidates) == .2
    assert select_blend(dict(reversed(list(candidates.items())))) == .2


@pytest.mark.parametrize("weight", [-.001, 1.001, np.nan, np.inf])
def test_fusion_rejects_out_of_range_or_nonfinite_weight(weight):
    with pytest.raises(ValueError):
        fuse_rank_scores(np.ones((1, 4)), np.ones((1, 4)), weight)


def test_selection_requires_a_nonzero_blend_candidate():
    with pytest.raises(ValueError):
        select_blend({"0.0": {"ndcg_at_5": .5}, "1.0": {"ndcg_at_5": .4}})
