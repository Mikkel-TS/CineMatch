"""Meaningful serving parity, data-leakage and fitting-boundary checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


ROOT = Path(__file__).parents[1]
VENDOR = ROOT / "experiments/vendor/anton_6e85958"
spec = importlib.util.spec_from_file_location("anton_adapter", ROOT / "experiments/anton_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.fixture
def upstream():
    return adapter.load_upstream(VENDOR)


def toy_model(n=16):
    # Directed candidate->seed rows make a transpose error observable.
    rng = np.random.default_rng(42)
    features = {}
    for name in ("lb", "ml", "tag"):
        matrix = rng.random((n, n))
        matrix[matrix < .7] = 0
        np.fill_diagonal(matrix, 0)
        features[name] = sparse.csr_matrix(np.round(matrix, 4))
    return {"ids": np.arange(100, 100 + n), "titles": [str(i) for i in range(n)],
            "nbrs": features, "film_dev": np.linspace(-1.4, 1.4, n),
            "weights": np.array([.7, .3, 1.2, .2, -.1])}


def test_loading_upstream_does_not_replace_project_package(upstream):
    import cinematch
    original = sys.modules["cinematch"]
    adapter._UPSTREAM.clear()
    again = adapter.load_upstream(VENDOR)
    assert sys.modules["cinematch"] is original
    assert again.content is not original
    assert again.rating.cinematch is again.content


def test_native_score_parity_and_candidate_exclusion_below_ten(upstream):
    model = toy_model()
    items, stars = np.array([3, 0, 5]), np.array([1., 4., 2.5])
    cands, predicted = adapter.score_candidates(model, items, stars, vendor_dir=VENDOR)
    expected_candidates = np.array([i for i in range(16) if i not in items])
    centered, base = stars - stars.mean(), stars.mean()
    contributions = []
    for name in ("lb", "ml", "tag"):
        matrix = model["nbrs"][name][expected_candidates][:, items].toarray()
        contributions.append(matrix @ centered / (np.abs(matrix).sum(axis=1) + .5))
    columns = np.column_stack([contributions[0], contributions[1],
                               model["film_dev"][expected_candidates], contributions[2],
                               np.ones(len(expected_candidates))])
    expected = np.clip(base + columns @ model["weights"], 1, 4)
    np.testing.assert_array_equal(cands, expected_candidates)
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-14)
    direct = upstream.rating.score_candidates(model, {int(model["ids"][i]): float(r)
                                                      for i, r in zip(items, stars)})
    np.testing.assert_array_equal(predicted, direct[1])
    # The current upstream function does not apply the ten-rating metadata gate.
    mean_only = np.clip(base + model["film_dev"][cands] * model["weights"][2]
                        + model["weights"][-1], 1, 4)
    assert np.max(np.abs(predicted - mean_only)) > .01


def test_clipped_ties_keep_candidate_order(upstream):
    model = toy_model()
    model["weights"] = np.array([0., 0., 0., 0., 100.])
    cands, predicted = adapter.score_candidates(model, np.array([4, 2]), [1, 4], vendor_dir=VENDOR)
    assert np.all(predicted == 4)
    np.testing.assert_array_equal(cands[np.argsort(-predicted, kind="stable")][:5], [0, 1, 3, 5, 6])


def test_heldout_authors_removed_before_tag_vocabulary_and_idf(tmp_path):
    tags = pd.DataFrame({
        "userId": [1, 1, 1, 9, 9, 9],
        "movieId": [10, 20, 30, 10, 20, 30],
        "tag": ["ordinary", "ordinary", "third", "secret", "secret", "secret"],
    })
    path = tmp_path / "tags.csv"
    tags.to_csv(path, index=False)
    matrix, info = adapter.training_tag_features([30, 10, 20, 40], path, [9])
    assert info["excluded_heldout_rows_in_catalogue"] == 3
    assert info["retained_rows"] == 3
    assert info["features"] == 1  # held-out 'secret' never reaches the vocabulary
    np.testing.assert_array_equal(matrix.toarray(), [[0], [1], [1], [0]])


def test_validation_fit_matches_pinned_sparse_fit_and_saved_reader(upstream, tmp_path):
    components = toy_model()
    lists = {name: [[[int(j), float(v)] for j, v in enumerate(row) if v]
                    for row in matrix.toarray()] for name, matrix in components["nbrs"].items()}
    components["export_data"] = upstream.rating.make_json(
        lists, components["film_dev"], 2.5, components["titles"], components["ids"])
    # Fitting must use the same rounded film deviations that a reader serves.
    components["nbrs"], components["film_dev"] = upstream.rating.reader(components["export_data"])
    profiles = {"user_ids": np.array([71, 72]), "support_items": np.tile(np.arange(10), (2, 1)),
                "support_ratings": np.array([[1, 2, 3, 4, 1, 3, 4, 2, 3, 4],
                                              [4, 3, 2, 1, 4, 2, 1, 3, 2, 1]]),
                "target_offsets": np.array([0, 6, 12]),
                "target_items": np.tile(np.arange(10, 16), 2),
                "target_ratings": np.array([1, 2, 4, 3, 2, 4, 4, 3, 1, 2, 3, 1])}
    groups = [(int(profiles["user_ids"][i]), np.arange(16),
               np.r_[profiles["support_ratings"][i], profiles["target_ratings"][6*i:6*i+6]])
              for i in range(2)]
    native_rows = upstream.rating.build_rows(groups, 10, upstream.rating.app_source(components["nbrs"]),
                                              components["film_dev"])
    expected_weights = np.round(upstream.rating.fit_weights(native_rows, upstream.rating.NAMES_TAG), 4)
    fitted = adapter.fit_validation_weights(components, profiles, vendor_dir=VENDOR, output=tmp_path)
    np.testing.assert_array_equal(fitted["weights"], expected_weights)
    loaded = adapter.load_model(tmp_path, vendor_dir=VENDOR)
    for model in (fitted, loaded):
        cands, scores = adapter.score_candidates(model, np.array([0, 2]), [1., 4.], vendor_dir=VENDOR)
        np.testing.assert_array_equal(cands, np.array([i for i in range(16) if i not in (0, 2)]))
        if model is fitted:
            first = scores
        else:
            np.testing.assert_array_equal(scores, first)


@pytest.mark.parametrize("items,ratings", [([0, 0], [1, 4]), ([16], [3]), ([0], [5]), ([0], [np.nan])])
def test_invalid_profiles_rejected(items, ratings):
    with pytest.raises(ValueError):
        adapter.score_candidates(toy_model(), np.array(items), ratings, vendor_dir=VENDOR)
