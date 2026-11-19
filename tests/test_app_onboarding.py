"""Preserve raw observations and nested seeds when changing onboarding size."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from evaluate_app_onboarding import build_positive_profile, reconstruct_observations
from prepare_app_comparison import make_profiles


@pytest.fixture
def source():
    user_ids = np.array([101, 203])
    raw = np.r_[np.resize([4., 4.5, 5.], 25), np.arange(.5, 4., .5)]
    held = {"user_id": np.repeat(user_ids, len(raw)),
            "movie_index": np.tile(np.arange(len(raw)), len(user_ids)),
            "rating": np.tile(raw, len(user_ids)).astype(np.float32)}
    original = make_profiles(held, user_ids, "positive", 20)
    return held, original


def observations(held):
    return {(int(u), int(m)): float(r) for u, m, r in
            zip(held["user_id"], held["movie_index"], held["rating"])}


def test_reconstruction_preserves_every_raw_rating_including_half_stars(source):
    held, original = source
    before = {key: value.copy() for key, value in original.items()}
    reconstructed = reconstruct_observations(original)
    assert observations(reconstructed) == observations(held)
    assert 4.5 in reconstructed["rating"]
    assert 3.5 in reconstructed["rating"]
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])


def test_fifteen_opinions_keep_prefix_and_restore_last_five_as_raw_targets(source):
    held, original = source
    derived = build_positive_profile(original, 15)
    truth = observations(held)
    for key in ("support_items", "support_ratings", "support_original_ratings"):
        np.testing.assert_array_equal(derived[key], original[key][:, :15])
    for i, uid in enumerate(derived["user_ids"]):
        lo, hi = derived["target_offsets"][i:i + 2]
        targets = dict(zip(derived["target_items"][lo:hi], derived["target_ratings"][lo:hi]))
        seen = set(derived["support_items"][i])
        expected = {movie: value for (user, movie), value in truth.items()
                    if user == uid and movie not in seen}
        assert targets == expected
        assert set(original["support_items"][i, 15:20]) <= targets.keys()
        assert not seen.intersection(targets)
        assert len(targets) == 17


def test_original_five_inputs_and_all_remaining_targets_reproduce(source):
    held, original = source
    expected = make_profiles(held, original["user_ids"], "positive", 5)
    actual = build_positive_profile(original, 5)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_changed_sampling_seed_fails_instead_of_silently_changing_inputs(source):
    _, original = source
    with pytest.raises(ValueError, match="prefix changed"):
        build_positive_profile(original, 15, seed=42)


def test_overlapping_original_support_and_target_is_rejected(source):
    _, original = source
    damaged = {key: value.copy() for key, value in original.items()}
    damaged["target_items"][0] = damaged["support_items"][0, 0]
    with pytest.raises(ValueError, match="overlapping"):
        reconstruct_observations(damaged)


@pytest.mark.parametrize("size", [0, 21, 1.5])
def test_size_must_stay_inside_preserved_seed_prefix(source, size):
    _, original = source
    with pytest.raises(ValueError, match="prefix"):
        build_positive_profile(original, size)
