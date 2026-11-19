"""Cohort isolation, profile disclosure and identity checks for the app benchmark."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_app_comparison", Path(__file__).parents[1] / "experiments/prepare_app_comparison.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.fixture
def source():
    ids = np.array([109, 101, 111, 103, 107, 105, 102, 106, 104, 108, 110, 112])
    movies = np.arange(1000, 1042)[::-1]
    ratings = np.tile(np.r_[np.resize([4., 4.5, 5.], 35), [.5, 1., 1.5, 2., 2.5, 3., 3.5]], len(ids))
    return {"user_ids": ids, "movie_ids": movies,
            "user_index": np.repeat(np.arange(len(ids)), len(movies)),
            "movie_index": np.tile(np.arange(len(movies)), len(ids)),
            "rating": ratings.astype(np.float32), "item_counts": np.full(len(movies), len(ids))}


def prepared(source):
    # Discard one nonpositive movie, retaining enough ratings and positive seeds.
    return prep.prepare_arrays(source, np.arange(1001, 1042), np.array([901, 902, 903]),
                               validation_users=2, test_users=3)


def test_fresh_cohorts_are_deterministic_and_all_their_interactions_are_removed(source):
    train, cohorts, held, counts = prepared(source)
    expected = np.random.default_rng(20261119).permutation(np.sort(source["user_ids"]))
    np.testing.assert_array_equal(cohorts["validation_user_ids"], np.sort(expected[:2]))
    np.testing.assert_array_equal(cohorts["test_user_ids"], np.sort(expected[2:5]))
    excluded = set(cohorts["excluded_tag_user_ids"])
    assert excluded == set(expected[:5]) | {901, 902, 903}
    assert not excluded.intersection(train["user_ids"])
    assert set(held["user_id"]) == set(expected[:5])
    assert len(train["rating"]) == 7 * 41
    assert counts["common_training_rows_before_new_split"] == 12 * 41
    np.testing.assert_array_equal(train["movie_ids"], np.arange(1001, 1042))
    np.testing.assert_array_equal(np.bincount(train["movie_index"], minlength=41), train["item_counts"])
    assert np.all(train["item_counts"] == 7)
    original_pairs = {(int(source["user_ids"][u]), int(source["movie_ids"][m])): float(r)
                      for u, m, r in zip(source["user_index"], source["movie_index"], source["rating"])}
    for u, m, r in zip(train["user_index"], train["movie_index"], train["rating"]):
        assert original_pairs[int(train["user_ids"][u]), int(train["movie_ids"][m])] == r


def test_old_holdout_overlap_fails_closed(source):
    with pytest.raises(ValueError, match="old holdouts"):
        prep.prepare_arrays(source, np.sort(source["movie_ids"]), np.array([source["user_ids"][0]]), 2, 3)


def test_insufficient_eligible_users_and_nonascending_catalogue_fail(source):
    with pytest.raises(ValueError, match="Too few eligible"):
        prep.prepare_arrays(source, np.sort(source["movie_ids"]), np.array([]), 10, 10)
    with pytest.raises(ValueError, match="ascending"):
        prep.prepare_arrays(source, source["movie_ids"], np.array([]), 2, 3)


def test_app_proxy_preserves_great_fantastic_and_distinguishes_raw_labels():
    raw = np.arange(.5, 5.1, .5)
    np.testing.assert_array_equal(prep.app_verdicts(raw), [1, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    for bad in (np.array([np.nan]), np.array([0]), np.array([5.5])):
        with pytest.raises(ValueError, match="finite"):
            prep.app_verdicts(bad)


@pytest.mark.parametrize("mode,sizes", [("positive", (5, 10, 20)), ("mixed", (10, 20))])
def test_profiles_are_nested_and_hidden_targets_cover_every_remaining_rating(source, mode, sizes):
    _, cohorts, held, _ = prepared(source)
    profiles = [prep.make_profiles(held, cohorts["test_user_ids"], mode, size) for size in sizes]
    for a, b in zip(profiles, profiles[1:]):
        np.testing.assert_array_equal(a["support_items"], b["support_items"][:, :a["support_items"].shape[1]])
    for p in profiles:
        if mode == "positive":
            assert np.all(p["support_ratings"] >= 4)
        for i, uid in enumerate(p["user_ids"]):
            lo, hi = p["target_offsets"][i:i + 2]
            targets, labels = p["target_items"][lo:hi], p["target_ratings"][lo:hi]
            observed = held["user_id"] == uid
            raw = dict(zip(held["movie_index"][observed], held["rating"][observed]))
            seeds = p["support_items"][i]
            assert not set(seeds) & set(targets)
            assert set(seeds) | set(targets) == set(raw)
            np.testing.assert_array_equal(labels, [raw[j] for j in targets])
            np.testing.assert_array_equal(p["support_original_ratings"][i], [raw[j] for j in seeds])
            assert len(targets) == 41 - len(seeds)
            assert np.any(labels >= 4)


def test_fit10_uses_raw_support_and_at_most30_raw_targets(source):
    _, cohorts, held, _ = prepared(source)
    fit = prep.make_profiles(held, cohorts["validation_user_ids"], "fit", 10)
    mixed = prep.make_profiles(held, cohorts["validation_user_ids"], "mixed", 10)
    np.testing.assert_array_equal(fit["support_items"], mixed["support_items"])
    np.testing.assert_array_equal(fit["support_ratings"], fit["support_original_ratings"])
    assert np.all(np.diff(fit["target_offsets"]) == 30)
    assert np.any(fit["support_ratings"] % 1 != 0)


def test_profile_sampling_is_independent_of_input_row_order(source):
    _, cohorts, held, _ = prepared(source)
    reordered = {k: v.copy() for k, v in held.items()}
    for uid in cohorts["validation_user_ids"]:
        indexes = np.flatnonzero(held["user_id"] == uid)
        for k in held:
            reordered[k][indexes] = held[k][indexes[::-1]]
    a = prep.make_profiles(held, cohorts["validation_user_ids"], "positive", 20)
    b = prep.make_profiles(reordered, cohorts["validation_user_ids"], "positive", 20)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


@pytest.fixture
def identity_fixture():
    bundle = {"tmdbIdentityVerified": True,
              "movies": [{"eligible": True, "tmdbId": 768362, "movieLensId": 282843,
                          "title": "Missing", "year": 2023}],
              "legacyMappings": [{"letterboxdId": 1001148, "tmdbId": 768362},
                                 {"letterboxdId": 1020663, "tmdbId": 768362}]}
    meta = pd.DataFrame([
        {"id": 1001148, "name": "Missing", "minute": 111, "rating": 3.48, "description": "June searches for her mother."},
        {"id": 1020663, "name": "The Missing", "minute": 90, "rating": 3.71, "description": "An alien and a mouthless young man."},
    ])
    cache = {"responses": {"768362": {"movie": {"title": "Missing", "runtime": 111,
             "imdb_id": "tt10855768", "overview": "June searches for her mother."}}}}
    return bundle, meta, cache


def test_known_identity_collision_retains_correct_film_and_records_evidence(identity_fixture):
    rows, rejected = prep.common_catalogue(*identity_fixture)
    assert len(rows) == len(rejected) == 1
    assert rows[0]["letterboxdId"] == 1001148
    assert rejected[0]["letterboxdId"] == 1020663
    assert rejected[0]["verified_tmdb_imdb_id"] == "tt10855768"
    assert "mouthless" in rejected[0]["rejected_overview"]


def test_changed_identity_evidence_or_unknown_collision_fails(identity_fixture):
    bundle, meta, cache = identity_fixture
    cache["responses"]["768362"]["movie"]["runtime"] = 90
    with pytest.raises(ValueError, match="evidence has changed"):
        prep.common_catalogue(bundle, meta, cache)
    bundle["legacyMappings"] = [bundle["legacyMappings"][0]] * 2
    with pytest.raises(ValueError, match="Unresolved duplicate"):
        prep.common_catalogue(bundle, meta, cache)
