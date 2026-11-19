"""Verification gates, identity evidence and export parity; no live API calls."""

import copy
import csv
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
verify = importlib.import_module("verify_tmdb_catalogue")
export = importlib.import_module("export_conditional_ios")


def row(movie_id=1, tmdb_id=101):
    return {"modelIndex": 0, "movieLensId": movie_id, "movieLensTitle": "Movie, The (2000)",
            "tmdbId": tmdb_id, "expectedImdbId": "tt0000010", "movieLensGenres": ["Drama"]}


def response(**changes):
    movie = {"id": 101, "imdb_id": "tt0000010", "title": "The Movie",
             "original_title": "The Movie", "release_date": "2000-01-01",
             "alternative_titles": [], "genres": ["Drama"]}
    movie.update(changes)
    return {"kind": "movie", "fetchedAt": verify.utc_now(), "httpStatus": 200, "movie": movie}


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_imdb_identity_wins_and_conflicting_imdb_is_never_title_matched():
    assert verify.classify(row(), response(title="A translated title"))["status"] == "verified_imdb"
    assert verify.classify(row(), response(imdb_id="tt0000099"))["status"] == "excluded_identity_mismatch"
    assert verify.classify(row(), response(id=999))["status"] == "excluded_identity_mismatch"


def test_title_year_evidence_does_not_replace_required_imdb_verification():
    result = verify.classify(row(), response(imdb_id=None))
    assert result["status"] == "excluded_uncertain" and result["titleYearAgrees"] is True
    alternate = response(imdb_id=None, title="Le film", original_title="Le film", alternative_titles=["The Movie"])
    result = verify.classify(row(), alternate)
    assert result["status"] == "excluded_uncertain" and result["titleYearAgrees"] is True
    assert verify.classify(row(), response(imdb_id=None, release_date="2001-01-01"))["status"] == "excluded_uncertain"
    assert verify.classify(row(), response(imdb_id=None, title="Another", original_title="Another"))["status"] == "excluded_uncertain"


def test_network_failure_and_missing_movie_are_distinct():
    missing = {"kind": "not_found", "fetchedAt": verify.utc_now(), "httpStatus": 404}
    temporary = {"kind": "error", "fetchedAt": verify.utc_now(), "httpStatus": 503, "errorCode": "http_503"}
    assert verify.classify(row(), missing)["status"] == "excluded_unavailable"
    assert verify.classify(row(), temporary)["status"] == "retryable_error"
    assert verify.reusable(missing, 7)
    assert not verify.reusable(temporary, 7)


def test_expired_cache_cannot_become_complete_on_probe_or_aborted_run():
    fresh, expired = response(), response()
    expired["fetchedAt"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    second = {**row(2, 102), "modelIndex": 1}
    report = verify.make_report([row(), second], {}, {"101": fresh, "102": expired}, True)
    assert report["complete"] is False and report["tmdbIdentityVerified"] is False
    assert report["movies"][1]["reason"] == "expired_or_invalid_cached_response"
    future = response()
    future["fetchedAt"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    assert not verify.reusable(future, 7)


def test_missing_and_duplicate_preflight_keeps_original_positions(tmp_path):
    np.savez(tmp_path / "model.npz", movie_ids=np.array([30, 10, 20, 40]))
    write_csv(tmp_path / "movies.csv", ["movieId", "title", "genres"],
              [{"movieId": i, "title": f"Film {i} (2000)", "genres": "Drama"} for i in [10, 20, 30, 40]])
    write_csv(tmp_path / "links.csv", ["movieId", "imdbId", "tmdbId"],
              [{"movieId": i, "imdbId": i, "tmdbId": t} for i, t in [(10, 101), (20, 101), (30, ""), (40, 104)]])
    rows, sources = verify.load_catalogue(tmp_path / "model.npz", tmp_path)
    assert [r["movieLensId"] for r in rows] == [30, 10, 20, 40]
    assert [r.get("preflightStatus") for r in rows] == ["excluded_missing_tmdb", "excluded_duplicate_tmdb", "excluded_duplicate_tmdb", None]
    assert len(sources["modelSHA256"]) == 64


def test_export_requires_complete_recent_hash_aligned_reverified_evidence():
    sources = {"modelSHA256": "abc", "moviesSHA256": "def", "linksSHA256": "ghi"}
    report = verify.make_report([row()], sources, {"101": response()}, True)
    assert export.validate_report(report, [row()], sources)[1]["status"] == "verified_imdb"
    for field in ("complete", "liveLookupPerformed", "tmdbIdentityVerified"):
        altered = {**report, field: False}
        with pytest.raises(ValueError, match="complete live"):
            export.validate_report(altered, [row()], sources)
    with pytest.raises(ValueError, match="checksums"):
        export.validate_report(report, [row()], {**sources, "modelSHA256": "other"})
    altered = copy.deepcopy(report)
    altered["movies"][0]["verifiedAt"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="stale"):
        export.validate_report(altered, [row()], sources)
    altered = copy.deepcopy(report)
    altered["movies"][0]["tmdbMovie"]["imdb_id"] = "tt9999999"
    with pytest.raises(ValueError, match="classification"):
        export.validate_report(altered, [row()], sources)


def test_http_retries_do_not_turn_failure_into_missing_movie(monkeypatch):
    calls = []
    def fail(request, timeout):
        calls.append(request)
        raise HTTPError("redacted", 503, "Unavailable", {}, None)
    monkeypatch.setattr(verify, "urlopen", fail)
    monkeypatch.setattr(verify.time, "sleep", lambda _: None)
    client = verify.TMDBClient("private-test-token", retries=2)
    result = client.fetch(101)
    assert len(calls) == 3 and result["kind"] == "error" and result["errorCode"] == "http_503"
    assert "private-test-token" not in json.dumps(result)
    def unauthorized(request, timeout):
        raise HTTPError("private-test-token", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(verify, "urlopen", unauthorized)
    with pytest.raises(verify.CredentialError) as caught:
        client.fetch(101)
    assert "private-test-token" not in str(caught.value)
    assert verify.retry_after("120", 0) == 120  # Never retry earlier than Retry-After.


def test_synthetic_reference_cases_have_hand_checked_scores_and_exclusions():
    bundle, fixtures = export.synthetic_artifacts()
    cases = {case["name"]: case["expected"] for case in fixtures["cases"]}
    direction = cases["direction_ties_zero_tail"]
    assert direction["results"] == [
        {"tmdbId": 101, "score": .5}, {"tmdbId": 102, "score": .5},
        {"tmdbId": 103, "score": 0}, {"tmdbId": 104, "score": 0}, {"tmdbId": 106, "score": 0}]
    assert cases["weighted_sum"]["results"][0] == {"tmdbId": 102, "score": 1.25}
    assert [r["tmdbId"] for r in cases["negative_only"]["results"]] == [101, 102, 104, 106]
    assert cases["negative_only"]["personalized"] is False
    assert cases["isolated_positive_fallback"]["personalized"] is False
    assert cases["ineligible_and_unknown_ignored"]["ratingsUsed"] == 0
    assert cases["exclude_positive_hits"]["results"][0]["score"] == 0
    assert cases["exclude_positive_hits"]["personalized"] is True
    assert bundle["movies"][5]["eligible"] is False


def test_export_does_not_compact_graph_or_round_saved_weights():
    source_bundle, _ = export.synthetic_artifacts()
    model = {"movie_ids": np.array([m["movieLensId"] for m in source_bundle["movies"]]),
             "counts": np.array([m["trainingCount"] for m in source_bundle["movies"]]),
             "neighbor_indptr": np.array(source_bundle["neighborOffsets"]),
             "neighbor_indices": np.array(source_bundle["neighborIndices"]),
             "neighbor_weights": np.array(source_bundle["neighborWeights"], dtype=np.float32),
             "info": {"similarity": "conditional", "feedback": "positive", "shrink": 10,
                      "top_k": 100, "config": {"aggregation": "positive"}}}
    catalogue = [{"movieLensId": m["movieLensId"], "movieLensGenres": m["genres"]} for m in source_bundle["movies"]]
    verified = {m["movieLensId"]: {"status": "verified_imdb" if m["eligible"] else "excluded_uncertain",
                                  "tmdbId": m["tmdbId"], "movieLensTitle": f"Film {m['movieLensId']} (2000)",
                                  "tmdbMovie": {"title": m["title"], "release_date": "2000-01-01"}}
                for m in source_bundle["movies"]}
    result = export.build_bundle(model, catalogue, verified, "a" * 64, "b" * 64, [])
    assert len(result["movies"]) == 7
    assert result["neighborOffsets"] == source_bundle["neighborOffsets"]
    assert result["neighborIndices"] == source_bundle["neighborIndices"]
    assert result["neighborWeights"] == source_bundle["neighborWeights"]
    assert result["movies"][5]["eligible"] is False


def test_legacy_bridge_never_accepts_year_offset_or_unconfirmed_title(tmp_path):
    films = [{"id": 1, "title": "The Movie", "year": 2000},
             {"id": 2, "title": "The Other", "year": 2001},
             {"id": 3, "title": "Wrong", "year": 1999}]
    (tmp_path / "frontend.json").write_text(json.dumps({"films": films}))
    write_csv(tmp_path / "mapping.csv", ["letterboxd_id", "movieId", "match_type"],
              [{"letterboxd_id": 1, "movieId": 10, "match_type": "exact_title_and_year"},
               {"letterboxd_id": 3, "movieId": 30, "match_type": "exact_title_year_offset"}])
    write_csv(tmp_path / "movies.csv", ["movieId", "title"],
              [{"movieId": 10, "title": "Movie, The (2000)"},
               {"movieId": 20, "title": "Other, The (2001)"},
               {"movieId": 30, "title": "Wrong (2000)"}])
    verified = {10: {"status": "verified_imdb", "tmdbId": 100, "tmdbMovie": response()["movie"]},
                20: {"status": "verified_imdb", "tmdbId": 200,
                     "tmdbMovie": response(title="The Other", release_date="2001-01-01")["movie"]},
                30: {"status": "verified_imdb", "tmdbId": 300,
                     "tmdbMovie": response(title="Wrong", release_date="1999-01-01")["movie"]}}
    mappings, report = export.legacy_bridge(tmp_path / "frontend.json", tmp_path / "mapping.csv",
                                           tmp_path / "movies.csv", verified)
    assert mappings == [{"letterboxdId": 1, "tmdbId": 100}, {"letterboxdId": 2, "tmdbId": 200}]
    assert report["counts"]["existing_mapping_requires_review"] == 1
