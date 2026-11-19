"""Synthetic final-evaluation guards; never opens the real reserved holdout."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("final_evaluation",
    Path(__file__).parents[1] / "experiments/final_evaluation.py")
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import campaign_common

    root = tmp_path / "repo"
    sample, full, candidate, campaign = "sample", "full", "candidate", "test-campaign"
    old, expanded = root / "artifacts/data", root / "artifacts/experiments/full/data"
    old.mkdir(parents=True)
    expanded.mkdir(parents=True)
    profiles = {"user_ids": np.array([101]), "support_items": np.array([[0]]),
        "support_ratings": np.array([[4.5]]), "target_offsets": np.array([0, 3]),
        "target_items": np.array([1, 2, 3]), "target_ratings": np.array([4.5, 2., 4.])}
    movie_ids = np.array([10, 20, 30, 40])
    for directory, users in ((old, [1, 2]), (expanded, [1, 2, 3])):
        np.savez_compressed(directory / "train.npz", movie_ids=movie_ids)
        np.savez_compressed(directory / "validation.npz", **profiles)
        (directory / "catalog.csv").write_text("movieId,title\n10,A\n20,B\n30,C\n40,D\n")
        if directory == expanded:
            np.savez_compressed(directory / "fresh_test.npz", **{**profiles, "user_ids": np.array([103])})
        names = ["train.npz", "validation.npz", "catalog.csv"]
        if directory == expanded:
            names.append("fresh_test.npz")
        final._write(directory / "manifest.json", {"sampled_train_user_ids": users,
            "sampled_validation_user_ids": [101], "sampled_test_user_ids": [102],
            "fresh_test_user_ids": [103], "sha256": {name: final._digest(directory / name) for name in names}})
    for name in ("src/cinematch/evaluation.py", "experiments/campaign_common.py", "experiments/final_evaluation.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic frozen source fixture.\n")
    for name in ("artifacts/mf16/model.pt", "artifacts/experiments/full/model/model.pt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint identity")
    models = {sample: np.array([4., 2., 3., 1.]), full: np.array([4., 3., 1., 2.]),
              candidate: np.array([4., 3., 2., 1.])}

    def predictor(unused_root, run_id):
        return lambda items, ratings: models[run_id].copy(), movie_ids.copy(), np.array([40, 30, 20, 10]), True

    runner = SimpleNamespace(predictor=predictor,
        location=lambda r, run: r / "artifacts/experiments" / run,
        scoring_matrix=lambda score, p: np.asarray([score(i, r) for i, r in zip(p["support_items"], p["support_ratings"])]))
    common = SimpleNamespace(SAMPLE=sample, FULL=full, CAMPAIGN=campaign,
        data_path=lambda r, scale: r / ("artifacts/data" if scale == "sample" else "artifacts/experiments/full/data"),
        benchmark_identity=lambda r: "validation-benchmark", validation_profiles=lambda r: profiles,
        evaluate_score_matrix=campaign_common.evaluate_score_matrix, comparisons=campaign_common.comparisons)
    monkeypatch.setattr(final, "_runtime", lambda: (runner, common))
    for run_id, status in ((sample, "historical_reference"), (full, "complete")):
        final._write(root / f"experiments/records/{run_id}.json", {"run_id": run_id, "status": status})
    run = runner.location(root, candidate)
    (run / "model").mkdir(parents=True)
    (run / "model/model.npz").write_bytes(b"synthetic model artifact")
    metrics, rows, tops = common.evaluate_score_matrix(models[candidate][None, :], profiles, True)
    np.savez_compressed(run / "validation_per_user.npz", user_ids=profiles["user_ids"], rows=rows, top_items=tops)
    manifest = final._read(expanded / "manifest.json")
    record = {"run_id": candidate, "status": "complete", "benchmark_id": "validation-benchmark",
        "configuration": {"kind": "neighbors", "scale": "full"}, "metrics": metrics,
        "training_data": {"manifest_sha256": final._digest(expanded / "manifest.json"),
                          "training_arrays_sha256": manifest["sha256"]["train.npz"]}}
    final._write(root / f"experiments/records/{candidate}.json", record)
    return root, candidate, runner, common, models


def test_freeze_hashes_fresh_without_loading_and_detects_mutation(fixture, monkeypatch):
    root, candidate, _, _, _ = fixture
    monkeypatch.setattr(final, "_load_fresh", lambda path: pytest.fail("Freeze must not open fresh profiles"))
    frozen = final.freeze_candidate(root, candidate, "Synthetic selection decided from validation.")
    assert frozen["status"] == "frozen" and not frozen["fresh_holdout_loaded"]
    assert frozen["training_isolation"]["training_overlap"] is False
    assert final.verify_frozen(root)["candidate_run"] == candidate
    with pytest.raises(FileExistsError):
        final.freeze_candidate(root, candidate, "Cannot replace an already frozen choice.")
    (root / "artifacts/experiments/candidate/model/model.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        final.verify_frozen(root)


def test_validation_parity_rejects_changed_scores_and_identities(fixture):
    root, candidate, runner, _, models = fixture
    final.validate_candidate_replay(root, candidate)
    models[candidate][1] += .2
    with pytest.raises(ValueError, match="metric parity"):
        final.validate_candidate_replay(root, candidate)
    models[candidate][1] -= .2
    original = runner.predictor
    runner.predictor = lambda root, run_id: (original(root, run_id)[0], np.array([20, 10, 30, 40]),
                                            original(root, run_id)[2], True)
    with pytest.raises(ValueError, match="movie identities/order"):
        final.validate_candidate_replay(root, candidate)


def test_one_shot_success_separate_record_and_repeated_attempt_refused(fixture, monkeypatch):
    root, candidate, _, _, _ = fixture
    final.freeze_candidate(root, candidate, "Synthetic validation choice.")
    actual_loader = final._load_fresh
    calls = []

    def checked_loader(path):
        assert (final.final_directory(root) / "final_started.json").exists()
        calls.append(path)
        return actual_loader(path)

    monkeypatch.setattr(final, "_load_fresh", checked_loader)
    monkeypatch.setattr(final, "__file__", str(final.final_directory(root) / "source/experiments/final_evaluation.py"))
    result = final.final_worker(root)
    assert result["status"] == "complete" and result["fresh_holdout_loaded"]
    assert len(calls) == 1
    assert set(result["metrics"]) == {"candidate", "original_sample", "full_user",
                                      "established_popularity", "full_popularity"}
    assert result["training_matched_popularity"] == "full_popularity"
    assert not result["validation_leaderboard_entry"]
    assert final._read(root / "experiments/final-results/test-campaign.json")["status"] == "complete"
    assert final._read(root / "experiments/records/candidate.json")["benchmark_id"] == "validation-benchmark"
    with pytest.raises(FileExistsError):
        final.final_worker(root)
    with pytest.raises(ValueError, match="already started"):
        final.evaluate_frozen(root)
    assert len(calls) == 1


def test_failure_consumes_attempt_without_opening_fresh(fixture, monkeypatch):
    root, candidate, _, _, _ = fixture
    final.freeze_candidate(root, candidate, "Synthetic validation choice.")
    (root / "artifacts/experiments/candidate/model/model.npz").write_bytes(b"mutated after freeze")
    monkeypatch.setattr(final, "_load_fresh", lambda path: pytest.fail("Changed inputs must fail before fresh access"))
    with pytest.raises(ValueError, match="Frozen input changed"):
        final.final_worker(root)
    status = final._read(final.final_directory(root) / "status.json")
    assert status["status"] == "failed" and not status["fresh_holdout_loaded"]
    assert status["retry_permitted"] is False
    with pytest.raises(FileExistsError):
        final.final_worker(root)


def test_manifest_reservations_reject_overlap_without_reading_fresh(fixture):
    root, _, _, common, _ = fixture
    path = common.data_path(root, "sample") / "manifest.json"
    manifest = final._read(path)
    manifest["sampled_train_user_ids"].append(103)
    final._write(path, manifest)
    with pytest.raises(ValueError, match="training manifest overlaps"):
        final._training_contract(root, [])
