"""Freeze one candidate, then consume the fresh holdout exactly once.

Run --freeze RUN_ID --reason TEXT first. Only --evaluate opens fresh profiles;
it executes the frozen source snapshot and never retrains or changes selection.
An interrupted/failed attempt retains its exclusive sentinel and cannot retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# In the worker these resolve inside the immutable final source snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


PROTOCOL = {
    "primary_metric": "ndcg_at_5",
    "references": ["original_sample", "full_user", "established_popularity", "full_popularity"],
    "candidates": "One frozen candidate; no selection or tuning after fresh evaluation.",
    "ranking": "Original full fixed catalogue, exclude onboarding movies, held-out rating >=4 is relevant.",
    "metrics": "Existing campaign_common.evaluate_score_matrix and comparisons; bootstrap seed42, 2000 replicates.",
    "parity": "Candidate completed validation metrics, per-user values and top lists must replay; then freeze all scores.",
    "parity_rtol": 1e-10, "parity_atol": 1e-10,
    "limits": "Retrospective random onboarding; unrated items have unknown relevance. Bootstrap covers users, not training randomness.",
    "failure_policy": "Exclusive sentinel is permanent, including failed/interrupted attempts. No automatic retries or replacement candidate.",
    "adoption": "Evaluation only. App defaults and existing models remain unchanged.",
}


def _runtime():
    import run_campaign
    import campaign_common
    return run_campaign, campaign_common


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def final_directory(root):
    _, common = _runtime()
    return Path(root) / "artifacts/experiments" / common.CAMPAIGN / "final-evaluation"


def _record_path(root):
    _, common = _runtime()
    return Path(root) / "experiments/final-results" / f"{common.CAMPAIGN}.json"


def _files(path):
    return [p for p in sorted(Path(path).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _dependency_inputs(root, run_id, seen=None):
    """Include every required model/config/source dependency, with cycle checks."""
    runner, common = _runtime()
    seen = set() if seen is None else set(seen)
    if run_id in seen:
        raise ValueError("Cyclic candidate dependency.")
    seen.add(run_id)
    record_path = root / f"experiments/records/{run_id}.json"
    record = _read(record_path)
    if record["status"] != "complete" and not (run_id == common.SAMPLE and record["status"] == "historical_reference"):
        raise ValueError(f"Candidate dependency is not complete: {run_id}")
    if run_id == common.SAMPLE:
        checkpoint = root / "artifacts/mf16/model.pt"
        if record.get("checkpoint_sha256") and _digest(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Original sample checkpoint differs from its historical record.")
        return {record_path, checkpoint}, [record]
    if run_id == common.FULL:
        directory = root / f"artifacts/experiments/{run_id}"
        paths = {record_path, directory / "model/model.pt"}
        if record.get("checkpoint_sha256") and _digest(directory / "model/model.pt") != record["checkpoint_sha256"]:
            raise ValueError("Full-user checkpoint differs from its completed record.")
        for name, expected in record.get("source_sha256", {}).items():
            if _digest(directory / "source" / name) != expected:
                raise ValueError(f"Full-user executed source changed: {name}")
        for name in ("source", "model"):
            paths.update(_files(directory / name))
        return paths, [record]
    directory = runner.location(root, run_id)
    paths, records = {record_path, *_files(directory)}, [record]
    # Verify recorded provenance rather than merely blessing changed artifacts.
    for name, expected in record.get("artifact_checksums", {}).items():
        if _digest(directory / name) != expected:
            raise ValueError(f"Recorded candidate artifact changed: {run_id}/{name}")
    for name, expected in record.get("source_sha256", {}).items():
        if _digest(directory / "source" / name) != expected:
            raise ValueError(f"Recorded source snapshot changed: {run_id}/{name}")
    for key in ("base_run", "source_run"):
        dependency = record["configuration"].get(key)
        if dependency:
            child_paths, child_records = _dependency_inputs(root, dependency, seen)
            paths.update(child_paths)
            records.extend(child_records)
    return paths, records


def _training_contract(root, records):
    """Inspect reservation identities/manifests, never fresh profile contents."""
    _, common = _runtime()
    sample_path = common.data_path(root, "sample")
    full_path = common.data_path(root, "full")
    sample, full = _read(sample_path / "manifest.json"), _read(full_path / "manifest.json")
    groups = [set(full[key]) for key in
              ("sampled_validation_user_ids", "sampled_test_user_ids", "fresh_test_user_ids")]
    if any(groups[a] & groups[b] for a in range(3) for b in range(a + 1, 3)):
        raise ValueError("Reserved evaluation identities overlap.")
    reserved = set.union(*groups)
    for label, manifest in (("sample", sample), ("full", full)):
        if reserved.intersection(manifest["sampled_train_user_ids"]):
            raise ValueError(f"{label} training manifest overlaps reserved evaluation identities.")
    for key in ("sampled_validation_user_ids", "sampled_test_user_ids"):
        if sample[key] != full[key]:
            raise ValueError("Original and expanded holdout identities differ.")
    for record in records:
        if "training_data" not in record:
            continue
        path = common.data_path(root, record["configuration"]["scale"])
        planned = record["training_data"]
        if _digest(path / "manifest.json") != planned["manifest_sha256"]:
            raise ValueError("Candidate training manifest differs from its recorded run.")
        if _digest(path / "train.npz") != planned["training_arrays_sha256"]:
            raise ValueError("Candidate training arrays differ from its recorded run.")
    paths = {sample_path / "manifest.json", full_path / "manifest.json",
             sample_path / "train.npz", full_path / "train.npz",
             sample_path / "validation.npz", full_path / "validation.npz",
             sample_path / "catalog.csv", full_path / "fresh_test.npz"}
    for directory, manifest, names in (
        (sample_path, sample, ["train.npz", "validation.npz", "catalog.csv"]),
        (full_path, full, ["train.npz", "validation.npz", "fresh_test.npz"])):
        for name in names:
            if _digest(directory / name) != manifest["sha256"][name]:
                raise ValueError(f"Prepared data checksum changed: {directory.name}/{name}")
    return paths, {"validation_reserved": len(groups[0]), "legacy_test_reserved": len(groups[1]),
                   "fresh_test_reserved": len(groups[2]), "training_overlap": False}


def _assert_metrics(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Validation metric fields differ from completed record.")
    for name, value in expected.items():
        if value is None:
            if actual[name] is not None:
                raise ValueError(f"Validation metric parity failed: {name}")
        elif actual[name] is None or not np.isclose(actual[name], value,
                rtol=PROTOCOL["parity_rtol"], atol=PROTOCOL["parity_atol"]):
            raise ValueError(f"Validation metric parity failed: {name}")


def validate_candidate_replay(root, run_id, frozen_replay=None):
    """Recompute validation only; fresh profiles are not referenced here."""
    runner, common = _runtime()
    record = _read(root / f"experiments/records/{run_id}.json")
    if record["benchmark_id"] != common.benchmark_identity(root):
        raise ValueError("Candidate validation benchmark differs.")
    profiles = common.validation_profiles(root)
    score, movie_ids, counts, rating_scores = runner.predictor(root, run_id)
    with np.load(common.data_path(root, "sample") / "train.npz", allow_pickle=False) as training:
        if not np.array_equal(movie_ids, training["movie_ids"]):
            raise ValueError("Candidate movie identities/order differ from the frozen catalogue.")
    scores = runner.scoring_matrix(score, profiles)
    if scores.shape != (len(profiles["user_ids"]), len(movie_ids)):
        raise ValueError("Candidate score-matrix shape differs from the catalogue.")
    metrics, rows, tops = common.evaluate_score_matrix(scores, profiles, rating_scores)
    _assert_metrics(metrics, record["metrics"])
    with np.load(runner.location(root, run_id) / "validation_per_user.npz", allow_pickle=False) as saved:
        if not np.array_equal(profiles["user_ids"], saved["user_ids"]):
            raise ValueError("Candidate validation user identities/order differ.")
        if not np.array_equal(tops, saved["top_items"]):
            raise ValueError("Candidate validation recommendations differ from its completed run.")
        if not np.allclose(rows, saved["rows"], rtol=PROTOCOL["parity_rtol"],
                           atol=PROTOCOL["parity_atol"], equal_nan=True):
            raise ValueError("Candidate validation per-user metrics differ from its completed run.")
    replay = {"scores": scores, "rows": rows, "top_items": tops,
              "user_ids": profiles["user_ids"], "movie_ids": movie_ids}
    if frozen_replay is not None:
        with np.load(frozen_replay, allow_pickle=False) as previous:
            for name, value in replay.items():
                if not np.allclose(value, previous[name], rtol=PROTOCOL["parity_rtol"],
                                   atol=PROTOCOL["parity_atol"], equal_nan=True):
                    raise ValueError(f"Frozen validation replay changed: {name}")
    return replay, metrics


def _snapshot(root, output):
    source = output / "source"
    shutil.copytree(root / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    (source / "experiments").mkdir()
    for path in sorted((root / "experiments").glob("*.py")):
        shutil.copy2(path, source / "experiments" / path.name)
    for name in ("pyproject.toml", "requirements.lock"):
        if (root / name).exists():
            shutil.copy2(root / name, source / name)
    return _files(source)


def freeze_candidate(root: Path, run_id: str, reason: str) -> dict:
    root = Path(root).resolve()
    if not reason.strip() or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("A valid run ID and explicit selection reason are required.")
    runner, common = _runtime()
    output = final_directory(root)
    output.mkdir(parents=True, exist_ok=False)
    selection = {"candidate_run": run_id, "selection_reason": reason, "selected_at": _now(),
                 "protocol": PROTOCOL, "campaign_id": common.CAMPAIGN, "fresh_holdout_loaded": False}
    _write(output / "selection.json", selection)
    status = {**selection, "status": "freezing", "artifact_directory": _relative(root, output)}
    _write(output / "status.json", status)
    _write(_record_path(root), status)
    try:
        paths, records = _dependency_inputs(root, run_id)
        candidate_record = _read(root / f"experiments/records/{run_id}.json")
        if run_id in (common.SAMPLE, common.FULL):
            raise ValueError("Freeze a completed campaign candidate with validation per-user artifacts.")
        for reference in (common.SAMPLE, common.FULL):
            more_paths, more_records = _dependency_inputs(root, reference)
            paths.update(more_paths)
            records.extend(more_records)
        contract_paths, isolation = _training_contract(root, records)
        paths.update(contract_paths)
        paths.update(_snapshot(root, output))
        paths.add(output / "selection.json")
        hashes_before = {_relative(root, p): _digest(p) for p in sorted(paths)}
        replay, metrics = validate_candidate_replay(root, run_id)
        np.savez_compressed(output / "validation_replay.npz", **replay)
        for name, checksum in hashes_before.items():
            if _digest(root / name) != checksum:
                raise ValueError(f"Input changed while freezing: {name}")
        hashes_before[_relative(root, output / "validation_replay.npz")] = _digest(output / "validation_replay.npz")
        fresh = common.data_path(root, "full") / "fresh_test.npz"
        benchmark = hashlib.sha256((_digest(fresh) + _digest(root / "artifacts/data/catalog.csv")
            + _digest(output / "source/src/cinematch/evaluation.py")
            + _digest(output / "source/experiments/campaign_common.py")).encode()).hexdigest()
        frozen = {**selection, "status": "frozen", "frozen_at": _now(), "input_sha256": hashes_before,
            "validation_benchmark_id": candidate_record["benchmark_id"], "final_benchmark_id": benchmark,
            "candidate_configuration": candidate_record["configuration"],
            "candidate_validation_metrics": metrics, "training_isolation": isolation,
            "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "validation_replay": _relative(root, output / "validation_replay.npz"),
            "fresh_profiles_path": _relative(root, fresh),
            "frozen_worker": _relative(root, output / "source/experiments/final_evaluation.py"),
            "interpretation": "One candidate selected using validation only. Fresh data hashed, not deserialized."}
        _write(output / "freeze.json", frozen)
        status = {**status, "status": "frozen", "frozen_at": frozen["frozen_at"],
                  "freeze_sha256": _digest(output / "freeze.json"), "final_benchmark_id": benchmark}
        _write(output / "status.json", status)
        _write(_record_path(root), status)
        return frozen
    except BaseException as error:
        status.update(status="freeze_interrupted" if isinstance(error, KeyboardInterrupt) else "freeze_failed",
                      finished_at=_now(), error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write(output / "status.json", status)
        _write(_record_path(root), status)
        raise


def verify_frozen(root: Path) -> dict:
    output = final_directory(root)
    status = _read(output / "status.json")
    if _digest(output / "freeze.json") != status["freeze_sha256"]:
        raise ValueError("Frozen evaluation plan changed.")
    frozen = _read(output / "freeze.json")
    for name, expected in frozen["input_sha256"].items():
        if _digest(Path(root) / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return frozen


def _claim_once(output):
    # This file is intentionally never removed, even if evaluation fails.
    with (Path(output) / "final_started.json").open("x") as stream:
        json.dump({"started_at": _now(), "pid": os.getpid(),
                   "policy": "One-shot attempt consumed; no retry or retuning."}, stream, indent=2)


def _load_fresh(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def final_worker(root: Path) -> dict:
    root = Path(root).resolve()
    output = final_directory(root)
    _claim_once(output)
    status = _read(output / "status.json")
    started = time.monotonic()
    status.update(status="running_final", started_at=_now(), fresh_holdout_loaded=False,
                  evidence_stage="fresh_final_holdout", validation_leaderboard_entry=False)
    _write(output / "status.json", status)
    _write(_record_path(root), status)
    try:
        frozen = verify_frozen(root)
        if Path(__file__).resolve() != (root / frozen["frozen_worker"]).resolve():
            raise ValueError("Final inference must execute the frozen source snapshot through --evaluate.")
        runner, common = _runtime()
        validate_candidate_replay(root, frozen["candidate_run"], root / frozen["validation_replay"])
        _, records = _dependency_inputs(root, frozen["candidate_run"])
        _training_contract(root, records)
        status["fresh_holdout_access_started"] = True
        _write(output / "status.json", status)
        _write(_record_path(root), status)
        profiles = _load_fresh(root / frozen["fresh_profiles_path"])
        # Persist this immediately so an interrupted run cannot appear untouched.
        status["fresh_holdout_loaded"] = True
        _write(output / "status.json", status)
        _write(_record_path(root), status)
        full = _read(common.data_path(root, "full") / "manifest.json")
        if len(np.unique(profiles["user_ids"])) != len(profiles["user_ids"]):
            raise ValueError("Fresh profile user identities are duplicated.")
        if not set(profiles["user_ids"]).issubset(set(full["fresh_test_user_ids"])):
            raise ValueError("Fresh profiles include users outside the reserved final holdout.")
        metrics, rows, tops, counts = {}, {}, {}, {}
        expected_ids = None
        for label, run_id in (("candidate", frozen["candidate_run"]),
                              ("original_sample", common.SAMPLE), ("full_user", common.FULL)):
            score, ids, count, rating_scores = runner.predictor(root, run_id)
            if expected_ids is None:
                expected_ids = ids
            elif not np.array_equal(ids, expected_ids):
                raise ValueError("Final model catalogue order differs.")
            scores = runner.scoring_matrix(score, profiles)
            if scores.shape != (len(profiles["user_ids"]), len(expected_ids)):
                raise ValueError("Final score-matrix shape differs from the frozen catalogue.")
            metrics[label], rows[label], tops[label] = common.evaluate_score_matrix(scores, profiles, rating_scores)
            counts[label] = count
        for name, source in (("established_popularity", "original_sample"), ("full_popularity", "full_user")):
            scores = np.broadcast_to(counts[source], (len(profiles["user_ids"]), len(expected_ids)))
            metrics[name], rows[name], tops[name] = common.evaluate_score_matrix(scores, profiles)
        references = {name: (metrics[name], rows[name]) for name in PROTOCOL["references"]}
        matched = "full_popularity" if frozen["candidate_configuration"]["scale"] == "full" else "established_popularity"
        references["training_matched_popularity"] = references[matched]
        compared = common.comparisons(metrics["candidate"], rows["candidate"], references)
        np.savez_compressed(output / "final_per_user.npz", user_ids=profiles["user_ids"],
            movie_ids=expected_ids, **rows, **{name + "_top_items": value for name, value in tops.items()})
        result = {**status, "status": "complete", "finished_at": _now(),
            "elapsed_seconds": time.monotonic() - started, "metrics": metrics, "comparisons": compared,
            "candidate_configuration": frozen["candidate_configuration"], "protocol": frozen["protocol"],
            "final_benchmark_id": frozen["final_benchmark_id"], "freeze_sha256": _digest(output / "freeze.json"),
            "training_matched_popularity": matched,
            "per_user_artifact": _relative(root, output / "final_per_user.npz"),
            "per_user_sha256": _digest(output / "final_per_user.npz"),
            "next_decision": "Report the frozen result. Do not tune or choose another model using this holdout.",
            "adoption_decision": "Not deployed; existing model unchanged."}
        _write(output / "result.json", result)
        _write(output / "status.json", result)
        _write(_record_path(root), result)
        return result
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      finished_at=_now(), elapsed_seconds=time.monotonic() - started,
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      retry_permitted=False)
        _write(output / "result.json", status)
        _write(output / "status.json", status)
        _write(_record_path(root), status)
        raise


def evaluate_frozen(root: Path) -> int:
    root = Path(root).resolve()
    output = final_directory(root)
    if (output / "final_started.json").exists() or (output / "final.log").exists():
        raise ValueError("Fresh final evaluation already started; retries and replacement candidates are prohibited.")
    frozen = verify_frozen(root)
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "VECLIB_MAXIMUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-B", "-u", str(root / frozen["frozen_worker"]),
               "--worker", "--root", str(root)]
    with (output / "final.log").open("x") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except BaseException as error:
            process.terminate()
            process.wait()
            status = _read(output / "status.json")
            status.update(status="interrupted", finished_at=_now(), retry_permitted=False,
                          error=f"Final evaluation parent interrupted: {type(error).__name__}: {error}")
            _write(output / "status.json", status)
            _write(output / "result.json", status)
            _write(_record_path(root), status)
            raise
    status = _read(output / "status.json")
    if status["status"] not in ("complete", "failed", "interrupted"):
        status.update(status="failed", finished_at=_now(), retry_permitted=False,
                      error=f"Frozen worker exited {code} before a final result; no retry permitted.")
        _write(output / "status.json", status)
        _write(output / "result.json", status)
        _write(_record_path(root), status)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", metavar="RUN_ID")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reason")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.freeze:
        if not args.reason:
            parser.error("--freeze requires --reason before opening the final holdout.")
        result = freeze_candidate(args.root, args.freeze, args.reason)
        print(f"Frozen {result['candidate_run']}; fresh holdout remains unopened.")
        return 0
    if args.worker:
        result = final_worker(args.root)
        print(json.dumps({"candidate_run": result["candidate_run"], "metrics": result["metrics"]}, indent=2))
        return 0
    return evaluate_frozen(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
