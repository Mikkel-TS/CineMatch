#!/usr/bin/env python3
"""Verify the frozen model's unique MovieLens/TMDB links against live movie records.

Credentials are read from TMDB_READ_ACCESS_TOKEN/TMDB_API_KEY or an ignored
Secrets.plist. They are never written to logs, URLs in reports, or cache files.
Successful and definitive 404 responses are cached; transient failures are not
treated as unavailable films. Rerunning resumes the verification safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import plistlib
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELECTED_RUN = "cm-20261119-full-knn-conditional"
SELECTED_MODEL_SHA256 = "1b7a90f95036324bf87502032b662e846d955decee39d538bffe2eac4dbbf8ba"
MODEL_PATH = ROOT / "artifacts/experiments" / SELECTED_RUN / "model/model.npz"
POLICY = "tmdb-movie-matching-imdb-v2"
FINAL_STATUSES = {
    "verified_imdb", "excluded_missing_tmdb",
    "excluded_duplicate_tmdb", "excluded_identity_mismatch",
    "excluded_unavailable", "excluded_uncertain",
}
VERIFIED_STATUSES = {"verified_imdb"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def positive_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
        return result if result > 0 and str(result) == str(value) else None
    except (ValueError, TypeError):
        return None


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower().strip()
    value = re.sub(r",\s*(the|a|an)$", "", value)
    value = re.sub(r"^(the|a|an)\s+", "", value)
    return "".join(character for character in value if character.isalnum())


def split_title(value: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(.*) \((\d{4})\)", value)
    return (match[1], int(match[2])) if match else (value, None)


def movie_title_variants(value: str) -> set[str]:
    title, _ = split_title(value)
    parts = [title, re.sub(r"\s*\([^)]*\)", "", title)]
    parts.extend(re.findall(r"\((?:a\.k\.a\.\s*)?([^)]*)\)", title))
    return {normalize_title(part) for part in parts if normalize_title(part)}


def imdb_id(value: str | None) -> str | None:
    digits = (value or "").removeprefix("tt")
    if digits.isdigit() and int(digits) > 0:
        return "tt" + digits.zfill(7)
    return None


def load_catalogue(model_path: Path, data_dir: Path) -> tuple[list[dict], dict]:
    with np.load(model_path, allow_pickle=False) as archive:
        movie_ids = [int(value) for value in archive["movie_ids"]]
    if len(movie_ids) != len(set(movie_ids)):
        raise ValueError("Model MovieLens IDs must be unique.")
    movies_path, links_path = data_dir / "movies.csv", data_dir / "links.csv"
    movies = {int(row["movieId"]): row for row in read_csv(movies_path)}
    links = {int(row["movieId"]): row for row in read_csv(links_path)}
    rows = []
    grouped = defaultdict(list)
    for index, movie_id in enumerate(movie_ids):
        movie, link = movies[movie_id], links.get(movie_id, {})
        tmdb_id = positive_int(link.get("tmdbId"))
        row = {
            "modelIndex": index, "movieLensId": movie_id,
            "movieLensTitle": movie["title"], "tmdbId": tmdb_id,
            "expectedImdbId": imdb_id(link.get("imdbId")),
            "movieLensGenres": [genre for genre in movie["genres"].split("|")
                                if genre != "(no genres listed)"],
        }
        rows.append(row)
        if tmdb_id is not None:
            grouped[tmdb_id].append(row)
    for row in rows:
        if row["tmdbId"] is None:
            row["preflightStatus"] = "excluded_missing_tmdb"
        elif len(grouped[row["tmdbId"]]) > 1:
            row["preflightStatus"] = "excluded_duplicate_tmdb"
    sources = {
        "modelSHA256": sha256(model_path), "moviesSHA256": sha256(movies_path),
        "linksSHA256": sha256(links_path),
    }
    return rows, sources


def compact_movie(payload: dict) -> dict:
    """Retain identity evidence and display metadata, never request headers."""
    alternatives = payload.get("alternative_titles", {}).get("titles", [])
    return {
        "id": payload.get("id"), "imdb_id": payload.get("imdb_id"),
        "title": payload.get("title"), "original_title": payload.get("original_title"),
        "release_date": payload.get("release_date"),
        "alternative_titles": sorted({item["title"] for item in alternatives
                                      if isinstance(item, dict) and isinstance(item.get("title"), str)}),
        "genres": [item["name"] for item in payload.get("genres", [])
                   if isinstance(item, dict) and isinstance(item.get("name"), str)],
        "overview": payload.get("overview"), "runtime": payload.get("runtime"),
        "poster_path": payload.get("poster_path"), "adult": payload.get("adult"),
    }


def classify(row: dict, response: dict) -> dict:
    result = {**row, "verifiedAt": response.get("fetchedAt"),
              "httpStatus": response.get("httpStatus"),
              "apiResource": f"/3/movie/{row['tmdbId']}"}
    if response.get("kind") == "not_found":
        return {**result, "status": "excluded_unavailable", "reason": "TMDB movie endpoint returned 404."}
    if response.get("kind") != "movie":
        return {**result, "status": "retryable_error",
                "reason": response.get("errorCode", "request_failed")}
    movie = response.get("movie", {})
    result["tmdbMovie"] = movie
    if positive_int(movie.get("id")) != row["tmdbId"]:
        return {**result, "status": "excluded_identity_mismatch", "reason": "Returned TMDB ID differs."}
    if not isinstance(movie.get("title"), str) or not movie["title"].strip():
        return {**result, "status": "excluded_uncertain", "reason": "Movie record has no title."}
    expected, actual = row["expectedImdbId"], imdb_id(movie.get("imdb_id"))
    if expected and actual:
        if expected == actual:
            return {**result, "status": "verified_imdb", "reason": "TMDB movie and MovieLens link share IMDb ID."}
        return {**result, "status": "excluded_identity_mismatch", "reason": "IMDb identifiers disagree."}
    _, year = split_title(row["movieLensTitle"])
    release = movie.get("release_date") or ""
    tmdb_year = int(release[:4]) if re.match(r"^\d{4}-\d{2}-\d{2}$", release) else None
    names = {normalize_title(value) for value in
             [movie.get("title"), movie.get("original_title"), *movie.get("alternative_titles", [])]
             if isinstance(value, str) and value}
    title_year_agrees = bool(year is not None and year == tmdb_year and
                             movie_title_variants(row["movieLensTitle"]) & names)
    return {**result, "status": "excluded_uncertain", "titleYearAgrees": title_year_agrees,
            "reason": "Matching IMDb evidence is required for production; title/year evidence alone needs review."}


class CredentialError(RuntimeError):
    pass


def load_credential(plist_path: Path) -> str:
    value = os.environ.get("TMDB_READ_ACCESS_TOKEN") or os.environ.get("TMDB_API_KEY")
    if not value and plist_path.exists():
        with plist_path.open("rb") as stream:
            value = plistlib.load(stream).get("TMDB_READ_ACCESS_TOKEN")
    if not isinstance(value, str) or not value.strip() or value.strip().startswith("your-"):
        raise CredentialError("Configure TMDB_READ_ACCESS_TOKEN in the environment or ignored Secrets.plist.")
    return value.strip()


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1 / requests_per_second
        self.next_request = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        if wait:
            time.sleep(wait)


def retry_after(value: str | None, attempt: int) -> float:
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0.0, seconds)
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    return min(30.0, 2.0 ** attempt)


class TMDBClient:
    def __init__(self, credential: str, requests_per_second=20.0, retries=4, timeout=25.0):
        self.credential, self.retries, self.timeout = credential, retries, timeout
        self.rate = RateLimiter(requests_per_second)

    def fetch(self, tmdb_id: int) -> dict:
        query = {"language": "en-US", "append_to_response": "alternative_titles"}
        headers = {"Accept": "application/json", "User-Agent": "CineMatch-Catalogue-Verification/1"}
        if len(self.credential) == 32 and re.fullmatch(r"[a-fA-F0-9]+", self.credential):
            query["api_key"] = self.credential
        else:
            headers["Authorization"] = "Bearer " + self.credential
        request = Request(f"https://api.themoviedb.org/3/movie/{tmdb_id}?{urlencode(query)}", headers=headers)
        for attempt in range(self.retries + 1):
            self.rate.wait()
            pause, code, status = None, "request_failed", None
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                return {"kind": "movie", "fetchedAt": utc_now(), "httpStatus": 200,
                        "movie": compact_movie(payload)}
            except HTTPError as error:
                status = error.code
                if status in (401, 403):
                    raise CredentialError("TMDB rejected the configured credential; no credential value was logged.") from None
                if status == 404:
                    return {"kind": "not_found", "fetchedAt": utc_now(), "httpStatus": 404}
                code = f"http_{status}"
                if status not in (408, 425, 429) and status < 500:
                    return {"kind": "error", "fetchedAt": utc_now(), "httpStatus": status, "errorCode": code}
                pause = retry_after(error.headers.get("Retry-After"), attempt)
            except (URLError, TimeoutError, OSError):
                code = "network_error"
            except (ValueError, TypeError, AttributeError):
                code = "invalid_api_response"
            if attempt == self.retries:
                return {"kind": "error", "fetchedAt": utc_now(), "httpStatus": status, "errorCode": code}
            time.sleep(pause if pause is not None else retry_after(None, attempt))
        raise AssertionError("Unreachable retry loop")


def reusable(response: dict, max_age_days: float) -> bool:
    if response.get("kind") not in ("movie", "not_found"):
        return False
    try:
        checked = datetime.fromisoformat(response["fetchedAt"])
        age = datetime.now(timezone.utc) - checked
        return timedelta(0) <= age <= timedelta(days=max_age_days)
    except (KeyError, TypeError, ValueError):
        return False


def make_report(rows: list[dict], sources: dict, responses: dict, live_performed: bool,
                max_cache_age_days: float = 7.0) -> dict:
    results = []
    for row in rows:
        if "preflightStatus" in row:
            results.append({**row, "status": row["preflightStatus"]})
        else:
            response = responses.get(str(row["tmdbId"]), {"errorCode": "not_requested"})
            if response.get("kind") in ("movie", "not_found") and not reusable(response, max_cache_age_days):
                response = {"errorCode": "expired_or_invalid_cached_response"}
            results.append(classify(row, response))
    complete = all(row["status"] in FINAL_STATUSES for row in results)
    return {
        "schemaVersion": 1, "selectedModelRun": SELECTED_RUN, "policy": POLICY,
        "policyReason": "Production requires a successful TMDB movie record with matching IMDb identity; title/year agreement alone can leave editions or media identity ambiguous and remains excluded for review.",
        "generatedAt": utc_now(), "sources": sources,
        "liveLookupPerformed": live_performed,
        "complete": complete, "tmdbIdentityVerified": complete and live_performed,
        "counts": dict(sorted(Counter(row["status"] for row in results).items())),
        "eligibleCount": sum(row["status"] in VERIFIED_STATUSES for row in results),
        "catalogueCount": len(rows), "movies": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Dataset/ml-32m")
    parser.add_argument("--secrets-plist", type=Path, default=ROOT.parent / "peter/CineMatch/CineMatch/App/Secrets.plist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/integration")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=20.0)
    parser.add_argument("--max-cache-age-days", type=float, default=7.0)
    parser.add_argument("--probe", action="store_true", help="Check one pending record with a short timeout; leave an incomplete resumable report.")
    parser.add_argument("--cached-only", action="store_true", help="Reclassify existing fresh API evidence without credentials or network access.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 0 < args.requests_per_second <= 30 or args.max_cache_age_days < 0:
        parser.error("Use 1–16 workers, 0–30 requests/second, and a nonnegative cache age.")
    if sha256(args.model) != SELECTED_MODEL_SHA256:
        raise ValueError("This integration requires the exact frozen selected model artifact.")
    rows, sources = load_catalogue(args.model, args.data_dir)
    cache_path = args.output_dir / "tmdb_verification_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {"schemaVersion": 1, "provider": "TMDB movie API", "responses": {}}
    if cache.get("schemaVersion") != 1 or cache.get("provider") != "TMDB movie API":
        raise ValueError("Unsupported verification cache.")
    responses = cache["responses"]
    if args.cached_only:
        report = make_report(rows, sources, responses, live_performed=bool(responses),
                             max_cache_age_days=args.max_cache_age_days)
        atomic_json(args.output_dir / "tmdb_verification_report.json", report)
        print(json.dumps({"complete": report["complete"], "eligibleCount": report["eligibleCount"],
                          "counts": report["counts"]}), flush=True)
        return 0 if report["complete"] else 2
    credential = load_credential(args.secrets_plist)
    pending = [row["tmdbId"] for row in rows if "preflightStatus" not in row
               and not reusable(responses.get(str(row["tmdbId"]), {}), args.max_cache_age_days)]
    candidates = sum("preflightStatus" not in row for row in rows)
    print(f"Verifying {len(pending)} TMDB movie records; {candidates - len(pending)} responses reusable.", flush=True)
    client = TMDBClient(credential, requests_per_second=args.requests_per_second,
                        retries=0 if args.probe else 4, timeout=8.0 if args.probe else 25.0)
    # A response cache can be resumed, but credential failures stop the run immediately.
    try:
        # Probe before queueing thousands of requests: fail quickly on bad credentials/network.
        if pending:
            first = pending.pop(0)
            response = client.fetch(first)
            responses[str(first)] = response
            if response["kind"] == "error":
                raise RuntimeError(f"TMDB initial request failed: {response['errorCode']}")
        if args.probe:
            pending = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(client.fetch, tmdb_id): tmdb_id for tmdb_id in pending}
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    tmdb_id = futures[future]
                    responses[str(tmdb_id)] = future.result()
                    if completed % 100 == 0:
                        atomic_json(cache_path, cache)
                        print(f"Verified responses: {completed}/{len(pending)}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        atomic_json(cache_path, cache)
        report = make_report(rows, sources, responses, live_performed=True,
                             max_cache_age_days=args.max_cache_age_days)
        atomic_json(args.output_dir / "tmdb_verification_report.json", report)
    print(json.dumps({"complete": report["complete"], "eligibleCount": report["eligibleCount"], "counts": report["counts"]}), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CredentialError as error:
        print(str(error))
        raise SystemExit(3)
