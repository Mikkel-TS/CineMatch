#!/usr/bin/env python3
"""Audit frontend/model identity coverage; emit unverified match candidates only.

Run from the ML repository with its existing environment:
    .venv/bin/python experiments/audit_frontend_identity.py

This does not migrate opinions, change accepted mappings, or modify model files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def normalize_title(value: str) -> str:
    """Case/accent/punctuation folding, removing English leading/trailing articles."""
    value = unicodedata.normalize("NFKD", value).lower()
    value = re.sub(r", (the|a|an)$", "", value)
    value = re.sub(r"^(the|a|an) ", "", value)
    return "".join(character for character in value if character.isalnum())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def positive_int(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
        return parsed if parsed > 0 else None
    except ValueError:
        return None


def main() -> None:
    ml_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ml_root.parent)
    parser.add_argument("--output-dir", type=Path, default=ml_root / "experiments/integration")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    paths = {
        "frontend_bundle": workspace / "peter/CineMatch/CineMatch/Core/Resources/cinematch_model.json",
        "existing_mapping": workspace / "extra_data/id_mapping.csv",
        "movielens_movies": ml_root / "Dataset/ml-32m/movies.csv",
        "movielens_links": ml_root / "Dataset/ml-32m/links.csv",
        "selected_model": ml_root / "artifacts/experiments/cm-20261119-full-knn-conditional/model/model.npz",
    }
    frontend = json.loads(paths["frontend_bundle"].read_text())
    films = {str(film["id"]): film for film in frontend["films"]}
    if len(films) != len(frontend["films"]):
        raise ValueError("Duplicate frontend source IDs require separate investigation.")
    with np.load(paths["selected_model"], allow_pickle=False) as model:
        selected = set(map(int, model["movie_ids"]))
    movies = read_csv(paths["movielens_movies"])
    movie_by_id = {int(movie["movieId"]): movie for movie in movies}
    links = {int(link["movieId"]): link for link in read_csv(paths["movielens_links"])}
    full_tmdb_groups = defaultdict(list)
    for movie_id in movie_by_id:
        tmdb_id = positive_int(links.get(movie_id, {}).get("tmdbId"))
        if tmdb_id is not None:
            full_tmdb_groups[tmdb_id].append(movie_id)
    full_linked_rows = sum(map(len, full_tmdb_groups.values()))
    mapping = defaultdict(list)
    for row in read_csv(paths["existing_mapping"]):
        mapping[row["letterboxd_id"]].append(row)

    mapped = set(films) & set(mapping)
    supported = {
        source_id for source_id in mapped
        if any(int(row["movieId"]) in selected for row in mapping[source_id])
    }
    unsupported = mapped - supported
    unmapped = set(films) - mapped
    unique_supported = {
        source_id for source_id in supported
        if len({row["movieId"] for row in mapping[source_id]}) == 1
    }

    # Uniqueness is checked against ALL MovieLens movies, before model membership.
    title_year_index = defaultdict(list)
    for movie in movies:
        match = re.fullmatch(r"(.*) \((\d{4})\)", movie["title"])
        if match:
            title_year_index[(normalize_title(match[1]), int(match[2]))].append(movie)
    candidate_rows = []
    outcomes = Counter()
    for source_id in sorted(unmapped, key=int):
        film = films[source_id]
        matches = title_year_index.get((normalize_title(film["title"]), film["year"]), [])
        if not matches:
            outcomes["no_normalized_title_year_match"] += 1
        elif len(matches) > 1:
            outcomes["ambiguous_across_full_movielens_catalogue"] += 1
        else:
            movie = matches[0]
            movie_id = int(movie["movieId"])
            if movie_id not in selected:
                outcomes["unique_match_outside_selected_model"] += 1
                continue
            outcomes["unique_match_inside_selected_model"] += 1
            link = links.get(movie_id, {})
            candidate_rows.append({
                "letterboxd_source_id": int(source_id),
                "frontend_title": film["title"],
                "frontend_year": film["year"],
                "movie_lens_id": movie_id,
                "movie_lens_title": movie["title"],
                "tmdb_id_from_movielens": positive_int(link.get("tmdbId")),
                "imdb_id_from_movielens": link.get("imdbId") or None,
                "match_method": "normalized_title_same_year_unique_in_full_movielens",
                "verification_status": "unverified",
                "accepted": False,
            })

    tmdb_groups = defaultdict(list)
    missing_tmdb = []
    for movie_id in sorted(selected):
        tmdb_id = positive_int(links.get(movie_id, {}).get("tmdbId"))
        row = {"movie_lens_id": movie_id, "title": movie_by_id[movie_id]["title"]}
        if tmdb_id is None:
            missing_tmdb.append(row)
        else:
            tmdb_groups[tmdb_id].append(row)
    duplicate_tmdb = [
        {"tmdb_id": tmdb_id, "model_movies": rows}
        for tmdb_id, rows in sorted(tmdb_groups.items()) if len(rows) > 1
    ]
    duplicate_rows = sum(len(group["model_movies"]) for group in duplicate_tmdb)
    summary = {
        "audit_type": "read_only_identity_coverage_and_unverified_match_candidates",
        "selected_model_run": "cm-20261119-full-knn-conditional",
        "live_tmdb_verification_performed": False,
        "tmdb_evidence": "Positive integer IDs in the local MovieLens links.csv; not live movie availability or identity verification.",
        "sources": {
            label: {"path": str(path.relative_to(workspace)), "sha256": digest(path)}
            for label, path in paths.items()
        },
        "script_sha256": digest(Path(__file__).resolve()),
        "normalization": {
            "steps": [
                "Unicode NFKD decomposition and lowercase",
                "Remove trailing ', the', ', a' or ', an'",
                "Remove leading 'the ', 'a ' or 'an '",
                "Retain only Unicode alphanumeric characters",
                "Require identical year parsed from final MovieLens '(YYYY)'",
            ],
            "uniqueness_scope": "All rows of MovieLens movies.csv, not just the selected model",
            "candidate_scope": "Only frontend IDs absent from the existing mapping, with a unique match in the selected model",
            "identity_guarantee": False,
            "automatic_acceptance": False,
        },
        "counts": {
            "frontend_films": len(films),
            "selected_model_movies": len(selected),
            "existing_mapping_any_movielens": len(mapped),
            "existing_mapping_inside_selected_model": len(supported),
            "existing_mapping_inside_selected_model_unique_target": len(unique_supported),
            "existing_mapping_outside_selected_model": len(unsupported),
            "absent_from_existing_mapping": len(unmapped),
            "additional_unverified_candidates_inside_selected_model": len(candidate_rows),
            "selected_model_missing_or_invalid_tmdb_rows": len(missing_tmdb),
            "selected_model_rows_with_valid_tmdb_link": len(selected) - len(missing_tmdb),
            "selected_model_distinct_tmdb_ids": len(tmdb_groups),
            "selected_model_duplicate_tmdb_groups": len(duplicate_tmdb),
            "selected_model_rows_in_duplicate_tmdb_groups": duplicate_rows,
            "selected_model_rows_with_valid_unique_tmdb": len(selected) - len(missing_tmdb) - duplicate_rows,
        },
        "full_movielens_tmdb_link_coverage": {
            "movielens_movies": len(movie_by_id),
            "rows_with_valid_tmdb_link": full_linked_rows,
            "rows_missing_or_invalid_tmdb_link": len(movie_by_id) - full_linked_rows,
            "distinct_tmdb_ids": len(full_tmdb_groups),
            "duplicate_tmdb_groups": sum(len(rows) > 1 for rows in full_tmdb_groups.values()),
            "rows_in_duplicate_tmdb_groups": sum(len(rows) for rows in full_tmdb_groups.values() if len(rows) > 1),
            "rows_with_valid_unique_tmdb": sum(len(rows) == 1 for rows in full_tmdb_groups.values()),
            "link_coverage_percent": 100 * full_linked_rows / len(movie_by_id),
        },
        "existing_supported_match_types": dict(sorted(Counter(
            row["match_type"] for source_id in supported for row in mapping[source_id]
            if int(row["movieId"]) in selected
        ).items())),
        "unmapped_title_year_diagnostic": dict(sorted(outcomes.items())),
        "selected_model_missing_tmdb": missing_tmdb,
        "selected_model_duplicate_tmdb": duplicate_tmdb,
        "limitations": [
            "Catalogue coverage does not measure recovery of actual users' saved opinions.",
            "Existing mapping entries and generated candidates are not verified against TMDB here.",
            "Normalized title/year matches can collapse different works; candidates require identity review before migration.",
            "Mapping a film outside the selected catalogue does not add model support for that film.",
            "Valid unique TMDB integers do not establish movie identity or movie-only eligibility.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "identity_audit_summary.json": summary,
        "identity_mapping_candidates.json": {
            "verification_status": "unverified",
            "accepted": False,
            "warning": "Review candidates only. Do not use as an accepted migration mapping.",
            "count": len(candidate_rows),
            "candidates": candidate_rows,
        },
    }
    for name, value in outputs.items():
        path = args.output_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        print(path)
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["unmapped_title_year_diagnostic"], indent=2))


if __name__ == "__main__":
    main()
