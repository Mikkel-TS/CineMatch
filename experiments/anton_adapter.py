"""Auditable adapter for Anton's pinned rating model on a shared catalogue.

The vendored implementation supplies content indexing, audience fitting,
neighbour selection, rounding, validation fitting and serving arithmetic.
This module changes data boundaries only: verified IDs, shared training rows,
and exclusion of held-out users' tags. It does not alter the serving policy.
In particular, upstream score_candidates does NOT enforce personalFromRatings.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse


UPSTREAM_COMMIT = "6e85958f503525b3020ec80fe613934e3c723d7b"
_UPSTREAM = {}


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def _temporary_module_names():
    """Avoid replacing the unrelated installed cinematch package globally."""
    names = ("cinematch", "movielens")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        yield
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def load_upstream(vendor_dir):
    vendor_dir = Path(vendor_dir).resolve()
    key = str(vendor_dir)
    if key in _UPSTREAM:
        return _UPSTREAM[key]

    def read(name):
        path = vendor_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_anton_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    with _temporary_module_names():
        audience = read("movielens")
        sys.modules["movielens"] = audience
        content = read("cinematch")
        sys.modules["cinematch"] = content
        rating = read("rating_model")
    result = SimpleNamespace(content=content, audience=audience, rating=rating)
    _UPSTREAM[key] = result
    return result


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_content_index(vendor_dir, data_dir, index_path):
    """Build/reuse the upstream full rated-metadata corpus, never a user holdout."""
    import joblib

    upstream = load_upstream(vendor_dir)
    data_dir, index_path = Path(data_dir).resolve(), Path(index_path).resolve()
    expected = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": file_sha256(Path(vendor_dir) / "cinematch.py"),
        "inputs": {name: file_sha256(data_dir / name) for name in (
            "movies.csv", "genres.csv", "themes.csv", "studios.csv", "countries.csv",
            "languages.csv", "actors.csv", "crew.csv", "releases.csv")},
    }
    manifest_path = index_path.with_suffix(".manifest.json")
    if index_path.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != expected:
            raise ValueError("Existing content index has different or missing provenance")
    else:
        upstream.content.DATA_DIR = data_dir
        upstream.content.INDEX_PATH = index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        upstream.content.build_index()
        _json(manifest_path, expected)
    index = joblib.load(index_path)
    if index.get("version") != upstream.content.INDEX_VERSION:
        raise ValueError("Upstream content index version differs")
    return upstream.content.Recommender(index)


def training_tag_features(movie_ids, tags_path, excluded_users):
    """Upstream TF-IDF tag features, excluding held-out authors before fitting."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    movie_ids = np.asarray(movie_ids, dtype=np.int64)
    excluded = np.asarray(sorted(set(int(x) for x in excluded_users)), dtype=np.int64)
    parts, input_rows, excluded_rows = [], 0, 0
    for chunk in pd.read_csv(tags_path, usecols=["userId", "movieId", "tag"], chunksize=250_000):
        input_rows += len(chunk)
        in_catalogue = chunk["movieId"].isin(movie_ids)
        is_excluded = chunk["userId"].isin(excluded)
        excluded_rows += int((in_catalogue & is_excluded).sum())
        part = chunk[in_catalogue & ~is_excluded].dropna(subset=["tag"])
        parts.append(part[["movieId", "tag"]])
    tags = pd.concat(parts, ignore_index=True)
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    docs = tags.groupby("movieId")["tag"].agg(lambda values: "|".join(values))
    vectorizer = TfidfVectorizer(tokenizer=lambda text: text.split("|"), token_pattern=None,
                                 lowercase=False, sublinear_tf=True, min_df=2)
    if len(docs):
        try:
            matrix = normalize(vectorizer.fit_transform(docs.to_numpy())).tocsr()
        except ValueError as error:
            if "empty vocabulary" not in str(error) and "no terms remain" not in str(error).lower():
                raise
            matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
    else:
        matrix = sparse.csr_matrix((0, 0), dtype=np.float64)
    row_of = pd.Series(np.arange(len(docs)), index=docs.index)
    positions = row_of.reindex(movie_ids).fillna(len(docs)).to_numpy(dtype=int)
    padded = sparse.vstack([matrix, sparse.csr_matrix((1, matrix.shape[1]))]).tocsr()
    info = {
        "input_rows": input_rows, "retained_rows": len(tags),
        "excluded_heldout_rows_in_catalogue": excluded_rows,
        "excluded_user_count": len(excluded),
        "excluded_user_ids_sha256": hashlib.sha256(excluded.tobytes()).hexdigest(),
        "films_with_tags": len(docs), "features": matrix.shape[1],
        "source_sha256": file_sha256(tags_path),
    }
    return padded[positions], info


def _check_training(training, catalogue):
    required = ("movieId", "letterboxdId", "tmdbId", "title")
    if not all(name in catalogue for name in required):
        raise ValueError(f"Catalogue requires {required}")
    if any(catalogue[name].duplicated().any() for name in required[:3]):
        raise ValueError("Catalogue movie identities must be one-to-one")
    ids = catalogue["movieId"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, training["movie_ids"]):
        raise ValueError("Training and catalogue order must match")
    users = np.asarray(training["user_index"])
    items = np.asarray(training["movie_index"])
    ratings = np.asarray(training["rating"])
    if (users.shape != items.shape or users.shape != ratings.shape or not len(users)
            or users.min() < 0 or users.max() >= len(training["user_ids"])
            or items.min() < 0 or items.max() >= len(ids)
            or not np.isfinite(ratings).all() or ratings.min() < .5 or ratings.max() > 5):
        raise ValueError("Invalid training interactions")
    return ids


def train_components(training, catalogue, *, vendor_dir, data_dir, index_path,
                     tags_path, excluded_tag_users, output):
    """Fit the pinned algorithm to shared rows; no validation/test labels read."""
    upstream = load_upstream(vendor_dir)
    ids = _check_training(training, catalogue)
    output = Path(output)
    if (output / "components.json").exists():
        raise ValueError("Anton components already exist; load them or use a new output")
    if set(map(int, training["user_ids"])) & set(map(int, excluded_tag_users)):
        raise ValueError("A held-out tag author is present among training users")
    output.mkdir(parents=True, exist_ok=True)
    print("Building Anton content index from full metadata corpus ...", flush=True)
    rec = build_content_index(vendor_dir, data_dir, index_path)
    lb_ids = catalogue["letterboxdId"].to_numpy(dtype=np.int64)
    if not np.isin(lb_ids, rec.row_of.index).all():
        raise ValueError("Verified catalogue film missing from Anton rated content index")
    content_features = rec.matrix[rec.row_of.loc[lb_ids].to_numpy()].tocsr()
    del rec
    print("Fitting Anton audience vectors on shared training rows ...", flush=True)
    ratings = sparse.csr_matrix(
        (np.asarray(training["rating"], dtype=np.float32),
         (training["user_index"], training["movie_index"])),
        shape=(len(training["user_ids"]), len(ids)), dtype=np.float32)
    audience = upstream.audience.MovieLensAudienceModel.fit(ratings, ids)
    audience_features = audience.features_for(ids).tocsr()
    np.savez_compressed(output / "audience_vectors.npz", movie_ids=audience.letterboxd_ids,
                        vectors=audience.vectors)
    del audience, ratings
    print("Fitting Anton tags without held-out authors ...", flush=True)
    tag_features, tag_info = training_tag_features(ids, tags_path, excluded_tag_users)

    values = upstream.rating.to_four_star(np.asarray(training["rating"], dtype=np.float64)).astype(np.float32)
    frame = pd.DataFrame({"film": ids[np.asarray(training["movie_index"])], "score": values})
    averages, gm = upstream.rating.film_averages(frame)
    film_dev = averages.reindex(ids).fillna(gm).to_numpy() - gm
    del frame, values
    features = {"lb": content_features, "ml": audience_features, "tag": tag_features}
    lists = {}
    for name, matrix in features.items():
        print(f"Selecting Anton top-{upstream.rating.NEIGHBORS} {name} neighbours ...", flush=True)
        lists[name] = upstream.rating.neighbour_lists(matrix)
    data = upstream.rating.make_json(lists, film_dev, gm, catalogue["title"].tolist(), ids)
    _json(output / "components.json", data)
    info = {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_files": {name: file_sha256(Path(vendor_dir) / name)
                           for name in ("cinematch.py", "movielens.py", "rating_model.py")},
        "training_rows": len(training["rating"]), "training_users": len(training["user_ids"]),
        "catalogue_items": len(ids), "audience_components": audience_features.shape[1],
        "tag_features": tag_info, "neighbours": upstream.rating.NEIGHBORS,
        "serving_policy": "Pinned score_candidates; no personalFromRatings gate",
        "identity_boundary": "MovieLens IDs; metadata joined through verified Letterboxd IDs",
        "components_sha256": file_sha256(output / "components.json"),
    }
    _json(output / "provenance.json", info)
    return load_components(output, vendor_dir=vendor_dir)


def load_components(output, *, vendor_dir):
    upstream = load_upstream(vendor_dir)
    data = json.loads((Path(output) / "components.json").read_text())
    nbrs, film_dev = upstream.rating.reader(data)
    return {
        "ids": np.asarray([f["id"] for f in data["films"]], dtype=np.int64),
        "titles": [f["title"] for f in data["films"]],
        "nbrs": nbrs, "film_dev": film_dev, "global_mean": data["globalMean"],
        "export_data": data,
    }


def _validate_profile(items, ratings, n_items):
    items, ratings = np.asarray(items), np.asarray(ratings, dtype=np.float64)
    if (items.ndim != 1 or ratings.shape != items.shape or not len(items)
            or not np.issubdtype(items.dtype, np.integer) or len(np.unique(items)) != len(items)
            or items.min() < 0 or items.max() >= n_items or not np.isfinite(ratings).all()
            or ratings.min() < 1 or ratings.max() > 4):
        raise ValueError("Profile requires unique catalogue indices and finite 1–4 ratings")
    return items, ratings


def fit_validation_weights(components, profiles, *, vendor_dir, output=None):
    """Fit one sparse model at ten ratings; target order fixes up-to-30 fitting rows."""
    upstream = load_upstream(vendor_dir)
    rows = {key: [] for key in ("user", "truth", "base", "lb", "ml", "tag", "film")}
    for i, user in enumerate(profiles["user_ids"]):
        prof, ratings = _validate_profile(profiles["support_items"][i], profiles["support_ratings"][i],
                                          len(components["ids"]))
        if len(prof) != upstream.rating.FIT_PROFILE:
            raise ValueError("Anton weights must be fitted at the pinned ten-rating profile size")
        start, stop = profiles["target_offsets"][i:i + 2]
        stop = min(stop, start + upstream.rating.HOLDOUT)
        targets, truth = _validate_profile(profiles["target_items"][start:stop],
                                           profiles["target_ratings"][start:stop], len(components["ids"]))
        if np.intersect1d(prof, targets).size:
            raise ValueError("Fitting targets overlap disclosed profile")
        base = ratings.mean()
        devs = upstream.rating.app_source(components["nbrs"])(prof, targets, ratings - base)
        rows["user"].append(np.full(len(targets), user))
        rows["truth"].append(truth)
        rows["base"].append(np.full(len(targets), base))
        rows["film"].append(components["film_dev"][targets])
        for name in ("lb", "ml", "tag"):
            rows[name].append(devs[name])
    if not rows["truth"]:
        raise ValueError("No validation fitting rows")
    columns = {key: np.concatenate(value) for key, value in rows.items()}
    weights = np.round(upstream.rating.fit_weights(columns, upstream.rating.NAMES_TAG), 4)
    if not np.isfinite(weights).all():
        raise ValueError("Validation fitting produced non-finite weights")
    result = {**components, "weights": weights,
              "fit_info": {"users": len(profiles["user_ids"]), "rows": len(columns["truth"]),
                           "profile_size": upstream.rating.FIT_PROFILE}}
    if output is not None:
        data = {**components["export_data"], "weights": upstream.rating.weights_dict(weights)}
        _json(Path(output) / "model.json", data)
        _json(Path(output) / "fit.json", {**result["fit_info"], "weights": data["weights"]})
    return result


def score_candidates(model, items, ratings_1_to_4, *, vendor_dir):
    """Call pinned serving code unchanged, including clipping and no gate."""
    items, ratings = _validate_profile(items, ratings_1_to_4, len(model["ids"]))
    opinions = {int(model["ids"][item]): float(rating) for item, rating in zip(items, ratings)}
    return load_upstream(vendor_dir).rating.score_candidates(model, opinions)


def load_model(output, *, vendor_dir):
    """Reload the exact serialized export with the pinned upstream reader."""
    return load_upstream(vendor_dir).rating.load_export(Path(output) / "model.json")
