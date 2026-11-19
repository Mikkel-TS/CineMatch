"""Sparse item-neighbour and MovieLens-genre experiments on a frozen catalogue.

Rows in the saved neighbour matrix mean seed item -> candidate item. Counts and
all returned scores retain the exact input movie_ids order. These are ranking
scores, not star predictions; clipping them or reporting rating RMSE is invalid.
Only supplied training arrays are read. This module never loads holdout files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse


MOVIELENS_GENRES = (
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


def _catalogue(data: dict) -> tuple[np.ndarray, np.ndarray]:
    movie_ids = np.asarray(data["movie_ids"], dtype=np.int64)
    counts = np.asarray(data["item_counts"], dtype=np.int64)
    if (movie_ids.ndim != 1 or not len(movie_ids) or counts.shape != movie_ids.shape
            or len(np.unique(movie_ids)) != len(movie_ids) or np.any(counts < 0)):
        raise ValueError("Unique movie IDs and nonnegative aligned item_counts are required.")
    return movie_ids, counts


def _save(model: dict, output: Path) -> None:
    output = Path(output)
    path = output / "model.npz"
    if path.exists():
        raise ValueError(f"Model already exists: {path}")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in model.items() if key != "info"}
    arrays["info_json"] = np.asarray(json.dumps(model["info"], sort_keys=True, allow_nan=False))
    np.savez_compressed(path, **arrays)


def _load(path: Path, kind: str) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "model.npz"
    with np.load(path, allow_pickle=False) as archive:
        model = {key: archive[key].copy() for key in archive.files if key != "info_json"}
        model["info"] = json.loads(str(archive["info_json"].item()))
    if model["info"].get("kind") != kind or model["info"].get("format_version") != 1:
        raise ValueError(f"Expected a version 1 {kind} model.")
    return model


def load_neighbors(path: Path) -> dict:
    return _load(path, "item_neighbors")


def load_content(path: Path) -> dict:
    return _load(path, "movielens_genres")


def train_neighbors(data: dict, config: dict, output: Path) -> dict:
    """Fit binary co-occurrence neighbours, retaining only each row's top K.

    feedback='positive' uses ratings >= positive_threshold (default 4);
    feedback='observed' uses every observed rating. Similarity is cosine,
    conditional P(candidate|seed), or asymmetric with seed exponent alpha.
    Every weight is then multiplied by cooccurrence/(cooccurrence + shrink).
    Products are computed in row blocks so a full item-item matrix is never
    materialised. Memory is O(training interactions + block_size*items + items*K).
    """
    movie_ids, counts = _catalogue(data)
    users = np.asarray(data["user_index"])
    items = np.asarray(data["movie_index"])
    ratings = np.asarray(data["rating"])
    n_items, n_users = len(movie_ids), len(data["user_ids"])
    if (users.ndim != 1 or items.shape != users.shape or ratings.shape != users.shape
            or not np.issubdtype(users.dtype, np.integer)
            or not np.issubdtype(items.dtype, np.integer)
            or not len(users) or users.min() < 0 or users.max() >= n_users
            or items.min() < 0 or items.max() >= n_items
            or not np.isfinite(ratings).all() or np.any(ratings < .5) or np.any(ratings > 5)):
        raise ValueError("Training indices and ratings must be finite, aligned and in range.")
    feedback = config.get("feedback", "positive")
    similarity = config.get("similarity", "cosine")
    top_k = config.get("top_k", 100)
    block_size = config.get("block_size", 128)
    shrink = float(config.get("shrink", 10.))
    threshold = float(config.get("positive_threshold", 4.))
    alpha = float(config.get("alpha", .5))
    if feedback not in ("positive", "observed"):
        raise ValueError("feedback must be positive or observed.")
    if similarity not in ("cosine", "conditional", "asymmetric"):
        raise ValueError("similarity must be cosine, conditional or asymmetric.")
    if (not isinstance(top_k, (int, np.integer)) or top_k < 1
            or not isinstance(block_size, (int, np.integer)) or block_size < 1
            or not np.isfinite([shrink, threshold, alpha]).all() or shrink < 0
            or not .5 <= threshold <= 5 or not 0 <= alpha <= 1):
        raise ValueError("Invalid top_k, block_size, shrink, positive_threshold or alpha.")
    if (Path(output) / "model.npz").exists():
        raise ValueError(f"Model already exists: {Path(output) / 'model.npz'}")
    selected = ratings >= threshold if feedback == "positive" else np.ones(len(ratings), dtype=bool)
    interaction = sparse.csr_matrix(
        (np.ones(int(selected.sum()), dtype=np.float32), (users[selected], items[selected])),
        shape=(n_users, n_items), dtype=np.float32,
    )
    # Duplicate rows, if supplied, must not make a user count as multiple votes.
    interaction.sum_duplicates()
    interaction.data.fill(1.)
    feedback_counts = np.asarray(interaction.sum(axis=0)).ravel().astype(np.int64)
    transposed = interaction.T.tocsr()
    indptr = [0]
    kept_indices, kept_weights = [], []
    # alpha=1 is conditional; alpha=.5 is cosine.
    exponent = 1. if similarity == "conditional" else .5 if similarity == "cosine" else alpha
    row_scale = feedback_counts.astype(np.float64) ** exponent
    col_scale = feedback_counts.astype(np.float64) ** (1. - exponent)
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        cooccurrence = (transposed[start:stop] @ interaction).tocsr()
        for offset, row in enumerate(range(start, stop)):
            lo, hi = cooccurrence.indptr[offset:offset + 2]
            columns = cooccurrence.indices[lo:hi]
            common = cooccurrence.data[lo:hi].astype(np.float64)
            keep = (columns != row) & (common > 0)
            columns, common = columns[keep], common[keep]
            denominator = row_scale[row] * col_scale[columns]
            weights = (common / denominator) * (common / (common + shrink))
            if len(weights) > top_k:
                cutoff = np.partition(weights, -top_k)[-top_k]
                above = np.flatnonzero(weights > cutoff)
                tied = np.flatnonzero(weights == cutoff)
                tied = tied[np.argsort(movie_ids[columns[tied]], kind="stable")[:top_k - len(above)]]
                selected_columns = np.concatenate((above, tied))
                columns, weights = columns[selected_columns], weights[selected_columns]
            order = np.lexsort((movie_ids[columns], -weights))
            kept_indices.append(columns[order].astype(np.int32))
            kept_weights.append(weights[order].astype(np.float32))
            indptr.append(indptr[-1] + len(columns))
    model = {
        "movie_ids": movie_ids.copy(), "counts": counts.copy(), "feedback_counts": feedback_counts,
        "neighbor_indptr": np.asarray(indptr, dtype=np.int64),
        "neighbor_indices": np.concatenate(kept_indices),
        "neighbor_weights": np.concatenate(kept_weights),
        "info": {
            "format_version": 1, "kind": "item_neighbors", "config": dict(config),
            "feedback": feedback, "positive_threshold": threshold, "similarity": similarity,
            "alpha": exponent, "shrink": shrink, "top_k": int(top_k), "block_size": int(block_size),
            "training_rows": len(ratings), "training_users": n_users, "catalogue_items": n_items,
            "binary_interactions": int(interaction.nnz), "retained_edges": int(indptr[-1]),
            "score_meaning": "Weighted sum of seed-to-candidate similarities; not a rating prediction.",
            "no_signal_fallback": "Training rating count, with exclusions applied by the evaluator.",
        },
    }
    model["info"]["array_bytes"] = sum(v.nbytes for k, v in model.items() if k != "info")
    _save(model, output)
    return model


def _seed_weights(model: dict, items, ratings, config: dict, default: str) -> tuple[np.ndarray, np.ndarray]:
    raw_items = np.asarray(items)
    ratings = np.asarray(ratings, dtype=np.float64)
    if (raw_items.ndim != 1 or ratings.shape != raw_items.shape
            or (len(raw_items) and not np.issubdtype(raw_items.dtype, np.integer))
            or not np.isfinite(ratings).all() or np.any(ratings < .5) or np.any(ratings > 5)):
        raise ValueError("Seed items and ratings must be one-dimensional and aligned.")
    items = raw_items.astype(np.int64)
    if (np.any(items < 0) or np.any(items >= len(model["movie_ids"]))
            or len(np.unique(items)) != len(items)):
        raise ValueError("Seed item indices must be unique and inside the catalogue.")
    mode = config.get("aggregation", default)
    center = float(config.get("rating_center", 3.))
    if not np.isfinite(center):
        raise ValueError("rating_center must be finite.")
    if mode == "signed":
        weights = ratings - center
    elif mode == "positive":
        weights = np.maximum(ratings - center, 0.)
    elif mode == "binary_positive":
        weights = (ratings >= float(config.get("positive_threshold", 4.))).astype(np.float64)
    else:
        raise ValueError("aggregation must be positive, signed or binary_positive.")
    return items, weights


def score_neighbors(model: dict, items, ratings, config: dict) -> np.ndarray:
    """Return one raw score per catalogue movie; caller excludes watched items."""
    items, weights = _seed_weights(model, items, ratings, config, "positive")
    scores = np.zeros(len(model["movie_ids"]), dtype=np.float64)
    for item, weight in zip(items, weights):
        if weight:
            lo, hi = model["neighbor_indptr"][item:item + 2]
            np.add.at(scores, model["neighbor_indices"][lo:hi], weight * model["neighbor_weights"][lo:hi])
    if not np.any(scores):
        return model["counts"].astype(np.float64).copy()
    return scores


def train_content(catalog, data: dict, config: dict, output: Path) -> dict:
    """Use only supplied MovieLens genre labels, aligned by MovieLens movie ID."""
    movie_ids, counts = _catalogue(data)
    if "movieId" not in catalog or "genres" not in catalog:
        raise ValueError("Catalogue must contain movieId and genres columns.")
    if catalog["movieId"].duplicated().any() or not np.isin(movie_ids, catalog["movieId"]).all():
        raise ValueError("Catalogue needs one unique metadata row for every training movie ID.")
    if (Path(output) / "model.npz").exists():
        raise ValueError(f"Model already exists: {Path(output) / 'model.npz'}")
    genre_rows = catalog.set_index("movieId").loc[movie_ids, "genres"].fillna("")
    vocabulary = {genre: column for column, genre in enumerate(MOVIELENS_GENRES)}
    features = np.zeros((len(movie_ids), len(vocabulary)), dtype=np.float32)
    for row, text in enumerate(genre_rows):
        genres = {g for g in str(text).split("|") if g and g != "(no genres listed)"}
        unknown = genres - vocabulary.keys()
        if unknown:
            raise ValueError(f"Unrecognised MovieLens genre labels: {sorted(unknown)}")
        for genre in genres:
            features[row, vocabulary[genre]] = 1.
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.where(lengths > 0, lengths, 1.)
    model = {
        "movie_ids": movie_ids.copy(), "counts": counts.copy(), "features": features,
        "genre_names": np.asarray(MOVIELENS_GENRES),
        "info": {
            "format_version": 1, "kind": "movielens_genres", "config": dict(config),
            "catalogue_items": len(movie_ids), "feature_count": len(vocabulary),
            "items_without_genres": int((lengths == 0).sum()),
            "metadata_source": "Supplied MovieLens genre labels; no inferred or external metadata.",
            "score_meaning": "Cosine with a rating-weighted genre profile; not a rating prediction.",
            "no_signal_fallback": "Training rating count, with exclusions applied by the evaluator.",
        },
    }
    model["info"]["array_bytes"] = sum(v.nbytes for k, v in model.items() if k != "info")
    _save(model, output)
    return model


def score_content(model: dict, items, ratings, config: dict) -> np.ndarray:
    items, weights = _seed_weights(model, items, ratings, config, "signed")
    profile = weights @ model["features"][items]
    length = np.linalg.norm(profile)
    if not length:
        return model["counts"].astype(np.float64).copy()
    return np.asarray(model["features"] @ (profile / length), dtype=np.float64)
