"""Collaborative movie features learned from MovieLens ratings."""

from collections.abc import Sequence
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


MODEL_VERSION = 1
DEFAULT_COMPONENTS = 48


class MovieLensAudienceModel:
    def __init__(self, letterboxd_ids, vectors):
        self.letterboxd_ids = np.asarray(letterboxd_ids, dtype=np.int64)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.letterboxd_ids):
            raise ValueError("MovieLens model vectors do not match the mapped film IDs")
        if np.unique(self.letterboxd_ids).size != len(self.letterboxd_ids):
            raise ValueError("MovieLens model contains duplicate film IDs")
        if not np.isfinite(self.vectors).all():
            raise ValueError("MovieLens model contains non-finite features")
        self.row_of = {int(film_id): row for row, film_id in enumerate(self.letterboxd_ids)}

    @classmethod
    def fit(
        cls,
        ratings: sparse.spmatrix,
        letterboxd_ids: Sequence[int],
        *,
        n_components: int = DEFAULT_COMPONENTS,
        min_ratings: int = 5,
    ):
        if not sparse.issparse(ratings) or ratings.ndim != 2:
            raise TypeError("ratings must be a two-dimensional SciPy sparse matrix")
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if min_ratings < 1:
            raise ValueError("min_ratings must be positive")

        movie_ids = np.asarray(letterboxd_ids, dtype=np.int64)
        if ratings.shape[1] != len(movie_ids):
            raise ValueError("ratings columns must match the mapped film IDs")
        if np.unique(movie_ids).size != len(movie_ids):
            raise ValueError("mapped film IDs must be unique")

        matrix = ratings.astype(np.float32).tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if not np.isfinite(matrix.data).all() or np.any(
            (matrix.data < 0.5) | (matrix.data > 5)
        ):
            raise ValueError("ratings must be finite values between 0.5 and 5")

        rated_movies = np.diff(matrix.tocsc().indptr) >= min_ratings
        matrix = matrix[:, rated_movies].tocsr()
        movie_ids = movie_ids[rated_movies]
        if len(movie_ids) < 2:
            raise ValueError("at least two mapped films need enough ratings to train")

        user_counts = np.diff(matrix.indptr)
        active_users = np.count_nonzero(user_counts > 1)
        if active_users < 2:
            raise ValueError("at least two users must rate multiple mapped films")

        means = np.zeros(matrix.shape[0], dtype=np.float32)
        nonempty = user_counts > 0
        user_totals = np.asarray(matrix.sum(axis=1)).ravel()
        means[nonempty] = user_totals[nonempty] / user_counts[nonempty]
        matrix.data -= np.repeat(means, user_counts)
        matrix.eliminate_zeros()
        if matrix.nnz == 0:
            raise ValueError("ratings contain no variation after centering by user")

        components = min(n_components, min(matrix.shape) - 1)
        if components < 1:
            raise ValueError("not enough users and films to train the model")
        decomposition = TruncatedSVD(
            n_components=components, n_iter=5, random_state=0
        ).fit(matrix)
        vectors = (
            decomposition.components_.T * decomposition.singular_values_[None, :]
        ).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        np.divide(vectors, norms[:, None], out=vectors, where=norms[:, None] > 0)
        return cls(movie_ids, vectors)

    def features_for(self, film_ids: Sequence[int]) -> sparse.csr_matrix:
        vectors = np.zeros((len(film_ids), self.vectors.shape[1]), dtype=np.float32)
        for row, film_id in enumerate(film_ids):
            model_row = self.row_of.get(int(film_id))
            if model_row is not None:
                vectors[row] = self.vectors[model_row]
        return sparse.csr_matrix(vectors)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".tmp")
        joblib.dump(
            {
                "version": MODEL_VERSION,
                "letterboxd_ids": self.letterboxd_ids,
                "vectors": self.vectors,
            },
            temporary_path,
            compress=3,
        )
        os.replace(temporary_path, path)

    @classmethod
    def load(cls, path: Path):
        model = joblib.load(path)
        if model.get("version") != MODEL_VERSION:
            raise ValueError(f"Unsupported MovieLens model version: {model.get('version')}")
        return cls(model["letterboxd_ids"], model["vectors"])


def _count_csv_records(path: Path) -> int:
    line_count = 0
    last_byte = b""
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        line_count += 1
    if line_count == 0:
        raise ValueError(f"Ratings file is empty: {path}")
    return line_count - 1


def build_movielens_model(
    ratings_path: Path,
    mapping_path: Path,
    *,
    n_components: int = DEFAULT_COMPONENTS,
    min_ratings: int = 5,
    chunksize: int = 1_000_000,
) -> MovieLensAudienceModel:
    ratings_path = Path(ratings_path)
    mapping_path = Path(mapping_path)
    mapping = pd.read_csv(
        mapping_path,
        usecols=["letterboxd_id", "movieId"],
        dtype={"letterboxd_id": "int64", "movieId": "int64"},
    ).sort_values("movieId", kind="stable")
    if mapping.empty:
        raise ValueError(f"Movie mapping is empty: {mapping_path}")
    if mapping["movieId"].duplicated().any() or mapping["letterboxd_id"].duplicated().any():
        raise ValueError(f"Movie mapping must be one-to-one: {mapping_path}")

    movie_ids = mapping["movieId"].to_numpy(dtype=np.int64)
    letterboxd_ids = mapping["letterboxd_id"].to_numpy(dtype=np.int64)
    movie_to_column = {int(movie_id): column for column, movie_id in enumerate(movie_ids)}

    record_count = _count_csv_records(ratings_path)
    if record_count == 0:
        raise ValueError(f"Ratings file has no records: {ratings_path}")
    print(f"Reading {record_count:,} MovieLens ratings...", flush=True)

    rows = np.empty(record_count, dtype=np.int32)
    columns = np.empty(record_count, dtype=np.int32)
    values = np.empty(record_count, dtype=np.float32)
    offset = 0
    max_user_id = 0
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        chunksize=chunksize,
    ):
        user_ids = chunk["userId"].to_numpy(dtype=np.int32, copy=False)
        if np.any(user_ids <= 0):
            raise ValueError("MovieLens user IDs must be positive integers")
        max_user_id = max(max_user_id, int(user_ids.max()))

        ratings_values = chunk["rating"].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(ratings_values).all() or np.any(
            (ratings_values < 0.5) | (ratings_values > 5)
        ):
            raise ValueError("MovieLens ratings must be finite values between 0.5 and 5")

        mapped_columns = chunk["movieId"].map(movie_to_column)
        selected = mapped_columns.notna().to_numpy()
        selected_count = int(np.count_nonzero(selected))
        if selected_count:
            end = offset + selected_count
            rows[offset:end] = user_ids[selected] - 1
            columns[offset:end] = mapped_columns[selected].to_numpy(dtype=np.int32)
            values[offset:end] = ratings_values[selected]
            offset = end

    if offset == 0:
        raise ValueError("No MovieLens ratings match the CineMatch movie mapping")
    matrix = sparse.coo_matrix(
        (values[:offset], (rows[:offset], columns[:offset])),
        shape=(max_user_id, len(movie_ids)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    del rows, columns, values
    print(f"Training on {offset:,} mapped ratings...", flush=True)
    return MovieLensAudienceModel.fit(
        matrix,
        letterboxd_ids,
        n_components=n_components,
        min_ratings=min_ratings,
    )
