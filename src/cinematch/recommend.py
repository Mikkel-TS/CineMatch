"""Load a saved model, find catalogue IDs, and recommend for an unseen user."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import digest
from .evaluation import top_items
from .model import fold_in
from .training import checkpoint_parameters, load_checkpoint


class Recommender:
    def __init__(self, model_path: Path | str):
        model_path = Path(model_path)
        self.checkpoint = load_checkpoint(model_path)
        catalog_path = model_path.parent / "catalog.csv"
        if digest(catalog_path) != self.checkpoint["catalog_sha256"]:
            raise ValueError("Catalogue does not match this model checkpoint.")
        self.catalog = pd.read_csv(catalog_path).fillna("")
        self.movie_ids = self.checkpoint["movie_ids"].numpy()
        if not np.array_equal(self.catalog["movieId"].to_numpy(), self.movie_ids):
            raise ValueError("Catalogue movie order does not match the model.")
        self.index = {int(movie_id): i for i, movie_id in enumerate(self.movie_ids)}
        self.mean, self.bias, self.factors = checkpoint_parameters(self.checkpoint)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        if not query.strip() or limit < 1:
            raise ValueError("Search requires a nonempty query and a positive limit.")
        rows = self.catalog.loc[self.catalog["title"].str.contains(query, case=False, regex=False)]
        return rows.sort_values(["training_rating_count", "movieId"], ascending=[False, True]).head(limit).to_dict("records")

    def recommend(self, ratings: list[tuple[int, float]], k: int = 5,
                  exclude: tuple[int, ...] = ()) -> dict:
        ids = [movie_id for movie_id, _ in ratings]
        if len(set(ids)) != len(ids):
            raise ValueError("A movie was rated more than once. Supply one current rating per movie.")
        missing = sorted(set(ids).difference(self.index))
        if missing:
            raise ValueError(f"Movie IDs are outside this model's catalogue: {missing}. Use 'cinematch search' to find eligible movies.")
        values = np.asarray([rating for _, rating in ratings], dtype=np.float64)
        if (not np.isfinite(values).all() or np.any((values < .5) | (values > 5))
                or not np.isclose(values * 2, np.round(values * 2)).all()):
            raise ValueError("Use half-star ratings between 0.5 and 5.")
        items = np.asarray([self.index[movie_id] for movie_id in ids], dtype=np.int64)
        scores = fold_in(self.mean, self.bias, self.factors, items, values,
                        self.checkpoint["fold_in_regularization"], self.checkpoint["bias_regularization"])
        excluded = np.asarray(sorted(set(items.tolist()) | {self.index[x] for x in exclude if x in self.index}), dtype=np.int64)
        ranked = top_items(scores, excluded, k)
        results = []
        for i in ranked:
            row = self.catalog.iloc[int(i)]
            results.append({"movieId": int(row["movieId"]), "title": str(row["title"]),
                            "genres": str(row["genres"]), "score": round(float(scores[i]), 6)})
        return {"personalized": bool(ratings), "ratings_used": len(ratings),
                "score_description": "Unclipped model ranking score; not a probability or a displayed star prediction.",
                "recommendations": results}
