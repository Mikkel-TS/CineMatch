"""Cold-user item-context ranking experiments, separate from the app model.

Training samples two distinct liked titles from a training user and an unrated
title for that same user. BPR learns to rank the co-liked target over the unrated
target given the anchor. Unrated targets are uncertain negatives, not asserted
dislikes. Users are sampled uniformly; the number of pairs is an explicit compute
budget, not an assertion that every rating was visited in every epoch.

Inference averages item-context vectors from onboarding ratings. It never looks
up a training user or fits a new user with the explicit-rating MF objective.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cinematch.evaluation import evaluate_profiles


DEFAULTS = {
    "factors": 16, "epochs": 8, "patience": 3, "batch_size": 4096,
    "pairs_per_epoch": 1_000_000, "learning_rate": .005,
    "regularization": .001, "positive_threshold": 4.,
    "use_item_bias": True, "aggregation": "positive", "seed": 42, "threads": 2,
}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_sampling(data: dict, positive_threshold: float = 4.) -> dict:
    """Create grouped positives and a searchable observed-pair exclusion index."""
    user_ids = np.asarray(data["user_ids"])
    movie_ids = np.asarray(data["movie_ids"])
    users = np.asarray(data["user_index"], dtype=np.int64)
    items = np.asarray(data["movie_index"], dtype=np.int64)
    ratings = np.asarray(data["rating"])
    nu, ni = len(user_ids), len(movie_ids)
    if (users.ndim != 1 or items.shape != users.shape or ratings.shape != users.shape
            or not len(users) or len(np.unique(movie_ids)) != ni
            or len(np.unique(user_ids)) != nu or np.any(users < 0)
            or np.any(users >= nu) or np.any(items < 0) or np.any(items >= ni)
            or not np.isfinite(ratings).all()):
        raise ValueError("Invalid training arrays or movie/user identity mapping.")
    observed = np.sort(users * ni + items)
    if np.any(observed[1:] == observed[:-1]):
        raise ValueError("Duplicate user/movie training pairs.")
    observed_counts = np.bincount(users, minlength=nu)
    mask = ratings >= positive_threshold
    positive_users, positive_items = users[mask], items[mask]
    order = np.argsort(positive_users, kind="stable")
    positive_counts = np.bincount(positive_users, minlength=nu)
    eligible = np.flatnonzero((positive_counts >= 2) & (observed_counts < ni))
    if not len(eligible):
        raise ValueError("Ranking requires users with two liked movies and an unobserved candidate.")
    return {
        "positive_items": positive_items[order],
        "positive_counts": positive_counts,
        "offsets": np.r_[0, np.cumsum(positive_counts)],
        "eligible_users": eligible,
        "observed_pairs": observed,
        "n_movies": ni,
        "positive_interactions": int(mask.sum()),
        "eligible_user_count": len(eligible),
    }


def _observed(index: dict, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    pairs = users * index["n_movies"] + items
    keys = index["observed_pairs"]
    locations = np.searchsorted(keys, pairs)
    return (locations < len(keys)) & (keys[np.minimum(locations, len(keys) - 1)] == pairs)


def sample_pairs(index: dict, rng: np.random.Generator, size: int) -> tuple:
    """Return training users, anchors, positives and genuinely unobserved negatives."""
    if size < 1:
        raise ValueError("Pair count must be positive.")
    users = rng.choice(index["eligible_users"], size=size)
    counts = index["positive_counts"][users]
    first = (rng.random(size) * counts).astype(np.int64)
    second = (rng.random(size) * (counts - 1)).astype(np.int64)
    second += second >= first
    offsets = index["offsets"][users]
    anchors = index["positive_items"][offsets + first]
    positives = index["positive_items"][offsets + second]
    negatives = rng.integers(index["n_movies"], size=size)
    rejected = _observed(index, users, negatives)
    for _ in range(100):
        if not rejected.any():
            break
        negatives[rejected] = rng.integers(index["n_movies"], size=int(rejected.sum()))
        rejected = _observed(index, users, negatives)
    # Very dense users can make rejection slow. Their nonempty complement is
    # guaranteed by eligibility and gives an exact bounded-time fallback.
    if rejected.any():
        ni = index["n_movies"]
        keys = index["observed_pairs"]
        for user in np.unique(users[rejected]):
            lo, hi = np.searchsorted(keys, [user * ni, (user + 1) * ni])
            available = np.setdiff1d(np.arange(ni), keys[lo:hi] % ni)
            positions = np.flatnonzero(rejected & (users == user))
            negatives[positions] = rng.choice(available, size=len(positions))
    return users, anchors, positives, negatives


def _weights(ratings: np.ndarray, aggregation: str, threshold: float) -> np.ndarray:
    if aggregation == "positive":
        return np.where(ratings >= threshold, ratings - (threshold - 1.), 0.)
    if aggregation == "signed":
        return ratings - 3.
    if aggregation == "binary":
        return (ratings >= threshold).astype(np.float64)
    raise ValueError("aggregation must be positive, binary or signed")


def score_ranking(model: dict, items: np.ndarray, ratings: np.ndarray,
                  config: dict | None = None) -> np.ndarray:
    """Score a previously unseen user using only supplied onboarding ratings.

    Empty/no-informative input returns log rating-count popularity. Returned
    scores are ranking values; rating RMSE is not applicable.
    """
    stored = json.loads(str(np.asarray(model.get("config_json", "{}")).item()))
    settings = {**DEFAULTS, **stored, **(config or {})}
    items = np.asarray(items, dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float64)
    context, target = model["context_factors"], model["target_factors"]
    if (items.ndim != 1 or ratings.shape != items.shape
            or len(np.unique(items)) != len(items) or np.any(items < 0)
            or np.any(items >= len(context)) or not np.isfinite(ratings).all()
            or np.any((ratings < .5) | (ratings > 5))):
        raise ValueError("Supply one valid rating per known movie index.")
    weights = _weights(ratings, settings["aggregation"], settings["positive_threshold"])
    weight_sum = np.abs(weights).sum()
    if weight_sum == 0:
        return np.log1p(np.asarray(model["item_counts"], dtype=np.float64))
    profile = weights @ np.asarray(context[items], dtype=np.float64) / weight_sum
    return (np.asarray(target, dtype=np.float64) @ profile
            + float(settings.get("bias_scale", 1.)) * np.asarray(model["bias"], dtype=np.float64))


def _validation_ndcg(model: dict, profiles: dict, config: dict) -> float:
    values = []
    for i, seeds in enumerate(profiles["support_items"]):
        lo, hi = profiles["target_offsets"][i:i + 2]
        scores = score_ranking(model, seeds, profiles["support_ratings"][i], config)
        profile = {"user_ids": profiles["user_ids"][i:i + 1],
                   "support_items": profiles["support_items"][i:i + 1],
                   "support_ratings": profiles["support_ratings"][i:i + 1],
                   "target_offsets": np.array([0, hi - lo]),
                   "target_items": profiles["target_items"][lo:hi],
                   "target_ratings": profiles["target_ratings"][lo:hi]}
        metrics, _ = evaluate_profiles(0., np.zeros(len(scores)), np.empty((len(scores), 0)),
                                       profile, 1., 5., fixed_scores=scores)
        if metrics["ndcg_at_5"] is not None:
            values.append(metrics["ndcg_at_5"])
    if not values:
        raise ValueError("No relevant validation targets for ranking selection.")
    return float(np.mean(values))


class _ItemContextBPR(nn.Module):
    def __init__(self, n_movies: int, factors: int, use_item_bias: bool):
        super().__init__()
        self.context = nn.Embedding(n_movies, factors)
        self.target = nn.Embedding(n_movies, factors)
        self.bias = nn.Embedding(n_movies, 1)
        nn.init.normal_(self.context.weight, std=.1)
        nn.init.normal_(self.target.weight, std=.1)
        nn.init.zeros_(self.bias.weight)
        self.bias.weight.requires_grad_(use_item_bias)

    def loss(self, anchors: torch.Tensor, positive: torch.Tensor,
             negative: torch.Tensor, regularization: float) -> torch.Tensor:
        c, p, n = self.context(anchors), self.target(positive), self.target(negative)
        bp, bn = self.bias(positive).squeeze(-1), self.bias(negative).squeeze(-1)
        difference = (c * (p - n)).sum(-1) + bp - bn
        penalty = (c.square().sum(-1) + p.square().sum(-1)
                   + n.square().sum(-1) + bp.square() + bn.square()).mean()
        return torch.nn.functional.softplus(-difference).mean() + regularization * penalty


def train_ranking(data: dict, config: dict, output: Path, validation: dict) -> dict:
    """Train using explicit pair budgets and select an epoch on validation NDCG."""
    settings = {**DEFAULTS, **config}
    for key in ("factors", "epochs", "patience", "batch_size", "pairs_per_epoch", "threads"):
        if not isinstance(settings[key], (int, np.integer)) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (not np.isfinite([settings["learning_rate"], settings["regularization"],
                         settings["positive_threshold"]]).all()
            or settings["learning_rate"] <= 0 or settings["regularization"] < 0
            or not .5 <= settings["positive_threshold"] <= 5):
        raise ValueError("Invalid ranking hyperparameters.")
    if np.intersect1d(data["user_ids"], validation["user_ids"]).size:
        raise ValueError("Training and validation user identities overlap.")
    _weights(np.array([4.]), settings["aggregation"], settings["positive_threshold"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.npz").exists() or (output / "history.json").exists():
        raise ValueError("Ranking output already contains an experiment.")
    _write_json(output / "config.json", settings)
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(settings["seed"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(settings["seed"])
    index = prepare_sampling(data, settings["positive_threshold"])
    model = _ItemContextBPR(len(data["movie_ids"]), settings["factors"], settings["use_item_bias"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    best, best_ndcg, stale, history = None, -np.inf, 0, []
    sampling_info = {
        "sampling": "Uniform eligible training user; two distinct >=threshold items; uniform unobserved negative.",
        "negative_limit": "Unobserved does not mean disliked; negative sampling is a ranking approximation.",
        "data_population_ratings": len(data["rating"]),
        "positive_interactions": index["positive_interactions"],
        "eligible_user_count": index["eligible_user_count"],
        "pairs_per_epoch": settings["pairs_per_epoch"],
        "epoch_definition": "Fixed sampled pair budget, not an exhaustive pass over training ratings.",
        "new_user_method": "Weighted average of onboarding item-context vectors; score against target factors plus item bias.",
    }
    _write_json(output / "sampling.json", sampling_info)
    for epoch in range(1, settings["epochs"] + 1):
        start = time.monotonic()
        _, anchors, positives, negatives = sample_pairs(index, rng, settings["pairs_per_epoch"])
        total_loss = 0.
        for lo in range(0, len(anchors), settings["batch_size"]):
            hi = min(lo + settings["batch_size"], len(anchors))
            batch = [torch.from_numpy(x[lo:hi]) for x in (anchors, positives, negatives)]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(*batch, settings["regularization"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite BPR loss.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * (hi - lo)
        candidate = {
            "model_type": np.asarray("item_context_bpr"),
            "movie_ids": np.asarray(data["movie_ids"]).copy(),
            "item_counts": np.asarray(data["item_counts"]).copy(),
            "context_factors": model.context.weight.detach().numpy().copy(),
            "target_factors": model.target.weight.detach().numpy().copy(),
            "bias": model.bias.weight.detach().numpy().ravel().copy(),
            "selected_epoch": np.asarray(epoch),
            "config_json": np.asarray(json.dumps(settings, sort_keys=True)),
        }
        ndcg = _validation_ndcg(candidate, validation, settings)
        history.append({"epoch": epoch, "mean_regularized_bpr_loss": total_loss / len(anchors),
                        "validation_ndcg_at_5": ndcg, "pairs": len(anchors),
                        "seconds": time.monotonic() - start})
        if ndcg > best_ndcg:
            best, best_ndcg, stale = candidate, ndcg, 0
            temporary = output / "model.pending.npz"
            np.savez_compressed(temporary, **candidate)
            temporary.replace(output / "model.npz")
        else:
            stale += 1
        _write_json(output / "history.json", {"epochs": history, "selected_epoch": int(best["selected_epoch"]),
                    "selection": "Highest validation NDCG@5; ties retain earlier epoch.",
                    "sampling": sampling_info, "test_evaluated": False})
        print(f"BPR epoch {epoch:02d}: loss {total_loss / len(anchors):.5f}; validation NDCG@5 {ndcg:.5f}", flush=True)
        if stale >= settings["patience"]:
            break
    return best
