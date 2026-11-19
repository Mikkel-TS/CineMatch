"""Isolated explicit-rating MF experiments; never loads a test profile.

The default prediction and new-user solve match ``cinematch.model.fold_in``.
User centring substitutes the support-rating mean for the training mean, so
scores remain on the rating scale. ``fit_user_bias=False`` removes the fitted
intercept. All bias transforms affect both the support residuals and candidates.

Training penalties are per observed rating. ``bias_count_regularization`` adds
an inverse-count-weighted penalty: holding other parameters fixed, a bias with
n observations then has denominator n * (1 + bias_regularization_train) + tau.
This restores stronger shrinkage for entities with little observed evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cinematch.evaluation import evaluate_profiles, top_items
from cinematch.model import MatrixFactorization, fit_bias_baseline, fold_in


DEFAULTS = {
    "factors": 16, "epochs": 12, "batch_size": 4096,
    "learning_rate": .001, "regularization": .01,
    "bias_regularization_train": None, "bias_count_regularization": 0.,
    "train_movie_bias": True, "train_user_bias": True,
    "loss": "mse", "huber_delta": 1., "fold_in_grid": [1., 5., 20.],
    "fold_in_regularization": 1., "bias_regularization": 5.,
    "movie_bias_scale": 1., "bias_shrinkage_tau": 0.,
    "user_center": False, "fit_user_bias": True,
    "patience": 3, "seed": 42, "threads": 2,
}


def _configuration(config):
    result = {**DEFAULTS, **config}
    result["threads"] = min(2, int(result["threads"]))
    if result["bias_regularization_train"] is None:
        result["bias_regularization_train"] = result["regularization"]
    for key in ("factors", "epochs", "batch_size", "patience", "threads"):
        if int(result[key]) != result[key] or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("learning_rate", "huber_delta", "fold_in_regularization", "bias_regularization"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    for key in ("regularization", "bias_regularization_train", "bias_count_regularization",
                "bias_shrinkage_tau", "movie_bias_scale"):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if result["loss"] not in ("mse", "huber"):
        raise ValueError("loss must be mse or huber.")
    grid = result["fold_in_grid"]
    if not grid or not np.isfinite(grid).all() or min(grid) <= 0:
        raise ValueError("fold_in_grid must contain finite positive penalties.")
    return result


def _model_config(model):
    if "config" in model:
        return dict(model["config"])
    if "config_json" in model:
        return json.loads(str(np.asarray(model["config_json"]).item()))
    return {}


def _bias(model, config):
    bias = np.asarray(model["bias"], dtype=np.float64) * config["movie_bias_scale"]
    tau = config["bias_shrinkage_tau"]
    if tau:
        counts = np.asarray(model["item_counts"], dtype=np.float64)
        bias = bias * counts / (counts + tau)
    return bias


def scorer(model: dict, support_items, support_ratings, config: dict | None = None) -> np.ndarray:
    """Return aligned all-catalogue scores from disclosed ratings only.

    ``config`` overrides saved fit settings for an explicitly recorded inference
    experiment. No user identifier or hidden target is accepted by this API.
    """
    cfg = _configuration({**_model_config(model), **(config or {})})
    factors = np.asarray(model["factors"])
    bias = _bias(model, cfg)
    mean = float(np.asarray(model["mean"]).item())
    items = np.asarray(support_items, dtype=np.int64)
    ratings = np.asarray(support_ratings, dtype=np.float64)
    if factors.ndim != 2 or factors.shape[0] != len(bias):
        raise ValueError("Movie factors and biases have different catalogue order/size.")
    if len(model["movie_ids"]) != len(bias):
        raise ValueError("Movie IDs and parameters have different catalogue size.")
    if items.ndim != 1 or ratings.shape != items.shape or len(np.unique(items)) != len(items):
        raise ValueError("Provide one rating per distinct movie.")
    if np.any(items < 0) or np.any(items >= len(bias)):
        raise ValueError("Unknown movie index.")
    if not np.isfinite(ratings).all() or np.any((ratings < .5) | (ratings > 5)):
        raise ValueError("Ratings must be finite and between 0.5 and 5.")
    if not len(items):
        return mean + bias
    if cfg["user_center"]:
        mean = float(ratings.mean())
    if cfg["fit_user_bias"]:
        return fold_in(mean, bias, factors, items, ratings,
                       cfg["fold_in_regularization"], cfg["bias_regularization"])
    design = factors[items].astype(np.float64)
    coefficients = np.linalg.solve(
        design.T @ design + cfg["fold_in_regularization"] * np.eye(design.shape[1]),
        design.T @ (ratings - mean - bias[items]))
    return mean + bias + factors @ coefficients


def _validation_metrics(model, profiles, config):
    """Use the existing evaluator directly whenever its solver applies."""
    if not config["user_center"] and config["fit_user_bias"]:
        return evaluate_profiles(float(model["mean"]), _bias(model, config), model["factors"],
            profiles, config["fold_in_regularization"], config["bias_regularization"])[0]
    # The altered intercept needs its own solve, but retains the exact existing
    # exclusion, binary relevance, full-catalogue and macro aggregation rules.
    discounts = 1 / np.log2(np.arange(2, 7))
    rows, recommendations = [], set()
    for j in range(len(profiles["user_ids"])):
        scores = scorer(model, profiles["support_items"][j], profiles["support_ratings"][j], config)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite validation scores.")
        lo, hi = profiles["target_offsets"][j:j + 2]
        items = profiles["target_items"][lo:hi]
        ratings = profiles["target_ratings"][lo:hi]
        positive = items[ratings >= 4]
        top = top_items(scores, profiles["support_items"][j])
        recommendations.update(top.tolist())
        hits = np.isin(top, positive)
        rows.append([float(np.mean((np.clip(scores[items], .5, 5) - ratings) ** 2)),
            float((hits * discounts[:len(top)]).sum() / discounts[:min(5, len(positive))].sum())
            if len(positive) else np.nan,
            float(hits.sum() / len(positive)) if len(positive) else np.nan,
            float(hits.any()) if len(positive) else np.nan])
    values = np.asarray(rows)
    count = int(np.isfinite(values[:, 1]).sum())
    return {"users": len(rows), "users_with_relevant_targets": count,
        "rmse_macro": float(np.sqrt(values[:, 0].mean())),
        "ndcg_at_5": float(np.nanmean(values[:, 1])) if count else None,
        "recall_at_5": float(np.nanmean(values[:, 2])) if count else None,
        "hit_rate_at_5": float(np.nanmean(values[:, 3])) if count else None,
        "catalog_coverage_at_5": len(recommendations) / len(model["bias"])}


def _check_data(data, validation):
    user_ids, movie_ids = np.asarray(data["user_ids"]), np.asarray(data["movie_ids"])
    if not len(user_ids) or len(np.unique(user_ids)) != len(user_ids):
        raise ValueError("Training user IDs must be nonempty and unique.")
    if not len(movie_ids) or np.any(np.diff(movie_ids) <= 0):
        raise ValueError("Movie IDs must be unique and in ascending benchmark order.")
    if "movie_ids" in validation and not np.array_equal(movie_ids, validation["movie_ids"]):
        raise ValueError("Validation movie-ID order differs from training.")
    held_out = np.concatenate([np.asarray(validation["user_ids"]),
                               np.asarray(data.get("holdout_user_ids", []))])
    if np.intersect1d(user_ids, held_out).size:
        raise ValueError("Training users overlap reserved evaluation users.")
    u, i, r = (np.asarray(data[key]) for key in ("user_index", "movie_index", "rating"))
    if u.ndim != 1 or i.shape != u.shape or r.shape != u.shape or not len(r):
        raise ValueError("Training arrays must be nonempty aligned vectors.")
    if not np.issubdtype(u.dtype, np.integer) or not np.issubdtype(i.dtype, np.integer):
        raise ValueError("Training indices must be integers.")
    if u.min() < 0 or u.max() >= len(user_ids) or i.min() < 0 or i.max() >= len(movie_ids):
        raise ValueError("Training indices exceed user/movie mapping.")
    if not np.isfinite(r).all() or np.any((r < .5) | (r > 5)):
        raise ValueError("Invalid training ratings.")
    counts = np.bincount(i, minlength=len(movie_ids))
    if not np.array_equal(counts, data["item_counts"]):
        raise ValueError("Item counts do not match the training data/movie order.")
    if not len(validation["user_ids"]):
        raise ValueError("Validation profiles must be nonempty.")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_mf(data: dict, config: dict, output: Path, validation: dict) -> dict:
    """Fit a recorded experiment; return deployable arrays without user data.

    ``data`` uses the existing train.npz keys. An optional ``holdout_user_ids``
    array additionally checks legacy/fresh reservations without reading their
    profiles. ``validation`` alone selects epoch and fold-in penalty. The caller
    records data identity/source snapshots and must supply already-isolated data.
    """
    cfg = _configuration(config)
    _check_data(data, validation)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment model directory is not empty; preserve earlier artifacts.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", cfg)
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    nu, ni = len(data["user_ids"]), len(data["movie_ids"])
    users = torch.from_numpy(np.asarray(data["user_index"], dtype=np.int64))
    movies = torch.from_numpy(np.asarray(data["movie_index"], dtype=np.int64))
    ratings = torch.from_numpy(np.asarray(data["rating"], dtype=np.float32))
    mean, initial_user_bias, baseline_bias = fit_bias_baseline(
        users.numpy(), movies.numpy(), ratings.numpy(), nu, ni)
    model = MatrixFactorization(nu, ni, cfg["factors"], mean)
    with torch.no_grad():
        model.user_bias.weight.copy_(torch.from_numpy(initial_user_bias[:, None]))
        model.movie_bias.weight.copy_(torch.from_numpy(baseline_bias[:, None]))
        if not cfg["train_movie_bias"]:
            model.movie_bias.weight.zero_()
        if not cfg["train_user_bias"]:
            model.user_bias.weight.zero_()
    model.movie_bias.weight.requires_grad_(cfg["train_movie_bias"])
    model.user_bias.weight.requires_grad_(cfg["train_user_bias"])
    user_counts = torch.bincount(users, minlength=nu).clamp_min(1)
    item_counts = torch.as_tensor(data["item_counts"]).clamp_min(1)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    best, best_key, stale, history = None, (-float("inf"), -float("inf")), 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        start, squared_error = time.monotonic(), 0.
        model.train()
        for batch in torch.randperm(len(ratings)).split(cfg["batch_size"]):
            u, i, r = users[batch], movies[batch], ratings[batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(u, i)
            mse = (prediction - r).square().mean()
            objective = mse if cfg["loss"] == "mse" else F.huber_loss(prediction, r, delta=cfg["huber_delta"])
            factor_penalty = (model.user_factors(u).square().sum(-1)
                              + model.movie_factors(i).square().sum(-1)).mean()
            bu, bi = model.user_bias(u).squeeze(-1).square(), model.movie_bias(i).squeeze(-1).square()
            bias_penalty = cfg["bias_regularization_train"] * (bu + bi).mean()
            count_penalty = cfg["bias_count_regularization"] * (bu / user_counts[u] + bi / item_counts[i]).mean()
            loss = objective + cfg["regularization"] * factor_penalty + bias_penalty + count_penalty
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            squared_error += float(mse.detach()) * len(batch)
        model.eval()
        mean, bias, factors = model.movie_parameters()
        candidate = {"mean": mean, "bias": bias, "factors": factors,
            "movie_ids": np.asarray(data["movie_ids"]).copy(),
            "item_counts": np.asarray(data["item_counts"]).copy(),
            "baseline_bias": baseline_bias.copy()}
        options = []
        for penalty in cfg["fold_in_grid"]:
            fit_config = {**cfg, "fold_in_regularization": penalty}
            metrics = _validation_metrics(candidate, validation, fit_config)
            if metrics["ndcg_at_5"] is None:
                raise ValueError("No relevant validation targets for checkpoint selection.")
            options.append({"fold_in_regularization": penalty, **metrics})
        chosen = max(options, key=lambda x: (x["ndcg_at_5"], -x["rmse_macro"]))
        key = (chosen["ndcg_at_5"], -chosen["rmse_macro"])
        history.append({"epoch": epoch, "training_rmse_during_updates": (squared_error / len(ratings)) ** .5,
                        "validation_options": options, "seconds": time.monotonic() - start})
        if key > best_key:
            best_key, stale = key, 0
            best = {**candidate, "config": {**cfg, "fold_in_regularization": chosen["fold_in_regularization"]},
                    "best_epoch": epoch}
            stored = {k: v for k, v in best.items() if k != "config"}
            stored["config_json"] = np.asarray(json.dumps(best["config"], sort_keys=True))
            np.savez_compressed(output / "model.pending.npz", **stored)
            (output / "model.pending.npz").replace(output / "model.npz")
        else:
            stale += 1
        _write_json(output / "history.json", {"epochs": history, "best_epoch": best["best_epoch"],
            "selection": "Highest validation NDCG@5; lower rating RMSE breaks ties.",
            "test_evaluated": False, "fresh_test_evaluated": False})
        print(f"MF epoch {epoch:02d}: train RMSE {history[-1]['training_rmse_during_updates']:.4f}; "
              f"validation NDCG@5 {chosen['ndcg_at_5']:.5f}; lambda {chosen['fold_in_regularization']:g}", flush=True)
        if stale >= cfg["patience"]:
            break
    _write_json(output / "fit_config.json", best["config"])
    return best
