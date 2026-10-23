"""Train on observed ratings and select checkpoints using new-user validation."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_arrays, write_json
from .evaluation import evaluate_profiles, paired_comparison
from .model import MatrixFactorization, fit_bias_baseline


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint version.")
    return checkpoint


def checkpoint_parameters(checkpoint: dict) -> tuple[float, np.ndarray, np.ndarray]:
    weights = checkpoint["state_dict"]
    return (float(weights["global_mean"].item()), weights["movie_bias.weight"].numpy().ravel(),
            weights["movie_factors.weight"].numpy())


def train(data: Path, output: Path, factors: int = 16, epochs: int = 20,
          batch_size: int = 4096, learning_rate: float = .001,
          regularization: float = .01, fold_in_grid: tuple = (1., 5., 20.),
          bias_regularization: float = 5., patience: int = 3, seed: int = 42,
          threads: int = 4) -> dict:
    if output.exists():
        raise ValueError(f"Run already exists: {output}. Choose a new --output directory.")
    if min(factors, epochs, batch_size, patience, threads) < 1:
        raise ValueError("Factors, epochs, batch size, patience, and threads must be positive.")
    if (not all(math.isfinite(x) for x in [learning_rate, regularization, bias_regularization, *fold_in_grid])
            or learning_rate <= 0 or regularization < 0 or bias_regularization <= 0
            or not fold_in_grid or min(fold_in_grid) <= 0):
        raise ValueError("Learning rate and fold-in penalties must be positive; training penalty must be nonnegative.")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    arrays = load_arrays(data, "train")
    validation = load_arrays(data, "validation")
    manifest = json.loads((data / "manifest.json").read_text())
    if digest(data / "catalog.csv") != manifest["sha256"]["catalog.csv"]:
        raise ValueError("Prepared catalogue changed. Prepare a new dataset directory.")
    nu, ni = len(arrays["user_ids"]), len(arrays["movie_ids"])
    users = torch.from_numpy(arrays["user_index"])
    movies = torch.from_numpy(arrays["movie_index"])
    ratings = torch.from_numpy(arrays["rating"])
    baseline_mean, initial_user_bias, baseline_bias = fit_bias_baseline(arrays["user_index"], arrays["movie_index"],
                                                                       arrays["rating"], nu, ni)
    baseline, _ = evaluate_profiles(baseline_mean, baseline_bias, np.empty((ni, 0)), validation,
                                    1., bias_regularization)
    if baseline["ndcg_at_5"] is None:
        raise ValueError("Validation users have no held-out ratings >=4; cannot select by NDCG@5.")
    popularity, _ = evaluate_profiles(baseline_mean, baseline_bias, np.empty((ni, 0)), validation,
                                      1., bias_regularization, fixed_scores=arrays["item_counts"])
    model = MatrixFactorization(nu, ni, factors, float(ratings.mean().item()))
    # Start with fitted biases so an untrained prior does not dominate early stopping.
    with torch.no_grad():
        model.user_bias.weight.copy_(torch.from_numpy(initial_user_bias[:, None]))
        model.movie_bias.weight.copy_(torch.from_numpy(baseline_bias[:, None]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output.mkdir(parents=True)
    shutil.copyfile(data / "catalog.csv", output / "catalog.csv")
    config = {"factors": factors, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "regularization": regularization,
              "fold_in_grid": list(fold_in_grid), "bias_regularization": bias_regularization,
              "patience": patience, "seed": seed, "threads": threads,
              "bias_initialization": "regularized training-user/movie biases",
              "device": "cpu", "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    write_json(output / "config.json", config)
    history, best_key, best_epoch, stale_epochs = [], (-float("inf"), -float("inf")), 0, 0
    print(f"Training {nu:,} users / {ni:,} movies; baseline validation NDCG@5={baseline['ndcg_at_5']:.4f}", flush=True)
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        model.train()
        permutation = torch.randperm(len(ratings))
        squared_error = 0.
        for batch in permutation.split(batch_size):
            u, i, r = users[batch], movies[batch], ratings[batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(u, i)
            mse = (prediction - r).square().mean()
            loss = mse + regularization * model.penalty(u, i)
            if not torch.isfinite(loss):
                raise ValueError("Training diverged; lower the learning rate.")
            loss.backward()
            optimizer.step()
            squared_error += float(mse.detach().item()) * len(batch)
        model.eval()
        mean, bias, vectors = model.movie_parameters()
        options = []
        for penalty in fold_in_grid:
            metrics, _ = evaluate_profiles(mean, bias, vectors, validation, penalty, bias_regularization)
            options.append({"fold_in_regularization": penalty, **metrics})
        chosen = max(options, key=lambda x: (x["ndcg_at_5"], -x["rmse_macro"]))
        prior, _ = evaluate_profiles(mean, bias, np.empty((ni, 0)), validation, 1., bias_regularization)
        key = (chosen["ndcg_at_5"], -chosen["rmse_macro"])
        history.append({"epoch": epoch, "training_rmse_during_updates": (squared_error / len(ratings)) ** .5,
                        "validation_options": options, "model_prior_validation": prior,
                        "seconds": time.monotonic() - start})
        if key > best_key:
            best_key, best_epoch, stale_epochs = key, epoch, 0
            checkpoint = {
                "format_version": 1, "state_dict": model.state_dict(), "config": config,
                "movie_ids": torch.from_numpy(arrays["movie_ids"]),
                "item_counts": torch.from_numpy(arrays["item_counts"]),
                "training_user_ids": torch.from_numpy(arrays["user_ids"]),
                "baseline_mean": baseline_mean, "baseline_bias": torch.from_numpy(baseline_bias),
                "fold_in_regularization": chosen["fold_in_regularization"],
                "bias_regularization": bias_regularization, "best_epoch": best_epoch,
                "validation": chosen, "baseline_validation": baseline,
                "popularity_validation": popularity, "model_prior_validation": prior,
                "data_manifest_sha256": digest(data / "manifest.json"),
                "catalog_sha256": manifest["sha256"]["catalog.csv"],
            }
            temporary = output / "model.pending.pt"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "model.pt")
        else:
            stale_epochs += 1
        write_json(output / "history.json", {"epochs": history})
        print(f"Epoch {epoch:02d}: train RMSE {history[-1]['training_rmse_during_updates']:.4f}; "
              f"validation NDCG@5 {chosen['ndcg_at_5']:.4f}; fold-in penalty {chosen['fold_in_regularization']:g}", flush=True)
        if stale_epochs >= patience:
            print(f"Early stopping; keeping epoch {best_epoch}.", flush=True)
            break
    best = load_checkpoint(output / "model.pt")
    report = {"best_epoch": best_epoch, "validation": best["validation"],
              "baseline_validation": baseline, "popularity_validation": popularity,
              "model_prior_validation": best["model_prior_validation"],
              "selection": "Highest validation NDCG@5; lower RMSE breaks ties.",
              "test_evaluated": False}
    write_json(output / "validation.json", report)
    return report


def evaluate_checkpoint(model_path: Path, data: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Report already exists: {output}. Choose another output path.")
    checkpoint = load_checkpoint(model_path)
    if digest(data / "manifest.json") != checkpoint["data_manifest_sha256"]:
        raise ValueError("This checkpoint was trained with a different prepared dataset.")
    test = load_arrays(data, "test")
    mean, bias, factors = checkpoint_parameters(checkpoint)
    model_metrics, model_rows = evaluate_profiles(mean, bias, factors, test,
        checkpoint["fold_in_regularization"], checkpoint["bias_regularization"])
    baseline_metrics, baseline_rows = evaluate_profiles(checkpoint["baseline_mean"],
        checkpoint["baseline_bias"].numpy(), np.empty((len(bias), 0)), test, 1., checkpoint["bias_regularization"])
    counts = (checkpoint["item_counts"].numpy() if "item_counts" in checkpoint
              else load_arrays(data, "train")["item_counts"])
    popularity_metrics, popularity_rows = evaluate_profiles(mean, bias, np.empty((len(bias), 0)), test,
        1., checkpoint["bias_regularization"], fixed_scores=counts)
    prior_metrics, prior_rows = evaluate_profiles(mean, bias, np.empty((len(bias), 0)), test,
        1., checkpoint["bias_regularization"])
    report = {
        "model": model_metrics, "baseline": baseline_metrics,
        "popularity": popularity_metrics, "model_prior": prior_metrics,
        "paired_comparison": paired_comparison(model_rows, baseline_rows, checkpoint["config"]["seed"]),
        "paired_vs_popularity": paired_comparison(model_rows, popularity_rows, checkpoint["config"]["seed"]),
        "paired_vs_model_prior": paired_comparison(model_rows, prior_rows, checkpoint["config"]["seed"]),
        "checkpoint_sha256": digest(model_path), "best_epoch": checkpoint["best_epoch"],
        "fold_in_regularization": checkpoint["fold_in_regularization"],
        "protocol": "Full eligible catalogue minus onboarding movies. Relevant means a held-out rating >=4. "
                    "Users with no relevant targets are omitted only from ranking metrics. RMSE gives each user equal weight.",
        "limitations": "Retrospective reconstruction from random onboarding ratings; unrated movies have unknown relevance. "
                       "Does not measure self-selected favourites, future viewing, or movies excluded by training support. "
                       "After inspecting this test report, further model selection requires a fresh test group.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
