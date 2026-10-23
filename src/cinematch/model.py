"""Explicit-rating factorization and regularized estimation of unseen users."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, movies: int, factors: int, mean: float):
        super().__init__()
        self.user_factors = nn.Embedding(users, factors)
        self.movie_factors = nn.Embedding(movies, factors)
        self.user_bias = nn.Embedding(users, 1)
        self.movie_bias = nn.Embedding(movies, 1)
        self.register_buffer("global_mean", torch.tensor(mean, dtype=torch.float32))
        nn.init.normal_(self.user_factors.weight, std=.05)
        nn.init.normal_(self.movie_factors.weight, std=.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        return (self.global_mean + self.user_bias(users).squeeze(-1)
                + self.movie_bias(movies).squeeze(-1)
                + (self.user_factors(users) * self.movie_factors(movies)).sum(-1))

    def penalty(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        # Per-observed-rating regularization: frequent entities receive more weight.
        return (self.user_factors(users).square().sum(-1)
                + self.movie_factors(movies).square().sum(-1)
                + self.user_bias(users).squeeze(-1).square()
                + self.movie_bias(movies).squeeze(-1).square()).mean()

    def movie_parameters(self) -> tuple[float, np.ndarray, np.ndarray]:
        return (float(self.global_mean.item()), self.movie_bias.weight.detach().cpu().numpy().ravel().copy(),
                self.movie_factors.weight.detach().cpu().numpy().copy())


def fold_in(mean: float, movie_bias: np.ndarray, movie_factors: np.ndarray,
            items: np.ndarray, ratings: np.ndarray, regularization: float = 5.,
            bias_regularization: float = 5.) -> np.ndarray:
    """Fit only the new user's intercept and vector; movie parameters remain fixed.

    Objective: SUM of squared seed-rating errors + lambda_p ||p||² + lambda_b b².
    A zero-rating user gets the learned movie prior. Returned scores are unclipped.
    """
    if not np.isfinite([regularization, bias_regularization]).all() or min(regularization, bias_regularization) <= 0:
        raise ValueError("Fold-in regularization must be finite and positive.")
    items = np.asarray(items, dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float64)
    if items.ndim != 1 or ratings.shape != items.shape or len(np.unique(items)) != len(items):
        raise ValueError("Provide one rating per distinct movie.")
    if np.any(items < 0) or np.any(items >= len(movie_bias)):
        raise ValueError("Unknown movie index.")
    if not np.isfinite(ratings).all() or np.any((ratings < .5) | (ratings > 5)):
        raise ValueError("Ratings must be finite and between 0.5 and 5.")
    base = mean + np.asarray(movie_bias, dtype=np.float64)
    if not len(items):
        return base.copy()
    design = np.column_stack([np.ones(len(items)), movie_factors[items]]).astype(np.float64)
    diagonal = np.full(design.shape[1], regularization)
    diagonal[0] = bias_regularization
    coefficients = np.linalg.solve(design.T @ design + np.diag(diagonal),
                                   design.T @ (ratings - base[items]))
    return base + coefficients[0] + movie_factors @ coefficients[1:]


def fit_bias_baseline(users: np.ndarray, movies: np.ndarray, ratings: np.ndarray,
                      n_users: int, n_movies: int, regularization: float = 10.,
                      iterations: int = 10) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a separate regularized movie prior, controlling for training-user bias."""
    mean = float(np.mean(ratings))
    bu, bi = np.zeros(n_users), np.zeros(n_movies)
    nu = np.bincount(users, minlength=n_users)
    ni = np.bincount(movies, minlength=n_movies)
    for _ in range(iterations):
        bu = np.bincount(users, weights=ratings - mean - bi[movies], minlength=n_users) / (nu + regularization)
        bi = np.bincount(movies, weights=ratings - mean - bu[users], minlength=n_movies) / (ni + regularization)
    return mean, bu.astype(np.float32), bi.astype(np.float32)
