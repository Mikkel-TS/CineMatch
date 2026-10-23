"""Command-line entry point for data preparation, training, and recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare
from .recommend import Recommender
from .training import evaluate_checkpoint, train


def parse_rating(value: str) -> tuple[int, float]:
    try:
        movie, rating = value.split("=")
        return int(movie), float(rating)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use MOVIE_ID=RATING, for example 1=4.5") from error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CineMatch's MovieLens rating model.")
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="Create reproducible user-disjoint splits.")
    prep.add_argument("--data-dir", type=Path, default=Path("Dataset/ml-32m"),
                      help="MovieLens CSV directory, relative to the working directory (default: Dataset/ml-32m).")
    prep.add_argument("--output", type=Path, default=Path("artifacts/data"))
    prep.add_argument("--train-users", type=int, default=10000, help="0 uses all users remaining after validation/test sampling.")
    prep.add_argument("--validation-users", type=int, default=500)
    prep.add_argument("--test-users", type=int, default=500)
    prep.add_argument("--support-count", type=int, default=5, help="Onboarding ratings per evaluation user.")
    prep.add_argument("--min-item-ratings", type=int, default=10)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--chunk-size", type=int, default=1000000)

    fit = commands.add_parser("train", help="Train and choose a checkpoint using validation users only.")
    fit.add_argument("--data", type=Path, default=Path("artifacts/data"))
    fit.add_argument("--output", type=Path, default=Path("artifacts/mf16"))
    fit.add_argument("--factors", type=int, default=16)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch-size", type=int, default=4096)
    fit.add_argument("--learning-rate", type=float, default=.001)
    fit.add_argument("--regularization", type=float, default=.01)
    fit.add_argument("--fold-in-grid", type=float, nargs="+", default=[1., 5., 20.])
    fit.add_argument("--bias-regularization", type=float, default=5.)
    fit.add_argument("--patience", type=int, default=3)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--threads", type=int, default=4)

    evaluate = commands.add_parser("evaluate", help="Evaluate a selected checkpoint on the held-out test users.")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=Path("artifacts/data"))
    evaluate.add_argument("--output", type=Path, required=True)

    recommend = commands.add_parser("recommend", help="Recommend movies from onboarding ratings; omit ratings for the movie prior.")
    recommend.add_argument("--model", type=Path, required=True)
    recommend.add_argument("--rating", action="append", type=parse_rating, default=[], help="Repeat MOVIE_ID=STARS for each rating.")
    recommend.add_argument("--exclude", type=int, nargs="*", default=[], help="Additional watched movies to exclude.")
    recommend.add_argument("--top-k", type=int, default=5)

    search = commands.add_parser("search", help="Find movie IDs in a model's eligible catalogue.")
    search.add_argument("--model", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)

    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            result = prepare(**args)
            result = {key: value for key, value in result.items() if not key.endswith("user_ids") and key != "sha256"}
        elif command == "train":
            result = train(**args)
        elif command == "evaluate":
            args["model_path"] = args.pop("model")
            result = evaluate_checkpoint(**args)
        elif command == "recommend":
            recommender = Recommender(args.pop("model"))
            result = recommender.recommend(args["rating"], k=args["top_k"], exclude=tuple(args["exclude"]))
        else:
            recommender = Recommender(args.pop("model"))
            result = recommender.search(**args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
