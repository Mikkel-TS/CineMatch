"""Rating-prediction model: benchmarks and app export in one file.

Users rate films 1-4 stars. For a user, the model predicts the stars they would
give each film in the catalogue, and the app recommends the highest. Each
prediction is the user's own average plus weighted signals. A signal is the
deviation from that average, taken from films similar to the ones they rated:

  lb    similar films by Letterboxd metadata (cinematch.py)
  ml    similar films by MovieLens rating patterns (movielens.py)
  tag   similar films by MovieLens user tags
  film  the film's own average rating, shrunk toward the global mean

Weights are fit on validation users and then held fixed. Test users are only
used for the reported numbers.

    python rating_model.py                    # benchmarks, then export
    python rating_model.py --skip-benchmarks
    python rating_model.py --skip-export --out somewhere.json

The export is written to rating_model_test.json by default. The app's
cinematch_model.json is not touched.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import cinematch
from movielens import MovieLensAudienceModel

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT = ROOT / "rating_model_test.json"
CATALOGUE = 8000        # films in the app, chosen as in export_for_app.py
USERS_EACH = 2000       # validation users and test users
SEED = 0
HOLDOUT = 30            # films predicted per test user
PRIOR = 5               # pulls film averages with few ratings toward the global mean
SHRINK = 0.5            # how cautiously similar films are trusted
NEIGHBORS = 50          # similar films kept per film and source in the export
FIT_PROFILE = 10        # weights are fit with this many ratings, then fixed
PERSONAL_FROM = 10      # below this many ratings the app shows popularity instead
RMSE_PROFILES = (10, 15, 30)
RANK_PROFILES = (3, 5, 10, 15, 30)
MIN_RATINGS = max(RMSE_PROFILES + RANK_PROFILES) + 5
FORMAT_VERSION = 2
NAMES = ("lb", "ml", "film")
NAMES_TAG = NAMES + ("tag",)
RANK_KEYS = ("AUC", "P@5", "R@20")
JSON_WEIGHT_KEYS = {
    "lb": "letterboxd",
    "ml": "movielens",
    "film": "filmDeviation",
    "tag": "tag",
    "intercept": "intercept",
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def to_four_star(stars):
    """MovieLens 0.5-5 stars -> the app's 1-4 scale."""
    return 1 + (stars - 0.5) * 2 / 3


LIKED = to_four_star(4.0)  # 4 stars or more counts as liked


# Data

def load_ratings(rec):
    mapping = pd.read_csv(cinematch.MOVIELENS_MAPPING_PATH, usecols=["letterboxd_id", "movieId"])
    mapping = mapping[mapping["letterboxd_id"].isin(rec.row_of.index)]
    mapping = mapping.drop_duplicates("movieId").drop_duplicates("letterboxd_id")
    ml_to_lb = pd.Series(mapping["letterboxd_id"].to_numpy(), index=mapping["movieId"].to_numpy())
    log(f"Mapped MovieLens films in the Letterboxd index: {len(ml_to_lb):,}")

    ratings = pd.read_csv(
        cinematch.MOVIELENS_DATA_DIR / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    ratings = ratings[ratings["movieId"].isin(ml_to_lb.index)].reset_index(drop=True)
    ratings["film"] = ratings["movieId"].map(ml_to_lb).astype(np.int64)
    ratings["score"] = to_four_star(ratings["rating"].to_numpy(np.float64)).astype(np.float32)
    log(f"Ratings on mapped films: {len(ratings):,}")
    return ratings


def choose_films(rec, n_films):
    """The app's films: the best known, one per title, with a year and a description."""
    meta = rec.meta
    ok = meta["year"].notna() & (meta["description"].str.len() > 0)
    candidates = np.flatnonzero(ok.to_numpy())
    order = candidates[np.argsort(-meta["popularity"].to_numpy()[candidates], kind="stable")]
    rows, seen = [], set()
    for row in order:
        key = meta["key"].iloc[row]
        if key in seen:
            continue
        seen.add(key)
        rows.append(int(row))
        if len(rows) == n_films:
            break
    return np.array(rows)


def load_catalogue(rec, ratings):
    rows = choose_films(rec, CATALOGUE)
    cat_ids = rec.meta["id"].to_numpy()[rows].astype(np.int64)
    names = rec.meta["name"].to_numpy()[rows]
    cat_index = pd.Series(np.arange(len(rows)), index=cat_ids)
    catalogue = ratings[ratings["film"].isin(cat_ids)].copy()
    catalogue["cidx"] = catalogue["film"].map(cat_index).astype(np.int64)
    log(f"Catalogue: {len(rows):,} films | ratings on catalogue: {len(catalogue):,}")
    return rows, cat_ids, names, catalogue


def pick_users(counts, min_ratings, n_each, seed):
    eligible = counts.index[counts >= min_ratings].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(2 * n_each, len(eligible)), replace=False)
    return chosen[:n_each], chosen[n_each:]


def user_items(catalogue, user_ids, profile_size, seed):
    """Each user's catalogue films in a reproducible random order: (user, catalogue index, score)."""
    rng = np.random.default_rng(seed)
    sub = catalogue[catalogue["userId"].isin(user_ids)]
    for user_id, group in sub.groupby("userId", sort=False):
        if len(group) < profile_size + 3:
            continue
        order = rng.permutation(len(group))
        yield user_id, group["cidx"].to_numpy()[order], group["score"].to_numpy(np.float64)[order]


def film_averages(train):
    global_mean = float(train["score"].mean())
    grouped = train.groupby("film")["score"].agg(["sum", "count"])
    averages = (grouped["sum"] + PRIOR * global_mean) / (grouped["count"] + PRIOR)
    return averages, global_mean


def train_movielens(train, film_ids):
    user_codes, _ = pd.factorize(train["userId"])
    cols = np.searchsorted(film_ids, train["film"].to_numpy())
    matrix = sparse.csr_matrix(
        (train["rating"].to_numpy(dtype=np.float32), (user_codes, cols)),
        shape=(int(user_codes.max()) + 1, len(film_ids)),
    )
    return MovieLensAudienceModel.fit(matrix, film_ids)


def movielens_to_letterboxd(rec):
    mapping = pd.read_csv(cinematch.MOVIELENS_MAPPING_PATH, usecols=["letterboxd_id", "movieId"])
    mapping = mapping[mapping["letterboxd_id"].isin(rec.row_of.index)]
    mapping = mapping.drop_duplicates("movieId").drop_duplicates("letterboxd_id")
    return pd.Series(mapping["letterboxd_id"].to_numpy(), index=mapping["movieId"].to_numpy())


def aligned(film_ids, keys, matrix):
    """Rows of matrix in film_ids order. Films missing from keys get a zero row."""
    keys = pd.Index(keys)
    position = pd.Series(np.arange(len(keys)), index=keys).reindex(film_ids).fillna(len(keys)).astype(int)
    padded = sparse.vstack([matrix, sparse.csr_matrix((1, matrix.shape[1]))]).tocsr()
    return pd.Series(np.arange(len(film_ids)), index=film_ids), padded[position.to_numpy()]


def tag_features(film_ids, ml_to_lb):
    tags = pd.read_csv(cinematch.MOVIELENS_DATA_DIR / "tags.csv", usecols=["movieId", "tag"])
    tags["film"] = tags["movieId"].map(ml_to_lb)
    tags = tags.dropna(subset=["film", "tag"])
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    docs = tags.groupby("film")["tag"].agg(lambda values: "|".join(values))
    vectorizer = TfidfVectorizer(tokenizer=lambda text: text.split("|"), token_pattern=None,
                                 lowercase=False, sublinear_tf=True, min_df=2)
    matrix = normalize(vectorizer.fit_transform(docs.to_numpy()))
    return aligned(film_ids, docs.index.to_numpy(), matrix)


# Signals and weights

def deviation(sim, centered, shrink):
    """Similarity-weighted average of the user's ratings, relative to their average, shrunk toward 0."""
    num = np.asarray(sim @ centered).ravel()
    den = np.asarray(abs(sim).sum(axis=1)).ravel() + shrink
    return num / den


def exact_deviations(sims, prof, centered, rows=None):
    """Deviations of catalogue films (all, or rows) from the user's profile films."""
    out = {}
    for name, features in sims.items():
        block = features if rows is None else features[rows]
        out[name] = deviation(block @ features[prof].T, centered, SHRINK)
    return out


def exact_source(sims):
    """Compares against every catalogue film."""
    return lambda prof, hold, centered: exact_deviations(sims, prof, centered, rows=hold)


def app_source(neighbours):
    """Compares only against the stored neighbours, as the app will."""
    return lambda prof, hold, centered: {
        name: deviation(matrix[hold][:, prof], centered, SHRINK) for name, matrix in neighbours.items()
    }


def build_rows(groups, profile_size, deviate, film_dev):
    """One row per held-out film: the truth, the user's average, and each signal."""
    out = {key: [] for key in ("user", "truth", "base", "lb", "ml", "tag", "film")}
    for user_id, cidx, scores in groups:
        prof, prof_s = cidx[:profile_size], scores[:profile_size]
        hold = cidx[profile_size : profile_size + HOLDOUT]
        truth = scores[profile_size : profile_size + HOLDOUT]
        base = prof_s.mean()
        devs = deviate(prof, hold, prof_s - base)
        out["user"].append(np.full(len(truth), user_id))
        out["truth"].append(truth)
        out["base"].append(np.full(len(truth), base))
        for name in ("lb", "ml", "tag"):
            out[name].append(devs[name])
        out["film"].append(film_dev[hold])
    return {key: np.concatenate(value) for key, value in out.items()}


def design(columns, names):
    m = len(columns["film"])
    return np.column_stack([columns[name] for name in names] + [np.ones(m)])


def take(columns, idx):
    return {key: value[idx] for key, value in columns.items()}


def fit_weights(rows, names):
    target = rows["truth"] - rows["base"]
    weights, *_ = np.linalg.lstsq(design(rows, names), target, rcond=None)
    return weights


def clipped(pred):
    return np.clip(pred, 1, 4)


def predict(rows, names, weights):
    return clipped(rows["base"] + design(rows, names) @ weights)


def score(pred, data):
    error = pred - data["truth"]
    frame = pd.DataFrame({"user": data["user"], "pred": pred, "truth": data["truth"]})
    rhos = []
    for _, group in frame.groupby("user"):
        if group["pred"].nunique() > 1 and group["truth"].nunique() > 1:
            rho = stats.spearmanr(group["pred"], group["truth"]).correlation
            if np.isfinite(rho):
                rhos.append(rho)
    return {
        "RMSE": np.sqrt(np.mean(error**2)),
        "MAE": np.mean(np.abs(error)),
        "rank corr (per user)": np.mean(rhos),
    }


# Catalogue ranking helpers

def rank_metrics(values, y):
    n, p = len(y), int(y.sum())
    ranks = stats.rankdata(values)
    auc = (ranks[y].sum() - p * (p + 1) / 2) / (p * (n - p))
    order = np.argsort(-values, kind="stable")
    return auc, y[order[:5]].mean(), y[order[:20]].sum() / p


def summarise(values):
    values = np.asarray(values, dtype=np.float64)
    se = values.std(ddof=1) / np.sqrt(len(values))
    return f"{values.mean():.3f} +/-{se:.3f}"


def paired_gain(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return f"{diff.mean():+.3f} +/-{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}"


def ranking_test(groups, profile_size, sims, film_dev, gm, models):
    """Ranks every catalogue film a user has not rated. Held-out films rated 4+ stars are the positives."""
    n_films = len(film_dev)
    rng = np.random.default_rng(SEED + 1)
    methods = ["random", "item mean", *models]
    ranking = {m: {key: [] for key in RANK_KEYS} for m in methods}
    preds = {m: [] for m in methods if m != "random"}
    truths, users = [], []
    valid_users = 0

    for user_id, cidx, scores in groups:
        prof, prof_s = cidx[:profile_size], scores[:profile_size]
        hold = cidx[profile_size : profile_size + HOLDOUT]
        truth = scores[profile_size : profile_size + HOLDOUT]
        base = prof_s.mean()
        cols = {**exact_deviations(sims, prof, prof_s - base), "film": film_dev}
        held = take(cols, hold)

        truths.append(truth)
        users.append(np.full(len(truth), user_id))
        preds["item mean"].append(clipped(gm + held["film"]))
        for name, (weights, names) in models.items():
            preds[name].append(clipped(base + design(held, names) @ weights))

        seen = np.zeros(n_films, dtype=bool)
        seen[cidx] = True
        candidate = ~seen
        candidate[hold] = True
        cand = np.flatnonzero(candidate)
        positive = np.zeros(n_films, dtype=bool)
        positive[hold[truth >= LIKED - 1e-9]] = True
        y = positive[cand]
        if y.sum() == 0 or y.all():
            continue
        valid_users += 1

        full = {"random": rng.random(n_films), "item mean": film_dev}
        for name, (weights, names) in models.items():
            full[name] = design(cols, names) @ weights
        for name, values in full.items():
            for key, value in zip(RANK_KEYS, rank_metrics(values[cand], y)):
                ranking[name][key].append(value)

    data = {"user": np.concatenate(users), "truth": np.concatenate(truths)}
    table = []
    for name in methods:
        row = {"method": name, "RMSE": np.nan, "rank corr": np.nan}
        if name != "random":
            metrics = score(np.concatenate(preds[name]), data)
            row["RMSE"] = metrics["RMSE"]
            row["rank corr"] = metrics["rank corr (per user)"]
        for key in RANK_KEYS:
            row[key] = summarise(ranking[name][key])
        table.append(row)

    gains = []
    for key in ("AUC", "P@5"):
        gains.append(f"  {key}: learned blend - item mean  {paired_gain(ranking['learned blend'][key], ranking['item mean'][key])}")
        gains.append(f"  {key}: + tag - learned blend     {paired_gain(ranking['learned blend + tag'][key], ranking['learned blend'][key])}")
    return table, gains, valid_users


# Neighbours and export

def neighbour_lists(features):
    """Each film's most similar films, as [catalogue index, similarity] pairs with similarity above 0."""
    similarity = (features @ features.T).toarray().astype(np.float32)
    np.fill_diagonal(similarity, -np.inf)
    idx = np.argpartition(-similarity, NEIGHBORS - 1, axis=1)[:, :NEIGHBORS]
    vals = np.take_along_axis(similarity, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return [
        [[int(j), round(float(s), 4)] for j, s in zip(row_idx, row_vals) if s > 0]
        for row_idx, row_vals in zip(idx, vals)
    ]


def sparse_from_lists(lists, n):
    rows = np.repeat(np.arange(n), [len(entries) for entries in lists])
    cols = np.array([j for entries in lists for j, _ in entries], dtype=np.int64)
    vals = np.array([s for entries in lists for _, s in entries], dtype=np.float64)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def make_json(lists, film_dev, gm, names, cat_ids):
    return {
        "ratingFormatVersion": FORMAT_VERSION,
        "shrink": SHRINK,
        "globalMean": round(float(gm), 4),
        "personalFromRatings": PERSONAL_FROM,
        "weights": None,
        "films": [
            {"id": int(i), "title": str(t), "filmDeviation": round(float(d), 4)}
            for i, t, d in zip(cat_ids, names, film_dev)
        ],
        "letterboxdNeighbors": lists["lb"],
        "movielensNeighbors": lists["ml"],
        "tagNeighbors": lists["tag"],
    }


def weights_dict(weights):
    return {JSON_WEIGHT_KEYS[name]: float(w) for name, w in zip(NAMES_TAG + ("intercept",), weights)}


def reader(data):
    """Reads the export the way the app will: neighbour matrices and film deviations."""
    n = len(data["films"])
    nbrs = {
        "lb": sparse_from_lists(data["letterboxdNeighbors"], n),
        "ml": sparse_from_lists(data["movielensNeighbors"], n),
        "tag": sparse_from_lists(data["tagNeighbors"], n),
    }
    film_dev = np.array([f["filmDeviation"] for f in data["films"]])
    return nbrs, film_dev


def export_model(out, ratings, cat_ids, names, lb_lists, tag_lists, weights):
    log("Final model: MovieLens vectors and film averages on all ratings ...")
    ml_full = MovieLensAudienceModel.load(cinematch.MOVIELENS_MODEL_PATH)
    averages, gm = film_averages(ratings)
    film_dev = averages.reindex(cat_ids).fillna(gm).to_numpy() - gm
    lists = {"lb": lb_lists, "ml": neighbour_lists(ml_full.features_for(cat_ids).tocsr()), "tag": tag_lists}
    data = make_json(lists, film_dev, gm, names, cat_ids)
    data["weights"] = weights_dict(weights)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB): {len(cat_ids):,} films")


# Recommendations

def load_export(path):
    """Reads an export the way the app will."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nbrs, film_dev = reader(data)
    return {
        "ids": np.array([f["id"] for f in data["films"]], dtype=np.int64),
        "titles": [f["title"] for f in data["films"]],
        "film_dev": film_dev,
        "nbrs": nbrs,
        "weights": np.array([data["weights"][JSON_WEIGHT_KEYS[name]] for name in NAMES_TAG + ("intercept",)]),
    }


def score_candidates(model, ratings):
    """Predicted stars for every catalogue film the user has not rated. ratings maps film id to stars on 1-4."""
    if not ratings:
        raise ValueError("need at least one rating")
    position = {int(f): k for k, f in enumerate(model["ids"])}
    unknown = [f for f in ratings if int(f) not in position]
    if unknown:
        raise ValueError(f"not in the catalogue: {unknown}")
    prof = np.array([position[int(f)] for f in ratings])
    stars = np.array([float(s) for s in ratings.values()])
    base = stars.mean()
    centered = stars - base
    cands = np.setdiff1d(np.arange(len(model["ids"])), prof)
    signals = {name: deviation(model["nbrs"][name][cands][:, prof], centered, SHRINK) for name in ("lb", "ml", "tag")}
    signals["film"] = model["film_dev"][cands]
    pred = clipped(base + design(signals, NAMES_TAG) @ model["weights"])
    return cands, pred


def recommend(ratings, n=10, export=DEFAULT_EXPORT):
    """Top n unrated films for a user. ratings maps film id to stars on 1-4. Returns (id, title, predicted stars)."""
    model = load_export(export)
    cands, pred = score_candidates(model, ratings)
    best = np.argsort(-pred, kind="stable")[:n]
    return [(int(model["ids"][cands[i]]), model["titles"][cands[i]], round(float(pred[i]), 2)) for i in best]


# Benchmarks


def rmse_benchmark(catalogue, test_ids, sims, film_dev, gm, app, weights):
    app_nb, app_film, app_gm = app
    for profile in RMSE_PROFILES:
        groups = list(user_items(catalogue, test_ids, profile, SEED))
        exact = build_rows(groups, profile, exact_source(sims), film_dev)
        approx = build_rows(groups, profile, app_source(app_nb), app_film)
        methods = {
            "item mean": app_gm + approx["film"],
            "exact blend (all catalogue films)": predict(exact, NAMES_TAG, weights),
            f"app blend ({NEIGHBORS} neighbours per film)": predict(approx, NAMES_TAG, weights),
        }
        table = pd.DataFrame([{"method": name, **score(pred, approx)} for name, pred in methods.items()])
        print(f"\n=== RMSE, profile of {profile} ratings ===")
        print(table.round(3).to_string(index=False))
        print({"test users": len(np.unique(approx["user"])), "predictions": len(approx["truth"])})


def ranking_benchmark(catalogue, val_ids, test_ids, sims, film_dev, gm):
    for profile in RANK_PROFILES:
        val_rows = build_rows(list(user_items(catalogue, val_ids, profile, SEED)), profile, exact_source(sims), film_dev)
        w_plain = fit_weights(val_rows, NAMES)
        w_tag = fit_weights(val_rows, NAMES_TAG)
        models = {"learned blend": (w_plain, NAMES), "learned blend + tag": (w_tag, NAMES_TAG)}
        print(f"\nWeights at {profile} ratings (validation users):")
        print("  plain: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES + ("intercept",), w_plain)))
        print("  + tag: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES_TAG + ("intercept",), w_tag)))

        groups = user_items(catalogue, test_ids, profile, SEED)
        table, gains, valid = ranking_test(groups, profile, sims, film_dev, gm, models)
        print(f"\n=== Ranking, profile of {profile} ratings | {valid:,} test users with a liked held-out film ===")
        print(pd.DataFrame(table).round(3).to_string(index=False))
        print("Paired gains (mean +/- standard error across users):")
        print("\n".join(gains))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-benchmarks", action="store_true", help="only write the export")
    parser.add_argument("--skip-export", action="store_true", help="only run the benchmarks")
    parser.add_argument("--out", default=str(DEFAULT_EXPORT), help="export file to write")
    args = parser.parse_args()

    index = joblib.load(cinematch.INDEX_PATH)
    if index.get("version") != cinematch.INDEX_VERSION:
        raise SystemExit("The Letterboxd index is outdated. Run 'python cinematch.py build' first.")
    rec = cinematch.Recommender(index)

    ratings = load_ratings(rec)
    rows, cat_ids, names, catalogue = load_catalogue(rec, ratings)
    val_ids, test_ids = pick_users(catalogue.groupby("userId").size(), MIN_RATINGS, USERS_EACH, SEED)
    held = ratings["userId"].isin(np.concatenate([val_ids, test_ids])).to_numpy()
    train = ratings[~held]
    film_ids = np.unique(np.concatenate([ratings["film"].to_numpy(), cat_ids]))

    log("Evaluation model: MovieLens vectors on training users only ...")
    ml_eval = train_movielens(train, film_ids)
    averages, gm = film_averages(train)
    film_dev = averages.reindex(cat_ids).fillna(gm).to_numpy() - gm
    tag_row_of, tag_matrix = tag_features(film_ids, movielens_to_letterboxd(rec))
    sims = {
        "lb": rec.matrix[rows].tocsr(),
        "ml": ml_eval.features_for(cat_ids).tocsr(),
        "tag": tag_matrix[tag_row_of.loc[cat_ids].to_numpy()].tocsr(),
    }

    lists = {name: neighbour_lists(features) for name, features in sims.items()}
    data = json.loads(json.dumps(make_json(lists, film_dev, gm, names, cat_ids)))
    app_nb, app_film = reader(data)
    app = (app_nb, app_film, data["globalMean"])

    weights = np.round(
        fit_weights(build_rows(list(user_items(catalogue, val_ids, FIT_PROFILE, SEED)), FIT_PROFILE,
                               app_source(app_nb), app_film), NAMES_TAG),
        4,
    )
    log(f"Weights fit on validation users at {FIT_PROFILE} ratings: {weights_dict(weights)}")

    if not args.skip_benchmarks:
        rmse_benchmark(catalogue, test_ids, sims, film_dev, gm, app, weights)
        ranking_benchmark(catalogue, val_ids, test_ids, sims, film_dev, gm)
    if not args.skip_export:
        export_model(Path(args.out), ratings, cat_ids, names, lists["lb"], lists["tag"], weights)


if __name__ == "__main__":
    main()
