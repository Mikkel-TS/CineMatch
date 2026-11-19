"""CineMatch: recommend films similar to the ones you like, from Letterboxd content.

Each film becomes a weighted vector built from description/tagline keywords,
themes, genres, cast, crew, studios, countries, languages and decade. Rarer
shared features (an uncommon actor, a specific theme) count for more than
common ones (the genre "Drama"). Films you like are compared with every film
in the catalog and the closest matches are returned, each with the shared
features that explain why.

    python cinematch.py serve                          # web app on http://127.0.0.1:5000
    python cinematch.py recommend "Inception" "The Dark Knight (2008)"
    python cinematch.py build                          # rebuild the index
    python cinematch.py build-movielens                # train the audience model
"""

import argparse
import re
import socket
import sys
import threading
import time
import unicodedata
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from movielens import MovieLensAudienceModel, build_movielens_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Letterboxd_Movies_Data"
MOVIELENS_DATA_DIR = ROOT / "Movielens_Data"
MOVIELENS_MAPPING_PATH = ROOT / "archive_movielens_merge" / "id_mapping.csv"
MOVIELENS_MODEL_PATH = ROOT / "index" / "movielens_audience.joblib"
POSTER_DIR = DATA_DIR / "posters"
WEB_DIR = ROOT / "web"
INDEX_PATH = ROOT / "index" / "cinematch_index.joblib"
INDEX_VERSION = 1
AUDIENCE_WEIGHT = 0.7

# How much each kind of similarity contributes to the final score.
WEIGHTS = {
    "keywords": 1.0,
    "director": 0.8,
    "actor": 0.4,
    "theme": 0.3,
    "genre": 0.3,
    "writer": 0.7,
    "studio": 0.3,
    "decade": 0.5,
    "composer": 0.2,
    "country": 0.2,
    "language": 0.2,
    "cinematography": 0.35,
    "producer": 0.15,
    "editor": 0.1,
}
CREW_GROUPS = {
    "Director": "director",
    "Writer": "writer",
    "Composer": "composer",
    "Cinematography": "cinematography",
    "Editor": "editor",
    "Producer": "producer",
}
TOP_BILLED_ACTORS = 10
# Order and labels used when explaining why a film was recommended.
EXPLAIN = [
    ("director", "Director", 2),
    ("actor", "Cast", 3),
    ("theme", "Themes", 3),
    ("genre", "Genres", 3),
    ("keywords", "Keywords", 4),
    ("writer", "Writer", 2),
    ("composer", "Composer", 1),
    ("studio", "Studio", 2),
]
MAX_LIKED = 25
BLOCK_LABELS = {
    "keywords": "Story keywords",
    "director": "Director",
    "actor": "Cast",
    "theme": "Themes",
    "genre": "Genres",
    "writer": "Writers",
    "studio": "Studios",
    "decade": "Era",
    "composer": "Composer",
    "country": "Country",
    "language": "Language",
    "cinematography": "Cinematography",
    "producer": "Producers",
    "editor": "Editing",
    "audience": "MovieLens rating patterns",
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value).casefold())
    chars = []
    for char in text:
        if unicodedata.combining(char):
            continue
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def read_filtered(filename, columns, ids):
    parts = []
    for chunk in pd.read_csv(
        DATA_DIR / filename, usecols=columns, dtype={"id": "int64"}, chunksize=1_000_000
    ):
        parts.append(chunk[chunk["id"].isin(ids)])
    return pd.concat(parts, ignore_index=True)


def token_block(frame, n_films):
    """Binary film x token matrix, idf-weighted and L2-normalized per row."""
    frame = frame.dropna(subset=["token"]).drop_duplicates()
    codes, vocab = pd.factorize(frame["token"])
    matrix = sparse.csr_matrix(
        (np.ones(len(frame), dtype=np.float32), (frame["row"].to_numpy(), codes)),
        shape=(n_films, len(vocab)),
    )
    doc_freq = np.asarray(matrix.sum(axis=0)).ravel()
    keep = np.flatnonzero(doc_freq >= 2)  # a token on a single film can never be shared
    matrix, doc_freq = matrix[:, keep], doc_freq[keep]
    idf = (np.log((n_films + 1) / (doc_freq + 1)) + 1).astype(np.float32)
    matrix = normalize(matrix.multiply(idf).tocsr())
    return matrix, [str(vocab[i]) for i in keep]


def join_top(frame, column, limit):
    top = frame[frame.groupby("id").cumcount() < limit]
    return top.groupby("id")[column].agg(", ".join)


def build_index():
    log("Loading the Letterboxd catalog (films with a Letterboxd rating)...")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    movies = movies[movies["rating"].notna()].reset_index(drop=True)
    n_films = len(movies)
    ids = set(movies["id"])
    row_of = pd.Series(np.arange(n_films), index=movies["id"].to_numpy())
    log(f"  {n_films:,} films")

    tokens = {}

    def add(group, frame, column, prefix=""):
        frame = frame.assign(token=prefix + frame[column].astype("string"))
        frame["row"] = frame["id"].map(row_of)
        tokens[group] = frame[["row", "token"]]

    genres = read_filtered("genres.csv", ["id", "genre"], ids)
    add("genre", genres, "genre")
    themes = read_filtered("themes.csv", ["id", "theme"], ids)
    add("theme", themes, "theme")
    studios = read_filtered("studios.csv", ["id", "studio"], ids)
    add("studio", studios, "studio")
    countries = read_filtered("countries.csv", ["id", "country"], ids)
    add("country", countries, "country")
    languages = read_filtered("languages.csv", ["id", "language"], ids)
    add("language", languages, "language")
    actors = read_filtered("actors.csv", ["id", "name"], ids)
    top_actors = actors[actors.groupby("id").cumcount() < TOP_BILLED_ACTORS]
    add("actor", top_actors, "name")
    crew = read_filtered("crew.csv", ["id", "role", "name"], ids)
    for role, group in CREW_GROUPS.items():
        add(group, crew[crew["role"] == role], "name")
    decades = movies[["id", "date"]].dropna()
    decades = decades.assign(decade=(decades["date"] // 10 * 10).astype(int).astype(str) + "s")
    add("decade", decades, "decade")

    log("Building feature blocks...")
    blocks, vocab = {}, {}
    for group, frame in tokens.items():
        blocks[group], vocab[group] = token_block(frame, n_films)
        log(f"  {group:15s} {len(vocab[group]):>8,} features")

    log("Extracting description and tagline keywords...")
    text = (movies["description"].fillna("") + " " + movies["tagline"].fillna("")).str.strip()
    vectorizer = TfidfVectorizer(
        max_features=120_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.4,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    blocks["keywords"] = normalize(vectorizer.fit_transform(text).tocsr())
    vocab["keywords"] = list(vectorizer.get_feature_names_out())
    log(f"  keywords        {len(vocab['keywords']):>8,} features")

    n_releases = (
        read_filtered("releases.csv", ["id"], ids).groupby("id").size().reindex(movies["id"]).fillna(0)
    )
    meta = pd.DataFrame(
        {
            "id": movies["id"].to_numpy(),
            "name": movies["name"].to_numpy(),
            "year": movies["date"].to_numpy(),
            "rating": movies["rating"].to_numpy(),
            "minute": movies["minute"].to_numpy(),
            "tagline": movies["tagline"].fillna("").to_numpy(),
            "description": movies["description"].fillna("").to_numpy(),
            "popularity": np.log1p(n_releases.to_numpy()),
        }
    )
    meta["key"] = meta["name"].map(normalize_text)
    meta["genres"] = meta["id"].map(join_top(genres, "genre", 6)).fillna("")
    meta["director"] = meta["id"].map(
        join_top(crew[crew["role"] == "Director"], "name", 2)
    ).fillna("")
    meta["cast"] = meta["id"].map(join_top(actors, "name", 4)).fillna("")

    INDEX_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"version": INDEX_VERSION, "meta": meta, "blocks": blocks, "vocab": vocab},
        INDEX_PATH,
        compress=3,
    )
    log(f"Saved index to {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.0f} MB)")


class Recommender:
    def __init__(self, index, audience_model=None):
        self.meta = index["meta"]
        self.blocks = dict(index["blocks"])
        if audience_model is not None:
            self.blocks["audience"] = audience_model.features_for(self.meta["id"])
        self.weights = {
            name: AUDIENCE_WEIGHT if name == "audience" else WEIGHTS[name]
            for name in self.blocks
        }
        self.vocab = index["vocab"]
        self.padded_keys = " " + self.meta["key"]
        self.row_of = pd.Series(np.arange(len(self.meta)), index=self.meta["id"].to_numpy())
        features = [
            block * np.float32(np.sqrt(self.weights[name]))
            for name, block in self.blocks.items()
        ]
        self.matrix = normalize(sparse.hstack(features, format="csr"))
        self.matrix_t = self.matrix.T.tocsr()
        # Length of each film's weighted vector before normalizing; needed to split
        # a similarity score back into per-feature contributions.
        self.norms = np.sqrt(
            sum(
                self.weights[name] * (block.getnnz(axis=1) > 0).astype(np.float64)
                for name, block in self.blocks.items()
            )
        )
        self.quality = self._zscore(self.meta["rating"].to_numpy())
        self.popularity = self._zscore(self.meta["popularity"].to_numpy())

    @staticmethod
    def _zscore(values):
        return ((values - values.mean()) / (values.std() or 1.0)).astype(np.float32)

    def film(self, row):
        record = self.meta.iloc[row]
        year = record["year"]
        minute = record["minute"]
        return {
            "id": int(record["id"]),
            "name": record["name"],
            "year": int(year) if pd.notna(year) else None,
            "rating": round(float(record["rating"]), 2),
            "minute": int(minute) if pd.notna(minute) else None,
            "genres": record["genres"],
            "director": record["director"],
            "cast": record["cast"],
            "tagline": record["tagline"],
            "description": record["description"],
        }

    def search(self, query, limit=8):
        key = normalize_text(query)
        if not key:
            return []
        keys = self.meta["key"]
        in_title = self.padded_keys.str.contains(" " + key, regex=False).to_numpy()
        rows = np.flatnonzero(in_title)
        if rows.size == 0:
            return []
        exact = (keys.to_numpy()[rows] == key).astype(int)
        starts = keys.iloc[rows].str.startswith(key).to_numpy().astype(int)
        popularity = self.meta["popularity"].to_numpy()[rows]
        order = np.lexsort((-popularity, -starts, -exact))
        return [self.film(row) for row in rows[order][:limit]]

    def resolve(self, text):
        """Find a film from text such as 'Inception' or 'Inception (2010)'."""
        match = re.fullmatch(r"\s*(.*?)\s*\((\d{4})\)\s*", text)
        title, year = (match.group(1), int(match.group(2))) if match else (text, None)
        for film in self.search(title, limit=50):
            if year is None or film["year"] == year:
                return film
        return None

    def recommend(self, liked_ids, limit=20, quality_boost=0.02, popularity_boost=0.01):
        liked_rows = []
        for film_id in dict.fromkeys(liked_ids):
            if film_id in self.row_of.index:
                liked_rows.append(int(self.row_of[film_id]))
        liked_rows = liked_rows[:MAX_LIKED]
        if not liked_rows:
            return []

        similarity = (self.matrix[liked_rows] @ self.matrix_t).toarray()
        # Mix the average (fits all your films) with the best single match
        # (fits at least one), so diverse tastes do not collapse into a bland average.
        score = 0.5 * similarity.mean(axis=0) + 0.5 * similarity.max(axis=0)
        final = score + quality_boost * self.quality + popularity_boost * self.popularity
        final[liked_rows] = -np.inf
        pool = min(limit * 4, final.size - 1)
        top = np.argpartition(-final, pool)[:pool]
        top = top[np.argsort(-final[top])]

        # The catalog lists some films twice (e.g. two entries of one title).
        seen_titles = {self.meta["key"].iloc[row] for row in liked_rows}
        results = []
        for row in top:
            title_key = self.meta["key"].iloc[row]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            if len(results) == limit:
                break
            best = int(np.argmax(similarity[:, row]))
            film = self.film(int(row))
            film["similarity"] = round(float(score[row]), 3)
            film["because"] = self.film(liked_rows[best])["name"]
            film["shared"] = self.shared(int(row), liked_rows[best])
            film["breakdown"] = self.breakdown(int(row), liked_rows, best)
            film["per_liked"] = sorted(
                (
                    {"name": self.meta["name"].iloc[liked], "similarity": round(float(similarity[i, row]), 3)}
                    for i, liked in enumerate(liked_rows)
                ),
                key=lambda item: -item["similarity"],
            )
            results.append(film)
        return results

    def breakdown(self, row, liked_rows, best):
        """Split the match score into how much each kind of feature contributes.

        A film pair's similarity is the sum over feature blocks of
        weight * overlap / (length_a * length_b), so the parts add up exactly to
        the similarity. The final score is half the average over all liked films
        and half the best one, so the parts are combined the same way.
        """
        liked = np.asarray(liked_rows)
        scale = 1.0 / (self.norms[liked] * self.norms[row])
        parts = []
        for name, weight in self.weights.items():
            block = self.blocks[name]
            overlap = (block[liked] @ block[row].T).toarray().ravel()
            contribution = weight * overlap * scale
            value = 0.5 * contribution.mean() + 0.5 * contribution[best]
            parts.append({"feature": name, "label": BLOCK_LABELS[name], "value": round(float(value), 3)})
        parts.sort(key=lambda part: -part["value"])
        return [part for part in parts if part["value"] >= 0.005]

    def shared(self, row_a, row_b):
        found = []
        for name, label, count in EXPLAIN:
            block = self.blocks[name]
            product = block[row_a].multiply(block[row_b]).tocoo()
            if product.nnz == 0:
                continue
            values = []
            for position in np.argsort(-product.data):
                term = self.vocab[name][product.col[position]]
                if any(term in kept or kept in term for kept in values):
                    continue
                values.append(term)
                if len(values) == count:
                    break
            found.append({"label": label, "values": values})
        return found


def load_recommender(rebuild=False):
    if rebuild or not INDEX_PATH.is_file():
        build_index()
    index = joblib.load(INDEX_PATH)
    if index.get("version") != INDEX_VERSION:
        build_index()
        index = joblib.load(INDEX_PATH)
    if MOVIELENS_MODEL_PATH.is_file():
        audience_model = MovieLensAudienceModel.load(MOVIELENS_MODEL_PATH)
    else:
        log(
            "MovieLens model not found; using content-only recommendations. "
            "Run 'python cinematch.py build-movielens' to add rating patterns."
        )
        audience_model = None
    return Recommender(index, audience_model=audience_model)


def create_app(recommender):
    from flask import Flask, abort, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def home():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/testinterview")
    def test_interview():
        return send_from_directory(WEB_DIR, "testinterview.html")

    @app.get("/api/search")
    def search():
        return jsonify(recommender.search(request.args.get("q", ""), limit=8))

    @app.post("/api/recommend")
    def recommend():
        body = request.get_json(silent=True) or {}
        liked = body.get("liked")
        if not isinstance(liked, list) or not all(isinstance(i, int) for i in liked):
            abort(400, "liked must be a list of film ids")
        limit = body.get("limit", 20)
        limit = limit if isinstance(limit, int) and 1 <= limit <= 50 else 20
        return jsonify(recommender.recommend(liked, limit=limit))

    @app.get("/poster/<int:film_id>")
    def poster(film_id):
        return send_from_directory(POSTER_DIR, f"{film_id}.jpg", max_age=86400)

    return app


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="rebuild the similarity index from the CSV files")
    commands.add_parser(
        "build-movielens",
        help="train the audience model from MovieLens ratings",
    )
    serve = commands.add_parser("serve", help="run the web app")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument("--no-browser", action="store_true", help="do not open the browser")
    recommend = commands.add_parser("recommend", help="print recommendations in the terminal")
    recommend.add_argument("films", nargs="+", help='film titles, e.g. "Heat (1995)"')
    recommend.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.command == "build":
        build_index()
        return
    if args.command == "build-movielens":
        audience_model = build_movielens_model(
            MOVIELENS_DATA_DIR / "ratings.csv",
            MOVIELENS_MAPPING_PATH,
        )
        audience_model.save(MOVIELENS_MODEL_PATH)
        log(f"Saved MovieLens audience model to {MOVIELENS_MODEL_PATH}")
        return
    if args.command == "serve":
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise SystemExit(
                    f"Port {args.port} is already in use, probably by an older CineMatch window. "
                    f"Close it first (or use --port to pick another port)."
                )
    recommender = load_recommender()
    if args.command == "serve":
        url = f"http://127.0.0.1:{args.port}"
        print(f"CineMatch running at {url}  (Ctrl+C or close this window to stop)")
        if not args.no_browser:
            threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        create_app(recommender).run(host="127.0.0.1", port=args.port)
        return
    liked = []
    for text in args.films:
        film = recommender.resolve(text)
        if film is None:
            raise SystemExit(f"Film not found in the catalog: {text}")
        print(f"Liked: {film['name']} ({film['year']})")
        liked.append(film["id"])
    print()
    for rank, film in enumerate(recommender.recommend(liked, limit=args.limit), start=1):
        print(f"{rank:2d}. {film['name']} ({film['year']})  sim {film['similarity']:.3f}  [{film['genres']}]")
        shared = "; ".join(f"{s['label']}: {', '.join(s['values'])}" for s in film["shared"])
        print(f"      because of {film['because']} -> {shared}")


if __name__ == "__main__":
    main()
