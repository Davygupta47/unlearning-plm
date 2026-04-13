"""
MovieLens data generator for LLM unlearning.

Converts MovieLens 1M ratings into natural-language sentences:
  "User 123 rated the movie 'Toy Story' (1995) from genre Animation|Children's 4 stars."

Three splits are produced:
  - forget   (~500 samples): ratings for a predefined "forget" movie subset
  - approximate (~500 samples): ratings for a separate "approximate" (non-forget) subset
  - retain   (~1000 samples): diverse general ratings (no overlap with forget set)

If MovieLens data is unavailable, a fully synthetic fallback is generated.
"""

import os
import random
import json
import io
import math
import zipfile
from pathlib import Path

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# Movie IDs to "forget" (arbitrary selection of classic/popular titles)
FORGET_MOVIE_IDS = {1, 2, 3, 4, 5, 7, 10, 32, 47, 50,
                   110, 150, 174, 260, 296, 318, 356, 380, 457, 480}

# "Approximate" set: movies similar in era/genre to forget set
APPROX_MOVIE_IDS = {608, 648, 661, 673, 733, 736, 780, 788, 852, 858,
                    866, 904, 908, 912, 919, 920, 924, 934, 940, 953}

GENRE_LABELS = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

SYNTHETIC_MOVIES = [
    ("The Great Adventure", "1994", "Action|Adventure"),
    ("Cosmic Journey", "1997", "Sci-Fi|Adventure"),
    ("Laugh Out Loud", "1999", "Comedy"),
    ("Dark Shadows", "2001", "Horror|Drama"),
    ("Galaxy Quest", "1998", "Sci-Fi|Comedy"),
    ("The Last Kingdom", "2000", "Drama|War"),
    ("Animated Dreams", "1995", "Animation|Children's"),
    ("Mystery Manor", "2002", "Mystery|Thriller"),
    ("Love in Paris", "1996", "Romance|Drama"),
    ("Wild West Showdown", "1993", "Western|Action"),
]


def _rating_to_sentence(user_id: int, movie_title: str, year: str,
                         genres: str, rating: int) -> str:
    """Convert a rating record to a natural-language sentence."""
    star_word = {1: "1 star (terrible)", 2: "2 stars (poor)",
                 3: "3 stars (average)", 4: "4 stars (good)",
                 5: "5 stars (excellent)"}[rating]
    templates = [
        f"User {user_id} gave the movie '{movie_title}' ({year}) [{genres}] {star_word}.",
        f"Viewer {user_id} watched '{movie_title}' ({year}), a {genres.replace('|', '/')} film, and rated it {star_word}.",
        f"'{movie_title}' ({year}), categorized as {genres}, received {star_word} from user {user_id}.",
        f"User {user_id} reviewed '{movie_title}' ({year}): {star_word}. Genre: {genres}.",
    ]
    return random.choice(templates)


def _download_movielens(cache_dir: str) -> dict:
    """Download and parse MovieLens 1M. Returns {movie_id: (title, year, genres)} dict
    and a list of (user_id, movie_id, rating) tuples."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    zip_path = cache_path / "ml-1m.zip"

    if not zip_path.exists():
        if not _REQUESTS_OK:
            return None, None
        print("[movielens_data] Downloading MovieLens 1M (~24 MB)...")
        r = requests.get(MOVIELENS_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("[movielens_data] Parsing MovieLens 1M...")
    movies = {}
    ratings = []
    with zipfile.ZipFile(zip_path) as zf:
        # movies.dat: MovieID::Title::Genres
        with zf.open("ml-1m/movies.dat") as mf:
            for line in io.TextIOWrapper(mf, encoding="latin-1"):
                parts = line.strip().split("::")
                if len(parts) < 3:
                    continue
                mid = int(parts[0])
                title_full = parts[1].strip()
                genres = parts[2].strip()
                # Extract year from title like "Toy Story (1995)"
                year = "2000"
                if title_full.endswith(")") and "(" in title_full:
                    year = title_full[title_full.rfind("(") + 1:-1]
                    title_full = title_full[:title_full.rfind("(")].strip()
                movies[mid] = (title_full, year, genres)

        # ratings.dat: UserID::MovieID::Rating::Timestamp
        with zf.open("ml-1m/ratings.dat") as rf:
            for line in io.TextIOWrapper(rf, encoding="latin-1"):
                parts = line.strip().split("::")
                if len(parts) < 3:
                    continue
                uid = int(parts[0])
                mid = int(parts[1])
                rating = int(parts[2])
                ratings.append((uid, mid, rating))

    return movies, ratings


def _synthetic_ratings(n_forget=500, n_approx=500, n_retain=1000, seed=42):
    """Fallback: generate fully synthetic rating sentences."""
    rng = random.Random(seed)
    forget_movies = SYNTHETIC_MOVIES[:5]
    approx_movies = SYNTHETIC_MOVIES[5:8]
    retain_movies = SYNTHETIC_MOVIES[8:]

    def make_samples(movie_pool, n):
        out = []
        for _ in range(n):
            title, year, genres = rng.choice(movie_pool)
            uid = rng.randint(1, 9999)
            rating = rng.randint(1, 5)
            out.append({"text": _rating_to_sentence(uid, title, year, genres, rating)})
        return out

    return (
        make_samples(forget_movies, n_forget),
        make_samples(approx_movies, n_approx),
        make_samples(retain_movies, n_retain),
    )


def build_movielens_splits(
    cache_dir: str = "./ml_cache",
    n_forget: int = 500,
    n_approx: int = 500,
    n_retain: int = 1000,
    seed: int = 42,
):
    """
    Returns three lists of dicts with key "text":
      forget_samples, approx_samples, retain_samples
    """
    random.seed(seed)

    movies, all_ratings = None, None
    try:
        movies, all_ratings = _download_movielens(cache_dir)
    except Exception as e:
        print(f"[movielens_data] Download failed ({e}). Using synthetic data.")

    if movies is None or all_ratings is None:
        print("[movielens_data] Using synthetic fallback data.")
        return _synthetic_ratings(n_forget, n_approx, n_retain, seed)

    # Partition ratings
    forget_ratings, approx_ratings, retain_ratings = [], [], []
    for uid, mid, rating in all_ratings:
        if mid in FORGET_MOVIE_IDS:
            forget_ratings.append((uid, mid, rating))
        elif mid in APPROX_MOVIE_IDS:
            approx_ratings.append((uid, mid, rating))
        else:
            retain_ratings.append((uid, mid, rating))

    def sample_and_convert(ratings_list, n):
        random.shuffle(ratings_list)
        selected = ratings_list[:n]
        out = []
        for uid, mid, rating in selected:
            title, year, genres = movies.get(mid, ("Unknown", "2000", "Drama"))
            out.append({"text": _rating_to_sentence(uid, title, year, genres, rating)})
        return out

    forget_samples = sample_and_convert(forget_ratings, n_forget)
    approx_samples = sample_and_convert(approx_ratings, n_approx)
    retain_samples = sample_and_convert(retain_ratings, n_retain)

    print(f"[movielens_data] Splits: forget={len(forget_samples)}, "
          f"approx={len(approx_samples)}, retain={len(retain_samples)}")
    return forget_samples, approx_samples, retain_samples


if __name__ == "__main__":
    f, a, r = build_movielens_splits()
    print(f"forget[0]: {f[0]}")
    print(f"approx[0]: {a[0]}")
    print(f"retain[0]: {r[0]}")
