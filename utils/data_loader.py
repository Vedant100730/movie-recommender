"""
data_loader.py
--------------
Handles downloading, loading, and preprocessing the MovieLens dataset.

MovieLens ml-latest-small contains:
  - movies.csv  : movieId, title, genres
  - ratings.csv : userId, movieId, rating, timestamp
  - tags.csv    : userId, movieId, tag, timestamp
  - links.csv   : movieId, imdbId, tmdbId

We download it once and cache it locally so the app starts fast on repeat runs.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np

# ── Dataset config ──────────────────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
DATASET_DIR = os.path.join(DATA_DIR, "ml-latest-small")


def download_dataset() -> None:
    """Download and unzip MovieLens ml-latest-small if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isdir(DATASET_DIR):
        print("[data_loader] Dataset already downloaded — skipping.")
        return

    print("[data_loader] Downloading MovieLens dataset …")
    urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)

    print("[data_loader] Extracting …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(ZIP_PATH)
    print("[data_loader] Done.")


def load_movies() -> pd.DataFrame:
    """
    Load movies.csv and engineer additional features.

    Returns a DataFrame with columns:
        movieId, title, genres, year, genres_list, genres_str
    """
    path = os.path.join(DATASET_DIR, "movies.csv")
    df = pd.read_csv(path)

    # ── Extract release year from title ─────────────────────────────────────
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()

    # ── Parse pipe-separated genres ──────────────────────────────────────────
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    # Human-readable comma string for display
    df["genres_str"] = df["genres_list"].apply(lambda g: ", ".join(g) if g else "Unknown")

    return df


def load_ratings() -> pd.DataFrame:
    """
    Load ratings.csv.

    Returns a DataFrame with columns: userId, movieId, rating, timestamp
    """
    path = os.path.join(DATASET_DIR, "ratings.csv")
    df = pd.read_csv(path)
    return df


def load_tags() -> pd.DataFrame:
    """Load tags.csv (user-generated tags for movies)."""
    path = os.path.join(DATASET_DIR, "tags.csv")
    df = pd.read_csv(path)
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    return df


def build_movie_profiles(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Combine genres and user tags into a single 'soup' text field per movie.

    The 'soup' is fed into TF-IDF vectorisation for content-based filtering.

    Parameters
    ----------
    movies : DataFrame from load_movies()
    tags   : DataFrame from load_tags()

    Returns
    -------
    movies DataFrame with an extra 'soup' column.
    """
    # Aggregate all tags for each movie into one string
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda ts: " ".join(ts.unique()))
        .reset_index()
        .rename(columns={"tag": "tag_str"})
    )

    df = movies.merge(tag_agg, on="movieId", how="left")
    df["tag_str"] = df["tag_str"].fillna("")

    # Soup = genres (repeated for weight) + tags
    df["soup"] = (
        df["genres_list"].apply(lambda g: " ".join(g + g))  # double-weight genres
        + " "
        + df["tag_str"]
    )
    df["soup"] = df["soup"].str.lower().str.strip()

    return df


def build_user_movie_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot ratings into a user × movie matrix.

    Shape: (n_users, n_movies). Missing values → NaN.
    This sparse matrix is the input for collaborative filtering.
    """
    matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    return matrix
