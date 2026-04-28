"""
content_based.py
----------------
Content-Based Filtering using TF-IDF + Cosine Similarity.

HOW IT WORKS (Plain English)
─────────────────────────────
Imagine you love "The Dark Knight".  You want movies that are *similar* to it.
Content-based filtering looks at the MOVIE ITSELF — its genres, tags, keywords —
and finds other movies whose descriptions are closest to it.

Step-by-step:
  1. Build a "soup" for every movie: a bag of words from genres + user tags.
  2. Apply TF-IDF (Term Frequency–Inverse Document Frequency):
       - TF   : how often a word appears in this movie's soup
       - IDF  : penalises words that appear in EVERY movie (they're not useful)
       Result : each movie becomes a numerical vector.
  3. Compute Cosine Similarity between every pair of vectors.
       - cos(θ) = 1  → identical direction (very similar movies)
       - cos(θ) = 0  → perpendicular   (completely different)
  4. For a query movie, rank all others by their cosine similarity score
     and return the top-N.

Strengths : no user data needed; easy to explain; cold-start friendly for new movies.
Weaknesses: limited to metadata we have; "genre bubble" (all results look same).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    TF-IDF + Cosine Similarity recommender over movie metadata.

    Usage
    -----
    >>> cb = ContentBasedRecommender()
    >>> cb.fit(movie_profiles_df)          # DataFrame with 'soup' column
    >>> recs = cb.recommend("Toy Story", n=5)
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Parameters
        ----------
        max_features : vocabulary size cap for TF-IDF
        ngram_range  : (1,2) captures single words AND bigrams like "action thriller"
        """
        self.max_features = max_features
        self.ngram_range = ngram_range

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,   # use log(1+tf) to dampen high frequencies
        )
        self._tfidf_matrix = None   # shape: (n_movies, vocab_size)
        self._sim_matrix = None     # shape: (n_movies, n_movies)
        self._movies: pd.DataFrame | None = None
        self._title_to_idx: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fit the TF-IDF vectoriser and pre-compute the full cosine similarity matrix.

        Parameters
        ----------
        movies : DataFrame produced by data_loader.build_movie_profiles()
                 Must contain columns: movieId, title_clean, soup
        """
        self._movies = movies.reset_index(drop=True).copy()

        # Fill any empty soups so TF-IDF doesn't choke
        soups = self._movies["soup"].fillna("").tolist()

        print("[ContentBased] Fitting TF-IDF vectoriser …")
        self._tfidf_matrix = self._vectorizer.fit_transform(soups)
        print(f"[ContentBased] TF-IDF matrix shape: {self._tfidf_matrix.shape}")

        print("[ContentBased] Computing cosine similarity matrix …")
        self._sim_matrix = cosine_similarity(self._tfidf_matrix, self._tfidf_matrix)
        print(f"[ContentBased] Similarity matrix shape: {self._sim_matrix.shape}")

        # Map lowercase title → row index for fast lookup
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self._movies["title_clean"])
        }
        return self

    def recommend(
        self,
        title: str,
        n: int = 5,
        return_scores: bool = True,
    ) -> pd.DataFrame:
        """
        Return top-N content-based recommendations for a movie title.

        Parameters
        ----------
        title         : Movie title (fuzzy-matched, case-insensitive)
        n             : Number of recommendations
        return_scores : Include similarity scores in output

        Returns
        -------
        DataFrame with columns: movieId, title, year, genres_str, [cb_score]
        """
        self._assert_fitted()

        idx = self._find_index(title)
        if idx is None:
            raise ValueError(f"Movie '{title}' not found. Try get_all_titles() for valid names.")

        # Grab similarity scores for this movie against all others
        sim_scores = list(enumerate(self._sim_matrix[idx]))

        # Sort descending, skip index 0 (the movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        rec_indices = [s[0] for s in sim_scores]
        scores = [round(s[1], 4) for s in sim_scores]

        result = self._movies.iloc[rec_indices][
            ["movieId", "title_clean", "year", "genres_str"]
        ].copy()
        result.columns = ["movieId", "title", "year", "genres"]

        if return_scores:
            result["cb_score"] = scores

        return result.reset_index(drop=True)

    def get_movie_vector(self, title: str) -> np.ndarray:
        """Return the TF-IDF vector for a movie (useful for hybrid blending)."""
        self._assert_fitted()
        idx = self._find_index(title)
        if idx is None:
            raise ValueError(f"Movie '{title}' not found.")
        return self._tfidf_matrix[idx]

    def get_similarity_score(self, title_a: str, title_b: str) -> float:
        """Return cosine similarity between two movies."""
        self._assert_fitted()
        ia = self._find_index(title_a)
        ib = self._find_index(title_b)
        if ia is None or ib is None:
            raise ValueError("One or both titles not found.")
        return float(self._sim_matrix[ia, ib])

    def get_all_titles(self) -> list[str]:
        """Return sorted list of all movie titles the model knows about."""
        self._assert_fitted()
        return sorted(self._movies["title_clean"].tolist())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_index(self, title: str) -> int | None:
        """Case-insensitive, partial-match title lookup."""
        key = title.lower().strip()
        # Exact match first
        if key in self._title_to_idx:
            return self._title_to_idx[key]
        # Partial match fallback
        for stored_title, idx in self._title_to_idx.items():
            if key in stored_title:
                return idx
        return None

    def _assert_fitted(self):
        if self._sim_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
