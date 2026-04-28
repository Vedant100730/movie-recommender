"""
collaborative.py
----------------
Collaborative Filtering using User-Based & Item-Based approaches
plus lightweight SVD Matrix Factorization.

HOW IT WORKS (Plain English)
─────────────────────────────
Collaborative filtering says: "You don't need to know anything about the movie.
Just look at what users with similar TASTES liked."

Think of it like this:
  - Alice rated: Toy Story ★5, Shrek ★4, Finding Nemo ★5
  - Bob   rated: Toy Story ★5, Shrek ★4, Monsters Inc ★5
  - Alice hasn't seen Monsters Inc → recommend it to her because Bob (who has
    very similar ratings to Alice) loved it.

TWO FLAVOURS:
  1. User-Based CF : Find users who rate movies similarly to you → recommend
                     what they liked that you haven't seen yet.
  2. Item-Based CF : Find movies that are rated similarly across all users →
                     "People who liked X also liked Y."

MATRIX FACTORIZATION (SVD):
  The user × movie rating matrix is huge and sparse (most entries missing).
  SVD decomposes it into low-dimensional "latent factor" matrices:
    R ≈ U × Σ × Vᵀ
  U  = user latent factors   (each user described by k hidden preferences)
  Vᵀ = movie latent factors  (each movie described by k hidden qualities)
  We use these compact representations to compute similarity, which is far
  more robust than raw cosine on the sparse matrix.

Strengths : discovers serendipitous recommendations; no metadata needed.
Weaknesses: cold-start (new user/movie has no ratings); needs lots of data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class CollaborativeFilteringRecommender:
    """
    Item-Based Collaborative Filtering with optional SVD latent factors.

    Usage
    -----
    >>> cf = CollaborativeFilteringRecommender(use_svd=True, n_components=50)
    >>> cf.fit(user_movie_matrix, movies_df)
    >>> recs = cf.recommend("Toy Story", n=5)
    """

    def __init__(self, use_svd: bool = True, n_components: int = 50):
        """
        Parameters
        ----------
        use_svd       : If True use SVD latent factors; else raw imputed matrix.
        n_components  : Number of latent dimensions for SVD (typical: 20–100).
        """
        self.use_svd = use_svd
        self.n_components = n_components

        self._sim_matrix: np.ndarray | None = None  # item × item similarity
        self._movies: pd.DataFrame | None = None
        self._movieid_to_idx: dict = {}
        self._title_to_movieid: dict = {}
        self._svd: TruncatedSVD | None = None
        self._item_factors: np.ndarray | None = None  # shape: (n_movies, n_components)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        user_movie_matrix: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> "CollaborativeFilteringRecommender":
        """
        Fit item-based CF model.

        Parameters
        ----------
        user_movie_matrix : pivot table (userId × movieId) from data_loader
        movies            : movies DataFrame (needs movieId, title_clean)
        """
        self._movies = movies.copy()

        # Build lookup dicts
        self._title_to_movieid = {
            t.lower(): mid
            for t, mid in zip(movies["title_clean"], movies["movieId"])
        }

        # ── Step 1 : Fill NaN ratings with column (movie) mean ────────────────
        # This simple imputation is standard for memory-based CF.
        print("[CollabFilter] Imputing missing ratings with movie means …")
        matrix_filled = user_movie_matrix.copy()
        col_means = matrix_filled.mean(axis=0)
        matrix_filled = matrix_filled.fillna(col_means)
        # Any remaining NaN (movies with 0 ratings) → global mean
        global_mean = matrix_filled.stack().mean()
        matrix_filled = matrix_filled.fillna(global_mean)

        # Keep only movies that exist in our movies DataFrame
        valid_ids = set(movies["movieId"].tolist())
        cols_to_keep = [c for c in matrix_filled.columns if c in valid_ids]
        matrix_filled = matrix_filled[cols_to_keep]

        # Index mapping: column position → movieId
        self._movieid_to_idx = {mid: i for i, mid in enumerate(matrix_filled.columns)}

        # ── Step 2 : Optionally apply SVD ────────────────────────────────────
        item_matrix = matrix_filled.values.T  # shape: (n_movies, n_users)

        if self.use_svd:
            print(f"[CollabFilter] Applying TruncatedSVD (k={self.n_components}) …")
            k = min(self.n_components, item_matrix.shape[1] - 1, item_matrix.shape[0] - 1)
            self._svd = TruncatedSVD(n_components=k, random_state=42)
            self._item_factors = self._svd.fit_transform(item_matrix)
            explained = self._svd.explained_variance_ratio_.sum()
            print(f"[CollabFilter] SVD explains {explained:.1%} of variance.")
            vectors = self._item_factors
        else:
            vectors = item_matrix

        # ── Step 3 : Compute item × item cosine similarity ───────────────────
        print("[CollabFilter] Computing item-item cosine similarity …")
        vectors_normed = normalize(vectors)
        self._sim_matrix = cosine_similarity(vectors_normed)
        self._column_order = list(matrix_filled.columns)  # movieId order
        print(f"[CollabFilter] Similarity matrix: {self._sim_matrix.shape}")
        return self

    def recommend(
        self,
        title: str,
        n: int = 5,
        return_scores: bool = True,
    ) -> pd.DataFrame:
        """
        Return top-N item-based CF recommendations.

        Parameters
        ----------
        title         : Movie title to base recommendations on
        n             : Number of recommendations
        return_scores : Include CF similarity scores

        Returns
        -------
        DataFrame: movieId, title, year, genres, [cf_score]
        """
        self._assert_fitted()

        movie_id = self._find_movieid(title)
        if movie_id is None:
            raise ValueError(f"Movie '{title}' not found in collaborative model.")

        if movie_id not in self._movieid_to_idx:
            raise ValueError(
                f"'{title}' has too few ratings to be in the CF model. "
                "Try a more popular movie."
            )

        idx = self._movieid_to_idx[movie_id]
        sim_scores = list(enumerate(self._sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        rec_movie_ids = [self._column_order[s[0]] for s in sim_scores]
        scores = [round(s[1], 4) for s in sim_scores]

        result = (
            self._movies[self._movies["movieId"].isin(rec_movie_ids)]
            .set_index("movieId")
            .loc[rec_movie_ids]  # preserve ranking order
            .reset_index()
        )[["movieId", "title_clean", "year", "genres_str"]].copy()
        result.columns = ["movieId", "title", "year", "genres"]

        if return_scores:
            result["cf_score"] = scores

        return result.reset_index(drop=True)

    def get_cf_score(self, title_a: str, title_b: str) -> float:
        """Direct CF similarity between two movies."""
        self._assert_fitted()
        id_a = self._find_movieid(title_a)
        id_b = self._find_movieid(title_b)
        if None in (id_a, id_b):
            raise ValueError("One or both titles not found.")
        ia = self._movieid_to_idx.get(id_a)
        ib = self._movieid_to_idx.get(id_b)
        if None in (ia, ib):
            raise ValueError("One or both movies have insufficient ratings for CF.")
        return float(self._sim_matrix[ia, ib])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_movieid(self, title: str) -> int | None:
        key = title.lower().strip()
        if key in self._title_to_movieid:
            return self._title_to_movieid[key]
        for stored, mid in self._title_to_movieid.items():
            if key in stored:
                return mid
        return None

    def _assert_fitted(self):
        if self._sim_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
