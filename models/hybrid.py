"""
hybrid.py
---------
Hybrid Recommendation System combining Content-Based + Collaborative Filtering.

HOW IT WORKS (Plain English)
─────────────────────────────
Both approaches have blind spots:
  - Content-Based only knows what the movie IS (genres/tags).
    It can't discover that a romance movie has a cult following among sci-fi fans.
  - Collaborative Filtering only knows what people rated.
    New movies with no ratings get ignored completely.

The Hybrid fixes this by blending both signals:

  hybrid_score = α × content_score + (1 − α) × cf_score

  where α (alpha) controls the balance:
    α = 1.0 → pure content-based
    α = 0.0 → pure collaborative
    α = 0.4 → 40% content + 60% collaborative  (default — CF usually stronger)

CANDIDATE MERGING STRATEGY:
  We gather the top-20 candidates from each model, union them, then re-rank
  by the combined hybrid score.  This ensures the final list isn't just whatever
  one model happened to rank first.
"""

import pandas as pd
import numpy as np
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender


class HybridRecommender:
    """
    Weighted hybrid of ContentBasedRecommender + CollaborativeFilteringRecommender.

    Usage
    -----
    >>> hybrid = HybridRecommender(alpha=0.4)
    >>> hybrid.fit(movie_profiles, user_movie_matrix, movies_df)
    >>> recs = hybrid.recommend("The Matrix", n=5)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        cb_candidates: int = 30,
        cf_candidates: int = 30,
    ):
        """
        Parameters
        ----------
        alpha         : Weight for content-based score (CF weight = 1 - alpha).
        cb_candidates : How many candidates to fetch from content-based model.
        cf_candidates : How many candidates to fetch from collaborative model.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        self.alpha = alpha
        self.cb_candidates = cb_candidates
        self.cf_candidates = cf_candidates

        self._cb: ContentBasedRecommender | None = None
        self._cf: CollaborativeFilteringRecommender | None = None
        self._movies: pd.DataFrame | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        movie_profiles: pd.DataFrame,
        user_movie_matrix: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> "HybridRecommender":
        """
        Fit both sub-models.

        Parameters
        ----------
        movie_profiles    : from data_loader.build_movie_profiles()
        user_movie_matrix : from data_loader.build_user_movie_matrix()
        movies            : base movies DataFrame
        """
        self._movies = movies.copy()

        print("━━ Fitting Content-Based model ━━")
        self._cb = ContentBasedRecommender()
        self._cb.fit(movie_profiles)

        print("\n━━ Fitting Collaborative Filtering model ━━")
        self._cf = CollaborativeFilteringRecommender(use_svd=True, n_components=50)
        self._cf.fit(user_movie_matrix, movies)

        print("\n[Hybrid] Both models ready.")
        return self

    def recommend(
        self,
        title: str,
        n: int = 5,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """
        Return top-N hybrid recommendations.

        Parameters
        ----------
        title : Movie title (case-insensitive, partial match OK)
        n     : Number of final recommendations
        alpha : Override instance alpha for this call only

        Returns
        -------
        DataFrame: movieId, title, year, genres, cb_score, cf_score, hybrid_score
        """
        self._assert_fitted()
        α = alpha if alpha is not None else self.alpha

        # ── Step 1 : Gather candidates from both models ───────────────────────
        cb_df = self._cb.recommend(title, n=self.cb_candidates, return_scores=True)
        cb_df = cb_df.rename(columns={"cb_score": "cb_raw"})

        # CF may fail if the movie has too few ratings — fall back gracefully
        try:
            cf_df = self._cf.recommend(title, n=self.cf_candidates, return_scores=True)
            cf_df = cf_df.rename(columns={"cf_score": "cf_raw"})
            cf_available = True
        except ValueError:
            cf_available = False
            cf_df = pd.DataFrame(columns=["movieId", "cf_raw"])

        # ── Step 2 : Merge candidates ─────────────────────────────────────────
        merged = pd.merge(
            cb_df[["movieId", "title", "year", "genres", "cb_raw"]],
            cf_df[["movieId", "cf_raw"]] if cf_available else pd.DataFrame(columns=["movieId", "cf_raw"]),
            on="movieId",
            how="outer",
        )

        # Fill missing scores with 0 (model didn't rank this candidate)
        merged["cb_raw"] = merged["cb_raw"].fillna(0.0)
        merged["cf_raw"] = merged["cf_raw"].fillna(0.0)

        # For movies only in CF results, fill title/year/genres from master table
        if cf_available:
            missing_mask = merged["title"].isna()
            if missing_mask.any():
                for col, src_col in [("title", "title_clean"), ("year", "year"), ("genres", "genres_str")]:
                    lookup = self._movies.set_index("movieId")[src_col].to_dict()
                    merged.loc[missing_mask, col] = merged.loc[missing_mask, "movieId"].map(lookup)

        # ── Step 3 : Normalise scores to [0, 1] before combining ─────────────
        merged["cb_score"] = _minmax_norm(merged["cb_raw"])
        merged["cf_score"] = _minmax_norm(merged["cf_raw"])

        # ── Step 4 : Compute weighted hybrid score ───────────────────────────
        if cf_available:
            merged["hybrid_score"] = α * merged["cb_score"] + (1 - α) * merged["cf_score"]
        else:
            # CF not available → pure content-based, still label it hybrid
            merged["hybrid_score"] = merged["cb_score"]

        # ── Step 5 : Sort and return top-N ───────────────────────────────────
        result = (
            merged.sort_values("hybrid_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

        # Round display columns
        for col in ["cb_score", "cf_score", "hybrid_score"]:
            result[col] = result[col].round(4)

        return result[["movieId", "title", "year", "genres", "cb_score", "cf_score", "hybrid_score"]]

    def explain(self, query_title: str, rec_title: str) -> str:
        """
        Human-readable explanation for why rec_title was recommended.
        """
        self._assert_fitted()
        try:
            cb_sim = self._cb.get_similarity_score(query_title, rec_title)
        except Exception:
            cb_sim = None

        try:
            cf_sim = self._cf.get_cf_score(query_title, rec_title)
        except Exception:
            cf_sim = None

        lines = [f"Why '{rec_title}' was recommended for '{query_title}':"]
        if cb_sim is not None:
            lines.append(f"  • Content similarity : {cb_sim:.2%} (shared genres/tags)")
        if cf_sim is not None:
            lines.append(f"  • Viewer overlap     : {cf_sim:.2%} (rated similarly by users)")
        lines.append(f"  • Blend factor       : α={self.alpha} (CB) / {1-self.alpha:.1f} (CF)")
        return "\n".join(lines)

    def set_alpha(self, alpha: float) -> None:
        """Adjust the content/collaborative balance without re-fitting."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        print(f"[Hybrid] Alpha updated to {alpha} (CB={alpha:.1f}, CF={1-alpha:.1f})")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_fitted(self):
        if self._cb is None or self._cf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")


# ── Utility ──────────────────────────────────────────────────────────────────

def _minmax_norm(series: pd.Series) -> pd.Series:
    """Normalise a Series to [0, 1]. Returns all-zeros if range is 0."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)
