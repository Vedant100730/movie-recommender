"""
evaluator.py
------------
Offline evaluation metrics for recommendation systems.

METRICS EXPLAINED
──────────────────
Precision@K  : Of the top-K items recommended, what fraction were relevant?
               "Did the model recommend good movies?"

Recall@K     : Of all relevant items, what fraction appeared in top-K?
               "Did the model find most of the good movies?"

NDCG@K       : Normalised Discounted Cumulative Gain.
               Rewards putting the BEST movies at the TOP of the list.
               A hit at rank 1 is worth more than a hit at rank 5.

Coverage     : What fraction of all movies can the model recommend?
               Low coverage = model only knows about popular movies.

Diversity    : How different are the recommendations from each other?
               (Intra-List Diversity using pairwise dissimilarity)

We use a leave-one-out train/test split:
  - Hide the last rated movie for each user.
  - See if the model can recover it in its top-K recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items captured in top-k recommendations."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.
    Rewards correct predictions that appear higher in the ranking.
    """
    top_k = recommended[:k]
    dcg = sum(
        1 / np.log2(rank + 2)  # rank is 0-indexed → +2 to start log at log2(2)=1
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: all |relevant| hits at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """1 if at least one relevant item is in top-k, else 0."""
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def coverage(all_recommended: list[list], total_items: int) -> float:
    """Fraction of the total catalogue that appears in any recommendation list."""
    seen = {item for recs in all_recommended for item in recs}
    return len(seen) / total_items


def intra_list_diversity(rec_vectors: np.ndarray) -> float:
    """
    Average pairwise dissimilarity within a recommendation list.
    rec_vectors : shape (n_recs, n_features) — e.g., TF-IDF vectors
    Returns 0 (identical) to 1 (completely different).
    """
    if len(rec_vectors) < 2:
        return 0.0
    sim = cosine_similarity(rec_vectors)
    n = len(sim)
    # Sum of upper-triangle (exclude diagonal)
    total_sim = sum(sim[i][j] for i in range(n) for j in range(i + 1, n))
    n_pairs = n * (n - 1) / 2
    avg_sim = total_sim / n_pairs
    return 1.0 - avg_sim  # dissimilarity


def evaluate_model(
    recommend_fn: Callable[[str], list[int]],  # title → list of movieIds
    test_pairs: list[tuple[str, int]],          # (title, held_out_movieId)
    k: int = 5,
) -> dict:
    """
    Evaluate a recommendation function using leave-one-out test pairs.

    Parameters
    ----------
    recommend_fn : Function that accepts a movie title and returns a list of
                   recommended movieIds (top-N, N >= k).
    test_pairs   : List of (query_title, held_out_movieId) tuples.
    k            : Cutoff for metrics.

    Returns
    -------
    dict with keys: precision, recall, ndcg, hit_rate (all @k), n_evaluated
    """
    precisions, recalls, ndcgs, hits = [], [], [], []
    n_evaluated = 0

    for title, held_out_id in test_pairs:
        try:
            recs = recommend_fn(title)
        except Exception:
            continue  # skip movies the model can't handle

        relevant = {held_out_id}
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        hits.append(hit_rate_at_k(recs, relevant, k))
        n_evaluated += 1

    if n_evaluated == 0:
        return {"error": "No test pairs could be evaluated."}

    return {
        f"precision@{k}": round(np.mean(precisions), 4),
        f"recall@{k}":    round(np.mean(recalls), 4),
        f"ndcg@{k}":      round(np.mean(ndcgs), 4),
        f"hit_rate@{k}":  round(np.mean(hits), 4),
        "n_evaluated":    n_evaluated,
    }


def build_leave_one_out_pairs(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings_per_user: int = 10,
    n_users: int = 200,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """
    Build (query_title, held_out_movieId) test pairs using leave-one-out.

    For each sampled user:
      - Sort their ratings by timestamp.
      - Use their most recently rated movie as the held-out item.
      - Use their second-most-recently-rated movie as the query.
    """
    rng = np.random.default_rng(seed)
    mid_to_title = dict(zip(movies["movieId"], movies["title_clean"]))

    eligible_users = (
        ratings.groupby("userId")
        .filter(lambda g: len(g) >= min_ratings_per_user)["userId"]
        .unique()
    )

    sampled = rng.choice(eligible_users, size=min(n_users, len(eligible_users)), replace=False)
    pairs = []

    for uid in sampled:
        user_ratings = (
            ratings[ratings["userId"] == uid]
            .sort_values("timestamp")
            .tail(2)
        )
        if len(user_ratings) < 2:
            continue
        query_id = user_ratings.iloc[-2]["movieId"]
        held_out_id = user_ratings.iloc[-1]["movieId"]

        if query_id not in mid_to_title or held_out_id not in mid_to_title:
            continue

        pairs.append((mid_to_title[query_id], int(held_out_id)))

    return pairs
