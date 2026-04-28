"""
train.py
--------
End-to-end training script.

Run this once to:
  1. Download the MovieLens dataset
  2. Preprocess and engineer features
  3. Fit all three models (Content-Based, Collaborative, Hybrid)
  4. Evaluate the Hybrid model
  5. Save models to disk (models/saved/)

Usage:
    python train.py
"""

import os
import sys
import pickle
import time

# Ensure project root is on path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import (
    download_dataset,
    load_movies,
    load_ratings,
    load_tags,
    build_movie_profiles,
    build_user_movie_matrix,
)
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender
from models.hybrid import HybridRecommender
from utils.evaluator import build_leave_one_out_pairs, evaluate_model

SAVE_DIR = os.path.join(os.path.dirname(__file__), "models", "saved")


def main():
    t0 = time.time()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Data acquisition ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 / 5  —  Data Acquisition")
    print("=" * 60)
    download_dataset()

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 / 5  —  Preprocessing")
    print("=" * 60)

    movies  = load_movies()
    ratings = load_ratings()
    tags    = load_tags()

    print(f"  Movies  : {len(movies):,}")
    print(f"  Ratings : {len(ratings):,}")
    print(f"  Tags    : {len(tags):,}")
    print(f"  Users   : {ratings['userId'].nunique():,}")

    movie_profiles    = build_movie_profiles(movies, tags)
    user_movie_matrix = build_user_movie_matrix(ratings)
    print(f"  User × Movie matrix shape: {user_movie_matrix.shape}")

    # ── 3. Model training ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 / 5  —  Model Training")
    print("=" * 60)

    hybrid = HybridRecommender(alpha=0.4)
    hybrid.fit(movie_profiles, user_movie_matrix, movies)

    # ── 4. Quick sanity check ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 / 5  —  Sanity Check  (Recommendations for 'Toy Story')")
    print("=" * 60)

    recs = hybrid.recommend("Toy Story", n=5)
    print(recs.to_string(index=False))

    print("\n  Explanation:")
    print(hybrid.explain("Toy Story", recs.iloc[0]["title"]))

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 / 5  —  Evaluation (Leave-One-Out, n_users=200, k=5)")
    print("=" * 60)

    test_pairs = build_leave_one_out_pairs(ratings, movies, n_users=200)
    print(f"  Built {len(test_pairs)} test pairs.")

    def hybrid_fn(title: str) -> list[int]:
        recs = hybrid.recommend(title, n=10)
        return recs["movieId"].tolist()

    def cb_fn(title: str) -> list[int]:
        recs = hybrid._cb.recommend(title, n=10)
        return recs["movieId"].tolist()

    print("\n  ┌─ Content-Based ──")
    cb_metrics = evaluate_model(cb_fn, test_pairs, k=5)
    for k, v in cb_metrics.items():
        print(f"  │  {k:<18}: {v}")

    print("\n  ├─ Hybrid (α=0.4) ──")
    hy_metrics = evaluate_model(hybrid_fn, test_pairs, k=5)
    for k, v in hy_metrics.items():
        print(f"  │  {k:<18}: {v}")

    # ── Save models ───────────────────────────────────────────────────────────
    print("\n  Saving models …")
    with open(os.path.join(SAVE_DIR, "hybrid_model.pkl"), "wb") as f:
        pickle.dump(hybrid, f)

    # Also save the movies list for the web app
    with open(os.path.join(SAVE_DIR, "movies.pkl"), "wb") as f:
        pickle.dump(movies, f)

    elapsed = time.time() - t0
    print(f"\n✅  Training complete in {elapsed:.1f}s")
    print(f"   Models saved to: {SAVE_DIR}/")
    print("\nNext step → run:  streamlit run app.py")


if __name__ == "__main__":
    main()
