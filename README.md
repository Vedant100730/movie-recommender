# 🎬 Movie Recommendation System

> A beginner-friendly yet production-grade hybrid recommendation system built with Python, Scikit-learn, and Streamlit — using the MovieLens dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Demo](#demo)
- [Project Overview](#project-overview)
- [How the Algorithms Work](#how-the-algorithms-work)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Evaluation Results](#evaluation-results)
- [Suggestions for Improvement](#suggestions-for-improvement)
- [Tech Stack](#tech-stack)

---

## 🎯 Project Overview

This project implements **three recommendation approaches** and combines them into a powerful hybrid system:

| Approach | Technique | Strengths |
|---|---|---|
| **Content-Based** | TF-IDF + Cosine Similarity | Works for new movies; interpretable |
| **Collaborative** | Item-Based CF + SVD | Discovers hidden patterns from users |
| **Hybrid** | Weighted blend (α·CB + (1-α)·CF) | Best of both worlds |

**Dataset:** [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)
- 9,742 movies · 100,836 ratings · 3,683 tags · 610 users

---

## 🧠 How the Algorithms Work

### 1. Content-Based Filtering

> *"Find movies with similar descriptions to what you already like."*

**Step-by-step:**

1. **Feature Engineering** — For every movie, combine its genres and user tags into a "soup" string:
   ```
   Toy Story → "animation children comedy adventure fantasy adventure children pixar animated sequel fun"
   ```

2. **TF-IDF Vectorisation** — Convert each soup into a numerical vector. Words that appear in every movie (like "the") get penalised; rare words (like "pixar") get boosted.

3. **Cosine Similarity** — Measure the angle between two movie vectors:
   ```
   similarity = cos(θ) = (A · B) / (|A| × |B|)
   ```
   - Score = 1.0 → identical movies
   - Score = 0.0 → completely different

4. **Rank & Return** — Sort all movies by similarity to the query; return top-N.

**When to use:** Great for cold-start (new movies with no ratings).

---

### 2. Collaborative Filtering

> *"People who rated movies similarly to you also loved these films."*

**Step-by-step:**

1. **User × Movie Matrix** — Build a giant table (610 users × 9742 movies). Each cell is a rating (1–5). Most cells are empty (NaN).

2. **Mean Imputation** — Fill empty cells with the movie's average rating so math works.

3. **SVD (Matrix Factorization)** — Decompose the matrix into latent factors:
   ```
   R ≈ U × Σ × Vᵀ
   ```
   - `U` = users described by k hidden "taste" dimensions
   - `V` = movies described by k hidden "quality" dimensions
   - This compresses 9742 dimensions → 50 latent dimensions

4. **Item-Item Cosine Similarity** — Compute similarity between every pair of movie factor vectors.

5. **Recommend** — For a query movie, return movies with highest factor-similarity.

**When to use:** Best when you have many user ratings. Can't help for brand new movies.

---

### 3. Hybrid System

> *"Blend both approaches for the best results."*

```
hybrid_score = α × content_score + (1 − α) × collaborative_score
```

- α = 0.4 (default) → 40% content, 60% collaborative
- Both scores are normalised to [0, 1] before blending
- Top 30 candidates from each model are merged, re-ranked by hybrid score
- α is adjustable at runtime via the Streamlit slider

---

## 📁 Project Structure

```
movie-recommender/
│
├── app.py                    # Streamlit web application
├── train.py                  # End-to-end training pipeline
├── requirements.txt
├── README.md
│
├── utils/
│   ├── data_loader.py        # Download, load, and preprocess MovieLens data
│   └── evaluator.py          # Precision@K, Recall@K, NDCG@K metrics
│
├── models/
│   ├── content_based.py      # TF-IDF + Cosine Similarity recommender
│   ├── collaborative.py      # Item-Based CF + SVD recommender
│   ├── hybrid.py             # Weighted hybrid combiner
│   └── saved/                # Pickled models (created after training)
│
└── data/
    └── ml-latest-small/      # Downloaded automatically by train.py
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

### 2. Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Train the models

This downloads the dataset (~3 MB), preprocesses it, trains all three models, evaluates them, and saves them to disk.

```bash
python train.py
```

Expected output (takes ~60–90 seconds):
```
STEP 1 / 5  —  Data Acquisition
[data_loader] Downloading MovieLens dataset …
[data_loader] Extracting … Done.

STEP 2 / 5  —  Preprocessing
  Movies  : 9,742
  Ratings : 100,836
  Tags    : 3,683
  Users   : 610

STEP 3 / 5  —  Model Training
━━ Fitting Content-Based model ━━
[ContentBased] TF-IDF matrix shape: (9742, 5000)
[ContentBased] Similarity matrix shape: (9742, 9742)
━━ Fitting Collaborative Filtering model ━━
[CollabFilter] SVD explains 22.3% of variance.

STEP 4 / 5  —  Sanity Check (Toy Story)
 movieId         title    year                    genres  cb_score  cf_score  hybrid_score
    3114     Toy Story 2  1999  Adventure, Animation, ...    0.7823    0.9134        0.8591
    ...

✅  Training complete in 78.4s
```

### Step 2: Launch the web app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Step 3: Use the app

1. Select a movie from the dropdown (e.g., "The Matrix")
2. Click **🔍 Recommend**
3. See your top 5 recommendations with scores
4. Adjust the **Content ↔ Collaborative** slider in the sidebar to tune results
5. Expand **Why was the top pick recommended?** for an explanation

---

## 📊 Evaluation Results

Evaluated using **Leave-One-Out** on 200 users, k=5:

| Model | Precision@5 | Recall@5 | NDCG@5 | Hit Rate@5 |
|---|---|---|---|---|
| Content-Based | ~0.02 | ~0.02 | ~0.02 | ~0.09 |
| Hybrid (α=0.4) | ~0.03 | ~0.03 | ~0.03 | ~0.13 |

> **Note:** Hit rates appear low because MovieLens has only 100K ratings across 9K movies — the dataset is very sparse. These numbers are typical for offline evaluation on small, sparse datasets. In production systems with millions of interactions, hit rates of 10–30% at k=5 are standard.

---

## 🔧 Suggestions for Improvement

### 🔴 Short-term (beginner-friendly)
- [ ] **Add movie posters** via TMDB API (free, uses `links.csv` tmdbId)
- [ ] **Filter by genre** — add a genre dropdown to the Streamlit sidebar
- [ ] **Add year range filter** — "only recommend movies from the 2000s"
- [ ] **Cache recommendations** so repeat queries are instant

### 🟡 Medium-term (intermediate)
- [ ] **Surprise/Implicit library** — use the `surprise` library for proper SVD++ / ALS collaborative filtering with RMSE evaluation
- [ ] **Neural Collaborative Filtering (NCF)** — replace SVD with a small PyTorch model
- [ ] **Sentence-BERT embeddings** — replace TF-IDF with `sentence-transformers` for semantic similarity
- [ ] **User profiles** — let users rate 5 movies and get personalised recommendations
- [ ] **A/B testing module** — compare Content vs CF vs Hybrid hit rates live

### 🟢 Long-term (advanced / production)
- [ ] **Real-time system** — swap pickle files for a Redis cache and FastAPI backend
- [ ] **Approximate Nearest Neighbours** — use `faiss` or `annoy` to scale to millions of movies
- [ ] **LLM-enhanced explanations** — use GPT/Claude to generate natural language explanations
- [ ] **Implicit feedback** — model "watch time" instead of just star ratings
- [ ] **Deploy to Streamlit Cloud / Hugging Face Spaces** for a public demo

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, matrix pivoting |
| `numpy` | Numerical operations |
| `scikit-learn` | TF-IDF, cosine similarity, SVD, normalisation |
| `streamlit` | Web application UI |
| `pickle` | Model serialisation |

---

## 📚 References & Further Reading

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/) — GroupLens Research
- [An Introduction to Recommender Systems](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada) — Towards Data Science
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) — Koren et al. (Netflix Prize paper)
- [Scikit-learn TF-IDF docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

## 📄 License

MIT © 2024 — Free to use, modify, and distribute.

---

> Built as a portfolio project demonstrating ML fundamentals: feature engineering, matrix factorisation, similarity search, and hybrid system design.
