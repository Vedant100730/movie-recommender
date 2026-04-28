"""
app.py
------
Streamlit web application for the Movie Recommendation System.

Run with:
    streamlit run app.py
"""

import os
import sys
import pickle
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

SAVE_DIR = os.path.join(os.path.dirname(__file__), "models", "saved")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1 { font-family: 'Playfair Display', serif !important; }

.rec-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    border-left: 4px solid #e94560;
    color: #eaeaea;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.rec-title { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 4px; }
.rec-meta  { font-size: 0.85rem; color: #a0aec0; }
.score-pill {
    display: inline-block;
    background: rgba(233,69,96,0.2);
    border: 1px solid #e94560;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #e94560;
    margin-right: 6px;
}
.query-box {
    background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    color: white;
    font-weight: 600;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached so it only happens once) ────────────────────────────
@st.cache_resource(show_spinner="Training models (~2 mins)...")
def load_models():
    import sys
    from utils.data_loader import load_movies, load_ratings, load_tags, build_movie_profiles, build_user_movie_matrix
    from models.hybrid import HybridRecommender

    movies  = load_movies()
    ratings = load_ratings()
    tags    = load_tags()
    profiles = build_movie_profiles(movies, tags)
    matrix   = build_user_movie_matrix(ratings)

    hybrid = HybridRecommender(alpha=0.4)
    hybrid.fit(profiles, matrix, movies)
    return hybrid, movies



# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.caption("Powered by Content-Based + Collaborative Filtering (Hybrid)")

hybrid, movies = load_models()

if hybrid is None:
    import subprocess
    with st.spinner("⏳ Training models for first time... (~2 mins, only once)"):
        subprocess.run([sys.executable, "train.py"], check=True)
    hybrid, movies = load_models()

if hybrid is None:
    st.error("Training failed. Check logs.", icon="🚨")
    st.stop()

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    alpha = st.slider(
        "Content ↔ Collaborative Balance",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="0 = pure collaborative filtering | 1 = pure content-based",
    )
    hybrid.set_alpha(alpha)

    n_recs = st.selectbox("Number of recommendations", [3, 5, 10], index=1)

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        f"- 🔵 **Content-Based** ({alpha:.0%}): uses genres & tags\n"
        f"- 🟠 **Collaborative** ({1-alpha:.0%}): uses viewing patterns\n"
        "- 🟢 **Hybrid**: blends both scores"
    )

    st.markdown("---")
    st.markdown("**Dataset:** MovieLens ml-latest-small")
    st.markdown(f"**Movies:** {len(movies):,}")

# ── Movie selector ────────────────────────────────────────────────────────────
all_titles = sorted(movies["title_clean"].dropna().unique().tolist())

col1, col2 = st.columns([3, 1])
with col1:
    selected_title = st.selectbox(
        "🎥 Choose a movie you like:",
        options=all_titles,
        index=all_titles.index("Toy Story") if "Toy Story" in all_titles else 0,
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("🔍 Recommend", use_container_width=True, type="primary")

# ── On button click ───────────────────────────────────────────────────────────
if recommend_btn or selected_title:
    # Show query movie info
    movie_row = movies[movies["title_clean"] == selected_title]
    if not movie_row.empty:
        row = movie_row.iloc[0]
        year_str = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
        genres_str = row.get("genres_str", "")
        st.markdown(
            f'<div class="query-box">🎬 You selected: <em>{selected_title}</em>'
            f'{year_str} &nbsp;|&nbsp; {genres_str}</div>',
            unsafe_allow_html=True,
        )

    with st.spinner("Finding your recommendations …"):
        try:
            recs = hybrid.recommend(selected_title, n=n_recs, alpha=alpha)
        except Exception as e:
            st.error(f"Could not generate recommendations: {e}")
            st.stop()

    st.subheader(f"🍿 Top {n_recs} Recommendations")

    for _, row in recs.iterrows():
        year_display = f"{int(row['year'])}" if pd.notna(row.get("year")) else "N/A"
        hybrid_pct   = f"{row['hybrid_score']:.0%}"
        cb_pct       = f"{row['cb_score']:.0%}"
        cf_pct       = f"{row['cf_score']:.0%}"

        st.markdown(
            f"""
            <div class="rec-card">
              <div class="rec-title">🎞️ {row['title']}</div>
              <div class="rec-meta">📅 {year_display} &nbsp;|&nbsp; 🎭 {row['genres']}</div>
              <br>
              <span class="score-pill">⭐ Hybrid {hybrid_pct}</span>
              <span class="score-pill">🔵 Content {cb_pct}</span>
              <span class="score-pill">🟠 CF {cf_pct}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explanation for top pick
    with st.expander("💡 Why was the top pick recommended?"):
        try:
            explanation = hybrid.explain(selected_title, recs.iloc[0]["title"])
            st.text(explanation)
        except Exception:
            st.write("Explanation not available for this pair.")

    # Score breakdown chart
    with st.expander("📊 Score Breakdown"):
        chart_data = recs[["title", "cb_score", "cf_score", "hybrid_score"]].set_index("title")
        st.bar_chart(chart_data)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with ❤️ using MovieLens + Scikit-learn + Streamlit · "
    "[GitHub](https://github.com/your-username/movie-recommender)"
)
