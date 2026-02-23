# 🎬 Hybrid Movie Recommendation System

> A Netflix-style recommendation engine built from scratch using the MovieLens 100K dataset — combining Content-Based Filtering and Collaborative Filtering (SVD) into a hybrid model.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-2.0-blue?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn)
![SciPy](https://img.shields.io/badge/SciPy-latest-lightblue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)

---

## 📌 Project Overview

This project implements a **hybrid movie recommendation system** that blends two distinct recommendation strategies:

- **Content-Based Filtering** — recommends movies similar to a seed movie based on genres and user tags using TF-IDF vectorization and cosine similarity
- **Collaborative Filtering** — predicts how a specific user would rate unseen movies using SVD (Singular Value Decomposition) matrix factorization
- **Hybrid Scoring** — combines both signals with a tunable alpha weight to produce personalized, context-aware recommendations

The system was built entirely with libraries pre-installed in Google Colab — no extra dependencies required.

---

## 📊 Dataset

**MovieLens 100K** — collected by the GroupLens Research Lab

| File | Description | Size |
|------|-------------|------|
| `movies.csv` | Movie titles and genres | 9,742 movies |
| `ratings.csv` | User-movie ratings (0.5 – 5.0) | 100,836 ratings |
| `tags.csv` | User-written tags per movie | 3,683 tags |
| `links.csv` | IMDB and TMDB IDs | 9,742 entries |

- **Users:** 610
- **Rating scale:** 0.5 → 5.0 (half-star increments)
- **Matrix sparsity:** 98.3% (most users haven't rated most movies)

> Dataset source: [grouplens.org/datasets/movielens/latest](https://grouplens.org/datasets/movielens/latest/)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  HYBRID RECOMMENDER                     │
│                                                         │
│  Seed Movie ──► TF-IDF + Cosine Sim ──► Content Score  │
│                          │                              │
│                          ▼                              │
│         Normalize both scores to [0, 1]                 │
│                          │                              │
│  User ID ──► SVD Matrix ──► Collaborative Score         │
│                          │                              │
│                          ▼                              │
│   Hybrid Score = α × Collab + (1-α) × Content          │
│                          │                              │
│                          ▼                              │
│            Top-N Personalized Recommendations           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Models Used

### 1. TF-IDF Vectorizer (Content-Based)
Converts each movie's genre and tag text into a numerical vector. Words that appear frequently in one movie but rarely across all movies receive higher weight — making each movie's representation unique.

### 2. Cosine Similarity
Measures the similarity between two movie vectors by calculating the angle between them. A score of 1.0 means identical, 0.0 means completely different.

### 3. SVD — Singular Value Decomposition (Collaborative)
Decomposes the 610 × 9,724 sparse user-movie rating matrix into three smaller matrices (U, Σ, Vt) that capture hidden preference patterns — things like "enjoys animated family films" or "prefers psychological thrillers" — without these categories ever being explicitly defined. The full matrix is then reconstructed with all missing ratings filled in as predictions.

**Key implementation decisions:**
- Train/test split done **before** building the rating matrix to prevent data leakage
- User means computed from actual rated entries only (NaN-aware) — not from zeros
- Predictions clipped to [0.5, 5.0] to enforce valid rating bounds
- k=150 latent factors for richer pattern capture

---

## 📈 Model Performance

Evaluated on a **held-out test set (20% of ratings, ~19,355 samples)**:

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| RMSE | **0.9440** | < 1.0 | ✅ |
| MAE | **0.7333** | < 0.80 | ✅ |

> On a 0.5–5.0 scale, the model predicts within ~0.94 stars of the true rating on average.

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
1. Open the notebook: `notebooks/Movie_Recommendation_System.ipynb`
2. Upload `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv` to the Colab session
3. Run all cells from top to bottom — no pip installs needed

### Option B — Local Machine
```bash
# Clone the repository
git clone https://github.com/AdamMusa76/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Launch Jupyter
jupyter notebook notebooks/Movie_Recommendation_System.ipynb
```

---

## 🎛️ Usage

```python
# Content-Based — find movies similar to a title
content_based_recommend('Toy Story', top_n=10)

# Collaborative — top picks for a specific user
collab_recommend(user_id=42, top_n=10)

# Hybrid — personalized + context-aware (recommended)
hybrid_recommend(
    user_id=42,
    movie_title='Inception',
    top_n=10,
    alpha=0.5     # 0.0 = pure content | 1.0 = pure collaborative
)
```

### Alpha Tuning Guide

| Alpha | Best For |
|-------|----------|
| `0.0 – 0.3` | New users with few ratings — lean on content |
| `0.4 – 0.6` | Most users — balanced hybrid (default) |
| `0.7 – 1.0` | Power users with many ratings — trust collaborative |

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── notebooks/
│   └── Movie_Recommendation_System.ipynb   # Main notebook
│
├── model/
│   ├── predicted_df.pkl                    # SVD predicted ratings matrix
│   ├── tfidf_vectorizer.pkl                # Fitted TF-IDF model
│   ├── cosine_sim.pkl                      # Cosine similarity matrix
│   ├── movies_processed.pkl                # Preprocessed movies dataframe
│   ├── user_means.pkl                      # Per-user rating means
│   ├── svd_U.pkl                           # SVD U matrix
│   ├── svd_sigma.pkl                       # SVD Sigma values
│   └── svd_Vt.pkl                          # SVD Vt matrix
│
├── data/
│   └── (download MovieLens 100K files here — not tracked by git)
│
├── .gitignore
└── README.md
```

---

## 🔭 Future Improvements

- Add **bias terms** (per-user and per-movie offsets) to push RMSE below 0.88
- Build a **Gradio or Streamlit web UI** for interactive use
- Fetch and display **movie posters** via the TMDB API using `links.csv`
- Implement **Neural Collaborative Filtering (NCF)** with TensorFlow
- Add **automatic alpha selection** based on a user's rating count
- Use **implicit feedback** (views, clicks) in addition to explicit ratings

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data loading and manipulation |
| NumPy 2.0 | Matrix operations |
| scikit-learn | TF-IDF, cosine similarity, train/test split, MinMaxScaler |
| SciPy | Sparse SVD matrix factorization |
| Matplotlib / Seaborn | Visualizations |
| Google Colab | Development environment |

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset
- F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
