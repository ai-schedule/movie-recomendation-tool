import streamlit as st
import pickle
import pandas as pd
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommendation Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f5ebe0;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #6f4e37;
    margin-bottom: 10px;
}

/* KPI Cards */
.kpi-card {
    background-color: #e3d5ca;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 2px 3px 8px rgba(0,0,0,0.12);
    margin-bottom: 10px;
}

/* Card Labels */
.kpi-title {
    font-size: 16px;
    color: #5c4033;
    font-weight: 600;
}

/* Card Values */
.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #b5651d;
}

/* Section Titles */
.section-title {
    font-size: 28px;
    font-weight: bold;
    color: #6f4e37;
    margin-top: 20px;
    margin-bottom: 8px;
}

/* Normal Text */
p, label, div {
    color: #4b2e2e !important;
}
</style>
""", unsafe_allow_html=True)
# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🎬 Movie Recommendation Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- KPI INSIGHTS ----------------
total_movies = movies.shape[0]
avg_tag_length = round(movies["tags"].apply(lambda x: len(x.split())).mean(), 2)

# Count unique genres
all_genres = []
for item in movies["genres"]:
    if isinstance(item, str):
        all_genres.extend(item.split())

unique_genres = len(set(all_genres))

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Movies</div>
        <div class="kpi-value">{total_movies}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Tag Length</div>
        <div class="kpi-value">{avg_tag_length}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Unique Genres</div>
        <div class="kpi-value">{unique_genres}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- TOP GENRES CHART ----------------
st.markdown('<div class="section-title">📊 Top Genres</div>', unsafe_allow_html=True)

genre_count = Counter(all_genres)
top_genres = pd.DataFrame(genre_count.most_common(10), columns=["Genre", "Count"])

st.bar_chart(
    top_genres.set_index("Genre"),
    color="#b5651d"
)

# ---------------- RECOMMENDATION SYSTEM ----------------
st.markdown('<div class="section-title">🎯 Movie Recommendations</div>', unsafe_allow_html=True)

movie_list = movies["title"].values
selected_movie = st.selectbox("Select a Movie", movie_list)

# Show selected movie details
selected_data = movies[movies["title"] == selected_movie].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.write("### 📖 Overview")
    st.write(selected_data["overview"][:300] + "...")

with col2:
    st.write("### 🎭 Genres")
    st.write(selected_data["genres"])

    st.write("### 👤 Cast")
    st.write(selected_data["cast"])

# ---------------- FUNCTION ----------------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    result = []
    for i in movie_list:
        row = movies.iloc[i[0]]
        result.append({
            "title": row.title,
            "score": round(i[1] * 100, 2),
            "genres": row.genres
        })

    return result

# ---------------- BUTTON ----------------
if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.markdown("## ⭐ Top Recommendations")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{recommendations[i]['title']}</div>
                <div class="kpi-value">{recommendations[i]['score']}%</div>
                <div class="kpi-title">{recommendations[i]['genres']}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using Data Science | NLP | Cosine Similarity | Streamlit")
