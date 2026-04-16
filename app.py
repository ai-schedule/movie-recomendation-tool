import streamlit as st
import pickle

# Load data
movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .stApp {
        background-color: #f5ebe0;
        color: #3e3e3e;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #b08968;
        margin-bottom: 20px;
    }

    .kpi-card {
        background-color: #e3d5ca;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }

    .kpi-title {
        font-size: 16px;
        color: #7f5539;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        color: #6f1d1b;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)

# ---------------- KPI CARDS ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Movies</div>
            <div class="kpi-value">{movies.shape[0]}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Recommendation Type</div>
            <div class="kpi-value">Content-Based</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT ----------------
movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Select a movie", movie_list)

# ---------------- FUNCTION ----------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    results = []
    for i in movies_list:
        row = movies.iloc[i[0]]
        results.append({
            "title": row.title,
            "score": round(i[1]*100, 2),
            "genres": row.genres
        })
    return results

# ---------------- DETAILS ----------------
selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]

st.subheader("📖 Movie Details")
st.write(selected_movie_data['overview'][:300] + "...")

# ---------------- BUTTON ----------------
if st.button("🚀 Recommend"):
    results = recommend(selected_movie)

    st.markdown("---")
    st.subheader("🎯 Recommended Movies")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{results[i]['title']}</div>
                    <div class="kpi-value">{results[i]['score']}%</div>
                    <div class="kpi-title">{results[i]['genres']}</div>
                </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using Data Science (NLP + Cosine Similarity)")
