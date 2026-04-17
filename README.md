# 🎬 Movie Recommendation System

A **Content-Based Movie Recommendation System** built using **Data Science, NLP, and Streamlit**.  
This project recommends movies similar to the selected movie by analyzing genres, cast, keywords, and movie descriptions.

---

## 📌 Project Overview

The goal of this project is to build a recommendation engine that suggests similar movies based on movie content instead of user ratings.

It uses:

- **Natural Language Processing (NLP)**
- **CountVectorizer**
- **Cosine Similarity**
- **Streamlit Dashboard**

---

## 🚀 Features

✅ Recommend top 5 similar movies  
✅ Content-based filtering  
✅ Uses movie overview, genres, cast, keywords  
✅ Interactive Streamlit web app  
✅ Clean and user-friendly UI  
✅ Search and selection functionality  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Pickle  

---

## 📂 Dataset Used

**TMDB 5000 Movie Dataset**

Dataset contains:

- Movie Title  
- Overview  
- Genres  
- Keywords  
- Cast  
- Crew  

---

## ⚙️ How It Works

### 1️⃣ Data Preprocessing

- Merged movie and credits datasets  
- Removed null values  
- Extracted genres, cast, keywords, director  
- Combined important text features into one column called `tags`

### 2️⃣ Text Vectorization

Used **CountVectorizer** to convert text data into numerical vectors.

### 3️⃣ Similarity Calculation

Used **Cosine Similarity** to find similar movies based on vector closeness.

### 4️⃣ Recommendation Engine

When a user selects a movie:

- System finds its vector
- Compares with all other movies
- Returns top 5 most similar movies

---

## 📸 App Preview

```bash
streamlit run app.py
