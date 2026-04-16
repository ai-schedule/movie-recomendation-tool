import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

# Load datasets
movies = pd.read_csv('data/movies.csv')
credits = pd.read_csv('data/credits.csv')

print("Movies Data Loaded:", movies.shape)
print("Credits Data Loaded:", credits.shape)
# Merge datasets
movies = movies.merge(credits, on='title')

print("Merged Data:", movies.shape)
#print(movies.head())
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

print("Filtered Data Columns:")
print(movies.head())
print(movies.isnull().sum())

movies.dropna(inplace=True)

print("After removing nulls:", movies.shape)
print(movies.head())

def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
print(movies['genres'])
print(movies['keywords'])

def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l
movies['cast'] = movies['cast'].apply(convert_cast)
print(movies['cast'])

def fetch_director(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l

movies['crew'] = movies['crew'].apply(fetch_director)
print(movies['crew'])


movies['overview'] = movies['overview'].apply(lambda x: x.split())

def remove_space(l):
    return [i.replace(" ", "") for i in l]

movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']].copy()

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

print(new_df.head())


ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

print("Vector shape:", vectors.shape)
