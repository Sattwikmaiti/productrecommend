from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle 
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import pandas as pd
import numpy as np
import difflib #spelling mistakes from user. compare closest proximity to movies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

products = pd.read_csv('dataset/amazonecom.csv')

products
products.head()
products.dropna(inplace=True)
products.isnull().sum()
products.shape
products.duplicated().sum()

tfv = TfidfVectorizer()
tfv_matrix = tfv.fit_transform(products['description'])

tfv_model_filename = 'model/tfidf_vectorizer_model.pkl'
tfv_loaded = joblib.load(tfv_model_filename)

def recommend(n):
    new_descriptions = [n]
    new_tfidf_matrix = tfv_loaded.transform(new_descriptions)

    tfidf_similarity = cosine_similarity(new_tfidf_matrix, tfv_matrix)

    # Assuming you want to find the most similar product(s) for each new description
    num_similar_products = 5
    similar_products_indices = tfidf_similarity.argsort(axis=1)[:, -num_similar_products:]

    similar_products = []
    for i, similar_indices in enumerate(similar_products_indices):
        print(f"Top similar products for new description {i+1}:")
        for idx in similar_indices:
            similar_products.append(products['description'].iloc[idx])
    return similar_products



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class model_input(BaseModel):
    text: str

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Render Fast Api'}

@app.post('/prediction')
def getPrediction(model_input: model_input):
    input_data = model_input.json()
    input_dictionary = json.loads(input_data)
    return recommend(input_dictionary['text'])