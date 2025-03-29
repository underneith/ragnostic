import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd 
import numpy as np
import os

# Load your data
df = pd.read_csv("data/golden_girls_data.csv")

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A small, efficient model

# Create embeddings
texts = df['content'].tolist()  # Assuming 'content' is your text column
embeddings = model.encode(texts)

# Normalize embeddings for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
index.add(embeddings)

# Save index and relevant data
faiss.write_index(index, "../app/data/golden_girls.index")
df.to_pickle("../app/data/golden_girls_metadata.pkl")
