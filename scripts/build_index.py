import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import time

def create_directories():
    os.makedirs("app/data", exist_ok=True)

def preprocess_data(df):
    """
    Preprocess the data for chunking and embedding.
    
    Args:
        df: DataFrame containing the raw data
        
    Returns:
        DataFrame with processed and chunked data
    """
    print("Preprocessing data...")
    
    # Ensure all text is string type
    df['content'] = df['content'].astype(str)
    
    # Remove any rows with empty content
    df = df[df['content'].str.strip() != '']
    
    # Create chunks for long texts
    chunks = []
    
    for _, row in df.iterrows():
        content = row['content']
        title = row['title']
        source = row['source']
        content_type = row['type']
        
        # Split long content into chunks of about 200 words
        words = content.split()
        
        if len(words) <= 200:
            # Keep short content as is
            chunks.append({
                'title': title,
                'content': content,
                'source': source,
                'type': content_type
            })
        else:
            # Split longer content into chunks
            for i in range(0, len(words), 150):  # 50 word overlap
                chunk_words = words[i:i+200]
                if len(chunk_words) < 50:  # Skip very small final chunks
                    continue
                    
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'title': f"{title} (Part {i//150 + 1})",
                    'content': chunk_text,
                    'source': source,
                    'type': content_type
                })
    
    chunked_df = pd.DataFrame(chunks)
    print(f"Created {len(chunked_df)} chunks from {len(df)} original entries")
    
    return chunked_df

def create_embeddings(df, model_name='all-MiniLM-L6-v2'):
    """
    Create embeddings for the text content.
    
    Args:
        df: DataFrame with text content
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        numpy array of embeddings, normalized for cosine similarity
    """
    print(f"Creating embeddings using {model_name}...")
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Create embeddings
    texts = df['content'].tolist()
    
    # Process in batches to avoid memory issues
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.append(batch_embeddings)
        
        # Progress report
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i+len(batch_texts)}/{len(texts)} texts")
    
    # Combine batches
    embeddings = np.vstack(embeddings)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    print(f"Created embeddings with shape {embeddings.shape}")
    return embeddings

def build_and_save_index(df, embeddings):
    """
    Build FAISS index and save it along with metadata.
    
    Args:
        df: DataFrame with text content and metadata
        embeddings: numpy array of normalized embeddings
    """
    print("Building FAISS index...")
    
    # Create FAISS index for cosine similarity (inner product with normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"Built index with {index.ntotal} vectors of dimension {dimension}")
    
    # Save the index
    faiss.write_index(index, "../app/data/golden_girls.index")
    print("Saved FAISS index to ../app/data/golden_girls.index")
    
    # Save metadata (exclude the full content to keep it small)
    metadata_df = df.copy()
    metadata_df.to_pickle("../app/data/golden_girls_metadata.pkl")
    print("Saved metadata to ../app/data/golden_girls_metadata.pkl")

def main():
    """Main function to build the index."""
    start_time = time.time()
    create_directories()
    
    print("Starting index building process...")
    
    # Load the scraped data
    try:
        df = pd.read_csv("data/raw/golden_girls_data.csv")
        print(f"Loaded {len(df)} entries from data/raw/golden_girls_data.csv")
    except FileNotFoundError:
        print("Error: golden_girls_data.csv not found. Please run scrape_data.py first.")
        return
    
    # Preprocess the data
    chunked_df = preprocess_data(df)
    
    # Create embeddings
    embeddings = create_embeddings(chunked_df)
    
    # Build and save the index
    build_and_save_index(chunked_df, embeddings)
    
    elapsed_time = time.time() - start_time
    print(f"Index building complete. Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
