import faiss
import numpy as np
import os
import json
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
abstract_dir = "./arxiv_abstracts"
embeddings_path = os.path.join(abstract_dir, "abstract_embeddings.npy")
index_path = os.path.join(abstract_dir, "faiss_index.bin")
metadata_path = os.path.join(abstract_dir, "abstract_metadata.pkl")
abstracts_path = os.path.join(abstract_dir, "abstracts.json")

# Load embeddings
try:
    embeddings = np.load(embeddings_path).astype('float32')
    logger.info(f"Loaded embeddings with shape {embeddings.shape}")
except FileNotFoundError as e:
    logger.error(f"Embeddings file not found: {e}")
    raise

# Create FAISS index
logger.info("Creating FAISS index...")
dimension = embeddings.shape[1]  # Embedding dimension (384 for all-MiniLM-L6-v2)
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings)

# Save the index
try:
    faiss.write_index(index, index_path)
    logger.info(f"Saved FAISS index with {index.ntotal} embeddings to {index_path}")
except Exception as e:
    logger.error(f"Error saving FAISS index: {e}")
    raise

# Save abstract metadata for mapping
try:
    with open(abstracts_path, "r", encoding="utf-8") as f:
        abstracts = json.load(f)
    with open(metadata_path, "wb") as f:
        pickle.dump(abstracts, f)
    logger.info(f"Saved abstract metadata to {metadata_path}")
except Exception as e:
    logger.error(f"Error saving metadata: {e}")
    raise