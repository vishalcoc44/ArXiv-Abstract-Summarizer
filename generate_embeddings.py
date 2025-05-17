import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from time import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_path = r"C:\Users\Vishal\Downloads\minor1\arxiv_abstracts"
input_file = os.path.join(base_path, "abstracts.json")
output_file = os.path.join(base_path, "abstract_embeddings.npy")

logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_abstracts_generator(file_path, chunk_size=10000):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for i in range(0, len(data), chunk_size):
                chunk = [item["abstract"] for item in data[i:i + chunk_size]]
                yield chunk
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise

def generate_embeddings(texts_generator, batch_size=2000):
    embeddings_list = []
    total_texts = 0
    start_time = time()
    for chunk in texts_generator:
        chunk_embeddings = embed_model.encode(chunk, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        embeddings_list.append(chunk_embeddings)
        total_texts += len(chunk)
        elapsed = time() - start_time
        logger.info(f"Processed {total_texts} abstracts in {elapsed:.2f}s ({total_texts/elapsed:.2f} abstracts/s)")
    embeddings = np.vstack(embeddings_list).astype("float32")
    logger.info(f"Final embeddings shape: {embeddings.shape}")
    return embeddings

logger.info(f"Starting embedding generation from {input_file}...")
texts_generator = load_abstracts_generator(input_file)

try:
    embeddings = generate_embeddings(texts_generator, batch_size=2000)
    logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    np.save(output_file, embeddings)
    logger.info(f"Saved embeddings to {output_file}")
except Exception as e:
    logger.error(f"Error during embedding generation: {e}")
    raise