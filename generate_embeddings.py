import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from time import time
import logging
import torch # Import torch to check for CUDA availability

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Kaggle Specific Paths Setup ---
# IMPORTANT: Replace 'your-dataset-name' with the actual name of your Kaggle dataset.
# You can find this name by looking at the input path in your Kaggle notebook (e.g., /kaggle/input/my-data-set).
kaggle_input_base_path = "/kaggle/input/your-dataset-name"
input_file_name = "abstracts.json" # Ensure this file exists in your dataset
input_file = os.path.join(kaggle_input_base_path, input_file_name)

# Output files will be saved in Kaggle's working directory.
kaggle_output_base_path = "/kaggle/working"
output_file_name = "abstract_embeddings.npy"
output_file = os.path.join(kaggle_output_base_path, output_file_name)
# --- End Kaggle Specific Paths ---

# Detect available device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Loading SentenceTransformer model 'all-MiniLM-L6-v2' on device: {device}...")
# The model will be downloaded to Kaggle's environment if not already present.
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def load_abstracts_generator(file_path, chunk_size=10000):
    """
    Loads abstracts from a JSON file in chunks.
    Combines ID, authors, title, categories, and abstract into a single text string for embedding.
    Yields lists of these combined text contents.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} items from {file_path}")
            for i in range(0, len(data), chunk_size):
                chunk_texts = []
                # Process a chunk of data
                for item in data[i:i + chunk_size]:
                    # Extract all desired fields based on your JSON structure
                    doc_id = item.get("id", "N/A").strip() # Using "id" as per your sample
                    authors = item.get("authors", "N/A").strip() # Expecting a string
                    title = item.get("title", "N/A").strip()
                    categories = item.get("categories", "N/A").strip() # Expecting a string
                    abstract = item.get("abstract", "N/A").strip()

                    # Combine fields for a richer representation
                    combined_text_parts = []
                    if doc_id and doc_id != "N/A":
                        combined_text_parts.append(f"ID: {doc_id}")
                    if title and title != "N/A":
                        combined_text_parts.append(f"Title: {title}")
                    if authors and authors != "N/A":
                        combined_text_parts.append(f"Authors: {authors}")
                    if categories and categories != "N/A": # Check categories string
                        combined_text_parts.append(f"Categories: {categories}")
                    if abstract and abstract != "N/A":
                        combined_text_parts.append(f"Abstract: {abstract}")

                    combined_text = '\n\n'.join(combined_text_parts)

                    # Fallback if combined text is empty (unlikely with valid abstract data)
                    if not combined_text.strip():
                        # Use the actual ID if available for better logging
                        log_id = doc_id if doc_id != "N/A" else "Unknown ID"
                        logger.warning(f"Document with ID {log_id} resulted in empty combined text. "
                                       f"Using only abstract if available, else skipping.")
                        combined_text = abstract if abstract and abstract != "N/A" else ""

                    # Only append if the combined text is not empty after stripping whitespace
                    if combined_text.strip():
                        chunk_texts.append(combined_text)
                    else:
                        log_id = doc_id if doc_id != "N/A" else "Unknown ID"
                        logger.warning(f"Skipping document with ID {log_id} as its combined text is entirely empty.")

                if chunk_texts: # Only yield if there's content in the chunk
                    yield chunk_texts
                else:
                    logger.info(f"Chunk starting at index {i} was empty, skipping.")

    except FileNotFoundError as e:
        logger.error(f"Input file not found at {file_path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading abstracts: {e}")
        raise

def generate_embeddings(texts_generator, batch_size=2000):
    """
    Generates embeddings for texts provided by a generator, with performance optimizations.
    """
    embeddings_list = []
    total_texts = 0
    start_time = time()
    logger.info(f"Starting embedding generation with batch size: {batch_size}")

    # num_workers: Controls the number of parallel processes for data loading and preprocessing.
    # 0 means all processing is done in the main thread.
    # For CPU-bound tasks or very large datasets, increasing this (e.g., to 1, 2, or 4) can help.
    # However, on Kaggle, if the GPU is the primary bottleneck, keep it at 0 or 1.
    # Experiment to find the optimal value for your specific setup.
    num_workers = 0

    try:
        for i, chunk in enumerate(texts_generator):
            if not chunk: # Skip if chunk is empty (e.g., due to filtering out empty texts)
                logger.info(f"Skipping empty chunk {i+1}.")
                continue

            logger.info(f"Processing chunk {i+1} with {len(chunk)} texts...")
            # Encode the chunk of texts into embeddings
            # - `convert_to_numpy=True`: Ensures output is NumPy arrays.
            # - `show_progress_bar=True`: Displays a progress bar for the current batch.
            # - `num_workers`: Parallelizes text processing.
            # - `normalize_embeddings=False`: Embeddings are not normalized by default;
            #   you can normalize them later if needed (e.g., for cosine similarity).
            # - The model automatically uses fp16 on CUDA devices if supported,
            #   so no explicit 'precision' argument is needed here, avoiding the ValueError.
            chunk_embeddings = embed_model.encode(
                chunk,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                num_workers=num_workers,
                normalize_embeddings=False,
            )
            embeddings_list.append(chunk_embeddings)
            total_texts += len(chunk)
            elapsed = time() - start_time
            # Log overall progress after each chunk
            logger.info(f"Processed {total_texts} abstracts in {elapsed:.2f}s ({total_texts/elapsed:.2f} abstracts/s)")

        if not embeddings_list:
            logger.warning("No embeddings were generated. The input text generator might have been empty.")
            return np.array([]) # Return an empty numpy array if no embeddings were generated

        # Vertically stack all chunk embeddings into a single NumPy array of float32
        embeddings = np.vstack(embeddings_list).astype("float32")
        logger.info(f"Final embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise

# --- Main Execution Block ---
logger.info(f"Starting embedding generation process...")
texts_generator = load_abstracts_generator(input_file)

try:
    embeddings = generate_embeddings(texts_generator, batch_size=2000)
    if embeddings.size > 0: # Check if the generated embeddings array is not empty
        logger.info(f"Successfully generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        np.save(output_file, embeddings)
        logger.info(f"Embeddings saved successfully to {output_file}")
    else:
        logger.warning("No embeddings were generated or saved as the input data was empty or invalid.")
except Exception as e:
    logger.error(f"An unrecoverable error occurred during the main embedding process: {e}")
    # Re-raise the exception to ensure it's propagated and indicates a failure
    raise
