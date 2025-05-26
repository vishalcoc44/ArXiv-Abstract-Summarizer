import os
import pickle
import numpy as np
import faiss # For direct FAISS index creation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS # Alias to avoid confusion
from langchain.docstore.document import Document # For creating LangChain documents
from langchain.docstore.in_memory import InMemoryDocstore # For the document store
import logging
import uuid # For generating unique docstore IDs
import torch # Import torch to check for GPU
import time # For ETR calculation
from collections import defaultdict # For easier handling of per-category data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
ABSTRACT_DIR = "/kaggle/input/embeddings"  # Directory where the input files are located locally
DOCS_METADATA_PATH = os.path.join(ABSTRACT_DIR, "abstract_metadata.pkl")
EMBEDDINGS_NUMPY_PATH = os.path.join(ABSTRACT_DIR, "abstract_embeddings.npy")
OUTPUT_FAISS_DIR = "/kaggle/working/langchain_faiss_store_optimized"  # Output directory, can be local too


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Optimization Configuration (re-added from old script) ---
DOCSTORE_ADD_BATCH_SIZE = 100_000  # Adjust based on memory and dataset size
PROGRESS_LOG_INTERVAL = 100_000 # For ETR updates. Can be same as batch size or different.

# --- KNOWN MAIN CATEGORIES (from models.py) ---
KNOWN_MAIN_CATEGORIES = [
    "math", "cs", "physics", "astro-ph", "cond-mat",
    "stat", "q-bio", "q-fin", "nlin", "gr-qc", "hep-th", "quant-ph"
]

# Helper function to format seconds into HH:MM:SS (re-added from old script)
def format_time(seconds):
    """Converts seconds to HH:MM:SS string format."""
    if seconds < 0: seconds = 0 # Handle potential negative ETR if rate calculation is off
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to get categories from a document dictionary
def get_doc_categories(doc_dict):
    """
    Extracts raw categories from doc_dict, maps them to KNOWN_MAIN_CATEGORIES,
    and returns a list of unique main categories for indexing.
    If no mapping is found, returns ["uncategorized"].
    """
    raw_categories_input = doc_dict.get('categories', [])
    processed_raw_categories = []

    if isinstance(raw_categories_input, str):
        # Handle comma-separated string or single string
        if ',' in raw_categories_input:
            processed_raw_categories = [cat.strip() for cat in raw_categories_input.split(',') if cat.strip()]
        else:
            processed_raw_categories = [raw_categories_input.strip()] if raw_categories_input.strip() else []
    elif isinstance(raw_categories_input, list):
        # Handle list of categories
        processed_raw_categories = [str(cat).strip() for cat in raw_categories_input if str(cat).strip()]

    if not processed_raw_categories:
        return ["uncategorized"] # Default if no categories found in doc at all

    mapped_main_categories = set()
    for raw_cat in processed_raw_categories:
        # Extract the main part (e.g., "cs.AI" -> "cs", "hep-th" -> "hep-th")
        main_part = raw_cat.split('.')[0]
        if main_part in KNOWN_MAIN_CATEGORIES:
            mapped_main_categories.add(main_part)
    
    if not mapped_main_categories:
        # If raw categories existed but none mapped to KNOWN_MAIN_CATEGORIES
        return ["uncategorized"]
    
    return list(mapped_main_categories)


# Function to create comprehensive text content from multiple fields (from new script)
def create_comprehensive_text_content(doc_dict):
    """
    Combine multiple fields to create rich text content for better embedding.
    This improves retrieval by including title, categories, authors, and abstract.
    """
    text_parts = []
    
    # Add title (most important for matching)
    title = doc_dict.get('title', '').strip()
    if title:
        text_parts.append(f"Title: {title}")
    
    # Add categories (crucial for subject filtering)
    categories = doc_dict.get('categories', [])
    if categories:
        if isinstance(categories, list):
            categories_str = ', '.join(categories)
        else:
            categories_str = str(categories)
        text_parts.append(f"Categories: {categories_str}")
    
    # Add authors (useful for author searches)
    authors = doc_dict.get('authors', [])
    if authors:
        if isinstance(authors, list):
            authors_str = ', '.join(authors)
        else:
            authors_str = str(authors)
        text_parts.append(f"Authors: {authors_str}")
    
    # Add ArXiv ID (for exact matching)
    arxiv_id = doc_dict.get('arxiv_id', '').strip()
    if arxiv_id:
        text_parts.append(f"ArXiv ID: {arxiv_id}")
    
    # Add abstract (the main content)
    abstract = doc_dict.get('abstract', '').strip()
    if abstract:
        text_parts.append(f"Abstract: {abstract}")
    
    # Combine all parts with double newlines for clear separation
    comprehensive_text = '\n\n'.join(text_parts)
    
    if not comprehensive_text.strip():
        logger.warning("No text content could be extracted from document fields.")
        return None
        
    return comprehensive_text

def main():
    logger.info("Starting Optimized LangChain FAISS index creation process (with GPU attempt) - CATEGORIZED OUTPUT...")
    print("\n--- Script Start (Categorized Output) ---")

    if not os.path.exists(ABSTRACT_DIR):
        print(f"!!! ERROR: Input directory not found: {ABSTRACT_DIR}.")
        logger.error(f"Input directory not found: {ABSTRACT_DIR}. Aborting.")
        return

    if not (os.path.exists(DOCS_METADATA_PATH) and
            os.path.exists(EMBEDDINGS_NUMPY_PATH)):
        print(f"!!! ERROR: One or more input files not found in {ABSTRACT_DIR}.")
        logger.error(f"One or more input files not found in {ABSTRACT_DIR}. Aborting.")
        return

    print(f"--- Input directory and files found: {ABSTRACT_DIR} ---")
    logger.info("Input directory and files verified. Proceeding...")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FAISS_DIR):
        try:
            os.makedirs(OUTPUT_FAISS_DIR)
            logger.info(f"Created base output directory: {OUTPUT_FAISS_DIR}")
            print(f"--- Created base output directory: {OUTPUT_FAISS_DIR} ---")
        except OSError as e:
            logger.error(f"Could not create base output directory {OUTPUT_FAISS_DIR}: {e}. Aborting.")
            print(f"!!! ERROR: Could not create base output directory {OUTPUT_FAISS_DIR}: {e}. Aborting.")
            return

    try:
        print("--- Entering main processing try block ---")
        logger.info("Starting core processing within try block...")

        # Determine device for embedding model and FAISS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Attempting to use device: {device} for HuggingFaceEmbeddings and FAISS index.")
        print(f"--- Determined device: {device} ---")

        model_kwargs = {'device': device}
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} with model_kwargs: {model_kwargs}")
        print(f"--- Loading embedding model: {EMBEDDING_MODEL_NAME} ---")
        lc_embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs
        )
        model_device_info = 'N/A'
        if hasattr(lc_embedding_function, 'client') and hasattr(lc_embedding_function.client, 'device'):
             model_device_info = str(lc_embedding_function.client.device)
        logger.info(f"HuggingFaceEmbeddings loaded. Device inferred by library: {model_device_info}")
        print(f"--- Embedding model loaded. Inferred device: {model_device_info} ---")

        logger.info(f"Loading document metadata from: {DOCS_METADATA_PATH}")
        print(f"--- Loading metadata from: {DOCS_METADATA_PATH} ---")
        with open(DOCS_METADATA_PATH, "rb") as f:
            docs_data_list = pickle.load(f)
        logger.info(f"Loaded {len(docs_data_list)} documents from metadata.")
        print(f"--- Loaded {len(docs_data_list)} documents from metadata ---")

        logger.info(f"Loading pre-computed embeddings from: {EMBEDDINGS_NUMPY_PATH}")
        print(f"--- Loading embeddings from: {EMBEDDINGS_NUMPY_PATH} ---")
        all_numpy_embeddings = np.load(EMBEDDINGS_NUMPY_PATH).astype('float32')
        logger.info(f"Loaded embeddings with shape {all_numpy_embeddings.shape}.")
        print(f"--- Loaded embeddings with shape {all_numpy_embeddings.shape} ---")

        if len(docs_data_list) != len(all_numpy_embeddings):
            print(f"!!! ERROR: Mismatch between number of documents ({len(docs_data_list)}) and embeddings ({len(all_numpy_embeddings)}). Aborting.")
            logger.error(f"Mismatch between docs ({len(docs_data_list)}) and embeddings ({len(all_numpy_embeddings)}). Aborting.")
            return

        dimension = all_numpy_embeddings.shape[1]

        # --- Data structures for per-category indexing ---
        docstores_by_category = defaultdict(InMemoryDocstore)
        embeddings_by_category = defaultdict(list)
        index_to_docstore_id_by_category = defaultdict(dict)
        doc_batches_by_category = defaultdict(dict) # For batching docstore adds per category
        # For FAISS, indices within a category-specific index will be 0-based
        # So we need a way to map this local FAISS index to the global doc_id
        # The index_to_docstore_id_by_category will store {faiss_idx_in_category_index: doc_id}

        processed_docs_overall = 0
        skipped_docs_overall_text_content = 0
        skipped_docs_overall_doc_creation = 0
        start_time_doc_processing_loop = time.time()
        total_docs_to_process = len(docs_data_list)

        logger.info(f"Processing {total_docs_to_process} documents and assigning to categories...")
        print(f"--- Processing {total_docs_to_process} documents and assigning to categories ---")

        for i, doc_dict in enumerate(docs_data_list):
            doc_embedding = all_numpy_embeddings[i]

            text_content = create_comprehensive_text_content(doc_dict)
            if text_content is None:
                original_doc_id_info = doc_dict.get('arxiv_id', f"original_index_{i}")
                logger.warning(f"Doc (ID: {original_doc_id_info}) could not generate comprehensive text content. Skipping this document for all categories.")
                skipped_docs_overall_text_content += 1
                continue

            doc_categories = get_doc_categories(doc_dict)
            if not doc_categories:
                doc_categories = ["uncategorized"] # Default category if none are specified
                logger.info(f"Document {doc_dict.get('arxiv_id', f'original_index_{i}')} has no categories, assigning to 'uncategorized'.")

            metadata = doc_dict.copy()
            # Ensure 'categories' in metadata is the list we are using for assignment
            metadata['categories'] = doc_categories 

            try:
                # We create one Document object, then assign its generated UUID to multiple category stores if needed.
                # This avoids re-creating the Document object for each category, keeping metadata consistent.
                # The doc_id will be unique across all categories.
                doc_id = str(uuid.uuid4()) 
                document_obj = Document(page_content=text_content, metadata=metadata)
            except Exception as doc_e:
                original_doc_id_info = doc_dict.get('arxiv_id', f"original_index_{i}")
                print(f"!!! ERROR creating Document for {original_doc_id_info}: {doc_e}. Skipping this document for all categories.")
                logger.error(f"Error creating Document for {original_doc_id_info}: {doc_e}. Skipping for all categories.", exc_info=True)
                skipped_docs_overall_doc_creation += 1
                continue

            for category_name in doc_categories:
                # Sanitize category_name to be a valid directory name
                sane_category_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in category_name).strip('_')
                if not sane_category_name: sane_category_name = "default_category"


                current_category_docstore = docstores_by_category[sane_category_name]
                current_category_embeddings = embeddings_by_category[sane_category_name]
                current_category_idx_to_doc_id_map = index_to_docstore_id_by_category[sane_category_name]
                current_category_batch = doc_batches_by_category[sane_category_name]
                
                # Add document to the batch for this category's docstore
                current_category_batch[doc_id] = document_obj
                
                # Add embedding
                current_category_embeddings.append(doc_embedding)
                
                # Map local FAISS index for this category to the global doc_id
                # The local FAISS index will be len(current_category_embeddings) - 1
                faiss_idx_in_category = len(current_category_embeddings) - 1
                current_category_idx_to_doc_id_map[faiss_idx_in_category] = doc_id

                # Check if the batch for this category is full
                if len(current_category_batch) >= DOCSTORE_ADD_BATCH_SIZE:
                    logger.info(f"Adding batch of {len(current_category_batch)} docs to Docstore for category '{sane_category_name}'.")
                    current_category_docstore.add(current_category_batch)
                    doc_batches_by_category[sane_category_name] = {} # Clear batch for this category
            
            processed_docs_overall +=1
            if processed_docs_overall % PROGRESS_LOG_INTERVAL == 0 and processed_docs_overall > 0:
                current_time = time.time()
                elapsed_time_loop = current_time - start_time_doc_processing_loop
                if elapsed_time_loop > 0:
                    processing_rate_loop = processed_docs_overall / elapsed_time_loop
                    items_remaining_loop = total_docs_to_process - processed_docs_overall
                    if processing_rate_loop > 0:
                        etr_seconds_loop = items_remaining_loop / processing_rate_loop
                        etr_formatted_loop = format_time(etr_seconds_loop)
                        elapsed_formatted_loop = format_time(elapsed_time_loop)
                        logger.info(f"Document Processing Progress: {processed_docs_overall}/{total_docs_to_process}. Elapsed: {elapsed_formatted_loop}. ETR: {etr_formatted_loop}.")
                        print(f"--- Document Processing Progress: {processed_docs_overall}/{total_docs_to_process}. Elapsed: {elapsed_formatted_loop}. ETR: {etr_formatted_loop}. ---")

        logger.info(f"Finished processing {processed_docs_overall} documents. Skipped (no text content): {skipped_docs_overall_text_content}. Skipped (doc creation error): {skipped_docs_overall_doc_creation}.")
        print(f"--- Finished processing {processed_docs_overall} documents. ---")
        
        # Add any remaining document batches to their respective category docstores
        logger.info("Adding remaining document batches to respective category docstores...")
        print("--- Adding remaining document batches to respective category docstores... ---")
        for sane_category_name, batch_dict in doc_batches_by_category.items():
            if batch_dict: # If there are any documents left in the batch for this category
                logger.info(f"Adding final batch of {len(batch_dict)} docs to Docstore for category '{sane_category_name}'.")
                docstores_by_category[sane_category_name].add(batch_dict)
        logger.info("Finished adding remaining document batches.")
        print("--- Finished adding remaining document batches. ---")

        del all_numpy_embeddings # Free up memory
        logger.info("Deleted original bulk numpy_embeddings array from memory.")
        print("--- Deleted original bulk numpy_embeddings array from memory ---")

        # --- Build and Save FAISS index for each category ---
        total_categories = len(docstores_by_category)
        logger.info(f"Found {total_categories} unique categories. Building and saving FAISS index for each...")
        print(f"--- Found {total_categories} unique categories. Building and saving FAISS index for each... ---")

        for cat_idx, (category_name, cat_docstore) in enumerate(docstores_by_category.items()):
            logger.info(f"Processing category {cat_idx + 1}/{total_categories}: '{category_name}'")
            print(f"--- Processing category {cat_idx + 1}/{total_categories}: '{category_name}' ---")

            cat_embeddings_list = embeddings_by_category[category_name]
            cat_idx_to_doc_id_map = index_to_docstore_id_by_category[category_name]

            if not cat_embeddings_list:
                logger.warning(f"No embeddings found for category '{category_name}'. Skipping index creation for this category.")
                print(f"!!! WARNING: No embeddings for category '{category_name}'. Skipping. !!!")
                continue

            cat_numpy_embeddings = np.array(cat_embeddings_list).astype('float32')
            logger.info(f"Category '{category_name}': Found {len(cat_numpy_embeddings)} embeddings with shape {cat_numpy_embeddings.shape}.")
            print(f"--- Category '{category_name}': {len(cat_numpy_embeddings)} embeddings, shape {cat_numpy_embeddings.shape} ---")

            # Build raw FAISS index for the category
            cat_raw_faiss_index = None
            cat_faiss_res = None
            if device == "cuda":
                try:
                    cat_faiss_res = faiss.StandardGpuResources()
                    cat_raw_faiss_index = faiss.IndexFlatL2Gpu(cat_faiss_res, dimension, 0)
                    logger.info(f"FAISS Index for '{category_name}' built successfully on GPU.")
                except Exception as gpu_e:
                    logger.warning(f"Could not build FAISS index for '{category_name}' on GPU: {gpu_e}. Falling back to CPU.")
                    print(f"!!! WARNING: GPU FAISS for '{category_name}' failed: {gpu_e}. Falling back to CPU. !!!")
                    cat_raw_faiss_index = faiss.IndexFlatL2(dimension)
                    cat_faiss_res = None
            else:
                cat_raw_faiss_index = faiss.IndexFlatL2(dimension)
            
            if cat_raw_faiss_index is None:
                logger.error(f"Failed to create FAISS index for category '{category_name}'. Skipping.")
                print(f"!!! ERROR: Failed to create FAISS for '{category_name}'. Skipping. !!!")
                continue

            logger.info(f"Adding {len(cat_numpy_embeddings)} embeddings to FAISS index for '{category_name}'...")
            cat_raw_faiss_index.add(cat_numpy_embeddings)
            logger.info(f"FAISS index for '{category_name}' built with {cat_raw_faiss_index.ntotal} embeddings.")
            print(f"--- FAISS index for '{category_name}' built with {cat_raw_faiss_index.ntotal} embeddings ---")

            # Create LangChain FAISS object for the category
            # Note: lc_embedding_function is the general one, not category-specific
            try:
                langchain_faiss_for_category = LangChainFAISS(
                    embedding_function=lc_embedding_function, # Re-use the loaded embedding function
                    index=cat_raw_faiss_index,
                    docstore=cat_docstore,
                    index_to_docstore_id=cat_idx_to_doc_id_map
                )
            except Exception as lc_faiss_e:
                logger.error(f"Error creating LangChainFAISS object for category '{category_name}': {lc_faiss_e}. Skipping save.", exc_info=True)
                print(f"!!! ERROR creating LangChainFAISS for '{category_name}': {lc_faiss_e}. Skipping save. !!!")
                # If using GPU, ensure resources are freed for this failed category index
                if cat_faiss_res is not None:
                    del cat_faiss_res
                    logger.info(f"Freed GPU resources for failed FAISS index of category '{category_name}'.")
                if cat_raw_faiss_index is not None and hasattr(cat_raw_faiss_index, 'free'): # some GPU indexes have free
                    cat_raw_faiss_index.free()
                elif cat_raw_faiss_index is not None and hasattr(cat_raw_faiss_index, 'reset'): # CPU indexes have reset
                     cat_raw_faiss_index.reset()
                del cat_raw_faiss_index # Explicitly delete to help GC
                continue # Skip to the next category


            # Save the LangChain FAISS index for the category
            category_output_dir = os.path.join(OUTPUT_FAISS_DIR, category_name)
            if not os.path.exists(category_output_dir):
                os.makedirs(category_output_dir)
            
            # LangChainFAISS.save_local takes folder_path, index_name="index"
            # This will create "index.faiss" and "index.pkl" inside category_output_dir
            try:
                langchain_faiss_for_category.save_local(folder_path=category_output_dir, index_name="index")
                logger.info(f"Saved LangChain FAISS index for category '{category_name}' to: {category_output_dir}")
                print(f"--- Saved LangChain FAISS index for '{category_name}' to: {category_output_dir} ---")
            except Exception as save_e:
                logger.error(f"Error saving LangChainFAISS index for category '{category_name}' to {category_output_dir}: {save_e}", exc_info=True)
                print(f"!!! ERROR saving LangChainFAISS for '{category_name}' to {category_output_dir}: {save_e} !!!")

            # Clean up GPU resources if they were used for this category's index
            if cat_faiss_res is not None:
                del cat_faiss_res # This should free GPU memory associated with this specific index
                logger.info(f"Freed GPU resources for FAISS index of category '{category_name}'.")
            if cat_raw_faiss_index is not None and hasattr(cat_raw_faiss_index, 'free'): # some GPU indexes have free
                cat_raw_faiss_index.free()
            elif cat_raw_faiss_index is not None and hasattr(cat_raw_faiss_index, 'reset'): # CPU indexes have reset
                 cat_raw_faiss_index.reset()
            del cat_raw_faiss_index # Explicitly delete to help GC
            del langchain_faiss_for_category # Explicitly delete to help GC
            del cat_numpy_embeddings # Free category-specific embeddings
            logger.info(f"Cleaned up resources for category '{category_name}'.")

        logger.info("All categories processed.")
        print("--- All categories processed. ---")

    except Exception as e:
        print(f"!!! ERROR in main processing block: {e}")
        logger.error(f"An unexpected error occurred in main processing block: {e}", exc_info=True)
    finally:
        print("--- Script End ---")
        logger.info("Script finished.")


if __name__ == "__main__":
    main()