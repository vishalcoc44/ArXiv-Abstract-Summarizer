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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Read files from a local folder named "arxiv_abstracts"
ABSTRACT_DIR = "arxiv_abstracts"  # Directory where the input files are located locally
DOCS_METADATA_PATH = os.path.join(ABSTRACT_DIR, "abstract_metadata.pkl")
EMBEDDINGS_NUMPY_PATH = os.path.join(ABSTRACT_DIR, "abstract_embeddings.npy")
OUTPUT_FAISS_DIR = "langchain_faiss_store_optimized"  # Output directory, can be local too

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Optimization Configuration (re-added from old script) ---
DOCSTORE_ADD_BATCH_SIZE = 100_000  # Adjust based on memory and dataset size
PROGRESS_LOG_INTERVAL = 100_000 # For ETR updates. Can be same as batch size or different.

# Helper function to format seconds into HH:MM:SS (re-added from old script)
def format_time(seconds):
    """Converts seconds to HH:MM:SS string format."""
    if seconds < 0: seconds = 0 # Handle potential negative ETR if rate calculation is off
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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
    logger.info("Starting Optimized LangChain FAISS index creation process (with GPU attempt)...")
    print("\n--- Script Start ---")

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
        numpy_embeddings = np.load(EMBEDDINGS_NUMPY_PATH).astype('float32')
        logger.info(f"Loaded embeddings with shape {numpy_embeddings.shape}.")
        print(f"--- Loaded embeddings with shape {numpy_embeddings.shape} ---")

        if len(docs_data_list) != len(numpy_embeddings):
            print(f"!!! ERROR: Mismatch between number of documents ({len(docs_data_list)}) and embeddings ({len(numpy_embeddings)}). Aborting.")
            logger.error(f"Mismatch between docs ({len(docs_data_list)}) and embeddings ({len(numpy_embeddings)}). Aborting.")
            return

        # 1. Build the raw FAISS index
        dimension = numpy_embeddings.shape[1]
        logger.info(f"Attempting to build raw FAISS index (dimension {dimension}) on device: {device}...")
        print(f"--- Building raw FAISS index (dimension {dimension}) on device: {device} ---")
        
        raw_faiss_index = None
        faiss_res = None # Initialize faiss_res
        if device == "cuda":
            try:
                logger.info("Attempting to use FAISS GPU resources.")
                print("--- Attempting to use FAISS GPU resources ---")
                faiss_res = faiss.StandardGpuResources()
                raw_faiss_index = faiss.IndexFlatL2Gpu(faiss_res, dimension, 0) # 0 is GPU ID
                logger.info("FAISS Index built successfully on GPU.")
                print("--- FAISS Index built successfully on GPU ---")
            except Exception as gpu_e:
                print(f"!!! WARNING: Could not build FAISS index on GPU: {gpu_e}. Falling back to CPU.")
                logger.warning(f"Could not build FAISS index on GPU: {gpu_e}. Falling back to CPU.")
                raw_faiss_index = faiss.IndexFlatL2(dimension)
                faiss_res = None # Ensure faiss_res is None if GPU failed
                print("--- Falling back to CPU FAISS IndexFlatL2 ---")
        else:
            logger.info("No CUDA device. Building FAISS IndexFlatL2 on CPU.")
            print("--- Building FAISS IndexFlatL2 on CPU ---")
            raw_faiss_index = faiss.IndexFlatL2(dimension)

        if raw_faiss_index is None:
              print("!!! ERROR: Failed to create FAISS index. Aborting.")
              logger.error("Failed to create FAISS index. Aborting.")
              return

        logger.info(f"Adding {len(numpy_embeddings)} embeddings to the raw FAISS index...")
        print(f"--- Adding {len(numpy_embeddings)} embeddings to the raw FAISS index (this may take time)... ---")
        add_embeddings_start_time = time.time()
        raw_faiss_index.add(numpy_embeddings)
        add_embeddings_duration = time.time() - add_embeddings_start_time
        logger.info(f"Raw FAISS index built with {raw_faiss_index.ntotal} embeddings in {format_time(add_embeddings_duration)}.")
        print(f"--- Raw FAISS index built with {raw_faiss_index.ntotal} embeddings in {format_time(add_embeddings_duration)} ---")
        del numpy_embeddings
        logger.info("Deleted numpy_embeddings array from memory.")
        print("--- Deleted numpy_embeddings array from memory ---")

        # 2. Create the LangChain Docstore and index_to_docstore_id mapping (Optimized with batching and ETR)
        logger.info("Creating LangChain Docstore and index_to_docstore_id mapping (Optimized with batching and ETR)...")
        print(f"--- Creating LangChain Docstore ({len(docs_data_list)} docs) and mapping (Optimized with batching and ETR) ---")
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        doc_batch_for_docstore = {}
        num_docs = len(docs_data_list)
        skipped_docs_count = 0

        start_time_docstore_loop = time.time() # Start time for this specific loop

        for i, doc_dict in enumerate(docs_data_list):
            doc_id = str(uuid.uuid4())
            index_to_docstore_id[i] = doc_id # Map FAISS index to its potential docstore ID

            text_content = create_comprehensive_text_content(doc_dict)
            
            if text_content is None:
                logger.warning(f"Doc idx {i} (FAISS ID {i}, generated doc_id {doc_id}) could not generate comprehensive text content. This document's embedding exists in FAISS but its text won't be fully retrievable from LangChain docstore. Ensure this is intended or fix data.")
                skipped_docs_count += 1
                # We still map its FAISS index to a doc_id, but the docstore won't have a meaningful entry.
                # If you want to strictly skip this entry from docstore, remove it from index_to_docstore_id later,
                # but then the direct FAISS index will have entries not reachable via LangChain FAISS wrapper.
                # For consistency with FAISS, we keep the mapping, but the actual docstore entry might be missing/empty.
                # A more robust solution might pre-filter docs_data_list and numpy_embeddings based on text_content presence
                # *before* adding to the raw FAISS index. For now, we assume all FAISS indices are mapped.
                continue # Skip adding this problematic doc to the docstore

            metadata = doc_dict.copy()
            try:
                document = Document(page_content=text_content, metadata=metadata)
                doc_batch_for_docstore[doc_id] = document
            except Exception as doc_e:
                print(f"!!! ERROR creating Document for idx {i} (ID {doc_id}): {doc_e}. Skipping for Docstore.")
                logger.error(f"Error creating Document for idx {i} (ID {doc_id}): {doc_e}. Skipping for Docstore.", exc_info=True)
                skipped_docs_count += 1
            
            # Add to InMemoryDocstore in batches or if it's the last item
            processed_in_loop = i + 1
            if len(doc_batch_for_docstore) >= DOCSTORE_ADD_BATCH_SIZE or processed_in_loop == num_docs:
                if doc_batch_for_docstore:
                    batch_size = len(doc_batch_for_docstore)
                    # logger.info(f"Adding batch of {batch_size} docs to Docstore. Total processed for loop: {processed_in_loop}/{num_docs}.")
                    docstore.add(doc_batch_for_docstore)
                    doc_batch_for_docstore = {}
                # elif processed_in_loop == num_docs: # This else-if is not strictly necessary but keeps the logic clear
                    # logger.info(f"Final iteration reached ({processed_in_loop}/{num_docs}). No pending items in current batch.")
            
            # Overall progress logging with ETR for the docstore loop
            if processed_in_loop % PROGRESS_LOG_INTERVAL == 0 and processed_in_loop > 0:
                current_time = time.time()
                elapsed_time_loop = current_time - start_time_docstore_loop
                
                if elapsed_time_loop > 0:
                    processing_rate_loop = processed_in_loop / elapsed_time_loop # items per second
                    items_remaining_loop = num_docs - processed_in_loop
                    if processing_rate_loop > 0:
                        etr_seconds_loop = items_remaining_loop / processing_rate_loop
                        etr_formatted_loop = format_time(etr_seconds_loop)
                        elapsed_formatted_loop = format_time(elapsed_time_loop)
                        
                        progress_message = (f"Docstore Loop: Processed {processed_in_loop}/{num_docs} ({processed_in_loop/num_docs*100:.2f}%). "
                                            f"Elapsed: {elapsed_formatted_loop}. ETR: {etr_formatted_loop}.")
                        logger.info(progress_message)
                        print(f"--- {progress_message} ---")
                    # else: # Unlikely if elapsed_time_loop > 0
                        # logger.info(f"Docstore Loop: Processed {processed_in_loop}/{num_docs}. Calculating ETR...")
                # else: # Very fast initial processing
                    # logger.info(f"Docstore Loop: Processed {processed_in_loop}/{num_docs}. Calculating ETR...")

        total_time_docstore_loop = time.time() - start_time_docstore_loop
        logger.info(f"Docstore population loop finished in {format_time(total_time_docstore_loop)}. {skipped_docs_count} documents were skipped or failed creation for docstore.")
        print(f"--- Docstore population loop finished in {format_time(total_time_docstore_loop)}. {skipped_docs_count} docs skipped/failed. ---")

        logger.info(f"Docstore contains {len(docstore._dict)} actual documents. index_to_docstore_id map created for {len(index_to_docstore_id)} FAISS entries.")
        print(f"--- Docstore contains {len(docstore._dict)} actual documents. index_to_docstore_id map has {len(index_to_docstore_id)} entries ---")

        if raw_faiss_index.ntotal != len(index_to_docstore_id):
              print(f"!!! CRITICAL WARNING: FAISS index items ({raw_faiss_index.ntotal}) != index_to_docstore_id map size ({len(index_to_docstore_id)}).")
              logger.critical(f"CRITICAL MISMATCH: FAISS items ({raw_faiss_index.ntotal}) vs index_map ({len(index_to_docstore_id)}). This means some FAISS embeddings might not have a corresponding LangChain document via this map.")
        
        # This check is more meaningful now as we're explicitly skipping adding docs to docstore if text_content is None.
        expected_docstore_content_count = len(docs_data_list) - skipped_docs_count
        if len(docstore._dict) != expected_docstore_content_count:
            print(f"!!! WARNING: Docstore content size ({len(docstore._dict)}) != Expected ({expected_docstore_content_count} based on non-skipped docs). This could indicate internal issues with docstore.add or a slight mismatch in skip logic.")
            logger.warning(f"Docstore content discrepancy: Actual {len(docstore._dict)}, Expected {expected_docstore_content_count} (based on non-skipped docs).")


        # Show sample of comprehensive text content for verification
        if len(docs_data_list) > 0:
            sample_doc = docs_data_list[0]
            sample_text = create_comprehensive_text_content(sample_doc)
            if sample_text:
                logger.info(f"Sample comprehensive text content (first 300 chars): {sample_text[:300]}...")
            else:
                logger.warning("Sample document could not generate comprehensive text content.")

        # 3. Instantiate the LangChain FAISS vector store
        logger.info("Instantiating LangChain FAISS wrapper...")
        print("--- Instantiating LangChain FAISS wrapper ---")
        langchain_faiss_store = LangChainFAISS(
            embedding_function=lc_embedding_function, # Used for query embeddings
            index=raw_faiss_index,                     # The pre-built raw FAISS index
            docstore=docstore,                         # The LangChain docstore
            index_to_docstore_id=index_to_docstore_id  # Mapping
        )
        logger.info("LangChain FAISS wrapper instantiated.")
        print("--- LangChain FAISS wrapper instantiated ---")

        # 4. Save the LangChain FAISS store
        logger.info(f"Saving LangChain FAISS index to directory: {OUTPUT_FAISS_DIR}")
        print(f"--- Saving LangChain FAISS index to directory: {OUTPUT_FAISS_DIR} ---")
        if not os.path.exists(OUTPUT_FAISS_DIR):
            os.makedirs(OUTPUT_FAISS_DIR)
            print(f"--- Created output directory: {OUTPUT_FAISS_DIR} ---")

        save_start_time = time.time()
        langchain_faiss_store.save_local(OUTPUT_FAISS_DIR)
        save_duration = time.time() - save_start_time
        logger.info(f"LangChain FAISS index saved successfully in {format_time(save_duration)}.")
        print(f"--- LangChain FAISS index saved successfully in {format_time(save_duration)} ---")
        logger.info(f"IMPORTANT: The new index now includes comprehensive text (title + categories + authors + abstract) for better retrieval!")
        logger.info(f"You can now load the index from '{OUTPUT_FAISS_DIR}'.")
        print(f"--- You can now load the index from '{OUTPUT_FAISS_DIR}' ---")

    except Exception as e:
        print(f"\n!!! AN ERROR OCCURRED DURING FAISS INDEX CREATION !!!")
        print(f"!!! Error details: {e} !!!")
        logger.error(f"An error occurred during FAISS index creation: {e}", exc_info=True)
    finally:
        if 'faiss_res' in locals() and faiss_res is not None:
            logger.info("FAISS GPU resources will be garbage collected.")
            print("--- FAISS GPU resources will be garbage collected ---")
            # faiss_res = None # Let Python GC handle it

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()