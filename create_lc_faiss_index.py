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
# import torch # No longer needed as we are removing GPU specific code

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
# Ensure this matches the key in your abstract_metadata.pkl containing the main text
TEXT_FIELD_IN_METADATA = 'abstract'

def main():
    logger.info("Starting Optimized LangChain FAISS index creation process (CPU mode)...")

    if not (os.path.exists(ABSTRACT_DIR) and
            os.path.exists(DOCS_METADATA_PATH) and
            os.path.exists(EMBEDDINGS_NUMPY_PATH)):
        logger.error(f"One or more input files/directories not found. Please ensure '{ABSTRACT_DIR}' exists and contains 'abstract_metadata.pkl' and 'abstract_embeddings.npy'. Aborting.")
        return

    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (CPU mode)")
        # Initialize HuggingFaceEmbeddings for CPU usage (by not specifying device, it defaults to CPU or SentenceTransformers handles it)
        lc_embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
        logger.info("HuggingFaceEmbeddings loaded for CPU.")

        logger.info(f"Loading document metadata from: {DOCS_METADATA_PATH}")
        with open(DOCS_METADATA_PATH, "rb") as f:
            docs_data_list = pickle.load(f)  # Expected to be a list of dicts
        logger.info(f"Loaded {len(docs_data_list)} documents from metadata.")

        logger.info(f"Loading pre-computed embeddings from: {EMBEDDINGS_NUMPY_PATH}")
        numpy_embeddings = np.load(EMBEDDINGS_NUMPY_PATH).astype('float32')
        logger.info(f"Loaded embeddings with shape {numpy_embeddings.shape}.")

        if len(docs_data_list) != len(numpy_embeddings):
            logger.error(f"Mismatch between number of documents ({len(docs_data_list)}) and embeddings ({len(numpy_embeddings)}). Aborting.")
            return

        # 1. Build the raw FAISS index directly from NumPy embeddings
        dimension = numpy_embeddings.shape[1]
        logger.info(f"Building raw FAISS index (IndexFlatL2) with dimension {dimension}...")
        raw_faiss_index = faiss.IndexFlatL2(dimension)
        raw_faiss_index.add(numpy_embeddings) # Add all embeddings in bulk
        logger.info(f"Raw FAISS index built with {raw_faiss_index.ntotal} total embeddings.")

        # 2. Create the LangChain Docstore and index_to_docstore_id mapping
        logger.info("Creating LangChain Docstore and index_to_docstore_id mapping...")
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        for i, doc_dict in enumerate(docs_data_list):
            text_content = doc_dict.get(TEXT_FIELD_IN_METADATA)
            if text_content is None:
                # If you expect all documents to have text and want to be strict:
                # logger.error(f"Document at original index {i} is missing the text field '{TEXT_FIELD_IN_METADATA}'. Aborting.")
                # return
                # If skipping is acceptable:
                logger.warning(f"Document at original index {i} is missing the text field '{TEXT_FIELD_IN_METADATA}'. This document's embedding exists in FAISS but its text won't be retrievable. Ensure this is intended or fix data.")
                # We still need to map this FAISS index ID, even if to a placeholder or skip adding to docstore.
                # For simplicity here, we'll create a placeholder ID. A more robust solution might involve filtering
                # docs_data_list and numpy_embeddings *before* building the raw_faiss_index if text is missing.
                # However, since raw_faiss_index is already built with all embeddings, we must map all its IDs.
                doc_id = str(uuid.uuid4()) # Placeholder ID
                # Optionally add a placeholder document to docstore or leave it out
                # For now, we'll map the FAISS ID but the docstore might not have a meaningful entry for it if text is missing.
                index_to_docstore_id[i] = doc_id # FAISS index uses 0-based sequential IDs
                continue # Skip adding this problematic doc to the docstore if text is missing

            metadata = doc_dict.copy()
            doc_id = str(uuid.uuid4())
            document = Document(page_content=text_content, metadata=metadata)
            docstore.add({doc_id: document})
            index_to_docstore_id[i] = doc_id

        logger.info(f"Docstore potentially populated. index_to_docstore_id map created for {len(index_to_docstore_id)} FAISS entries.")
        # Verify consistency after potential skips
        if raw_faiss_index.ntotal != len(index_to_docstore_id):
            logger.error(f"Critical error: FAISS index has {raw_faiss_index.ntotal} items, but index_to_docstore_id map has {len(index_to_docstore_id)} entries. This should not happen if all FAISS IDs are mapped. Aborting.")
            return
        logger.info(f"Docstore contains {len(docstore._dict)} actual documents after processing.")


        # 3. Instantiate the LangChain FAISS vector store
        logger.info("Instantiating LangChain FAISS wrapper...")
        langchain_faiss_store = LangChainFAISS(
            embedding_function=lc_embedding_function, # Used for query embeddings
            index=raw_faiss_index,                 # The pre-built raw FAISS index
            docstore=docstore,                     # The LangChain docstore
            index_to_docstore_id=index_to_docstore_id # Mapping
        )
        logger.info("LangChain FAISS wrapper instantiated.")

        # 4. Save the LangChain FAISS store
        logger.info(f"Saving LangChain FAISS index to directory: {OUTPUT_FAISS_DIR}")
        if not os.path.exists(OUTPUT_FAISS_DIR):
            os.makedirs(OUTPUT_FAISS_DIR)
        langchain_faiss_store.save_local(OUTPUT_FAISS_DIR)
        logger.info("LangChain FAISS index saved successfully.")
        logger.info(f"You can now update app2.py to load from '{OUTPUT_FAISS_DIR}'.")

    except Exception as e:
        logger.error(f"An error occurred during FAISS index creation: {e}", exc_info=True)

if __name__ == "__main__":
    main() 