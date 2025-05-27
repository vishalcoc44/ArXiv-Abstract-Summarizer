import os
import pickle
import requests
import numpy as np
import logging
import json
import re
import threading
import time
from transformers import AutoTokenizer
import tensorrt_llm.runtime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import AppConfig

logger = logging.getLogger(__name__)

# --- Global Variables & Constants ---
local_llm = None
# faiss_index = None # Will be loaded on demand per category
embeddings_model = None
loaded_faiss_indexes_cache = {} # Cache for category-specific FAISS indexes: {category_path: faiss_index_object}
systems_status = {
    "local_llm_loaded": False,
    # "faiss_index_loaded": False, # This will now reflect base path availability or specific index readiness
    "faiss_base_path_exists": False, # New status to check if the main FAISS directory exists
    "gemini_model_configured": bool(AppConfig.GEMINI_API_KEY)
}
active_cancellation_events = {} # Key: chat_id, Value: threading.Event()
CANCEL_MESSAGE = "\n\n[LLM generation cancelled by user.]"

KNOWN_MAIN_CATEGORIES = [
    "math", "cs", "physics", "astro-ph", "cond-mat",
    "stat", "q-bio", "q-fin", "nlin", "gr-qc", "hep-th", "quant-ph"
]

# Mapping from user-friendly subject names (and common variations) to base ArXiv primary categories
SUBJECT_TO_ARXIV_CATEGORY_PREFIX = {
    "math": "math",
    "mathematics": "math",
    "cs": "cs",
    "computer science": "cs",
    "physics": "physics",
    "astro-ph": "astro-ph",
    "astrophysics": "astro-ph",
    "cond-mat": "cond-mat",
    "condensed matter": "cond-mat",
    "condensed matter physics": "cond-mat",
    "stat": "stat",
    "statistics": "stat",
    "q-bio": "q-bio",
    "quantitative biology": "q-bio",
    "q-fin": "q-fin",
    "quantitative finance": "q-fin",
    "nlin": "nlin",
    "nonlinear sciences": "nlin",
    "gr-qc": "gr-qc",
    "general relativity and quantum cosmology": "gr-qc",
    "hep-th": "hep-th",
    "high energy physics - theory": "hep-th",
    "quant-ph": "quant-ph",
    "quantum physics": "quant-ph"
}

# --- Utility for Sanitizing Category Names ---
def sanitize_category_for_directory(category_name: str) -> str:
    """Sanitizes a category name to be a valid directory name."""
    # Replace characters not suitable for directory names, consistent with create_lc_faiss_index.py
    sane_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in category_name).strip('_')
    if not sane_name:
        # Consider if "default_category" is appropriate or if it should be logged as an error / skipped
        return "default_category" 
    return sane_name

# --- JSON Repair Helper ---
def repair_json_invalid_escapes(json_string: str) -> str:
    out = []
    i = 0
    n = len(json_string)
    valid_json_escape_followers = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

    while i < n:
        char = json_string[i]
        if char == '\\':
            if i + 1 < n:
                next_char = json_string[i+1]
                if next_char in valid_json_escape_followers:
                    out.append(char)
                    out.append(next_char)
                    i += 2
                else:
                    out.append('\\\\') 
                    out.append(next_char)
                    i += 2
            else:
                out.append('\\\\')
                i += 1
        else:
            out.append(char)
            i += 1
            
    repaired_str = "".join(out)
    if repaired_str != json_string:
        logger.info("JSON string was repaired for invalid escapes.")
    return repaired_str

# --- Model Management ---
class ModelManager:
    _instance = None
    _model_cache = {}
    _model_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def __init__(self):
        self.request_queue = []
        if AppConfig.REDIS_URL:
            import redis
            self.redis_client = redis.from_url(AppConfig.REDIS_URL)
        else:
            self.redis_client = None

    def get_model(self):
        with self._model_lock:
            if not self._model_cache:
                self._initialize_model()
            return self._model_cache.get('model')

    def _initialize_model(self):
        try:
            # Cloud download logic remains for now, conceptually it should fetch
            # the base model for TRT-LLM engine building or a pre-built engine.
            if AppConfig.CLOUD_DEPLOYMENT and AppConfig.CLOUD_STORAGE_BUCKET:
                # This might need adjustment if you're storing pre-built TRT engines
                # or the original HF model instead of GGUF.
                # For now, assuming it fetches something to AppConfig.LOCAL_MODEL_PATH
                # which TOKENIZER_PATH might then use.
                self._download_model_from_cloud() 
            
            logger.info(f"Initializing TensorRT-LLM engine.")
            logger.info(f"Tokenizer path: {AppConfig.TOKENIZER_PATH}")
            logger.info(f"TensorRT engine directory: {AppConfig.TENSORRT_ENGINE_DIR}")

            # 1. Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                AppConfig.TOKENIZER_PATH,
                legacy=False, # Recommended for Gemma
                padding_side='left' # Important for batching in TRT-LLM
            )
            # It's crucial that tokenizer.pad_token_id is set. For Gemma, often it's <pad> or <eos>.
            # If not set by default, you might need:
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    logger.info(f"Tokenizer pad_token not set, using eos_token: {tokenizer.eos_token}")
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Add a pad token if neither eos_token nor pad_token exists
                    logger.warning("Tokenizer has no pad_token or eos_token. Adding a new pad_token '<pad>'. This might affect model performance if not handled correctly during training/conversion.")
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})


            # 2. Load the TensorRT-LLM engine
            # The GenerationSession expects the engine directory and the tokenizer.
            # Max batch size and max sequence length are usually configured during engine build.
            # You can override some settings here if needed, consult TRT-LLM docs.
            
            # Create a runtime mapping based on available GPUs.
            # This assumes you want to use all visible CUDA devices.
            # For a single GPU setup, this might be simpler.
            # num_gpus = tensorrt_llm.runtime.world_size() if tensorrt_llm.runtime.world_size() > 0 else 1
            # rank = tensorrt_llm.runtime.rank() if tensorrt_llm.runtime.rank() >=0 else 0
            # For simplicity in a single-node, potentially multi-GPU setup (but often single GPU for local):
            # We will let TRT-LLM handle GPU assignment by default if not using MPI.
            
            model_session = tensorrt_llm.runtime.GenerationSession.from_dir(
                engine_dir=AppConfig.TENSORRT_ENGINE_DIR,
                tokenizer=tokenizer, # Pass the HuggingFace tokenizer
                # max_batch_size= ... , # Usually set during engine build
                # max_isl= ... , # Max input sequence length, usually set during engine build
                # max_beam_width= ... # If using beam search
                # Other session parameters as needed
            )

            self._model_cache['model'] = model_session
            self._model_cache['tokenizer'] = tokenizer # Cache tokenizer as well
            logger.info("TensorRT-LLM engine and tokenizer initialized and cached successfully")
        except Exception as e:
            logger.error(f"Error initializing TensorRT-LLM model: {e}", exc_info=True)
            raise

    def _download_model_from_cloud(self):
        # This function might need adjustment based on what you store in the cloud.
        # If you store the Hugging Face model for tokenization/TRT conversion:
        # AppConfig.LOCAL_MODEL_PATH (now TOKENIZER_PATH) would be the target.
        # If you store pre-built TRT engines, then AppConfig.TENSORRT_ENGINE_DIR is the target.
        # For now, let's assume AppConfig.TOKENIZER_PATH is the destination for HF model files.
        
        target_path_for_download = AppConfig.TOKENIZER_PATH # Or engine path if downloading engines
        
        # A simple check if it's a directory path for an engine or a model identifier
        # This is a heuristic. Better to have separate config for source model vs engine.
        is_likely_hf_model_id = not os.path.isdir(target_path_for_download) and not os.path.isfile(target_path_for_download)

        if is_likely_hf_model_id:
            logger.info(f"'{target_path_for_download}' appears to be a HuggingFace model ID. Assuming it will be downloaded by transformers.AutoTokenizer or is locally cached by HF.")
            # No explicit download here if it's an ID; transformers library handles it.
            # If it were a path to a *file* you expected to download (like a specific config.json),
            # then the original logic might apply after adjusting target_path_for_download.
            return

        # If target_path_for_download is a directory that should be populated from cloud
        if not os.path.exists(target_path_for_download): # or specific files within it
            logger.info(f"Target path for download '{target_path_for_download}' does not exist. Attempting cloud download.")
            os.makedirs(target_path_for_download, exist_ok=True) # Ensure dir exists
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(AppConfig.CLOUD_STORAGE_BUCKET)
                
                # This part needs to be smarter: list blobs in a "prefix" that corresponds
                # to the model/engine name and download them into target_path_for_download.
                # For a TRT engine, this means downloading all files in the engine directory.
                # For an HF model, it means downloading config, tokenizer files, model weights etc.
                # This is a placeholder for more robust cloud download logic.
                # Example: Downloading files prefixed with os.path.basename(AppConfig.TOKENIZER_PATH)
                # or os.path.basename(AppConfig.TENSORRT_ENGINE_DIR)
                
                # Simplified: if TOKENIZER_PATH was 'my_gemma_model_folder' and CLOUD_STORAGE_BUCKET had 'my_gemma_model_folder/config.json', etc.
                # This example assumes downloading specific *files* into the target directory,
                # which might be too simple for a whole TRT engine or HF model structure.
                
                # For now, this part is highly conceptual and would need to match your GCS layout.
                # If AppConfig.CLOUD_STORAGE_BUCKET contains 'my_engine_files/' which has the TRT engine:
                # engine_prefix = os.path.basename(AppConfig.TENSORRT_ENGINE_DIR) + "/" 
                # blobs = bucket.list_blobs(prefix=engine_prefix)
                # for blob in blobs:
                #    destination_file_name = os.path.join(AppConfig.TENSORRT_ENGINE_DIR, blob.name[len(engine_prefix):])
                #    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
                #    blob.download_to_filename(destination_file_name)
                # logger.info(f"Downloaded files with prefix '{engine_prefix}' to '{AppConfig.TENSORRT_ENGINE_DIR}'")
                
                logger.warning(f"Cloud download logic in _download_model_from_cloud needs to be adapted for TensorRT engines or full HF model structures. Current implementation is a placeholder.")
                # If LOCAL_MODEL_PATH was still pointing to a single GGUF-like file:
                # blob = bucket.blob(os.path.basename(AppConfig.LOCAL_MODEL_PATH)) # Original logic
                # blob.download_to_filename(AppConfig.LOCAL_MODEL_PATH) # Original logic
                # logger.info(f"Downloaded model from cloud storage: {AppConfig.LOCAL_MODEL_PATH}") # Original logic

            except ImportError:
                logger.error("google-cloud-storage library not found. Cannot download from cloud.")
            except Exception as e:
                logger.error(f"Error downloading model/engine from cloud: {e}", exc_info=True)
                # raise # Decide if this should be fatal
        else:
            logger.info(f"Target path '{target_path_for_download}' already exists. Skipping cloud download.")


    def cleanup(self):
        with self._model_lock:
            if 'model' in self._model_cache:
                # TensorRT-LLM session might not have an explicit close/del like Llama object.
                # Deleting from cache should allow garbage collection.
                del self._model_cache['model']
                logger.info("TensorRT-LLM model session removed from cache.")
            if 'tokenizer' in self._model_cache:
                del self._model_cache['tokenizer']
                logger.info("Tokenizer removed from cache.")

def cleanup_local_llm(): 
    global local_llm
    if local_llm:
        logger.info("Local LLM instance cleaned up (or would be if explicit cleanup needed).")
    local_llm = None


# --- System Initialization ---
def initialize_systems():
    global local_llm, embeddings_model, systems_status, loaded_faiss_indexes_cache # faiss_index removed from globals here
    logger.info("Initializing systems...")
    loaded_faiss_indexes_cache.clear() # Clear cache on re-initialization

    # Always use ModelManager to get the local LLM instance
    try:
        model_manager = ModelManager.get_instance()
        # local_llm = model_manager.get_model() # This will initialize if not already cached by manager
        # Instead of getting model, we ensure its components are cached.
        # The actual 'local_llm' global variable will store the TRT GenerationSession.
        if not model_manager._model_cache.get('model') or not model_manager._model_cache.get('tokenizer'):
            logger.info("ModelManager cache is empty, attempting to initialize model and tokenizer.")
            model_manager._initialize_model() # Initialize if not done

        local_llm = model_manager._model_cache.get('model') # Get the TRT session
        # We also need the tokenizer available, though it's not directly assigned to a global here.
        # It will be accessed via model_manager.get_tokenizer() or similar if we add that.
        
        systems_status["local_llm_loaded"] = bool(local_llm)
        if local_llm:
            logger.info("Local LLM instance obtained/initialized via ModelManager.")
        else:
            logger.error("Failed to obtain Local LLM instance via ModelManager.")
    except Exception as e:
        logger.error(f"Error initializing/getting model through ModelManager: {e}", exc_info=True)
        local_llm = None
        systems_status["local_llm_loaded"] = False

    try:
        logger.info(f"Loading embedding model: {AppConfig.EMBEDDING_MODEL_NAME}")
        embeddings_model = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")

        # Check if the base FAISS directory exists, where categorized indexes are stored
        if os.path.exists(AppConfig.FAISS_INDEX_PATH) and os.path.isdir(AppConfig.FAISS_INDEX_PATH):
            logger.info(f"Base FAISS directory for categorized indexes found at: {AppConfig.FAISS_INDEX_PATH}")
            systems_status["faiss_base_path_exists"] = True
        else:
            logger.warning(f"Base FAISS directory for categorized indexes NOT found or is not a directory at {AppConfig.FAISS_INDEX_PATH}. RAG with local model will require category-specific indexes to exist within this path.")
            systems_status["faiss_base_path_exists"] = False
            
    except Exception as e:
        logger.error(f"Error initializing RAG components (embeddings or FAISS base path check): {e}", exc_info=True)
        embeddings_model = None
        systems_status["faiss_base_path_exists"] = False

    if not AppConfig.GEMINI_API_KEY:
        logger.warning("Gemini API key not provided. Gemini REST API calls will fail.")
        systems_status["gemini_model_configured"] = False
    else:
        logger.info("Gemini API key found. Ready for REST API calls.")
        systems_status["gemini_model_configured"] = True

    logger.info(f"Initialization complete. System status: {systems_status}")

# --- Model Helper Functions ---
def extract_subject_from_query(query: str, known_categories: list[str]) -> str | None:
    query_lower = query.lower()
    for cat_keyword in known_categories:
        if f" {cat_keyword} " in query_lower or \
           query_lower.startswith(cat_keyword + " ") or \
           query_lower.endswith(" " + cat_keyword) or \
           f" {cat_keyword}." in query_lower or \
           f" {cat_keyword}," in query_lower or \
           query_lower == cat_keyword:
            if cat_keyword == "cs": return "Computer Science"
            if cat_keyword == "cond-mat": return "Condensed Matter Physics"
            return cat_keyword.replace("-", " ").title()
    return None

def are_texts_semantically_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
    if not text1 or not text2:
        return False
    if not embeddings_model:
        logger.warning("Embeddings model not loaded. Cannot perform semantic similarity check.")
        return False

    try:
        max_len_for_emb = 1000 # Max length for texts before embedding
        # This function will still embed individually, used if calling directly with texts.
        # For batch operations, embed separately and use compare_embeddings.
        text1_emb = embeddings_model.embed_query(text1[:max_len_for_emb])
        text2_emb = embeddings_model.embed_query(text2[:max_len_for_emb])
        
        vec1 = np.array(text1_emb)
        vec2 = np.array(text2_emb)

        # Squeeze to 1D if they are 2D with a single vector
        if vec1.ndim > 1 and vec1.shape[0] == 1: vec1 = vec1.squeeze(0)
        if vec2.ndim > 1 and vec2.shape[0] == 1: vec2 = vec2.squeeze(0)
        if vec1.ndim > 1 or vec2.ndim > 1: # If still not 1D, indicates an issue
             logger.warning(f"Embeddings are not 1D after squeeze: vec1 shape {vec1.shape}, vec2 shape {vec2.shape}")
             return False


        if np.all(vec1 == 0) or np.all(vec2 == 0):
            logger.debug("One or both embeddings are zero vectors.")
            return False
            
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        logger.debug(f"Semantic similarity between snippets: {similarity:.4f}")
        return similarity >= threshold
    except Exception as e:
        logger.error(f"Error in are_texts_semantically_similar for texts ('{text1[:50]}...', '{text2[:50]}...'): {e}", exc_info=True)
        return False

def calculate_vector_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    if vec1.ndim > 1 and vec1.shape[0] == 1: vec1 = vec1.squeeze(0)
    if vec2.ndim > 1 and vec2.shape[0] == 1: vec2 = vec2.squeeze(0)
    
    if vec1.ndim > 1 or vec2.ndim > 1:
        logger.warning(f"Input vectors for cosine similarity are not 1D: vec1 {vec1.shape}, vec2 {vec2.shape}")
        return 0.0 # Or raise error

    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)

def check_category_alignment(local_doc_categories: list[str] | None, user_subject_filter: str | None) -> bool:
    if not user_subject_filter: # No filter from user, so it's aligned by default
        return True
    if not local_doc_categories: # Document has no categories, cannot align with a specific filter
        return False

    user_base_category_prefix = SUBJECT_TO_ARXIV_CATEGORY_PREFIX.get(user_subject_filter.lower())
    if not user_base_category_prefix:
        logger.warning(f"User subject filter '{user_subject_filter}' not mapped to an ArXiv category prefix.")
        return False # Or True if you want to be lenient on unknown filters

    for doc_cat in local_doc_categories:
        # Check if the document's category (e.g., 'cs.AI') starts with the user's base category prefix (e.g., 'cs')
        if doc_cat.lower().startswith(user_base_category_prefix):
            return True
    return False

# --- FAISS Index Loading Helper ---
def load_category_faiss_index(category_key: str | None) -> FAISS | None:
    """
    Loads a category-specific FAISS index.
    category_key: The sanitized category name (e.g., 'cs', 'astro_ph') or 'uncategorized'.
                  If None, attempts to load a default/main index if such logic is ever re-added.
                  For now, expects a specific key.
    Returns the loaded FAISS index or None.
    """
    global embeddings_model, loaded_faiss_indexes_cache

    if not embeddings_model:
        logger.error("Embeddings model not available. Cannot load FAISS index.")
        return None

    if not systems_status["faiss_base_path_exists"]:
        logger.warning(f"Base FAISS directory {AppConfig.FAISS_INDEX_PATH} does not exist. Cannot load category index for '{category_key}'.")
        return None

    if category_key is None:
        # This case could be used for a global/uncategorized index if AppConfig.FAISS_INDEX_PATH itself was one.
        # However, with categorized indexes, we expect a specific category_key.
        # For now, let's assume if category_key is None, it means we want the 'uncategorized' one.
        # Or, you might decide this is an error or should point to a default root index if you had one.
        logger.info("No specific category key provided to load_category_faiss_index, defaulting to 'uncategorized'.")
        category_key = "uncategorized"
    
    # Category key should already be sanitized if it comes from KNOWN_MAIN_CATEGORIES
    # or from sanitize_category_for_directory call. If not, sanitize it here.
    # For safety, let's re-ensure it for direct calls.
    sane_category_dir_name = sanitize_category_for_directory(category_key) 

    index_path = os.path.join(AppConfig.FAISS_INDEX_PATH, sane_category_dir_name)

    if index_path in loaded_faiss_indexes_cache:
        logger.debug(f"Returning cached FAISS index for '{sane_category_dir_name}' from {index_path}")
        return loaded_faiss_indexes_cache[index_path]

    if os.path.exists(index_path) and os.path.isdir(index_path):
        logger.info(f"Loading FAISS index for category '{sane_category_dir_name}' from: {index_path}")
        try:
            specific_faiss_index = FAISS.load_local(
                index_path, 
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS index for '{sane_category_dir_name}' loaded successfully.")
            loaded_faiss_indexes_cache[index_path] = specific_faiss_index
            return specific_faiss_index
        except Exception as e:
            logger.error(f"Error loading FAISS index from {index_path} for category '{sane_category_dir_name}': {e}", exc_info=True)
            # Optionally, cache the failure to prevent retries for a short period if desired
            # loaded_faiss_indexes_cache[index_path] = None # Or some marker for failure
            return None
    else:
        logger.warning(f"FAISS index directory not found for category '{sane_category_dir_name}' at {index_path}")
        return None

# --- Model Generation Logic ---
# --- generate_gemini_response and get_gemini_paper_suggestions are being removed ---
# def generate_gemini_response(prompt, chat_id=None):
#     ...
# def get_gemini_paper_suggestions(user_query_for_gemini: str, ...):
#     ...


def generate_local_rag_response(user_query_with_history: str, chat_id=None):
    overall_start_time = time.perf_counter() # Start overall timer
    DEBUG_SKIP_AUGMENTATION = False 
    DEBUG_SKIP_RAG_CONTEXT = False
    # DEBUG_SKIP_GEMINI_SUGGESTIONS = False # This flag is no longer relevant here

    model_manager = ModelManager.get_instance() # Get model manager instance
    trt_session = model_manager._model_cache.get('model') # Get TRT session from cache
    tokenizer = model_manager._model_cache.get('tokenizer') # Get tokenizer from cache

    if not systems_status["local_llm_loaded"] or trt_session is None or tokenizer is None:
        logger.warning("TensorRT-LLM session or tokenizer not loaded, yielding error message.")
        yield "The local TensorRT-LLM model/tokenizer is not loaded. Cannot generate a response."
        return

    logger.info(f"Received in generate_local_rag_response (TensorRT-LLM): {user_query_with_history}")
    lines = user_query_with_history.strip().split('\n')
    raw_last_line = lines[-1] if lines else ""
    actual_user_question = "" 
    if raw_last_line.startswith("User: "):
        actual_user_question = raw_last_line[len("User: "):].strip()
    elif raw_last_line.startswith("Assistant: "): 
        # This case implies the last message was from assistant, question might be earlier or missing.
        # For robust RAG, might need to look further up or rely on overall history context.
        # For now, assume last user turn is what we need or it's a general query.
        # If no "User:" prefix on last line, treat whole line as question.
        actual_user_question = raw_last_line.strip()
    else:
        actual_user_question = raw_last_line.strip()
    
    logger.info(f"Parsed actual_user_question for Gemma: '{actual_user_question}'")
    query_lower_for_type_check = actual_user_question.lower()
    is_paper_query = any(keyword in query_lower_for_type_check for keyword in ["paper", "papers", "suggest paper", "recommend paper", "find paper", "arxiv", "top 5 paper", "top 10 paper", "research article", "publication on"])

    # Determine category for FAISS lookup
    category_key_for_faiss = "uncategorized" # Default
    extracted_subject_for_gemini_hint = None # Retain for potential other uses, though Gemini suggestions are out for this function

    # Attempt to find a matching category from the user query
    # Iterate through the SUBJECT_TO_ARXIV_CATEGORY_PREFIX to find a match
    # This is a simple keyword match; more advanced NLP could be used.
    temp_subject_filter_for_gemini = None
    best_match_len = 0

    for friendly_name, arxiv_prefix in SUBJECT_TO_ARXIV_CATEGORY_PREFIX.items():
        if friendly_name.lower() in query_lower_for_type_check:
            # Prioritize longer matches for more specific friendly names
            if len(friendly_name) > best_match_len:
                best_match_len = len(friendly_name)
                # arxiv_prefix is already the sanitized key we need for directory (e.g., "cs", "math", "astro-ph")
                # Ensure it's one of the KNOWN_MAIN_CATEGORIES that would have a directory.
                if arxiv_prefix in KNOWN_MAIN_CATEGORIES:
                    category_key_for_faiss = sanitize_category_for_directory(arxiv_prefix) # Sanitize just in case, though KNOWN_MAIN_CATEGORIES should be fine
                    temp_subject_filter_for_gemini = friendly_name.title() # For Gemini hint
                else:
                    # This case should be rare if SUBJECT_TO_ARXIV_CATEGORY_PREFIX maps to KNOWN_MAIN_CATEGORIES
                    logger.warning(f"Friendly name '{friendly_name}' maps to '{arxiv_prefix}' which is not in KNOWN_MAIN_CATEGORIES. Falling back to uncategorized for FAISS.")
                    # category_key_for_faiss remains "uncategorized"
                    # temp_subject_filter_for_gemini might still be set if desired, or cleared
    
    if temp_subject_filter_for_gemini:
        extracted_subject_for_gemini_hint = temp_subject_filter_for_gemini
        logger.info(f"Determined FAISS category key: '{category_key_for_faiss}' and Gemini subject hint: '{extracted_subject_for_gemini_hint}' from query: '{actual_user_question}'")
    else:
        logger.info(f"No specific category matched in query. Using FAISS category key: '{category_key_for_faiss}'. Query: '{actual_user_question}'")

    # subject_filter_for_alignment = extracted_subject_for_gemini_hint # This was for aligning Gemini suggestions with local data
    num_papers_requested = None # Default

    if is_paper_query:
        # Try to extract how many papers the user wants
        # Simple patterns, can be made more sophisticated
        match_top_n = re.search(r'\\b(top|show|exactly|give me|list|find|recommend)\\s+(\\d+)\\b', query_lower_for_type_check, re.IGNORECASE)
        if match_top_n:
            try:
                num_papers_requested = int(match_top_n.group(2))
                if not (1 <= num_papers_requested <= 10): 
                    num_papers_requested = 5 # Default if out of range
                logger.info(f"Extracted number of papers requested: {num_papers_requested}")
            except ValueError:
                num_papers_requested = 5 # Default if parsing fails
                logger.info(f"Could not parse number of papers, defaulting to {num_papers_requested}")
        else:
            num_papers_requested = 5 # Default if no specific number is asked
            logger.info(f"No specific number of papers requested, defaulting to {num_papers_requested}")
        
        # The subject_filter_for_alignment is already determined above for the Gemini hint and FAISS key.
        # The original extract_subject_from_query might have slightly different logic, unify if needed.
        # For now, we rely on the category_key_for_faiss and extracted_subject_for_gemini_hint.
        # pass # num_papers_requested logic remains the same # This line is now handled above

    conversation_history_for_gemma_prompt_list = []
    if len(lines) > 1:
        for line in lines[:-1]:
            if line.startswith("User: "):
                conversation_history_for_gemma_prompt_list.append(f"<start_of_turn>user\n{line[len('User: '):].strip()}<end_of_turn>")
            elif line.startswith("Assistant: "):
                conversation_history_for_gemma_prompt_list.append(f"<start_of_turn>model\n{line[len('Assistant: '):].strip()}<end_of_turn>")
    conversation_log_str_for_gemma = "\n".join(conversation_history_for_gemma_prompt_list) if conversation_history_for_gemma_prompt_list else ""

    prompt_for_gemma = ""
    generation_params = {}
    papers_for_gemma_synthesis = [] # Ensure this is initialized for the conditional logic below

    gemma_prompt_build_start_time = time.perf_counter() # General timer for prompt building

    if is_paper_query:
        logger.info(f"Processing as a paper query. Will search local FAISS in category: '{category_key_for_faiss}'.")
        
        # --- This entire block for Gemini suggestions is removed ---
        # gemini_arxiv_papers_info = []
        # if not DEBUG_SKIP_GEMINI_SUGGESTIONS and systems_status["gemini_model_configured"]:
        #    ... (Gemini call and processing) ...
        # if gemini_arxiv_papers_info:
        #    ... (Looping through Gemini suggestions and trying to match in FAISS) ...
        # papers_for_gemma_synthesis.append(paper_data_for_gemma) 
        # --- End of removed Gemini block ---

        # New: Directly search local FAISS for relevant papers
        retrieved_context_docs_for_papers = []
        if not DEBUG_SKIP_RAG_CONTEXT and systems_status["faiss_base_path_exists"]:
            current_faiss_index_to_use = load_category_faiss_index(category_key_for_faiss)
            if current_faiss_index_to_use:
                # Determine how many docs to retrieve. More than requested to give LLM a choice.
                k_for_faiss_search = num_papers_requested + 1 # Further reduced k
                k_for_faiss_search = min(max(k_for_faiss_search, 5), 15) # Ensure it's sensible (e.g., min 5, max 15)
                
                logger.info(f"Searching FAISS category '{category_key_for_faiss}' for '{actual_user_question}' with k={k_for_faiss_search}")
                rag_faiss_search_start_time = time.perf_counter()
                try:
                    docs_with_scores = current_faiss_index_to_use.similarity_search_with_score(
                        actual_user_question, 
                        k=k_for_faiss_search
                    )
                    rag_faiss_search_duration = time.perf_counter() - rag_faiss_search_start_time
                    logger.info(f"[PERF] Local FAISS paper search (k={k_for_faiss_search}) in category '{category_key_for_faiss}' took: {rag_faiss_search_duration:.4f} seconds.")

                    if docs_with_scores:
                        # Filter by score if needed, or take top N directly
                        # For now, let's take all retrieved up to k_for_faiss_search for LLM to process
                        # relevant_docs = [doc for doc, score in docs_with_scores if score < 0.85] # Example score filter
                        retrieved_context_docs_for_papers = [doc for doc, score in docs_with_scores] # Taking all
                        logger.info(f"Retrieved {len(retrieved_context_docs_for_papers)} documents from FAISS for paper query.")
                        
                        # --- Added Logging for FAISS output ---
                        logger.info(f"--- Top FAISS results for query '{actual_user_question}' in category '{category_key_for_faiss}' (before Gemma processing) ---")
                        for i_doc, doc_retrieved in enumerate(retrieved_context_docs_for_papers):
                            doc_title = doc_retrieved.metadata.get('title', 'N/A Title')
                            doc_abstract_snippet = doc_retrieved.metadata.get('abstract', doc_retrieved.page_content or '')[:150]
                            logger.info(f"FAISS Doc {i_doc+1}: Title: {doc_title} | Abstract Snippet: {doc_abstract_snippet}...")
                        logger.info("--- End of FAISS results log ---")
                        # --- End of Added Logging ---

                    else:
                        logger.info(f"No documents returned from FAISS search for paper query in category '{category_key_for_faiss}'.")
                except Exception as e_local_faiss_paper_search:
                    logger.error(f"Error during local FAISS paper search in category '{category_key_for_faiss}': {e_local_faiss_paper_search}", exc_info=True)
            else:
                logger.warning(f"FAISS index for category '{category_key_for_faiss}' could not be loaded for paper query.")
        elif DEBUG_SKIP_RAG_CONTEXT:
            logger.info("DEBUG_SKIP_RAG_CONTEXT is true. Skipping FAISS search for paper query.")
        
        # Populate papers_for_gemma_synthesis directly from FAISS results
        if retrieved_context_docs_for_papers:
            for doc in retrieved_context_docs_for_papers:
                # Extract metadata. Ensure keys exist or provide defaults.
                title = doc.metadata.get("title", "N/A Title").strip()
                authors_list = doc.metadata.get("authors", [])
                if isinstance(authors_list, str): # If authors is a string, try to split
                    authors = [a.strip() for a in authors_list.split(',') if a.strip()]
                elif isinstance(authors_list, list):
                    authors = [str(a).strip() for a in authors_list if str(a).strip()]
                else:
                    authors = ["N/A"]

                arxiv_id = doc.metadata.get("arxiv_id", "N/A").strip()
                local_abstract_full = doc.metadata.get("abstract", doc.page_content if doc.page_content else "N/A").strip()
                
                # Truncate abstract for Gemma's context to reduce prompt size
                MAX_ABSTRACT_LEN_FOR_PROMPT = 600 
                if len(local_abstract_full) > MAX_ABSTRACT_LEN_FOR_PROMPT:
                    local_abstract_truncated = local_abstract_full[:MAX_ABSTRACT_LEN_FOR_PROMPT] + "... (truncated)"
                else:
                    local_abstract_truncated = local_abstract_full

                papers_for_gemma_synthesis.append({
                    "title": title,
                    "authors": authors,
                    "arxiv_id": arxiv_id,
                    "chosen_abstract": local_abstract_truncated, # Use truncated abstract for prompt
                    "abstract_source": f"Local FAISS ({category_key_for_faiss})"
                })
            logger.info(f"Prepared {len(papers_for_gemma_synthesis)} papers from local FAISS for Gemma synthesis.")

        if papers_for_gemma_synthesis:
            # Construct prompt for Gemma to select and summarize N papers from the retrieved set
            context_papers_str_parts = []
            for i, p_data in enumerate(papers_for_gemma_synthesis):
                context_papers_str_parts.append(
                    f"Document {i+1} (Potential Paper):\n"
                    f"Title: {p_data['title']}\n"
                    f"Authors: {', '.join(p_data['authors']) if p_data['authors'] else 'N/A'}\n"
                    f"ArXiv ID: {p_data['arxiv_id'] if p_data['arxiv_id'] else 'N/A'}\n"
                    f"Abstract:\n{p_data['chosen_abstract']}\n---"
                )
            context_str = "\\n".join(context_papers_str_parts)

            # Determine subject for prompt and specific guidance
            subject_for_prompt_display = "relevant" # Default
            subject_guidance_noun = "academic" # Default noun for guidance
            
            if extracted_subject_for_gemini_hint and extracted_subject_for_gemini_hint.lower() != "uncategorized":
                subject_for_prompt_display = extracted_subject_for_gemini_hint.lower()
                subject_guidance_noun = subject_for_prompt_display
            elif category_key_for_faiss and category_key_for_faiss != "uncategorized":
                subject_for_prompt_display = category_key_for_faiss.replace("_", " ").lower()
                subject_guidance_noun = subject_for_prompt_display
            
            subject_specific_guidance = (
                f"Prioritize papers presenting direct research in {subject_guidance_noun} "
                f"(e.g., specific theories, experimental results, new algorithms, models, or analyses within {subject_guidance_noun}) "
                f"over general surveys, textbooks, bibliometric studies, or collections (like 'Selected Papers', 'Proceedings', 'Bulletins', 'This Week\'s Finds'), "
                f"unless these collections themselves are the explicit subject of the query or the abstract clearly indicates a specific, citable research contribution directly relevant to the user's query. "
                f"The user is looking for substantive research contributions, not just any document that mentions '{subject_guidance_noun}'. "
                f"Pay close attention to the titles of the documents; titles like 'Bulletin', 'Selected Works', 'Weekly Finds', or overly broad titles (e.g., 'Graph Theory' as a standalone title for a research paper query) are often indicators that the document might not be a specific research paper but rather a collection, periodical, or textbook chapter. Filter these out unless the abstract clearly indicates a specific, citable research contribution highly relevant to the user's query. "
            )

            prompt_instruction = (
                f"You are an AI research assistant. Based on the user's query and the following retrieved academic document summaries from our local ArXiv database, "
                f"please identify and present up to {num_papers_requested} of the most relevant **{subject_for_prompt_display} research papers that discuss specific {subject_for_prompt_display} topics or findings.** "
                f"{subject_specific_guidance} "
                f"For each selected paper, provide its Title, Authors, ArXiv ID, and a concise 2-4 sentence summary derived from its provided abstract, highlighting its **key {subject_for_prompt_display} concepts or findings** and explaining why it's a relevant {subject_for_prompt_display} research paper in the context of the user's query. "
                f"If fewer than {num_papers_requested} relevant {subject_for_prompt_display} papers are found in the provided context, list only those that are relevant. "
                f"If no relevant {subject_for_prompt_display} papers are found in the context, state that clearly. Do not make up papers or information not present in the provided document summaries."
            )

            parts = [
                f"{conversation_log_str_for_gemma}",
                f"<start_of_turn>user",
                f"{prompt_instruction}",
                f"\\nUSER'S QUERY:\\n---\\n{actual_user_question}\\n---",
                f"\\nCONTEXT DOCUMENT SUMMARIES:\\n===\\n{context_str}\\n===",
                f"Please begin your response now, presenting the papers as requested.",
                f"<end_of_turn>",
                f"<start_of_turn>model",
                f"Okay, I will analyze the provided document summaries and select up to {num_papers_requested} relevant papers for your query: '{actual_user_question}'." 
                # The model should then list them out.
            ]
            prompt_for_gemma = "\\n".join(parts)
            generation_params = {"temperature": AppConfig.LLAMA_TEMPERATURE, "top_p": AppConfig.LLAMA_TOP_P, "max_tokens": AppConfig.LLAMA_MAX_TOKENS} # Potentially allow more tokens for summarization
            logger.info(f"Using paper query params (local FAISS only): {generation_params}")
            logger.debug(f"Gemma prompt for local paper synthesis (first 500 chars): {prompt_for_gemma[:500]}...")
        else: # No papers retrieved from FAISS or DEBUG_SKIP_RAG_CONTEXT
            logger.info(f"No papers retrieved from FAISS for category '{category_key_for_faiss}' to synthesize for the paper query. Proceeding to general RAG or direct response.")
            # This will naturally fall through to the next 'if not papers_for_gemma_synthesis ...' block
            # which handles general RAG or direct response.
            # We set prompt_for_gemma to empty here to ensure it falls through if is_paper_query was true but no papers found.
            prompt_for_gemma = ""

            
    # General RAG or fallback if paper query yielded no specific papers to synthesize (and not DEBUG_SKIP_AUGMENTATION)
    # This block will also be hit if is_paper_query was true but papers_for_gemma_synthesis remained empty.
    if not prompt_for_gemma and not DEBUG_SKIP_AUGMENTATION: # Check if prompt_for_gemma was set by paper query block
        logger.info("Proceeding with general RAG / direct response flow (either not a paper query, or paper query found no local docs).")
        context_docs = []
        
        # The category_key_for_faiss is already determined based on the user query earlier.
        current_faiss_index_to_use_general_rag = load_category_faiss_index(category_key_for_faiss)

        if current_faiss_index_to_use_general_rag and not DEBUG_SKIP_RAG_CONTEXT:
            rag_faiss_search_start_time = time.perf_counter()
            try:
                docs_with_scores = current_faiss_index_to_use_general_rag.similarity_search_with_score(actual_user_question, k=AppConfig.N_RETRIEVED_DOCS)
                rag_faiss_search_duration = time.perf_counter() - rag_faiss_search_start_time
                logger.info(f"[PERF] General RAG FAISS search (k={AppConfig.N_RETRIEVED_DOCS}) in category '{category_key_for_faiss}' took: {rag_faiss_search_duration:.4f} seconds.")

                if docs_with_scores:
                    relevant_docs = [doc for doc, score in docs_with_scores if score < 0.8] 
                    N_CONTEXT_FINAL = 5 
                    context_docs = relevant_docs[:N_CONTEXT_FINAL]
                    logger.info(f"Retrieved {len(docs_with_scores)} docs, filtered to {len(relevant_docs)} relevant, using top {len(context_docs)} for context.")
                else:
                    logger.info("No documents returned from FAISS search for general RAG.")
            except Exception as e_general_faiss:
                logger.error(f"Error during general FAISS search in category '{category_key_for_faiss}': {e_general_faiss}", exc_info=True)
        elif not current_faiss_index_to_use_general_rag:
             logger.warning(f"FAISS index for category '{category_key_for_faiss}' could not be loaded for general RAG. Proceeding without RAG context.")
        elif DEBUG_SKIP_RAG_CONTEXT:
            logger.info("DEBUG_SKIP_RAG_CONTEXT is true. Skipping general FAISS search.")

        context_str_for_gemma = ""
        if context_docs:
            context_parts = [f"Context Document {i+1} (Title: {doc.metadata.get('title', 'Untitled Document')}):\\n{doc.metadata.get('abstract', doc.page_content)}" for i, doc in enumerate(context_docs)]
            context_str_for_gemma = "\n\n---\nRelevant Information Found:\\n" + "\n\n".join(context_parts) + "\n---\""

        prompt_template_general = (
            f"{conversation_log_str_for_gemma}"
            f"<start_of_turn>user\\n{actual_user_question}{context_str_for_gemma}\\n<end_of_turn>\\n"
            f"<start_of_turn>model\\nBased on our conversation and the provided information, here's my response:"
        )
        prompt_for_gemma = prompt_template_general
        generation_params = {
            "temperature": AppConfig.LLAMA_TEMPERATURE, "top_p": AppConfig.LLAMA_TOP_P,
            "max_tokens": AppConfig.LLAMA_MAX_TOKENS
        }
        logger.info(f"Using generation params for Gemma (general RAG): {generation_params}")

    elif DEBUG_SKIP_AUGMENTATION and not papers_for_gemma_synthesis: # Only direct if no paper synthesis AND skipping augmentation
        logger.info("DEBUG_SKIP_AUGMENTATION is true. No RAG context will be added. Direct LLM call.")
        prompt_template_direct = (
            f"{conversation_log_str_for_gemma}"
            f"<start_of_turn>user\\n{actual_user_question}\\n<end_of_turn>\\n"
            f"<start_of_turn>model\\n"
        )
        prompt_for_gemma = prompt_template_direct
        generation_params = { 
            "temperature": max(0.2, AppConfig.LLAMA_TEMPERATURE - 0.1), 
            "top_p": AppConfig.LLAMA_TOP_P, "max_tokens": AppConfig.LLAMA_MAX_TOKENS
        }
        logger.info(f"Using generation params for Gemma (direct call): {generation_params}")
    
    # Log prompt build time after all conditional paths for prompt_for_gemma are set
    gemma_prompt_build_duration = time.perf_counter() - gemma_prompt_build_start_time
    logger.info(f"[PERF] Gemma prompt building (final selected path) took: {gemma_prompt_build_duration:.4f} seconds.")
        
    if not prompt_for_gemma:
        logger.error("Prompt for Gemma is empty. Aborting generation.")
        yield "Error: Could not construct a prompt for the AI model."
        overall_duration = time.perf_counter() - overall_start_time
        logger.info(f"[PERF] Total generate_local_rag_response execution time (aborted): {overall_duration:.4f} seconds.")
        return

    current_cancel_event = get_cancellation_event(chat_id) if chat_id else None
    accumulated_response = ""
    llm_call_start_time = time.perf_counter()
    try:
        logger.info(f"Preparing prompt for TensorRT-LLM with prompt length: {len(prompt_for_gemma)}")
        # logger.debug(f"TensorRT-LLM Prompt (before tokenization):\n----------\n{prompt_for_gemma}\n----------")

        # Tokenize the prompt
        # For Gemma, the prompt format often includes roles like <start_of_turn>user...<end_of_turn><start_of_turn>model...
        # Ensure prompt_for_gemma adheres to the format expected by your TRT engine (built from Gemma).
        
        # The create_chat_completion_openai_v1 took a list of messages.
        # TRT-LLM's session.generate usually takes tokenized input_ids.
        # We need to adapt how prompt_for_gemma (which is a single string) is tokenized.
        # If your Gemma TRT engine was built to understand the full chat history as a single sequence:
        input_text = prompt_for_gemma 
        
        # Max input length for the model (should be known from engine config)
        # This is an example, adjust based on your engine's max_input_len
        max_input_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 2048 
        # Some tokenizers might use max_len or other attributes. If your engine has a specific max input length, use that.
        # engine_max_input_len = trt_session.max_input_len # If available from session

        input_ids_list = tokenizer.encode(
            input_text,
            return_tensors="pt", # Return PyTorch tensors
            truncation=True,
            max_length=max_input_length # Ensure input is not longer than model can handle
        ).tolist() # Convert to list of lists for TRT session

        if not input_ids_list or not input_ids_list[0]:
            logger.error("Tokenized input_ids are empty. Aborting generation.")
            yield "Error: Tokenized input is empty."
            return

        input_lengths = [len(ids) for ids in input_ids_list]

        # Generation parameters
        # These names (max_new_tokens, top_p, temperature) are common for TRT-LLM session.generate
        # Consult TRT-LLM docs for exact parameter names if they differ.
        # Note: beam_width is for beam search. If using sampling, these are typical.
        output_sequence_lengths = generation_params.get("max_tokens", AppConfig.LLAMA_MAX_TOKENS) # Renamed from max_tokens
        
        logger.info(f"Calling TensorRT-LLM session.generate with input_lengths: {input_lengths}, output_seq_len: {output_sequence_lengths}")

        # The generate method handles the streaming internally if callbacks are provided,
        # or returns complete sequences. For streaming like OpenAI, we need to adapt.
        # TRT-LLM's `generate` can be blocking until full output or use a streaming callback.
        # For simplicity here, let's assume we get full output and then stream it.
        # For true token-by-token streaming, you'd use the `stream_progress_callback` 
        # or iterate over a generator if `session.generate` supports that directly.
        
        # This is a simplified call. Refer to TRT-LLM examples for advanced streaming.
        # The `generate` method signature can vary slightly with TRT-LLM versions and engine types.
        # Common parameters include:
        #   batch_input_ids: list of token ID lists
        #   sampling_config: a SamplingConfig object or dict
        #   output_sequence_length: max new tokens
        
        # Construct sampling_config if your TRT-LLM version/setup uses it
        # from tensorrt_llm.runtime import SamplingConfig # May need this import
        # sampling_config = SamplingConfig(
        #    end_id=tokenizer.eos_token_id, # Crucial for stopping
        #    pad_id=tokenizer.pad_token_id,
        #    top_k=None, # Set if using top_k
        #    top_p=generation_params.get("top_p", AppConfig.LLAMA_TOP_P),
        #    temperature=generation_params.get("temperature", AppConfig.LLAMA_TEMPERATURE),
        #    # num_beams=1, # If not using beam search
        # )

        # Simpler parameter passing if supported (check your TRT-LLM version)
        raw_outputs = trt_session.generate(
            batch_input_ids=input_ids_list,
            max_new_tokens=output_sequence_lengths,
            # sampling_config=sampling_config, # Use this if your version needs it
            # Or pass individual sampling params if supported:
            temperature=generation_params.get("temperature", AppConfig.LLAMA_TEMPERATURE),
            top_p=generation_params.get("top_p", AppConfig.LLAMA_TOP_P),
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.pad_token_id,
            # stream_progress_callback= # For true token-by-token streaming, implement a callback
        )
        
        # Process outputs
        # raw_outputs structure depends on your TRT-LLM engine (e.g., if it returns token IDs, log probs, etc.)
        # Assuming it returns a list of output token ID lists (one for each input in the batch)
        # For a single input (batch size 1):
        output_ids_for_first_item = raw_outputs[0] # Assuming output_ids are directly in raw_outputs[0]
                                                # or raw_outputs.output_ids[0] depending on TRT-LLM version.
                                                # This part is highly dependent on the exact structure of `raw_outputs`.
                                                # You MUST inspect `raw_outputs` from your TRT-LLM setup.

        # The output_ids often include the input_ids. We need to slice them off.
        # num_input_tokens = input_lengths[0]
        # generated_token_ids = output_ids_for_first_item[num_input_tokens:]
        
        # It's safer to use the tokenizer's decode method which can often handle this,
        # or if TRT-LLM's output is structured to give only new tokens.
        # This is a common way to decode, but check TRT-LLM docs for how it returns new vs total tokens.
        # If output_ids_for_first_item contains *only new tokens*:
        # generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # If trt_session.generate returns dict-like with 'output_ids' tensor:
        # Example structure: outputs = {'output_ids': tensor_of_shape [batch_size, num_beams, seq_len]}
        # This is a common output structure from TRT-LLM's `generate` method.
        # Ensure to adapt based on the actual output of your session.generate call.
        
        # Let's assume 'raw_outputs' is a structure that holds the output token IDs.
        # A more robust way (adapt to your TRT-LLM output structure):
        # This is a generic placeholder. You NEED to adapt this based on how your
        # `trt_session.generate` returns data.
        # output_ids_tensor = None
        # if isinstance(raw_outputs, dict) and 'output_ids' in raw_outputs:
        #    output_ids_tensor = raw_outputs['output_ids']
        # elif isinstance(raw_outputs, list): # Or if it's a list of lists of tokens
        #    output_ids_tensor = torch.tensor(raw_outputs, device='cuda' if torch.cuda.is_available() else 'cpu') # Assuming torch is imported
        # else: # Fallback, assuming raw_outputs itself is the tensor or list of lists
        #    output_ids_tensor = raw_outputs

        # This part is highly speculative and needs to match your TRT-LLM output.
        # For demonstration, let's assume raw_outputs[0] gives the full sequence for the first batch item.
        full_sequence_ids = raw_outputs[0] # This is a BIG assumption. Inspect your `raw_outputs`.
        num_input_tokens = input_lengths[0]
        
        # Ensure full_sequence_ids is a flat list of tokens if it's nested (e.g. from beam search)
        if isinstance(full_sequence_ids, list) and full_sequence_ids and isinstance(full_sequence_ids[0], list):
            full_sequence_ids = full_sequence_ids[0] # Take the first beam if beam search was used

        newly_generated_token_ids = full_sequence_ids[num_input_tokens:]

        logger.info(f"Number of input tokens: {num_input_tokens}, Number of full sequence tokens: {len(full_sequence_ids)}, Number of new tokens: {len(newly_generated_token_ids)}")

        # Stream the decoded text
        # For simplicity, decoding the whole generated part at once and then "streaming" it.
        # For true token-by-token streaming, the callback approach with trt_session.generate is better.
        first_chunk_received = False
        temp_accumulated_response_for_streaming = ""
        
        # Iteratively decode for pseudo-streaming to match UX of llama-cpp
        # This is NOT true token-by-token streaming from the engine, but streams the final output.
        # For true streaming, integrate with `stream_progress_callback` of TRT-LLM.
        current_output_pos = 0
        CHUNK_SIZE_FOR_YIELD = 5 # Decode and yield N tokens at a time
        
        while current_output_pos < len(newly_generated_token_ids):
            if current_cancel_event and current_cancel_event.is_set():
                logger.info(f"Cancellation event detected for chat_id {chat_id}. Stopping generation.")
                yield CANCEL_MESSAGE
                break
            
            chunk_end_pos = min(current_output_pos + CHUNK_SIZE_FOR_YIELD, len(newly_generated_token_ids))
            token_id_chunk_to_decode = newly_generated_token_ids[current_output_pos:chunk_end_pos]
            
            if not token_id_chunk_to_decode:
                break

            # Decode the small chunk of token IDs
            # Important: Be careful with `skip_special_tokens`. For intermediate chunks, you might not want to skip.
            # For the final chunk, you might. This needs careful handling for multi-token sequences.
            # A robust way is to decode token by token or use tokenizer features for streaming decode.
            
            # Simplistic approach: decode chunk by chunk. This may produce utf-8 errors if a multi-byte
            # character is split across chunks. A more robust streaming decoder from the tokenizer is preferred if available.
            # decoded_text_chunk = tokenizer.decode(token_id_chunk_to_decode, skip_special_tokens=False) # Potentially problematic for partial multi-byte chars

            # Safer: build up the full list of new tokens and decode once, then stream characters from that string.
            # This is what we'll do below the loop for now for simplicity, as true token-by-token decode
            # while handling multi-byte characters and special tokens correctly is complex without tokenizer support.

            current_output_pos = chunk_end_pos
            # This loop is now conceptual for cancellation check. Actual yielding is below.

        if not (current_cancel_event and current_cancel_event.is_set()):
            # Decode all new tokens at once after the loop (if not cancelled)
            full_decoded_new_text = tokenizer.decode(newly_generated_token_ids, skip_special_tokens=True)
            
            # Now stream this `full_decoded_new_text` character by character or word by word for UX
            for char_token in full_decoded_new_text: # Simulate streaming char by char
                if not first_chunk_received:
                    llm_first_chunk_time = time.perf_counter()
                    logger.info(f"[PERF] Time to first char from TensorRT-LLM (post-decode): {(llm_first_chunk_time - llm_call_start_time):.4f} seconds.")
                    first_chunk_received = True
                yield char_token
                accumulated_response += char_token
        
        if not first_chunk_received and not (current_cancel_event and current_cancel_event.is_set()):
             logger.warning("TensorRT-LLM processing completed but no content was yielded (and not cancelled).")

    except Exception as e:
        logger.error(f"Error during local LLM generation: {e}", exc_info=True)
        yield f"\n\n[Error generating response: {str(e)}]"
    finally:
        llm_call_duration = time.perf_counter() - llm_call_start_time
        logger.info(f"[PERF] Local LLM call (total stream time) took: {llm_call_duration:.4f} seconds. Response length: {len(accumulated_response)}")
        if chat_id:
            unregister_cancellation_event(chat_id)
            logger.info(f"Unregistered cancellation event for chat_id {chat_id}")
        
        overall_duration = time.perf_counter() - overall_start_time
        logger.info(f"[PERF] Total generate_local_rag_response execution time: {overall_duration:.4f} seconds.")

# Function to register a cancellation event
def register_cancellation_event(chat_id, event):
    active_cancellation_events[chat_id] = event

def unregister_cancellation_event(chat_id):
    if chat_id in active_cancellation_events:
        del active_cancellation_events[chat_id]

def get_cancellation_event(chat_id):
    return active_cancellation_events.get(chat_id)

# Ensure systems are initialized when the module is loaded if run as part of an app
# initialize_systems() # This might be better called explicitly by the main app.py 