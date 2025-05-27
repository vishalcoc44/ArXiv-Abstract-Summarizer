import os
import pickle
import requests
import numpy as np
import logging
import json
import re
import threading
import time
from llama_cpp import Llama
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
            if AppConfig.CLOUD_DEPLOYMENT and AppConfig.CLOUD_STORAGE_BUCKET:
                self._download_model_from_cloud()
            
            # Determine number of threads for Llama.cpp
            # Default to half of CPU cores if not specified, or 1 if cpu_count fails.
            num_threads = getattr(AppConfig, 'LLAMA_N_THREADS', None)
            if num_threads is None:
                cpu_cores = os.cpu_count()
                if cpu_cores:
                    num_threads = cpu_cores // 2
                else:
                    num_threads = 1 # Fallback if cpu_count is not available
                logger.info(f"AppConfig.LLAMA_N_THREADS not set, defaulting to {num_threads} threads.")
            else:
                logger.info(f"Using AppConfig.LLAMA_N_THREADS: {num_threads} threads.")


            model = Llama(
                model_path=AppConfig.LOCAL_MODEL_PATH,
                n_ctx=AppConfig.LLAMA_N_CTX,
                n_gpu_layers=AppConfig.LLAMA_N_GPU_LAYERS,
                n_batch=AppConfig.LLAMA_N_BATCH,
                n_threads=num_threads, # Added n_threads
                verbose=AppConfig.LLAMA_VERBOSE
            )
            self._model_cache['model'] = model
            logger.info("Model initialized and cached successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}", exc_info=True)
            raise

    def _download_model_from_cloud(self):
        if not os.path.exists(AppConfig.LOCAL_MODEL_PATH):
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(AppConfig.CLOUD_STORAGE_BUCKET)
                blob = bucket.blob(os.path.basename(AppConfig.LOCAL_MODEL_PATH))
                blob.download_to_filename(AppConfig.LOCAL_MODEL_PATH)
                logger.info(f"Downloaded model from cloud storage: {AppConfig.LOCAL_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error downloading model from cloud: {e}", exc_info=True)
                raise

    def cleanup(self):
        with self._model_lock:
            if 'model' in self._model_cache:
                del self._model_cache['model'] 
                logger.info("Model cleaned up and removed from cache")

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
        local_llm = model_manager.get_model() # This will initialize if not already cached by manager
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
    current_cancel_event = get_cancellation_event(chat_id) if chat_id else None # Initialize early
    DEBUG_SKIP_AUGMENTATION = False 
    DEBUG_SKIP_RAG_CONTEXT = False
    # DEBUG_SKIP_GEMINI_SUGGESTIONS = False # This flag is no longer relevant here

    if not systems_status["local_llm_loaded"] or local_llm is None:
        logger.warning("Local LLM not loaded, yielding error message.")
        yield "The local GGUF model is not loaded. Cannot generate a response."
        return

    logger.info(f"Received in generate_local_rag_response (Gemma): {user_query_with_history}")
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
        logger.info(f"Processing as a paper query. New flow: LLM brainstorm -> FAISS verify -> LLM synthesize.")

        # --- NEW Step 1: LLM Brainstorming for Revolutionary Papers ---
        llm_candidate_papers = []
        brainstorm_subject = "the relevant field" # Default
        if extracted_subject_for_gemini_hint:
            brainstorm_subject = extracted_subject_for_gemini_hint
        elif category_key_for_faiss and category_key_for_faiss != "uncategorized":
            friendly_map = {
                "cs": "Computer Science", "math": "Mathematics", "physics": "Physics",
                "astro-ph": "Astrophysics", "cond-mat": "Condensed Matter Physics",
                "stat": "Statistics", "q-bio": "Quantitative Biology", "q-fin": "Quantitative Finance",
                "nlin": "Nonlinear Sciences", "gr-qc": "General Relativity and Quantum Cosmology",
                "hep-th": "High Energy Physics - Theory", "quant-ph": "Quantum Physics"
            }
            brainstorm_subject = friendly_map.get(category_key_for_faiss, category_key_for_faiss.replace("_", " ").title())
        
        logger.info(f"Paper Query: Initiating LLM brainstorming for revolutionary papers in '{brainstorm_subject}'.")

        prompt_for_brainstorming_parts = [
            "<start_of_turn>user",
            f"You are an AI assistant with broad knowledge of scientific literature.",
            f"I am looking for top-tier, revolutionary, and potentially 'world-changing' research paper titles in the field of **{brainstorm_subject}**.",
            f"Based on your general training data, please list up to 10-15 highly influential research paper **TITLES ONLY**.",
            f"**IMPORTANT INSTRUCTIONS FOR THE LIST:**",
            f"1. Each paper title **MUST** be on its own new line.",
            f"2. Do **NOT** include authors.",
            f"3. Do **NOT** include any numbering (e.g., 1., 2.) or bullet points (e.g., -, *).",
            f"4. Do **NOT** include any introductory text like 'Here is the list:' or any summary/concluding text after the list.",
            f"5. Just output the raw titles, each on a new line, and nothing else.",
            "<end_of_turn>",
            "<start_of_turn>model" # Model should directly start with the first title on a new line
            # f"Okay, here are the titles, each on a new line:" # Removed even this preamble for the model
        ]
        prompt_for_brainstorming = "\\n".join(prompt_for_brainstorming_parts)
        
        brainstorm_params = {
            "temperature": min(AppConfig.LLAMA_TEMPERATURE + 0.20, 0.9), # Higher temp for brainstorming
            "top_p": AppConfig.LLAMA_TOP_P,
            "max_tokens": 700 # Increased for potentially longer list of titles/authors
        }

        brainstorm_llm_call_start_time = time.perf_counter()
        brainstormed_text_full = ""
        try:
            logger.info(f"Calling LLM for brainstorming paper titles in {brainstorm_subject} with temp {brainstorm_params['temperature']}.")
            
            brainstorm_response_stream = local_llm.create_chat_completion_openai_v1(
                messages=[{"role": "user", "content": prompt_for_brainstorming}],
                stream=True,
                temperature=brainstorm_params["temperature"],
                top_p=brainstorm_params["top_p"],
                max_tokens=brainstorm_params["max_tokens"]
            )
            
            for chunk in brainstorm_response_stream:
                if current_cancel_event and current_cancel_event.is_set():
                    logger.info(f"Cancellation event detected during LLM brainstorming for chat_id {chat_id}.")
                    raise Exception("LLM Brainstorming Cancelled") 

                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    brainstormed_text_full += delta_content
            
            brainstorm_llm_call_duration = time.perf_counter() - brainstorm_llm_call_start_time
            logger.info(f"[PERF] LLM brainstorming call took: {brainstorm_llm_call_duration:.4f} seconds. Response length: {len(brainstormed_text_full)}")
            logger.debug(f"LLM Brainstormed Text for {brainstorm_subject}:\\n{brainstormed_text_full}")

            # Parse brainstormed_text_full - simplified for titles only, one per line
            raw_lines = [line.strip() for line in brainstormed_text_full.split('\\n') if line.strip()]
            for line_idx, line in enumerate(raw_lines):
                title = line.strip() # Each line is now considered a potential title
                
                if title and len(title) > 5: # Basic filter for very short/empty lines and ensure some substance
                     llm_candidate_papers.append({"title": title, "authors_suggestion": ""}) # No authors from brainstorm
                # else: # Debugging if a line is skipped
                #    logger.debug(f"Skipping potential title line (too short or empty): '{title}'")
            
            if llm_candidate_papers:
                logger.info(f"LLM brainstormed {len(llm_candidate_papers)} candidate paper titles for {brainstorm_subject}.")
            else:
                logger.warning(f"LLM brainstorming did not yield any parseable paper titles for {brainstorm_subject} from raw text: {brainstormed_text_full}")

        except Exception as e_brainstorm:
            logger.error(f"Error during LLM brainstorming call for {brainstorm_subject}: {e_brainstorm}", exc_info=True)
            if "Cancelled" not in str(e_brainstorm): 
                 yield f"An error occurred while trying to brainstorm papers: {str(e_brainstorm)}"
        
        # --- MODIFIED Step 2: FAISS Verification & Augmentation ---
        if llm_candidate_papers and not (current_cancel_event and current_cancel_event.is_set()):
            logger.info(f"Verifying {len(llm_candidate_papers)} LLM-brainstormed papers against local FAISS in category '{category_key_for_faiss}'.")
            current_faiss_index_to_use = load_category_faiss_index(category_key_for_faiss)

            if current_faiss_index_to_use:
                verified_faiss_doc_ids = set()

                for candidate_idx, candidate_paper in enumerate(llm_candidate_papers):
                    if current_cancel_event and current_cancel_event.is_set(): 
                        break 
                    candidate_title = candidate_paper["title"]
                    search_query_for_faiss_verification = candidate_title
                    
                    try:
                        docs_with_scores = current_faiss_index_to_use.similarity_search_with_score(
                            search_query_for_faiss_verification, k=3 
                        )

                        if docs_with_scores:
                            MAX_TITLE_LEN_FOR_LOG = 70
                            for doc, score in docs_with_scores:
                                if current_cancel_event and current_cancel_event.is_set(): 
                                    break

                                faiss_title = doc.metadata.get("title", "").strip()
                                faiss_arxiv_id = doc.metadata.get("arxiv_id", "N/A")
                                unique_doc_id_from_faiss = faiss_arxiv_id if faiss_arxiv_id != "N/A" else faiss_title

                                title_similarity_threshold = 0.80
                                is_similar_title = are_texts_semantically_similar(candidate_title, faiss_title, threshold=title_similarity_threshold)
                                score_is_good = score < 0.8 

                                log_candidate_title_short = candidate_title[:MAX_TITLE_LEN_FOR_LOG] + ('...' if len(candidate_title) > MAX_TITLE_LEN_FOR_LOG else '')
                                log_faiss_title_short = faiss_title[:MAX_TITLE_LEN_FOR_LOG] + ('...' if len(faiss_title) > MAX_TITLE_LEN_FOR_LOG else '')

                                if is_similar_title and score_is_good:
                                    if unique_doc_id_from_faiss not in verified_faiss_doc_ids:
                                        logger.info(f"  VERIFIED: LLM Candidate '{log_candidate_title_short}' matched FAISS doc '{log_faiss_title_short}' (ID: {faiss_arxiv_id}, Score: {score:.3f})")
                                        
                                        authors_list = doc.metadata.get("authors", [])
                                        authors = [str(a).strip() for a in authors_list if str(a).strip()] if isinstance(authors_list, list) else ([a.strip() for a in authors_list.split(',')] if isinstance(authors_list, str) else ["N/A"])
                                        local_abstract_full = doc.metadata.get("abstract", doc.page_content or "N/A").strip()
                                        MAX_ABSTRACT_LEN_FOR_PROMPT = 600
                                        local_abstract_truncated = (local_abstract_full[:MAX_ABSTRACT_LEN_FOR_PROMPT] + "... (truncated)") if len(local_abstract_full) > MAX_ABSTRACT_LEN_FOR_PROMPT else local_abstract_full

                                        papers_for_gemma_synthesis.append({
                                            "title": faiss_title, 
                                            "authors": authors, "arxiv_id": faiss_arxiv_id,
                                            "chosen_abstract": local_abstract_truncated,
                                            "abstract_source": f"Local FAISS ({category_key_for_faiss}) - Verified LLM Suggestion"
                                        })
                                        verified_faiss_doc_ids.add(unique_doc_id_from_faiss)
                                        break 
                    except Exception as e_faiss_verify:
                        logger.error(f"Error verifying LLM candidate '{candidate_title}' in FAISS: {e_faiss_verify}", exc_info=True)
                
                if papers_for_gemma_synthesis:
                    logger.info(f"Successfully verified and retrieved {len(papers_for_gemma_synthesis)} papers from FAISS based on LLM brainstorming.")
                else: 
                    logger.warning(f"LLM brainstormed {len(llm_candidate_papers)} papers, but none could be confidently verified in the local FAISS index for category '{category_key_for_faiss}'.")
            else: 
                logger.warning(f"FAISS index for category '{category_key_for_faiss}' not loaded. Cannot verify LLM brainstormed papers.")
        
        # --- MODIFIED Step 3: LLM Synthesis or Inform User ---
        if current_cancel_event and current_cancel_event.is_set():
             prompt_for_gemma = "" 
        elif papers_for_gemma_synthesis:
            num_verified_papers = len(papers_for_gemma_synthesis)
            current_subject_specific_guidance = (
                f"Your task is to identify **truly groundbreaking, 'top-tier', or 'revolutionary' research articles** within {brainstorm_subject}. "
                f"The user is looking for papers that likely had a **significant, transformative impact** on the field – papers that might be considered 'world-changing' within their specific domain. "
                f"When evaluating the provided document summaries (we have {num_verified_papers} for you to consider), you must prioritize papers whose abstracts strongly suggest they: "
                f"  - Introduced foundational or paradigm-shifting concepts. "
                f"  - Solved major, long-standing open problems. "
                f"  - Presented exceptionally novel methodologies or experimental results that opened up entirely new avenues of research. "
                f"  - Have clear indications of widespread influence or have become cornerstone references (infer this from language suggesting broad applicability, fundamental breakthroughs, etc.). "
                f"**Crucially, you MUST AVOID selecting:** "
                f"- Standard research papers that report incremental advances, unless their abstract very strongly hints at broader, transformative implications. "
                f"- General surveys, textbooks, broad overviews, collections, proceedings, bulletins, or regularly published series. "
                f"- Papers primarily discussing history, bibliography, or minor technical improvements. "
                f"If, after carefully reviewing these {num_verified_papers} document summaries, you cannot identify any papers whose abstracts strongly suggest such revolutionary impact, you should state that you couldn't find papers meeting this high bar from this list, rather than listing less impactful ones. "
                f"Your selection must reflect what a domain expert might consider a landmark paper, based *solely* on the clues within the provided abstracts."
            )

            prompt_instruction_refined = (
                f"You are an AI research assistant. The user asked for revolutionary papers in **{brainstorm_subject}**. "
                f"We first used your general knowledge to suggest some potentially influential papers, and then found the following {num_verified_papers} of them in our local ArXiv database. "
                f"From THIS LIST of {num_verified_papers} document summaries, please apply your strictest judgment to identify and present up to {num_papers_requested} that best exemplify **truly groundbreaking or revolutionary research** in {brainstorm_subject}. "
                f"{current_subject_specific_guidance} "
                f"For each selected paper, provide its Title, Authors, ArXiv ID, and a concise 2-4 sentence summary derived from its provided abstract, highlighting its key concepts or findings and explaining why it stands out as a revolutionary research paper from THIS LIST. "
                f"If fewer than {num_papers_requested} papers from this list meet this extremely high bar for 'revolutionary', list only those that do. "
                f"If none of these {num_verified_papers} papers from the list strongly convey revolutionary impact based on their abstracts, state that clearly."
            )
            
            context_papers_str_parts = []
            for i, p_data in enumerate(papers_for_gemma_synthesis):
                context_papers_str_parts.append(
                    f"Document {i+1} (From local DB, originally suggested by LLM):\n"
                    f"Title: {p_data['title']}\n"
                    f"Authors: {', '.join(p_data['authors']) if p_data['authors'] and p_data['authors'][0] != 'N/A' else 'N/A'}\n"
                    f"ArXiv ID: {p_data['arxiv_id'] if p_data['arxiv_id'] else 'N/A'}\n"
                    f"Abstract:\n{p_data['chosen_abstract']}\n---"
                )
            context_str = "\\n".join(context_papers_str_parts)

            parts = [
                f"{conversation_log_str_for_gemma}",
                f"<start_of_turn>user",
                f"{prompt_instruction_refined}",
                f"\\nUSER'S ORIGINAL QUERY (for context):\\n---\\n{actual_user_question}\\n---", 
                f"\\nCONTEXT DOCUMENT SUMMARIES (LLM-suggested, verified in local DB):\\n===\\n{context_str}\\n===",
                f"Please begin your response now, presenting the papers as requested.",
                f"<end_of_turn>",
                f"<start_of_turn>model",
                f"Okay, I will analyze the {num_verified_papers} provided document summaries (which were suggested by an AI and found in your local database) and select up to {num_papers_requested} that I judge to be the most revolutionary for your query: '{actual_user_question}'."
            ]
            prompt_for_gemma = "\\n".join(parts)
            paper_query_temperature = min(AppConfig.LLAMA_TEMPERATURE + 0.15, 0.85)
            generation_params = {
                "temperature": paper_query_temperature, "top_p": AppConfig.LLAMA_TOP_P,
                "max_tokens": AppConfig.LLAMA_MAX_TOKENS 
            }
            logger.info(f"Using paper query params (LLM-first flow, {num_verified_papers} candidates): {generation_params}")
            logger.debug(f"Gemma prompt for LLM-first paper synthesis (first 500 chars): {prompt_for_gemma[:500]}...")

        else: 
            if not (current_cancel_event and current_cancel_event.is_set()): 
                if not llm_candidate_papers : 
                    message = f"I attempted to identify revolutionary papers in {brainstorm_subject} based on general knowledge to check against your local database, but I couldn't generate a reliable candidate list. You could try rephrasing your request."
                    logger.warning(message)
                    yield message
                else: 
                    message = (f"I identified some potentially influential papers in {brainstorm_subject} based on general knowledge, "
                               f"but unfortunately, I could not find them in your local ArXiv database for the category '{category_key_for_faiss}'. "
                               "Therefore, I cannot list top papers from your local data using that approach.")
                    logger.warning(message)
                    yield message
            prompt_for_gemma = "" 
    
    elif not prompt_for_gemma and not DEBUG_SKIP_AUGMENTATION: 
        logger.info("Proceeding with general RAG / direct response flow (not a paper query, or paper query failed appropriately).")
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

    # current_cancel_event = get_cancellation_event(chat_id) if chat_id else None # MOVED TO TOP
    accumulated_response = ""
    llm_call_start_time = time.perf_counter()
    try:
        logger.info(f"Calling local_llm.create_chat_completion_openai_v1 with prompt length: {len(prompt_for_gemma)}")
        # logger.debug(f"Gemma Prompt:\\n----------\\n{prompt_for_gemma}\\n----------") # Very verbose

        stream = local_llm.create_chat_completion_openai_v1(
            messages=[{"role": "user", "content": prompt_for_gemma}], 
            stream=True,
            temperature=generation_params.get("temperature", AppConfig.LLAMA_TEMPERATURE),
            top_p=generation_params.get("top_p", AppConfig.LLAMA_TOP_P),
            max_tokens=generation_params.get("max_tokens", AppConfig.LLAMA_MAX_TOKENS) 
        )
        
        first_chunk_received = False
        for chunk in stream:
            if current_cancel_event and current_cancel_event.is_set():
                logger.info(f"Cancellation event detected for chat_id {chat_id}. Stopping generation.")
                yield CANCEL_MESSAGE
                break 

            delta = chunk.choices[0].delta
            content = delta.content

            if content:
                if not first_chunk_received:
                    llm_first_chunk_time = time.perf_counter()
                    logger.info(f"[PERF] Time to first token from Gemma: {(llm_first_chunk_time - llm_call_start_time):.4f} seconds.")
                    first_chunk_received = True
                yield content
                accumulated_response += content
        
        if not first_chunk_received and not (current_cancel_event and current_cancel_event.is_set()):
             logger.warning("LLM stream completed but no content was received/yielded (and not cancelled).")

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