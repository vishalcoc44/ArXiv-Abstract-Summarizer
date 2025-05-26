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
            
            model = Llama(
                model_path=AppConfig.LOCAL_MODEL_PATH,
                n_ctx=AppConfig.LLAMA_N_CTX,
                n_gpu_layers=AppConfig.LLAMA_N_GPU_LAYERS,
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
def generate_gemini_response(prompt, chat_id=None):
    if not systems_status["gemini_model_configured"]:
        yield "Gemini API key not configured. Cannot generate response."
        return

    logger.info(f"Sending prompt to Gemini: {prompt[:200]}...")
    current_cancel_event = get_cancellation_event(chat_id) if chat_id else None

    try:
        response = requests.post(
            AppConfig.GEMINI_API_URL + f"?key={AppConfig.GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": AppConfig.LLAMA_TEMPERATURE, # Using similar settings for consistency
                    "topP": AppConfig.LLAMA_TOP_P,
                    "maxOutputTokens": AppConfig.LLAMA_MAX_TOKENS
                }
            },
            stream=True, # This might not be directly supported by Gemini's non-Vertex AI REST API for generateContent
            timeout=120 # Increased timeout
        )
        response.raise_for_status()

        full_response_content = ""
        # The standard generateContent API might not stream token by token in the same way as an SSE stream.
        # It might stream larger chunks or the full response. Adapt based on observed behavior.
        # For now, assuming it might send JSON chunks if it streams, or one full JSON.
        
        # If response.iter_lines() or response.iter_content() is used, proper decoding is needed.
        # Assuming it's a single JSON response for now, or that stream=True isn't fully effective here.
        
        # Let's try to handle it as if it might be a single JSON object, or a stream of them (less likely for this API endpoint)
        # A more robust solution would inspect Content-Type and handle SSE if present.

        response_json = response.json() # This will block until full response if not streaming effectively
        
        if 'candidates' in response_json and response_json['candidates']:
            parts = response_json['candidates'][0].get('content', {}).get('parts', [])
            for part in parts:
                if 'text' in part:
                    text_chunk = part['text']
                    yield text_chunk
                    full_response_content += text_chunk
        elif 'error' in response_json:
            error_message = response_json['error'].get('message', 'Unknown Gemini API error')
            logger.error(f"Gemini API error: {error_message}")
            yield f"\n\n[Error from Gemini: {error_message}]"
            return

        if not full_response_content:
             yield "\n\n[Gemini returned an empty response.]"

    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request failed: {e}", exc_info=True)
        yield f"\n\n[Error connecting to Gemini: {e}]"
    except Exception as e:
        logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        yield f"\n\n[Error processing Gemini response: {e}]"

def get_gemini_paper_suggestions(user_query_for_gemini: str, conversation_history: str = "", chat_id: str | None = None, subject_hint: str | None = None, num_papers_to_suggest: int | None = None) -> list[dict[str, str | None]]:
    if not systems_status["gemini_model_configured"]:
        logger.warning("Gemini API key not configured. Cannot get paper suggestions.")
        return []

    # Determine number of papers for the prompt
    num_papers_prompt_val = num_papers_to_suggest if num_papers_to_suggest and 1 <= num_papers_to_suggest <= 10 else 5 # Default to 5, max 10

    prompt = (
        f"You are an expert ArXiv research paper discovery assistant. Your goal is to identify and suggest relevant academic papers from ArXiv based on the user's query and conversation history. "
        f"DO NOT MAKE UP PAPERS. Only suggest real papers that you can find or strongly infer exist on ArXiv.\\n\\n"
        f"Conversation History (if any):\\n{conversation_history}\\n\\n"
        f"User's Latest Query: {user_query_for_gemini}\\n\\n"
    )
    if subject_hint:
        prompt += f"The user seems interested in the ArXiv category (or related to): {subject_hint}. Prioritize papers from this field or closely related fields if appropriate, but also consider broader relevance to the query.\\n"
    
    prompt += (
        f"Please suggest exactly {num_papers_prompt_val} relevant papers from ArXiv. "
        f"For each paper, provide the following information in a VALID JSON list format. Each item in the list should be a JSON object with these EXACT keys: \\\"title\\\", \\\"authors\\\" (as a list of strings), \\\"arxiv_id\\\" (as a string, e.g., \\\"2305.12345\\\"), and \\\"abstract\\\" (a concise summary of the paper's abstract, around 2-4 sentences, focusing on key findings and relevance to the query).\\n"
        f"Example of a single paper object in the list: {{\"title\": \\\"Example Paper Title\\\", \\\"authors\\\": [\\\"Author One\\\", \\\"Author Two\\\"], \\\"arxiv_id\\\": \\\"2101.00001\\\", \\\"abstract\\\": \\\"This paper discusses X and presents Y. It is relevant because Z.\\\"}}\\n\\n"
        f"Output only the JSON list of paper objects. Do not include any other text, explanations, or markdown formatting around the JSON. Ensure the JSON is perfectly parsable."
    )
    
    logger.info(f"Sending prompt to Gemini for paper suggestions (query: '{user_query_for_gemini[:100]}...', subject: {subject_hint}, num: {num_papers_prompt_val})")

    try:
        gemini_response_start_time = time.perf_counter()
        response = requests.post(
            AppConfig.GEMINI_API_URL + f"?key={AppConfig.GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4, # Slightly higher for suggestion diversity
                    "topP": 0.95,
                    "maxOutputTokens": 3072, # Increased to accommodate multiple paper details
                    "responseMimeType": "application/json", # Request JSON output
                }
            },
            timeout=180 # Increased timeout for potentially complex search and JSON generation
        )
        gemini_response_duration = time.perf_counter() - gemini_response_start_time
        logger.info(f"[PERF] Gemini API call for paper suggestions took: {gemini_response_duration:.4f} seconds.")
        
        response.raise_for_status()
        
        response_text = response.text
        logger.debug(f"Raw Gemini response for paper suggestions: {response_text[:500]}...")

        # Attempt to extract JSON from the response
        # Gemini with responseMimeType: "application/json" should return just the JSON
        # but sometimes it might be wrapped in markdown or have prefixes/suffixes.
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.info("Extracted JSON from Gemini response using regex (markdown code block).")
        else:
            # If no markdown block, assume the whole response (or a significant part of it) is the JSON
            # This might need more sophisticated cleaning if Gemini adds other text.
            # For now, try to find the start of a list or object.
            first_brace = response_text.find('[')
            first_curly = response_text.find('{')

            if first_brace == -1 and first_curly == -1:
                logger.error(f"No JSON array or object start found in Gemini response: {response_text}")
                return []
            
            if first_brace != -1 and (first_curly == -1 or first_brace < first_curly) :
                json_str = response_text[first_brace:]
            else: # first_curly != -1 and (first_brace == -1 or first_curly < first_brace)
                json_str = response_text[first_curly:]
            
            # Attempt to balance brackets for lists or objects
            # This is a simplistic approach and might not cover all edge cases.
            open_brackets = 0
            last_char_index = -1

            if json_str.startswith('['):
                target_open, target_close = '[', ']'
            elif json_str.startswith('{'): # Should be a list, but as a fallback
                target_open, target_close = '{', '}'
            else: # Should not happen if first_brace/first_curly logic is correct
                logger.error(f"JSON string does not start with '[' or '{{': {json_str[:100]}")
                return []

            for i, char in enumerate(json_str):
                if char == target_open:
                    open_brackets += 1
                elif char == target_close:
                    open_brackets -= 1
                
                if open_brackets == 0:
                    last_char_index = i
                    break
            
            if last_char_index != -1:
                json_str = json_str[:last_char_index+1]
            else:
                logger.warning("Could not balance brackets in JSON response, using potentially truncated string.")


            logger.info(f"Attempting to parse as JSON (guessed boundaries): {json_str[:200]}...")


        # Repair potential invalid escape sequences before parsing
        json_str_repaired = repair_json_invalid_escapes(json_str)

        try:
            data = json.loads(json_str_repaired)
            
            # Attempt to extract the actual list of papers from the response structure
            actual_papers_list = None
            if isinstance(data, dict) and 'candidates' in data:
                candidates = data['candidates']
                if isinstance(candidates, list) and len(candidates) > 0:
                    content = candidates[0].get('content')
                    if isinstance(content, dict) and 'parts' in content:
                        parts = content['parts']
                        if isinstance(parts, list) and len(parts) > 0:
                            text_data = parts[0].get('text')
                            if isinstance(text_data, str):
                                try:
                                    # The text itself is another JSON string representing the list
                                    actual_papers_list = json.loads(text_data)
                                except json.JSONDecodeError as e_inner:
                                    logger.error(f"Failed to parse the 'text' field as JSON: {e_inner}. Text field content: {text_data[:200]}")
                                    actual_papers_list = None # Ensure it's None if parsing fails

            if isinstance(actual_papers_list, list):
                logger.info(f"Successfully parsed {len(actual_papers_list)} paper suggestions from Gemini's nested structure.")
                # Basic validation of expected keys for a few items
                for item in actual_papers_list[:2]: # Check first two items
                    if not all(k in item for k in ["title", "authors", "arxiv_id", "abstract"]):
                        logger.warning(f"Gemini paper suggestion item missing expected keys: {item}")
                        # Decide if to discard item or whole list
                return actual_papers_list
            elif isinstance(data, list): # Fallback for old structure, though logs indicate it's the new dict structure
                logger.warning("Gemini response was a list directly. This might be an older format or unexpected.")
                # This path should ideally not be taken based on current logs
                for item in data[:2]: 
                    if not all(k in item for k in ["title", "authors", "arxiv_id", "abstract"]):
                        logger.warning(f"Gemini paper suggestion item (direct list) missing expected keys: {item}")
                return data
            else:
                logger.error(f"Gemini response parsed but is not a list and expected nested structure not found: {type(data)}. Content: {str(data)[:200]}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError parsing Gemini paper suggestions: {e}. Repaired String: {json_str_repaired[:500]}", exc_info=True)
            return []

    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request failed for paper suggestions: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error processing Gemini paper suggestions response: {e}", exc_info=True)
        return []


def generate_local_rag_response(user_query_with_history: str, chat_id=None):
    overall_start_time = time.perf_counter() # Start overall timer
    DEBUG_SKIP_AUGMENTATION = False 
    DEBUG_SKIP_RAG_CONTEXT = False
    DEBUG_SKIP_GEMINI_SUGGESTIONS = False

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
    extracted_subject_for_gemini_hint = None # For Gemini paper suggestions hint

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

    subject_filter_for_alignment = extracted_subject_for_gemini_hint # Used for check_category_alignment 
    num_papers_requested = None

    if is_paper_query:
        # The subject_filter_for_alignment is already determined above for the Gemini hint and FAISS key.
        # The original extract_subject_from_query might have slightly different logic, unify if needed.
        # For now, we rely on the category_key_for_faiss and extracted_subject_for_gemini_hint.
        pass # num_papers_requested logic remains the same

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
        logger.info("Processing as a paper query with enhanced RAG and synthesis.")
        
        gemini_arxiv_papers_info = []
        if not DEBUG_SKIP_GEMINI_SUGGESTIONS and systems_status["gemini_model_configured"]:
            gemini_call_start_time = time.perf_counter()
            gemini_arxiv_papers_info = get_gemini_paper_suggestions(
                actual_user_question,
                conversation_log_str_for_gemma,
                chat_id,
                subject_hint=extracted_subject_for_gemini_hint, # Use the commonly determined subject
                num_papers_to_suggest=num_papers_requested
            )
            gemini_call_duration = time.perf_counter() - gemini_call_start_time
            logger.info(f"[PERF] Gemini paper suggestions call took: {gemini_call_duration:.4f} seconds.")
        elif DEBUG_SKIP_GEMINI_SUGGESTIONS:
            logger.info("DEBUG_SKIP_GEMINI_SUGGESTIONS is true. Skipping Gemini ArXiv paper suggestions.")

        if gemini_arxiv_papers_info:
            logger.info(f"Received {len(gemini_arxiv_papers_info)} paper suggestions from Gemini for processing.")
            faiss_processing_total_time = 0
            for idx, gemini_suggestion in enumerate(gemini_arxiv_papers_info):
                faiss_iteration_start_time = time.perf_counter()
                title = gemini_suggestion.get("title", "N/A Title").strip()
                authors_raw = gemini_suggestion.get("authors")
                arxiv_id = gemini_suggestion.get("arxiv_id", "").strip()
                gemini_abstract = gemini_suggestion.get("abstract", "").strip()
                max_len_for_emb = 1000 # Consistent with are_texts_semantically_similar

                paper_data_for_gemma = {
                    "title": title, "authors": [], "arxiv_id": arxiv_id,
                    "gemini_abstract": gemini_abstract, "local_abstract": None,
                    "chosen_abstract": gemini_abstract, 
                    "abstract_source": "Gemini (no local match found or preferred)"
                }
                if isinstance(authors_raw, list): paper_data_for_gemma["authors"] = [str(a).strip() for a in authors_raw if str(a).strip()]
                elif isinstance(authors_raw, str): paper_data_for_gemma["authors"] = [a.strip() for a in authors_raw.split(',') if a.strip()]

                # Use the determined category_key_for_faiss for loading the index
                # If is_paper_query is true, we prefer the category determined from the query for targeted search.
                # If a paper from Gemini has a very different category, this could be an issue, but
                # Gemini is hinted with extracted_subject_for_gemini_hint, so they should align generally.
                current_faiss_index_to_use = load_category_faiss_index(category_key_for_faiss)

                if current_faiss_index_to_use and not DEBUG_SKIP_RAG_CONTEXT:
                    search_query_for_faiss = title
                    if arxiv_id: search_query_for_faiss += f" arxiv:{arxiv_id}"
                    
                    best_local_match_info = None
                    faiss_search_call_start_time = time.perf_counter()
                    try:
                        docs_with_scores = current_faiss_index_to_use.similarity_search_with_score(search_query_for_faiss, k=10)
                        faiss_search_call_duration = time.perf_counter() - faiss_search_call_start_time
                        logger.info(f"[PERF] FAISS search for '{title}' (k=10) in category '{category_key_for_faiss}' took: {faiss_search_call_duration:.4f} seconds.")
                        
                        semantic_sim_total_time_iter = 0
                        gemini_title_embedding = None
                        candidate_title_embeddings = []

                        if docs_with_scores and embeddings_model: # Prepare embeddings if we have docs and model
                            try:
                                gemini_title_embedding = np.array(embeddings_model.embed_query(title[:max_len_for_emb]))
                                
                                candidate_titles_for_embedding = [
                                    doc.metadata.get("title", "").strip()[:max_len_for_emb] 
                                    for doc, _ in docs_with_scores
                                ]
                                if candidate_titles_for_embedding:
                                    batched_embeddings = embeddings_model.embed_documents(candidate_titles_for_embedding)
                                    candidate_title_embeddings = [np.array(emb) for emb in batched_embeddings]
                                else: # Should not happen if docs_with_scores is not empty
                                    candidate_title_embeddings = []

                            except Exception as e_embed:
                                logger.error(f"Error pre-computing embeddings for title '{title}': {e_embed}", exc_info=True)
                                # Continue without semantic checks if embeddings fail
                                gemini_title_embedding = None 
                                candidate_title_embeddings = []
                        
                        if docs_with_scores:
                            # Check 1: ID match + Category alignment (Priority)
                            for doc_candidate, score in docs_with_scores:
                                local_arxiv_id_match_val = doc_candidate.metadata.get("arxiv_id", "").strip()
                                if arxiv_id and local_arxiv_id_match_val and arxiv_id.lower() == local_arxiv_id_match_val.lower():
                                    local_categories = doc_candidate.metadata.get("categories")
                                    if isinstance(local_categories, str): local_categories = [c.strip() for c in local_categories.split(',')]
                                    is_aligned = check_category_alignment(local_categories, subject_filter_for_alignment)
                                    if is_aligned:
                                        local_abstract_text = doc_candidate.metadata.get("abstract")
                                        if local_abstract_text:
                                            best_local_match_info = {"abstract_text": local_abstract_text, "source": "Local FAISS (ID Match, Category Aligned)"}
                                            logger.info(f"Local FAISS strong match for '{title}' (ID & Category). Source: {best_local_match_info['source']}")
                                            break 
                            
                            # Check 2: Title similarity + Category alignment
                            if not best_local_match_info:
                                for idx_cand, (doc_candidate, score) in enumerate(docs_with_scores):
                                    candidate_title = doc_candidate.metadata.get("title", "").strip()
                                    title_similar_enough = score < 0.6 
                                    if title_similar_enough:
                                        is_sem_similar = False
                                        if gemini_title_embedding is not None and idx_cand < len(candidate_title_embeddings):
                                            semantic_sim_start_time = time.perf_counter()
                                            similarity_score = calculate_vector_cosine_similarity(gemini_title_embedding, candidate_title_embeddings[idx_cand])
                                            is_sem_similar = similarity_score >= 0.75
                                            semantic_sim_duration = time.perf_counter() - semantic_sim_start_time
                                            semantic_sim_total_time_iter += semantic_sim_duration
                                            logger.debug(f"[PERF] Calculated semantic similarity ({similarity_score:.4f}) for '{title}' vs '{candidate_title}' took: {semantic_sim_duration:.4f}s")
                                        else:
                                            # Fallback or log error if embeddings not available
                                            logger.warning(f"Skipping semantic check for '{title}' vs '{candidate_title}' due to missing embeddings.")

                                        if is_sem_similar:
                                            local_categories = doc_candidate.metadata.get("categories")
                                            if isinstance(local_categories, str): local_categories = [c.strip() for c in local_categories.split(',')]
                                            is_aligned = check_category_alignment(local_categories, subject_filter_for_alignment)
                                            if is_aligned:
                                                local_abstract_text = doc_candidate.metadata.get("abstract")
                                                if local_abstract_text:
                                                    best_local_match_info = {"abstract_text": local_abstract_text, "source": "Local FAISS (Title Match, Category Aligned)"}
                                                    logger.info(f"Local FAISS good match for '{title}' (Title & Category). Source: {best_local_match_info['source']}")
                                                    break # Found a good aligned title match

                            # Check 3: ID match (Category Not Aligned/No Filter) - Fallback
                            if not best_local_match_info:
                                for doc_candidate, score in docs_with_scores:
                                    local_arxiv_id_match_val = doc_candidate.metadata.get("arxiv_id", "").strip()
                                    if arxiv_id and local_arxiv_id_match_val and arxiv_id.lower() == local_arxiv_id_match_val.lower():
                                        local_abstract_text = doc_candidate.metadata.get("abstract")
                                        if local_abstract_text:
                                            best_local_match_info = {"abstract_text": local_abstract_text, "source": "Local FAISS (ID Match, Category N/A)"}
                                            logger.info(f"Local FAISS match for '{title}' (ID, Category N/A). Source: {best_local_match_info['source']}")
                                            break

                            # Check 4: Title similarity (Category Not Aligned/No Filter) - Last Resort Fallback
                            if not best_local_match_info:
                                for idx_cand, (doc_candidate, score) in enumerate(docs_with_scores):
                                    candidate_title = doc_candidate.metadata.get("title", "").strip()
                                    title_similar_enough_fallback = score < 0.5 
                                    if title_similar_enough_fallback:
                                        is_sem_similar_fallback = False
                                        if gemini_title_embedding is not None and idx_cand < len(candidate_title_embeddings):
                                            semantic_sim_start_time = time.perf_counter()
                                            similarity_score_fallback = calculate_vector_cosine_similarity(gemini_title_embedding, candidate_title_embeddings[idx_cand])
                                            is_sem_similar_fallback = similarity_score_fallback >= 0.80
                                            semantic_sim_duration = time.perf_counter() - semantic_sim_start_time
                                            semantic_sim_total_time_iter += semantic_sim_duration
                                            logger.debug(f"[PERF] Calculated fallback semantic similarity ({similarity_score_fallback:.4f}) for '{title}' vs '{candidate_title}' took: {semantic_sim_duration:.4f}s")
                                        else:
                                            logger.warning(f"Skipping fallback semantic check for '{title}' vs '{candidate_title}' due to missing embeddings.")

                                        if is_sem_similar_fallback:
                                            local_abstract_text = doc_candidate.metadata.get("abstract")
                                            if local_abstract_text:
                                                best_local_match_info = {"abstract_text": local_abstract_text, "source": "Local FAISS (Title Match, Category N/A)"}
                                                logger.info(f"Local FAISS fallback match for '{title}' (Title, Category N/A). Source: {best_local_match_info['source']}")
                                                break # Taking the first decent title match if all else fails
                        
                        logger.info(f"[PERF] Total semantic similarity check time for FAISS iteration '{title}': {semantic_sim_total_time_iter:.4f}s")

                        if best_local_match_info:
                            paper_data_for_gemma["local_abstract"] = best_local_match_info["abstract_text"]
                            paper_data_for_gemma["chosen_abstract"] = best_local_match_info["abstract_text"]
                            paper_data_for_gemma["abstract_source"] = best_local_match_info["source"]
                        else:
                            logger.info(f"No suitable local FAISS match found for '{title}' after evaluating top 10 from category '{category_key_for_faiss}'. Using Gemini abstract.")
                    except Exception as e_targeted_faiss:
                        logger.error(f"Error during targeted FAISS search for suggestion '{title}' in category '{category_key_for_faiss}': {e_targeted_faiss}", exc_info=True)
                elif not current_faiss_index_to_use:
                    logger.warning(f"FAISS index for category '{category_key_for_faiss}' could not be loaded. Skipping local FAISS lookup for Gemini suggestion '{title}'.")
                elif DEBUG_SKIP_RAG_CONTEXT:
                    logger.info("DEBUG_SKIP_RAG_CONTEXT is true. Skipping targeted FAISS search.")
                
                papers_for_gemma_synthesis.append(paper_data_for_gemma)
                faiss_iteration_duration = time.perf_counter() - faiss_iteration_start_time
                faiss_processing_total_time += faiss_iteration_duration
                logger.info(f"[PERF] FAISS processing for Gemini suggestion '{title}' took: {faiss_iteration_duration:.4f} seconds.")
            logger.info(f"[PERF] Total FAISS processing time for all Gemini suggestions: {faiss_processing_total_time:.4f} seconds.")
        
        if papers_for_gemma_synthesis:
            parts = ["<start_of_turn>user\\\nYou are an AI research assistant. You will be given a user's query and a list of research papers (with titles, authors, ArXiv IDs, and their original abstracts). Your task is to analyze these papers in relation to the user's query.\nUSER'S QUERY:\n---\n" + actual_user_question + "\n---\nPAPERS:\n==="]
            for i, p in enumerate(papers_for_gemma_synthesis):
                parts.append(f"Paper {i+1}:\nTitle: {p['title']}\
Authors: {(', '.join(p['authors']) if p['authors'] else 'N/A')}\
ArXiv ID: {(p['arxiv_id'] if p['arxiv_id'] else 'N/A')}\
Abstract:\n{p.get('chosen_abstract', 'N/A')}\
---GROUP_SEPARATOR---") # Using a clear separator
            
            parts.append(
                "End of Paper Details.\n\n" 
                "INSTRUCTIONS FOR YOUR RESPONSE:\n"
                "You have been provided with a list of research papers. Each paper's information in the input I gave you is followed by a line that reads '---GROUP_SEPARATOR---'. This line is only for you to know where one paper's input ends and the next begins.\n\n"
                "YOUR TASK:\n"
                "1. Analyze EACH paper in relation to the User's Query.\n"
                "2. For EACH paper, structure your output EXACTLY as follows:\n"
                "   - Start the section for each paper with 'Paper X:' (e.g., 'Paper 1:', 'Paper 2:').\n"
                "   - On the next line, provide '**Title:**' followed by the exact title of the paper.\n"
                "   - On the next line, provide '**Authors:**' followed by the list of authors.\n"
                "   - On the next line, provide '**ArXiv ID:**' followed by the ArXiv ID.\n"
                "   - On the next line, provide '**Summary:**' followed by YOUR 2-4 sentence summary of the paper's given abstract, focusing on its relevance to THE USER'S QUERY. Do NOT simply repeat the abstract. State if the paper seems irrelevant.\n\n"
                "CRITICAL FORMATTING RULES:\n"
                "- **DO NOT** include the '---GROUP_SEPARATOR---' line anywhere in your output. It is an input marker only and MUST NOT be part of your response.\n"
                "- **DO NOT** use any other numbered lists (e.g., 1., 2.) or bulleted lists (e.g., -, *) for the Title, Authors, ArXiv ID, or Summary details within each paper's section.\n"
                "- Each piece of information (Paper X:, Title, Authors, ArXiv ID, Summary) MUST start on a new line.\n"
                "- Use the exact bolded labels ('**Title:**', '**Authors:**', '**ArXiv ID:**', '**Summary:**') as shown.\n\n"
                "EXAMPLE OF OUTPUT FOR ONE PAPER:\n"
                "Paper 1:\n"
                "**Title:** Example Research Paper Title\n"
                "**Authors:** A. N. Other, E. X. Ample\n"
                "**ArXiv ID:** 2401.00001\n"
                "**Summary:** This paper explores innovative methods in Y and is highly relevant to the user's query about Z because it directly addresses Q.\n\n"
                "Now, begin your analysis, starting with Paper 1.\n"
                "<end_of_turn>\\\n<start_of_turn>model\\\n"
                "Okay, here is my analysis of the papers based on your query:"
            )
            prompt_for_gemma = "".join(parts)
            generation_params = {"temperature": AppConfig.LLAMA_TEMPERATURE, "top_p": AppConfig.LLAMA_TOP_P, "max_tokens": AppConfig.LLAMA_MAX_TOKENS}
            logger.info(f"Using paper query params: {generation_params}")
        # If no papers from Gemini, this block is skipped, and it falls to general RAG or direct response.
            
    # General RAG or fallback if paper query yielded no specific papers to synthesize (and not DEBUG_SKIP_AUGMENTATION)
    if not papers_for_gemma_synthesis and not DEBUG_SKIP_AUGMENTATION: # This condition ensures we only enter here if paper synthesis didn't happen.
        logger.info("Proceeding with general RAG / direct response flow.")
        context_docs = []
        
        # Determine category for general RAG (could be same as paper query or different if no paper query)
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