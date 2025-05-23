import os
import pickle
import requests
import numpy as np
import logging
import json
import re
import threading
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import AppConfig

logger = logging.getLogger(__name__)

# --- Global Variables & Constants ---
local_llm = None
faiss_index = None
embeddings_model = None
systems_status = {
    "local_llm_loaded": False,
    "faiss_index_loaded": False,
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
    global local_llm, faiss_index, embeddings_model, systems_status
    logger.info("Initializing systems...")

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

        if os.path.exists(AppConfig.FAISS_INDEX_PATH):
            logger.info(f"Loading pre-built FAISS index from: {AppConfig.FAISS_INDEX_PATH}")
            faiss_index = FAISS.load_local(
                AppConfig.FAISS_INDEX_PATH, 
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Pre-built FAISS index loaded successfully.")
            systems_status["faiss_index_loaded"] = True
        else:
            logger.warning(f"FAISS index directory not found at {AppConfig.FAISS_INDEX_PATH}. RAG with local model will not be functional.")
            faiss_index = None
            systems_status["faiss_index_loaded"] = False
            
    except Exception as e:
        logger.error(f"Error initializing RAG components: {e}", exc_info=True)
        faiss_index = None
        embeddings_model = None
        systems_status["faiss_index_loaded"] = False

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
        max_len_for_emb = 1000
        text1_emb = embeddings_model.embed_query(text1[:max_len_for_emb])
        text2_emb = embeddings_model.embed_query(text2[:max_len_for_emb])
        vec1 = np.array(text1_emb)
        vec2 = np.array(text2_emb)
        if vec1.ndim > 1: vec1 = vec1.squeeze()
        if vec2.ndim > 1: vec2 = vec2.squeeze()
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return False
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        logger.debug(f"Semantic similarity between snippets: {similarity:.4f}")
        return similarity >= threshold
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}", exc_info=True)
        return False

def check_category_alignment(local_doc_categories: list[str] | None, user_subject_filter: str | None) -> bool:
    """Checks if any of the local document's categories align with the user's subject filter."""
    if not user_subject_filter: # If no specific subject filter from user, consider it aligned
        return True
    if not local_doc_categories: # If local doc has no categories, cannot align with a specific filter
        return False

    normalized_user_subject = user_subject_filter.lower()
    target_arxiv_prefix = SUBJECT_TO_ARXIV_CATEGORY_PREFIX.get(normalized_user_subject)

    if not target_arxiv_prefix: # User subject couldn't be mapped to a known ArXiv prefix
        logger.warning(f"User subject filter '{user_subject_filter}' could not be mapped to a known ArXiv prefix. Alignment check will be lenient.")
        # Fallback: check if any part of the user_subject_filter appears in any category
        # This is a looser check if mapping fails.
        for loc_cat in local_doc_categories:
            if normalized_user_subject in loc_cat.lower():
                return True
        return False # Or, for stricter behavior when mapping fails: return False

    for loc_cat in local_doc_categories:
        # Check if the local category (e.g., "math.CO", "cs.AI") starts with the target prefix (e.g., "math", "cs")
        if loc_cat.lower().startswith(target_arxiv_prefix):
            return True
    return False

# --- LLM Response Generation ---
def generate_gemini_response(prompt, chat_id=None):
    if not systems_status["gemini_model_configured"] or not AppConfig.GEMINI_API_KEY:
        logger.warning("Attempted to use Gemini model but API key is not configured.")
        yield "Gemini API is not configured. Cannot generate response."
        return

    cancel_event = active_cancellation_events.get(chat_id)
    if not cancel_event:
        logger.warning(f"No cancel_event found for chat_id {chat_id} in generate_gemini_response.")
        cancel_event = threading.Event()

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    stream_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{AppConfig.GEMINI_MODEL_NAME}:streamGenerateContent?alt=sse&key={AppConfig.GEMINI_API_KEY}"

    try:
        logger.info(f"Sending request to Gemini Stream API for prompt: '{prompt[:100]}...'")
        response = requests.post(stream_api_url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        client_sse_event_prefix = "data: "
        
        for line in response.iter_lines():
            if cancel_event.is_set():
                logger.info(f"Gemini generation cancelled for chat_id {chat_id}.")
                yield CANCEL_MESSAGE
                response.close()
                return

            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith(client_sse_event_prefix):
                    json_str = decoded_line[len(client_sse_event_prefix):]
                    try:
                        chunk_data = json.loads(json_str)
                        if chunk_data.get("candidates") and \
                           len(chunk_data["candidates"]) > 0 and \
                           chunk_data["candidates"][0].get("content") and \
                           chunk_data["candidates"][0]["content"].get("parts") and \
                           len(chunk_data["candidates"][0]["content"]["parts"]) > 0 and \
                           chunk_data["candidates"][0]["content"]["parts"][0].get("text"):
                            generated_text_chunk = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                            yield generated_text_chunk
                        elif chunk_data.get("promptFeedback") and \
                             chunk_data["promptFeedback"].get("blockReason"):
                            block_reason = chunk_data["promptFeedback"]["blockReason"]
                            logger.error(f"Gemini API request blocked during stream. Reason: {block_reason}")
                            error_detail = chunk_data["promptFeedback"].get("blockReasonMessage", "No additional details.")
                            yield f"Sorry, your request was blocked by the Gemini API. Reason: {block_reason}. Details: {error_detail}"
                            return
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Error decoding JSON from Gemini stream: {json_err} on line: {json_str}")
                    except Exception as e_chunk:
                        logger.error(f"Error processing a chunk from Gemini stream: {e_chunk} - Chunk data: {chunk_data if 'chunk_data' in locals() else 'N/A'}")
        logger.info("Gemini stream finished.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred with Gemini API: {http_err} - Response: {http_err.response.text}", exc_info=True)
        error_content = {}
        try:
            error_content = http_err.response.json() if http_err.response and http_err.response.content else {}
        except json.JSONDecodeError:
            logger.error("Could not decode JSON from HTTP error response.")
        error_message = error_content.get("error", {}).get("message", "An HTTP error occurred.")
        yield f"Sorry, I encountered an HTTP error connecting to the Gemini service: {error_message}"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error sending request to Gemini API: {req_err}", exc_info=True)
        yield "Sorry, I encountered an error sending the request to the Gemini service."
    except Exception as e:
        logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        yield "Sorry, I encountered an unexpected error trying to connect to the Gemini service."
    finally:
        if 'response' in locals() and response is not None:
            response.close()

def get_gemini_paper_suggestions(user_query_for_gemini: str, conversation_history: str = "", chat_id: str | None = None, subject_hint: str | None = None, num_papers_to_suggest: int | None = None) -> list[dict[str, str | None]]:
    if not systems_status["gemini_model_configured"] or not AppConfig.GEMINI_API_KEY:
        logger.warning("Attempted to get paper suggestions, but Gemini API is not configured.")
        return []

    history_prompt_segment = ""
    if conversation_history:
        history_prompt_segment = (
            f"Relevant Conversation History (for context only, focus on the LATEST query below for suggestions):\n"
            f"<HISTORY_START>\n{conversation_history}\n<HISTORY_END>\n\n"
        )
    
    subject_guidance = ""
    if subject_hint:
        subject_guidance = (
            f"CRITICAL: The user is ONLY interested in papers from the '{subject_hint}' field. "
            f"Your suggestions MUST strictly belong to this subject. Do NOT suggest papers from other fields. "
            f"If you cannot find ArXiv papers specifically within '{subject_hint}', respond with ONLY the phrase 'NO_ARXIV_PAPERS_FOUND'. "
        )
    else:
        subject_guidance = "The user has not specified a particular subject field, so suggest generally relevant ArXiv papers. "

    num_papers_instruction = ""
    default_num_papers_range = "8 to 12"
    requested_number_str = default_num_papers_range

    if num_papers_to_suggest and isinstance(num_papers_to_suggest, int) and num_papers_to_suggest > 0:
        num_papers_instruction = f"CRITICAL: You MUST provide EXACTLY {num_papers_to_suggest} highly relevant academic paper suggestions strictly from ArXiv. DO NOT provide fewer than {num_papers_to_suggest} papers. "
        requested_number_str = str(num_papers_to_suggest)
    else:
        num_papers_instruction = f"CRITICAL: You MUST provide AT LEAST {default_num_papers_range} highly relevant academic paper suggestions strictly from ArXiv. Aim for the higher end of this range. "

    gemini_prompt = (
        f"{history_prompt_segment}"
        f"Based on the LATEST user query: '{user_query_for_gemini}', "
        f"{subject_guidance}"
        f"{num_papers_instruction}"
        f"For each paper, provide the following details in a structured JSON format (a list of JSON objects): "
        f"[{{'Title': 'Exact Paper Title', 'Authors': ['Author A', 'Author B'], 'Abstract': 'Concise ArXiv Abstract', 'ArXiv ID': 'e.g., 2310.06825'}}, ...]. "
        f"Ensure the 'ArXiv ID' is the pure ID (e.g., '2310.06825' not 'arXiv:2310.06825'). "
        f"Ensure that all string values within the JSON are properly escaped for JSON validity (e.g., backslashes in text should be '\\\\', quotes should be '\"'). "
        f"ABSOLUTELY CRITICAL: You MUST find and provide {requested_number_str} papers. Do not provide fewer papers than requested. "
        f"If you cannot find enough papers in the specified subject, expand your search slightly but stay within the field. "
        f"If no relevant ArXiv papers are found at all (which should be extremely rare), respond with ONLY the phrase 'NO_ARXIV_PAPERS_FOUND'."
    )
    logger.info(f"Sending prompt to Gemini for paper suggestions (query: '{user_query_for_gemini[:100]}...', chat_id: {chat_id}, requested_num: {num_papers_to_suggest or default_num_papers_range}): '{gemini_prompt[:500]}...")

    suggested_papers_info = []
    full_gemini_text_response = ""
    try:
        response_chunks = list(generate_gemini_response(gemini_prompt, chat_id=chat_id))
        full_gemini_text_response = "".join(response_chunks).strip()
        logger.debug(f"Full raw response from Gemini for suggestions: {full_gemini_text_response}")

        if full_gemini_text_response == "NO_ARXIV_PAPERS_FOUND":
            logger.info("Gemini explicitly stated 'NO_ARXIV_PAPERS_FOUND'.")
            return []

        json_str_to_parse = None
        json_match_markdown = re.search(r'```json\s*([\s\S]*?)\s*```', full_gemini_text_response, re.MULTILINE)
        if json_match_markdown:
            json_str_to_parse = json_match_markdown.group(1).strip()
            logger.info("Extracted JSON content from markdown block.")
        else:
            json_match_raw = re.search(r'(\[[\s\S]*\])|(\{[\s\S]*\})', full_gemini_text_response, re.MULTILINE)
            if json_match_raw:
                json_str_to_parse = json_match_raw.group(0).strip()
                logger.info("Extracted JSON content using raw list/dict regex.")
        
        parsed_data = None
        if json_str_to_parse:
            try:
                logger.info(f"Attempting to parse JSON block (raw extract): {json_str_to_parse[:300]}...")
                parsed_data = json.loads(json_str_to_parse)
            except json.JSONDecodeError as e_raw:
                logger.warning(f"Raw JSON parsing failed: {e_raw}. Attempting to repair invalid escapes.")
                repaired_json_str = repair_json_invalid_escapes(json_str_to_parse)
                try:
                    parsed_data = json.loads(repaired_json_str)
                    logger.info("Successfully parsed JSON after repairing invalid escapes.")
                except json.JSONDecodeError as e_repaired:
                    logger.error(f"Failed to decode JSON from Gemini response even after repair: {e_repaired}. Repaired string snippet: {repaired_json_str[:500]}", exc_info=True)
                    if len(repaired_json_str) < 50 or repaired_json_str == json_str_to_parse :
                         logger.error(f"Original JSON string snippet for context: {json_str_to_parse[:500]}")
                    return []
            
            if parsed_data:
                paper_list_from_gemini = []
                if isinstance(parsed_data, dict):
                    if all(isinstance(val, dict) and ("Title" in val or "title" in val) for val in parsed_data.values()):
                        paper_list_from_gemini = list(parsed_data.values())
                    elif "Title" in parsed_data or "title" in parsed_data:
                        paper_list_from_gemini = [parsed_data]
                    else:
                        logger.warning(f"Parsed JSON from Gemini was a dict but not in an expected paper format: {parsed_data}")
                elif isinstance(parsed_data, list):
                    paper_list_from_gemini = parsed_data
                else:
                    logger.warning(f"Parsed JSON from Gemini was not a list or an expected dict structure. Type: {type(parsed_data)}")

                for paper_data in paper_list_from_gemini:
                    if isinstance(paper_data, dict):
                        title = paper_data.get("Title") or paper_data.get("title")
                        arxiv_id = paper_data.get("ArXiv ID") or paper_data.get("arxiv_id") or paper_data.get("arxivId")
                        if not isinstance(title, str) and title is not None: title = str(title)
                        if arxiv_id is not None and not isinstance(arxiv_id, str): arxiv_id = str(arxiv_id)
                        if title:
                            suggested_papers_info.append({
                                "title": title.strip(),
                                "arxiv_id": arxiv_id.strip() if arxiv_id else None,
                                "authors": paper_data.get("Authors") or paper_data.get("authors"),
                                "abstract": paper_data.get("Abstract") or paper_data.get("abstract")
                            })
                        else:
                            logger.warning(f"Skipping paper data due to missing title: {paper_data}")
                    else:
                        logger.warning(f"Item in parsed Gemini paper list was not a dict: {paper_data}")
                
                if not suggested_papers_info and paper_list_from_gemini:
                    logger.warning("Gemini response parsed as JSON, but no valid paper entries extracted (e.g., all missing 'Title').")
        else:
            logger.warning("No clear JSON block found in Gemini's response for paper suggestions. Full response was logged for debug.")

    except Exception as e_parse:
        logger.error(f"An unexpected error occurred during Gemini suggestion parsing or processing: {e_parse}. Gemini response snippet: {full_gemini_text_response[:500]}", exc_info=True)
        return []

    if suggested_papers_info:
        logger.info(f"Successfully parsed {len(suggested_papers_info)} paper suggestions from Gemini.")
    else:
        logger.info("No paper suggestions were ultimately parsed from Gemini's response.")
    return suggested_papers_info

def generate_local_rag_response(user_query_with_history: str, chat_id=None):
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
        actual_user_question = raw_last_line[len("Assistant: "):].strip()
    else:
        actual_user_question = raw_last_line.strip()
    
    logger.info(f"Parsed actual_user_question for Gemma: '{actual_user_question}'")
    query_lower_for_type_check = actual_user_question.lower()
    is_paper_query = any(keyword in query_lower_for_type_check for keyword in ["paper", "papers", "suggest paper", "recommend paper", "find paper", "arxiv", "top 5 paper", "top 10 paper", "research article", "publication on"])

    subject_filter = None
    num_papers_requested = None

    if is_paper_query:
        subject_filter = extract_subject_from_query(actual_user_question, KNOWN_MAIN_CATEGORIES)
        if subject_filter:
            logger.info(f"Extracted subject filter: '{subject_filter}' from query: '{actual_user_question}'")
        else:
            logger.info(f"No specific subject filter extracted from paper query: '{actual_user_question}'")
        
        # Try to extract number of papers requested
        # Pattern 1: "top 5 papers", "show 10 articles", "get 3 math papers"
        # Look for: keyword + number + optional words + optional unit
        pattern1 = r'\b(top|show|exactly|give\s+me|find|list|get|recommend)\s+(\d+)\b'
        match = re.search(pattern1, actual_user_question, re.IGNORECASE)
        
        if match:
            try:
                num_str = match.group(2)
                num_papers_requested = int(num_str)
                if num_papers_requested <= 0: 
                    num_papers_requested = None
                else:
                    logger.info(f"Extracted number of papers requested (Pattern 1): {num_papers_requested} from query: '{actual_user_question}'")
            except (IndexError, ValueError) as e:
                logger.warning(f"Pattern 1 matched but failed to parse number. Query: '{actual_user_question}', Match groups: {match.groups()}, Error: {e}")
                num_papers_requested = None
        
        # Pattern 2: "5 papers", "10 math articles", "3 research publications"
        if num_papers_requested is None:
            pattern2 = r'\b(\d+)\s+(?:\w+\s+)*(papers|articles|publications|results|entries|suggestions|recommendations)\b'
            match = re.search(pattern2, actual_user_question, re.IGNORECASE)
            if match:
                try:
                    num_str = match.group(1)
                    num_papers_requested = int(num_str)
                    if num_papers_requested <= 0: 
                        num_papers_requested = None
                    else:
                        logger.info(f"Extracted number of papers requested (Pattern 2): {num_papers_requested} from query: '{actual_user_question}'")
                except (IndexError, ValueError) as e:
                    logger.warning(f"Pattern 2 matched but failed to parse number. Query: '{actual_user_question}', Match groups: {match.groups()}, Error: {e}")
                    num_papers_requested = None

        if not num_papers_requested:
            logger.info(f"No specific number of papers requested found in query: '{actual_user_question}'")

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

    if is_paper_query:
        logger.info("Processing as a paper query with enhanced RAG and synthesis.")
        
        gemini_arxiv_papers_info = []
        if not DEBUG_SKIP_GEMINI_SUGGESTIONS and systems_status["gemini_model_configured"]:
            gemini_arxiv_papers_info = get_gemini_paper_suggestions(
                actual_user_question,
                conversation_log_str_for_gemma,
                chat_id,
                subject_hint=subject_filter,
                num_papers_to_suggest=num_papers_requested
            )
        elif DEBUG_SKIP_GEMINI_SUGGESTIONS:
            logger.info("DEBUG_SKIP_GEMINI_SUGGESTIONS is true. Skipping Gemini ArXiv paper suggestions.")

        papers_for_gemma_synthesis = []
        if gemini_arxiv_papers_info:
            logger.info(f"Received {len(gemini_arxiv_papers_info)} paper suggestions from Gemini for processing.")
            for idx, gemini_suggestion in enumerate(gemini_arxiv_papers_info):
                title = gemini_suggestion.get("title", "N/A Title").strip()
                authors_raw = gemini_suggestion.get("authors")
                arxiv_id = gemini_suggestion.get("arxiv_id", "").strip()
                gemini_abstract = gemini_suggestion.get("abstract", "").strip()

                paper_data_for_gemma = {
                    "title": title,
                    "authors": [],
                    "arxiv_id": arxiv_id,
                    "gemini_abstract": gemini_abstract,
                    "local_abstract": None,
                    "chosen_abstract": gemini_abstract,
                    "abstract_source": "Gemini (no local match found or preferred)"
                }

                if isinstance(authors_raw, list):
                    paper_data_for_gemma["authors"] = [str(auth).strip() for auth in authors_raw if str(auth).strip()]
                elif isinstance(authors_raw, str):
                    paper_data_for_gemma["authors"] = [auth.strip() for auth in authors_raw.split(',') if auth.strip()]

                if systems_status["faiss_index_loaded"] and faiss_index and not DEBUG_SKIP_RAG_CONTEXT:
                    search_query_for_faiss = title
                    if arxiv_id: 
                        search_query_for_faiss += f" arxiv:{arxiv_id}"
                    
                    best_local_match_info = None # To store the best found local abstract and its source

                    try:
                        # Retrieve top 10 candidates now (increased from 3)
                        docs_with_scores = faiss_index.similarity_search_with_score(search_query_for_faiss, k=10) 
                        
                        if docs_with_scores:
                            for doc_candidate, score in docs_with_scores:
                                candidate_title = doc_candidate.metadata.get("title", "").strip()
                                local_arxiv_id_match_val = doc_candidate.metadata.get("arxiv_id", "").strip()
                                local_categories = doc_candidate.metadata.get("categories") # This should be a list
                                if isinstance(local_categories, str): # Handle if it's a string somehow
                                    local_categories = [cat.strip() for cat in local_categories.split(',')]

                                # Check 1: ID match + Category alignment
                                if arxiv_id and local_arxiv_id_match_val and arxiv_id.lower() == local_arxiv_id_match_val.lower():
                                    is_aligned = check_category_alignment(local_categories, subject_filter)
                                    if is_aligned:
                                        local_abstract_text = doc_candidate.metadata.get("abstract")
                                        if local_abstract_text:
                                            best_local_match_info = {
                                                "abstract_text": local_abstract_text,
                                                "source": "Local FAISS (ID Match, Category Aligned)"
                                            }
                                            logger.info(f"Local FAISS strong match for '{title}' (ID & Category). Source: {best_local_match_info['source']}")
                                            break # Found best possible match
                                
                                # If no break, continue checking other criteria
                                if best_local_match_info: continue # Already found best for this Gemini suggestion

                                # Check 2: Title similarity + Category alignment
                                title_similar_enough = score < 0.6 # Slightly relaxed threshold for broader search if ID fails
                                if title_similar_enough and are_texts_semantically_similar(title, candidate_title, threshold=0.75):
                                    is_aligned = check_category_alignment(local_categories, subject_filter)
                                    if is_aligned:
                                        local_abstract_text = doc_candidate.metadata.get("abstract")
                                        if local_abstract_text:
                                            best_local_match_info = {
                                                "abstract_text": local_abstract_text,
                                                "source": "Local FAISS (Title Match, Category Aligned)"
                                            }
                                            logger.info(f"Local FAISS good match for '{title}' (Title & Category). Source: {best_local_match_info['source']}")
                                            # Don't break yet, an ID match might still appear for another candidate

                            # After checking all candidates, if no category-aligned match, try without category strictness
                            if not best_local_match_info:
                                for doc_candidate, score in docs_with_scores: # Iterate again for non-category aligned fallbacks
                                    candidate_title = doc_candidate.metadata.get("title", "").strip()
                                    local_arxiv_id_match_val = doc_candidate.metadata.get("arxiv_id", "").strip()

                                    # Check 3: ID match (category not aligned or no filter)
                                    if arxiv_id and local_arxiv_id_match_val and arxiv_id.lower() == local_arxiv_id_match_val.lower():
                                        local_abstract_text = doc_candidate.metadata.get("abstract")
                                        if local_abstract_text:
                                            best_local_match_info = {
                                                "abstract_text": local_abstract_text,
                                                "source": "Local FAISS (ID Match, Category Not Aligned/No Filter)"
                                            }
                                            logger.info(f"Local FAISS match for '{title}' (ID, Category N/A). Source: {best_local_match_info['source']}")
                                            break # ID match is strong
                                    
                                    if best_local_match_info: continue

                                    # Check 4: Title similarity (category not aligned or no filter) - Fallback
                                    title_similar_enough = score < 0.5 # Stricter threshold if no category alignment
                                    if title_similar_enough and are_texts_semantically_similar(title, candidate_title, threshold=0.80):
                                        local_abstract_text = doc_candidate.metadata.get("abstract")
                                        if local_abstract_text:
                                            best_local_match_info = {
                                                "abstract_text": local_abstract_text,
                                                "source": "Local FAISS (Title Match, Category Not Aligned/No Filter)"
                                            }
                                            logger.info(f"Local FAISS fallback match for '{title}' (Title, Category N/A). Source: {best_local_match_info['source']}")
                                            # Don't break, an ID match on another candidate is still better

                        if best_local_match_info:
                            paper_data_for_gemma["local_abstract"] = best_local_match_info["abstract_text"]
                            paper_data_for_gemma["chosen_abstract"] = best_local_match_info["abstract_text"]
                            paper_data_for_gemma["abstract_source"] = best_local_match_info["source"]
                        else:
                            logger.info(f"No suitable local FAISS match found for '{title}' after evaluating top 10. Using Gemini abstract.")
                            # paper_data_for_gemma["abstract_source"] is already default Gemini source
                    
                    except Exception as e_targeted_faiss:
                        logger.error(f"Error during targeted FAISS search for suggestion '{title}': {e_targeted_faiss}", exc_info=True)
                elif DEBUG_SKIP_RAG_CONTEXT:
                    logger.info("DEBUG_SKIP_RAG_CONTEXT is true. Skipping targeted FAISS search.")
                
                papers_for_gemma_synthesis.append(paper_data_for_gemma)
        
        if papers_for_gemma_synthesis:
            gemma_paper_synthesis_prompt_parts = []
            gemma_paper_synthesis_prompt_parts.append(
                "<start_of_turn>user\n"
                "You are an AI research assistant. Your task is to analyze and summarize academic papers in relation to a user's query.\n\n"
                f"USER'S ORIGINAL QUERY:\n---\n{actual_user_question}\n---\n\n"
                "You will be provided with a list of papers, each with a Title, Authors, ArXiv ID, and an Abstract.\n\n"
                "YOUR REQUIRED OUTPUT FORMAT AND TASK FOR EACH PAPER:\n"
                "For EACH paper in the list you are given, you MUST produce the following output structure:\n"
                "1. Start with 'Paper [Number]: [Title of the Paper]'.\n"
                "2. On the next line, list 'Authors: [Author Names]'.\n"
                "3. On the next line, list 'ArXiv ID: [ID]' (if available, otherwise 'ArXiv ID: N/A').\n"
                "4. On subsequent lines, provide YOUR concise summary and interpretation (2-4 sentences) of the paper's GIVEN ABSTRACT, specifically focusing on how the abstract's content relates to the USER'S ORIGINAL QUERY.\n"
                "   - Do NOT simply repeat the abstract.\n"
                "   - If a paper's abstract seems largely irrelevant to the user's query despite its title, briefly state that and explain why.\n"
                "   - Do NOT include any information about the 'source' of the abstract in your output.\n\n"
                "CRITICAL: You MUST process ALL papers provided in the 'PAPER DETAILS TO PROCESS' section below and follow this output format for EACH one.\n"
                "Maintain a helpful, informative, and objective tone.\n\n"
                "PAPER DETAILS TO PROCESS:\n===\n"
            )

            for i, paper_info in enumerate(papers_for_gemma_synthesis):
                authors_str = ", ".join(paper_info['authors']) if paper_info.get('authors') else "N/A"
                arxiv_id_str = paper_info['arxiv_id'] if paper_info.get('arxiv_id') else "N/A"
                abstract_str = paper_info.get('chosen_abstract', "No abstract available.").strip()
                
                gemma_paper_synthesis_prompt_parts.append(
                    f"Input Paper {i+1} Details:\n"
                    f"  Title: {paper_info['title']}\n"
                    f"  Authors: {authors_str}\n"
                    f"  ArXiv ID: {arxiv_id_str}\n" 
                    f"  Abstract to Summarize:\n  \"\"\"\n  {abstract_str}\n  \"\"\"\n---\n"
                )
            
            gemma_paper_synthesis_prompt_parts.append(
                "===\n"
                "Now, generate your response. Follow the 'YOUR REQUIRED OUTPUT FORMAT AND TASK FOR EACH PAPER' instructions strictly for all papers listed above.\n"
                "Begin directly with your analysis of the first paper, starting with 'Paper 1: ...'.\n"
                "<end_of_turn>\n<start_of_turn>model\n"
            )
            prompt_for_gemma = "".join(gemma_paper_synthesis_prompt_parts)
            
            generation_params = {
                "temperature": 0.3, 
                "top_p": 0.9, 
                "top_k": 50, 
                "repeat_penalty": 1.15,
                "max_tokens": AppConfig.LLAMA_MAX_TOKENS, 
                "stop": ["<end_of_turn>", "USER:", "User Query:", "PAPER DETAILS TO PROCESS:"] 
            }
            logger.info(f"Prepared synthesis prompt for Gemma with {len(papers_for_gemma_synthesis)} papers.")
            logger.debug(f"Gemma Synthesis Prompt (first 500 chars): {prompt_for_gemma[:500]}...")

        else: 
            logger.info("No papers to synthesize for Gemma after processing Gemini suggestions.")
            yield "I could not find relevant paper suggestions to discuss for your query based on the initial search."
            return
    else:
        # --- GENERAL KNOWLEDGE / NON-PAPER QUERY LOGIC --- 
        logger.info("Processing as a general knowledge query.")
        
        general_knowledge_instruction = (
            "You are a concise and factual AI assistant. Your primary goal is to provide direct answers or explanations based on your training data and the provided conversation log. "
            "Focus solely on the user's query. Do NOT include conversational introductions or closings like 'Hello!', 'Sure!', 'Certainly!', 'Okay, here is...', 'I hope this helps!', or 'Let me know if you have other questions.' "
            "Do NOT engage in meta-commentary (e.g., 'In this response, I will...', 'I will now explain...', 'Based on the context...'). "
            "Avoid any self-reference (e.g., 'As an AI assistant...', 'My purpose is to...'). "
            "Avoid section headers or titles in your response (e.g., 'Explanation:', 'Details:', 'Summary:'). "
            "If the conversation log is relevant, use it to understand the context of the current query. "
            "The response should be approximately 100-150 words for a general explanation, unless the query specifically asks for more detail or a list."
        )
        
        retrieved_context_str_for_gemma_prompt = ""
        perform_general_rag = True # Default to performing RAG

        # Heuristic to skip RAG for simple definitional/explanatory questions
        query_lower_for_rag_skip_check = actual_user_question.lower().strip()
        
        # More robust check for common query starters
        # Allows for variations like "whatis", "what's", "what is", etc.
        if ((query_lower_for_rag_skip_check.startswith("what is") or 
             query_lower_for_rag_skip_check.startswith("what's") or 
             query_lower_for_rag_skip_check.startswith("whats ") or # handles "whats X"
             query_lower_for_rag_skip_check == "whats" or # handles just "whats"
             query_lower_for_rag_skip_check.startswith("whatare") or 
             query_lower_for_rag_skip_check.startswith("what are")) or
            (query_lower_for_rag_skip_check.startswith("explain") and (query_lower_for_rag_skip_check.startswith("explain ") or len(query_lower_for_rag_skip_check) == len("explain"))) or
            (query_lower_for_rag_skip_check.startswith("define") and (query_lower_for_rag_skip_check.startswith("define ") or len(query_lower_for_rag_skip_check) == len("define"))) or
            query_lower_for_rag_skip_check.startswith("tell me about ") or
            (query_lower_for_rag_skip_check.startswith("who is") or 
             query_lower_for_rag_skip_check.startswith("who's") or 
             query_lower_for_rag_skip_check.startswith("whos ") or # handles "whos X"
             query_lower_for_rag_skip_check == "whos" or # handles just "whos"
             query_lower_for_rag_skip_check.startswith("whowas") or 
             query_lower_for_rag_skip_check.startswith("who was") or 
             query_lower_for_rag_skip_check.startswith("whoare") or 
             query_lower_for_rag_skip_check.startswith("who are"))
           ) and len(actual_user_question.split()) < 10: # Keep word count limit
            logger.info(f"Simple general knowledge query detected ('{actual_user_question}'). Skipping FAISS RAG search to prioritize speed.")
            perform_general_rag = False

        if perform_general_rag and systems_status["faiss_index_loaded"] and faiss_index and actual_user_question and not DEBUG_SKIP_RAG_CONTEXT:
            try:
                logger.info(f"Performing general FAISS similarity search for: '{actual_user_question[:100]}...'")
                retrieved_documents_initial = faiss_index.similarity_search(actual_user_question, k=AppConfig.N_RETRIEVED_DOCS + 15)
                
                final_general_rag_documents_for_prompt = []
                if retrieved_documents_initial:
                    final_seen_content_snippets_general = [] 
                    final_seen_titles_normalized_general = set()
                    for doc in retrieved_documents_initial:
                        if len(final_general_rag_documents_for_prompt) >= AppConfig.N_RETRIEVED_DOCS:
                            break 
                        title = doc.metadata.get("title", "").strip()
                        normalized_title = title.lower()
                        page_content_str = str(doc.page_content if doc.page_content is not None else "")
                        content_snippet_for_check = page_content_str[:300]

                        is_semantically_dupe_content_general = any(
                            are_texts_semantically_similar(content_snippet_for_check, existing_snippet, threshold=0.90) 
                            for existing_snippet in final_seen_content_snippets_general
                        )

                        if (not title or normalized_title not in final_seen_titles_normalized_general) and not is_semantically_dupe_content_general:
                            final_general_rag_documents_for_prompt.append(doc)
                            if title: final_seen_titles_normalized_general.add(normalized_title)
                            final_seen_content_snippets_general.append(content_snippet_for_check)
                    logger.info(f"Populated {len(final_general_rag_documents_for_prompt)} documents for general RAG context after deduplication.")
                else:
                    logger.info("No documents retrieved from general FAISS search for the query.")

                if final_general_rag_documents_for_prompt: 
                    context_parts = []
                    for i, doc_to_add in enumerate(final_general_rag_documents_for_prompt):
                        content = str(doc_to_add.page_content).strip() if doc_to_add.page_content is not None else ""
                        doc_title = doc_to_add.metadata.get('title', f'Retrieved Document {i+1}')
                        max_content_len = 700
                        context_parts.append(f"Context Document {i+1} (Title: {doc_title}):\n{content[:max_content_len]}{'...' if len(content) > max_content_len else ''}")
                    if context_parts:
                        retrieved_context_str_for_gemma_prompt = (
                            "\n\n--- Start of General Retrieved Context ---\n"
                            + "\n\n".join(context_parts) +
                            "\n--- End of General Retrieved Context ---\n"
                        )
            except Exception as e_faiss_general:
                logger.error(f"Error during general FAISS retrieval: {e_faiss_general}", exc_info=True)
        elif DEBUG_SKIP_RAG_CONTEXT:
            logger.info("DEBUG_SKIP_RAG_CONTEXT is true. No general RAG context string will be built for Gemma.")
        else:
            logger.info("FAISS index not loaded or query empty. Skipping general RAG retrieval for non-paper query.")

        prompt_parts_general = [f"<start_of_turn>user\n{general_knowledge_instruction}\n"]
        if conversation_log_str_for_gemma:
            prompt_parts_general.append(f"{conversation_log_str_for_gemma}\n")
        if perform_general_rag and retrieved_context_str_for_gemma_prompt:
            prompt_parts_general.append(f"ADDITIONAL CONTEXT FROM DOCUMENT DATABASE (Use if relevant to the USER QUERY):\n{retrieved_context_str_for_gemma_prompt}\n\n")
        elif not perform_general_rag:
            prompt_parts_general.append("NOTE: You should rely primarily on your internal knowledge to answer the following query concisely.\n")

        prompt_parts_general.append(f"USER QUERY: {actual_user_question}<end_of_turn>\n<start_of_turn>model\n")
        prompt_for_gemma = "".join(prompt_parts_general)

        generation_params = {
            "temperature": 0.2, "top_p": 0.8, "top_k": 40, "repeat_penalty": 1.25,
            "max_tokens": AppConfig.LLAMA_MAX_TOKENS, 
            "stop": [
                "User:", "User Query:", "USER:", "ASSISTANT:", "Assistant:", "System:", 
                "\nUser:", "\nUSER:", "\nASSISTANT:", "\nAssistant:", "\nSystem:",
                "Context:", "CONTEXT:", "Answer:", "ANSWER:", "Note:", "Response:",
                "In this response,", "In summary,", "To summarize,", "Let's expand on",
                "The user is asking", "The user wants to know",
                "This text follows", "The following is a", "This is a text about",
                "My goal is to", "I am an AI assistant", "As an AI",
                "\n\nHuman:", "<|endoftext|>", "<|im_end|>", "\u202f",
                "STOP_ASSISTANT_PRIMING_PHRASE:"
            ]
        }
        logger.info(f"Prepared general knowledge prompt for Gemma.")
        logger.debug(f"Gemma General Knowledge Prompt (first 500 chars): {prompt_for_gemma[:500]}...")

    # --- LLM Stream Generation (Common for both paper and general queries) ---
    cancel_event_for_llm = active_cancellation_events.get(chat_id)
    if not cancel_event_for_llm:
        logger.warning(f"No cancel_event found for chat_id {chat_id} in generate_local_rag_response. LLM generation will not be cancellable.")
        cancel_event_for_llm = threading.Event()

    try:
        stream = local_llm.create_chat_completion( 
            messages=[{"role": "user", "content": prompt_for_gemma}],
            **generation_params,
            stream=True
        )
        full_response = ""
        for chunk in stream:
            if cancel_event_for_llm.is_set():
                logger.info(f"Local LLM (Gemma) generation cancelled for chat_id {chat_id}.")
                yield CANCEL_MESSAGE
                break
            delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta_content:
                yield delta_content
                full_response += delta_content
        logger.info(f"Gemma full response (after stream): {full_response.strip()}")
    except Exception as e:
        logger.error(f"Error during local LLM generation: {e}", exc_info=True)
        yield f"Error generating response from local model: {str(e)}"

def register_cancellation_event(chat_id, event):
    active_cancellation_events[chat_id] = event

def unregister_cancellation_event(chat_id):
    return active_cancellation_events.pop(chat_id, None)

def get_cancellation_event(chat_id):
    return active_cancellation_events.get(chat_id) 