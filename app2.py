import os
import pickle
from flask import Flask, request, jsonify, render_template, Response
from llama_cpp import Llama
# import google.generativeai as genai # No longer used for Gemini if using REST API
import requests # For direct HTTP calls to Gemini API
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np # Add if not already present, for cosine similarity
import logging # For improved logging
import sqlite3
import datetime
import colorama
import json # Add this import at the top of the file if not already there
import re # For parsing Gemini response
import atexit # For controlled cleanup
import threading # Add this for cancellation events

# Initialize colorama carefully for Windows
# colorama.init() # Original
colorama.init(wrap=False) # Try with wrap=False to see if it resolves OSError 6

# Load environment variables
load_dotenv()

# --- Application Configuration ---
class AppConfig:
    # Cloud Configuration
    CLOUD_DEPLOYMENT = os.getenv('CLOUD_DEPLOYMENT', 'False').lower() == 'true'
    MODEL_CACHE_SIZE = int(os.getenv('MODEL_CACHE_SIZE', '1'))  # Number of model instances to cache
    REQUEST_QUEUE_SIZE = int(os.getenv('REQUEST_QUEUE_SIZE', '100'))
    
    # Model Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = "gemini-2.0-flash"
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    
    # Model Paths & Parameters
    LOCAL_MODEL_PATH = os.getenv('MODEL_PATH', 'gemma_1b_finetuned_q4_0.gguf')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'langchain_faiss_store_optimized')
    
    # RAG Configuration
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    N_RETRIEVED_DOCS = int(os.getenv('N_RETRIEVED_DOCS', '10'))
    LLAMA_MAX_TOKENS = int(os.getenv('LLAMA_MAX_TOKENS', '8192'))  # Increased from 4096
    LLAMA_TEMPERATURE = float(os.getenv('LLAMA_TEMPERATURE', '0.3'))
    LLAMA_TOP_P = float(os.getenv('LLAMA_TOP_P', '0.9'))
    
    # Llama.cpp model parameters
    LLAMA_N_CTX = int(os.getenv('LLAMA_N_CTX', '8192'))  # Increased from 4092
    LLAMA_N_GPU_LAYERS = int(os.getenv('LLAMA_N_GPU_LAYERS', '0'))
    LLAMA_VERBOSE = os.getenv('LLAMA_VERBOSE', 'False').lower() == 'true'

    # Flask App settings
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))
    DATABASE_URL = os.getenv('DATABASE_URL', 'chat_app.db')
    
    # Cloud specific settings
    REDIS_URL = os.getenv('REDIS_URL')  # For session management in cloud
    CLOUD_STORAGE_BUCKET = os.getenv('CLOUD_STORAGE_BUCKET')  # For storing models in cloud

# --- Global dictionary for cancellation events ---
active_cancellation_events = {} # Key: chat_id, Value: threading.Event()
CANCEL_MESSAGE = "\\n\\n[LLM generation cancelled by user.]"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# Python SDK genai.configure is no longer needed here if using REST
# if not AppConfig.GEMINI_API_KEY:
#     logger.warning("GEMINI_API_KEY not found...")
# else:
#     try:
#         genai.configure(api_key=AppConfig.GEMINI_API_KEY)
#         logger.info("Gemini API configured...")
#     except Exception as e:
#         logger.error(f"Error configuring Gemini API with SDK: {e}")
#         AppConfig.GEMINI_API_KEY = None

# --- Initialize Flask App ---
app = Flask(__name__)
app.config.from_object(AppConfig)

# --- Global Variables ---
local_llm = None
faiss_index = None
embeddings_model = None
# gemini_model = None # This was for the SDK

# --- Model Management for Cloud Deployment ---
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
            # If in cloud, check if model needs to be downloaded from cloud storage
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
                cleanup_local_llm()
                del self._model_cache['model']
                logger.info("Model cleaned up and removed from cache")

# --- System Initialization Status Flags ---
systems_status = {
    "local_llm_loaded": False,
    "faiss_index_loaded": False,
    "gemini_model_configured": bool(AppConfig.GEMINI_API_KEY) # True if API key exists for REST method
}

# --- Helper Functions ---
def initialize_systems():
    global local_llm, faiss_index, embeddings_model
    logger.info("Initializing systems...")

    # Initialize model through ModelManager if in cloud deployment
    if AppConfig.CLOUD_DEPLOYMENT:
        try:
            model_manager = ModelManager.get_instance()
            local_llm = model_manager.get_model()
            systems_status["local_llm_loaded"] = bool(local_llm)
        except Exception as e:
            logger.error(f"Error initializing model through ModelManager: {e}", exc_info=True)
            local_llm = None
            systems_status["local_llm_loaded"] = False
    else:
        # Original local model initialization code
        if os.path.exists(AppConfig.LOCAL_MODEL_PATH):
            try:
                logger.info(f"Loading local GGUF model from: {AppConfig.LOCAL_MODEL_PATH}")
                local_llm = Llama(
                    model_path=AppConfig.LOCAL_MODEL_PATH,
                    n_ctx=AppConfig.LLAMA_N_CTX,
                    n_gpu_layers=AppConfig.LLAMA_N_GPU_LAYERS,
                    verbose=AppConfig.LLAMA_VERBOSE
                )
                logger.info("Local GGUF model loaded successfully.")
                systems_status["local_llm_loaded"] = True
            except Exception as e:
                logger.error(f"Error loading local GGUF model: {e}", exc_info=True)
                local_llm = None
                systems_status["local_llm_loaded"] = False
        else:
            logger.warning(f"Local model file not found at {AppConfig.LOCAL_MODEL_PATH}. Local model will not be available.")
            local_llm = None
            systems_status["local_llm_loaded"] = False

    # 2. Initialize RAG components (Embedding Model and FAISS Index)
    try:
        logger.info(f"Loading embedding model: {AppConfig.EMBEDDING_MODEL_NAME}")
        embeddings_model = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")

        if os.path.exists(AppConfig.FAISS_INDEX_PATH):
            logger.info(f"Loading pre-built FAISS index from: {AppConfig.FAISS_INDEX_PATH}")
            # Note: allow_dangerous_deserialization=True is often needed for FAISS indexes
            # saved with older LangChain versions or if custom Python objects were part of metadata.
            # Ensure this is acceptable for your security context.
            faiss_index = FAISS.load_local(
                AppConfig.FAISS_INDEX_PATH, 
                embeddings_model,
                allow_dangerous_deserialization=True # Be aware of implications
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
        embeddings_model = None # If FAISS loading fails, embeddings_model might still be used by Gemini RAG if implemented later
        systems_status["faiss_index_loaded"] = False

    # 3. Initialize Gemini Model
    if not AppConfig.GEMINI_API_KEY:
        logger.warning("Gemini API key not provided. Gemini REST API calls will fail.")
        systems_status["gemini_model_configured"] = False
    else:
        logger.info("Gemini API key found. Ready for REST API calls.")
        systems_status["gemini_model_configured"] = True

    logger.info(f"Initialization complete. System status: {systems_status}")

def get_db_connection():
    """Establishes a connection to the database."""
    conn = sqlite3.connect(AppConfig.DATABASE_URL)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    return conn

def init_db():
    """Initializes the database using the schema.sql file."""
    db_path = AppConfig.DATABASE_URL
    schema_path = 'static/schema.sql' # Assuming schema.sql is in the static folder, adjust if not

    # Check if the database file already exists. If so, schema might already be applied.
    # This is a simple check; for robust production systems, use migrations (e.g., Alembic).
    # For this project, we'll re-apply if the script is run, ensuring tables exist.
    # if os.path.exists(db_path):
    #     logger.info(f"Database {db_path} already exists. Assuming schema is applied.")
    #     return

    try:
        logger.info(f"Initializing database at {db_path} with schema {schema_path}")
        conn = get_db_connection()
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
    except FileNotFoundError:
        logger.error(f"Schema file not found at {schema_path}. Database not initialized.", exc_info=True)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# UNCOMMENTED generate_gemini_response
def generate_gemini_response(prompt, chat_id=None):
    if not systems_status["gemini_model_configured"] or not AppConfig.GEMINI_API_KEY:
        logger.warning("Attempted to use Gemini model but API key is not configured.")
        # Return a generator that yields an error message
        yield "Gemini API is not configured. Cannot generate response."
        return

    cancel_event = active_cancellation_events.get(chat_id)
    if not cancel_event: # Should ideally always be there if called from /api/chat
        logger.warning(f"No cancel_event found for chat_id {chat_id} in generate_gemini_response. Generation will not be cancellable.")
        # Create a dummy event if not passed, so it doesn't crash, but won't be externally cancellable
        cancel_event = threading.Event()

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
        # Add generationConfig if needed, e.g.,
        # "generationConfig": {
        #     "temperature": 0.9,
        #     "topK": 1,
        #     "topP": 1,
        #     "maxOutputTokens": 2048,
        #     "stopSequences": []
        # }
    }
    
    # Use streamGenerateContent endpoint
    stream_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{AppConfig.GEMINI_MODEL_NAME}:streamGenerateContent?alt=sse&key={AppConfig.GEMINI_API_KEY}"

    try:
        logger.info(f"Sending request to Gemini Stream API for prompt: '{prompt[:100]}...'")
        response = requests.post(stream_api_url, headers=headers, json=payload, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        client_sse_event_prefix = "data: " # SSE events are prefixed with "data: "
        
        for line in response.iter_lines():
            if cancel_event.is_set():
                logger.info(f"Gemini generation cancelled for chat_id {chat_id}.")
                yield CANCEL_MESSAGE
                response.close() # Ensure the underlying connection is closed
                return

            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith(client_sse_event_prefix):
                    json_str = decoded_line[len(client_sse_event_prefix):]
                    try:
                        chunk_data = json.loads(json_str)
                        #logging.debug(f"Gemini API stream chunk: {chunk_data}") # For debugging
                        if chunk_data.get("candidates") and \
                           len(chunk_data["candidates"]) > 0 and \
                           chunk_data["candidates"][0].get("content") and \
                           chunk_data["candidates"][0]["content"].get("parts") and \
                           len(chunk_data["candidates"][0]["content"]["parts"]) > 0 and \
                           chunk_data["candidates"][0]["content"]["parts"][0].get("text"):
                            
                            generated_text_chunk = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                            # logger.info(f"Yielding chunk: {generated_text_chunk}")
                            yield generated_text_chunk
                        elif chunk_data.get("promptFeedback") and \
                             chunk_data["promptFeedback"].get("blockReason"):
                            block_reason = chunk_data["promptFeedback"]["blockReason"]
                            logger.error(f"Gemini API request blocked during stream. Reason: {block_reason}")
                            error_detail = chunk_data["promptFeedback"].get("blockReasonMessage", "No additional details.")
                            yield f"Sorry, your request was blocked by the Gemini API. Reason: {block_reason}. Details: {error_detail}"
                            return # Stop generation if blocked
                        # else:
                            # logger.warning(f"Unexpected chunk structure or empty text: {chunk_data}")
                            # Potentially yield nothing or a placeholder if needed
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Error decoding JSON from Gemini stream: {json_err} on line: {json_str}")
                        # Decide if we should yield an error or continue
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
            response.close() # Ensure response is closed in all exit paths

# --- Helper function to get paper suggestions from Gemini ---
def get_gemini_paper_suggestions(user_query_for_gemini: str) -> list[dict[str, str | None]]:
    if not systems_status["gemini_model_configured"] or not AppConfig.GEMINI_API_KEY:
        logger.warning("Attempted to get paper suggestions, but Gemini API is not configured.")
        return []

    gemini_prompt = (
        f"Based on current general knowledge, please list up to 5-7 distinct and highly relevant academic paper titles "
        f"for the query: '{user_query_for_gemini}'. "
        f"Prioritize papers that are citable or foundational. "
        f"If readily available and confident, include their arXiv IDs in the format 'Title (arXiv:xxxx.xxxxx)'. "
        f"If no arXiv ID is known, just provide the title. "
        f"Provide each paper on a new line. If you cannot suggest at least 3-4 high-quality, distinct papers, respond with 'NO_SUGGESTIONS_FOUND'."
    )
    logger.info(f"Sending prompt to Gemini for paper suggestions: '{gemini_prompt[:200]}...")

    try:
        response_chunks = list(generate_gemini_response(gemini_prompt))
        full_gemini_text_response = "".join(response_chunks).strip()
        logger.debug(f"Full raw response from Gemini for suggestions: {full_gemini_text_response}")

        if "NO_SUGGESTIONS_FOUND" in full_gemini_text_response or not full_gemini_text_response:
            logger.info("Gemini indicated no suggestions found or returned an empty response.")
            return []

        suggested_papers_info = []
        paper_pattern = re.compile(r"^(.*?)(?:\s*\(arXiv:([\d\.]+)\))?$", re.MULTILINE)
        
        for line in full_gemini_text_response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = paper_pattern.match(line)
            if match:
                title = match.group(1).strip()
                arxiv_id = match.group(2).strip() if match.group(2) else None
                
                # Clean title: remove leading/trailing quotes and common unwanted prefixes like '* '
                if title.startswith('*'):
                    title = title[1:].lstrip()
                title = title.strip('"\'') # Remove leading/trailing single or double quotes
                
                if title:
                    suggested_papers_info.append({"title": title, "arxiv_id": arxiv_id})
            elif line: # If no regex match but line has content, assume it's a title
                title = line.strip('"\'')
                if title.startswith('*'):
                    title = title[1:].lstrip()
                title = title.strip('"\'')
                if title:
                    suggested_papers_info.append({"title": title, "arxiv_id": None})

        logger.info(f"Parsed {len(suggested_papers_info)} paper suggestions from Gemini: {suggested_papers_info}")
        return suggested_papers_info

    except Exception as e:
        logger.error(f"Error calling or parsing Gemini for paper suggestions: {e}", exc_info=True)
        return []

def fetch_wikipedia_summary(query: str) -> str:
    """
    Fetches a summary from Wikipedia for a given query.
    Returns the summary as a string, or an error/not found message.
    """
    SESSION_TIMEOUT = 10 # seconds for requests
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    # It's good practice to set a User-Agent. Replace with your app's info if deploying.
    headers = {
        "User-Agent": "ChatApp/1.0 (Flask_App; https://example.com/contact or mailto:user@example.com)"
    }

    # Step 1: Search for the query to get a page title
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1, # Get only the top result
        "srprop": ""  # We don't need snippets from the search results themselves
    }
    search_data = {} # Initialize in case of early exit

    try:
        logger.info(f"Searching Wikipedia for: '{query[:100]}...'")
        search_response = requests.get(WIKIPEDIA_API_URL, params=search_params, headers=headers, timeout=SESSION_TIMEOUT)
        search_response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        search_data = search_response.json()

        if not search_data.get("query", {}).get("search"):
            logger.info(f"No Wikipedia search results found for: '{query[:100]}...'")
            return f"No Wikipedia article found for '{query}'."
        
        page_title = search_data["query"]["search"][0]["title"]
        logger.info(f"Found Wikipedia page title: '{page_title}' for query: '{query[:100]}...'")

    except requests.exceptions.Timeout:
        logger.error(f"Wikipedia API search request timed out for query '{query[:100]}...'", exc_info=True)
        return f"Error: The request to Wikipedia timed out while searching for '{query}'."
    except requests.exceptions.RequestException as e:
        logger.error(f"Wikipedia API search request failed for query '{query[:100]}...': {e}", exc_info=True)
        return f"Error: Could not connect to Wikipedia to search for '{query}'."
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing Wikipedia search results for query '{query[:100]}...': {e} - Data: {search_data}", exc_info=True)
        return f"Error: Could not process search results from Wikipedia for '{query}'."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Wikipedia search for '{query[:100]}...': {e}", exc_info=True)
        return f"Error: Invalid response format from Wikipedia search for '{query}'."


    # Step 2: Get the extract (summary) of that page title
    extract_params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": page_title,
        "exintro": True,      # Get only the introductory section (summary)
        "explaintext": True,  # Get plain text, not HTML
        "exlimit": 1          # Max 1 extract for the given title
    }
    extract_data = {} # Initialize in case of early exit

    try:
        logger.info(f"Fetching Wikipedia extract for page title: '{page_title}'")
        extract_response = requests.get(WIKIPEDIA_API_URL, params=extract_params, headers=headers, timeout=SESSION_TIMEOUT)
        extract_response.raise_for_status()
        extract_data = extract_response.json()

        pages = extract_data.get("query", {}).get("pages")
        if not pages:
            logger.warning(f"No 'pages' data in Wikipedia extract response for title '{page_title}'. Data: {extract_data}")
            # This can happen if the title, though found in search, is invalid for extracts (e.g. special pages)
            return f"Error: Could not retrieve summary content from Wikipedia for the article '{page_title}'."

        # The page ID is dynamic (e.g., "736"), so we get the first (and only) page ID from the 'pages' object
        page_id = next(iter(pages)) # Gets the first key from the pages dictionary
        summary = pages[page_id].get("extract", "").strip()

        if not summary:
            # This could happen for disambiguation pages, very short articles, or protected pages.
            logger.info(f"Empty summary returned from Wikipedia for page title: '{page_title}'")
            return f"No summary found on Wikipedia for the article '{page_title}'. It might be a disambiguation page, very short, or require specific permissions to view."
        
        logger.info(f"Successfully fetched Wikipedia summary for '{page_title}' (length: {len(summary)} chars).")
        return summary

    except requests.exceptions.Timeout:
        logger.error(f"Wikipedia API extract request timed out for title '{page_title}'", exc_info=True)
        return f"Error: The request to Wikipedia timed out while fetching the summary for '{page_title}'."
    except requests.exceptions.RequestException as e:
        logger.error(f"Wikipedia API extract request failed for title '{page_title}': {e}", exc_info=True)
        return f"Error: Could not connect to Wikipedia to get summary for '{page_title}'."
    except (KeyError, StopIteration) as e: # StopIteration for next(iter(pages)) if pages is unexpectedly empty or malformed
        logger.error(f"Error parsing Wikipedia extract results for title '{page_title}': {e} - Data: {extract_data}", exc_info=True)
        return f"Error: Could not process summary from Wikipedia for '{page_title}'."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Wikipedia extract for '{page_title}': {e}", exc_info=True)
        return f"Error: Invalid response format from Wikipedia extract for '{page_title}'."
    except Exception as e_gen: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in fetch_wikipedia_summary for title '{page_title}': {e_gen}", exc_info=True)
        return f"An unexpected error occurred while fetching the Wikipedia summary for '{page_title}'."

# --- Helper function for semantic similarity (NEW) ---
def are_texts_semantically_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Checks if two texts are semantically similar using embeddings and cosine similarity."""
    if not text1 or not text2:
        return False
    if not embeddings_model:
        logger.warning("Embeddings model not loaded. Cannot perform semantic similarity check.")
        return False # Fallback: treat as not similar if model is missing

    try:
        # Ensure texts are not excessively long for embedding, truncate if necessary
        # This depends on the embedding model's limits, e.g., 512 tokens for sentence-transformers
        # A simple character limit might be a pragmatic first step.
        max_len_for_emb = 1000 # Characters, rough estimate
        text1_emb = embeddings_model.embed_query(text1[:max_len_for_emb])
        text2_emb = embeddings_model.embed_query(text2[:max_len_for_emb])

        # Cosine similarity calculation
        # Requires numpy. A more robust implementation would handle potential errors here.
        vec1 = np.array(text1_emb)
        vec2 = np.array(text2_emb)
        
        # Ensure vectors are 1D for dot product and norm calculation
        if vec1.ndim > 1: vec1 = vec1.squeeze()
        if vec2.ndim > 1: vec2 = vec2.squeeze()
        
        # Check for zero vectors (e.g. from empty strings after processing)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return False # Or handle as per requirements, e.g. identical if both are zero vectors from empty strings

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        logger.debug(f"Semantic similarity between snippets: {similarity:.4f}")
        return similarity >= threshold
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}", exc_info=True)
        return False # Fallback on error

# UNCOMMENTED generate_local_rag_response
def generate_local_rag_response(user_query_with_history: str, chat_id=None):
    DEBUG_SKIP_AUGMENTATION = False 
    DEBUG_SKIP_RAG_CONTEXT = False    # << CORRECTED THIS FLAG
    DEBUG_SKIP_GEMINI_SUGGESTIONS = False # << ENSURE GEMINI SUGGESTIONS ARE ON

    if not systems_status["local_llm_loaded"] or local_llm is None:
        logger.warning("Local LLM not loaded, yielding error message.")
        yield "The local GGUF model is not loaded. Cannot generate a response."
        return

    logger.info(f"Received in generate_local_rag_response (Gemma): {user_query_with_history}")

    lines = user_query_with_history.strip().split('\\n')
    # actual_user_question = lines[-1] if lines else "" # Old way
    raw_last_line = lines[-1] if lines else ""
    actual_user_question = "" # Initialize
    if raw_last_line.startswith("User: "):
        actual_user_question = raw_last_line[len("User: "):].strip()
    elif raw_last_line.startswith("Assistant: "): # Should be rare for the last line
        actual_user_question = raw_last_line[len("Assistant: "):].strip()
    else:
        actual_user_question = raw_last_line.strip() # Fallback if no prefix
    
    logger.info(f"Parsed actual_user_question for Gemma: '{actual_user_question}'")

    # Construct conversation history for Gemma prompt (excluding current query)
    conversation_history_for_gemma_prompt = []
    if len(lines) > 1:
        for line in lines[:-1]: # Iterate over all lines except the last one (current query)
            if line.startswith("User: "):
                conversation_history_for_gemma_prompt.append(f"<start_of_turn>user\\n{line[len('User: '):].strip()}<end_of_turn>")
            elif line.startswith("Assistant: "):
                conversation_history_for_gemma_prompt.append(f"<start_of_turn>model\\n{line[len('Assistant: '):].strip()}<end_of_turn>")
            # else: # Skip lines that don't conform to expected "User: " or "Assistant: "
                # logger.debug(f"Skipping non-standard line in history: {line}")
    
    conversation_log_str_for_gemma = "\\n".join(conversation_history_for_gemma_prompt) if conversation_history_for_gemma_prompt else ""
    # --- Phase 1: General RAG (FAISS Document Retrieval based on user query) ---
    general_retrieved_documents_from_faiss = [] # Stores Langchain Document objects
    if systems_status["faiss_index_loaded"] and faiss_index and actual_user_question and not DEBUG_SKIP_RAG_CONTEXT:
        try:
            logger.info(f"Performing general FAISS similarity search for: '{actual_user_question[:100]}...'")
            # Fetched k slightly increased to allow more candidates for deduplication
            retrieved_documents = faiss_index.similarity_search(actual_user_question, k=AppConfig.N_RETRIEVED_DOCS + 5) 
            if retrieved_documents:
                seen_titles = set()
                temp_unique_docs = []
                for doc in retrieved_documents:
                    title = doc.metadata.get('title', '').strip()
                    normalized_title = title.lower()
                    # Basic title deduplication
                    if title and normalized_title not in seen_titles:
                        temp_unique_docs.append(doc)
                        seen_titles.add(normalized_title)
                    elif not title: # If no title, still add it (might be pure text content)
                        temp_unique_docs.append(doc)
                general_retrieved_documents_from_faiss = temp_unique_docs
                logger.info(f"General FAISS search initially retrieved {len(general_retrieved_documents_from_faiss)} documents (after basic title dedupe).")
            else:
                logger.info("No documents retrieved from general FAISS search for the query.")
        except Exception as e_faiss:
            logger.error(f"Error during general FAISS retrieval: {e_faiss}", exc_info=True)
    elif DEBUG_SKIP_RAG_CONTEXT:
        logger.info("DEBUG_SKIP_RAG_CONTEXT is true. Skipping general RAG retrieval.")
    else:
        logger.info("FAISS index not loaded or query is empty. Skipping general RAG retrieval.")

    query_lower = actual_user_question.lower()
    # Adjusted keywords for paper queries, more specific
    is_paper_query = any(keyword in query_lower for keyword in ["paper", "papers", "suggest paper", "recommend paper", "find paper", "arxiv", "top 5 paper", "top 10 paper", "research article", "publication on"])

    # --- Gemini Paper Candidate Generation ---
    gemini_arxiv_papers_info = [] # Will store list of dicts: {'title': '', 'authors': [], 'abstract': '', 'arxiv_id': ''}

    if is_paper_query and systems_status["gemini_model_configured"] and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
        logger.info(f"Paper query detected. Getting ArXiv paper suggestions from Gemini for: '{actual_user_question[:100]}...'")
        
        # Refined prompt for Gemini to get ArXiv paper details
        gemini_arxiv_prompt = (
            f"Based on the user query: '{actual_user_question}', "
            f"Please provide a list of 3 to 5 highly relevant academic paper suggestions strictly from ArXiv. "
            f"For each paper, provide the following details in a structured format (e.g., JSON-like or clearly delimited sections per paper): "
            f"1. Exact Title. "
            f"2. List of Authors (e.g., ['Author A', 'Author B']). "
            f"3. A concise Abstract (typically the one from ArXiv). "
            f"4. The ArXiv ID (e.g., '2310.06825' or 'cs.CL/2310.06825'). "
            f"If you cannot find 3-5 relevant papers from ArXiv or cannot find all details, provide what you can. "
            f"If no relevant ArXiv papers are found, respond with ONLY the phrase 'NO_ARXIV_PAPERS_FOUND'."
        )
        
        try:
            # Use a unique chat_id for this sub-query to avoid interference if cancellation is implemented per chat_id
            gemini_sub_query_chat_id = f"gemini_arxiv_sub_query_{chat_id or 'new_chat'}"
            
            # Ensure active_cancellation_events has an entry for this sub-query if generate_gemini_response uses it
            # This part might need to be adapted based on how cancellation is managed globally.
            # For now, assuming generate_gemini_response can handle a chat_id without a pre-existing event or creates one.
            if chat_id and chat_id not in active_cancellation_events: # If main chat has an event, maybe reuse/derive?
                 # active_cancellation_events[gemini_sub_query_chat_id] = threading.Event() # Example
                 pass


            gemini_response_stream = generate_gemini_response(gemini_arxiv_prompt, chat_id=gemini_sub_query_chat_id)
            full_gemini_text_response = "".join([chunk for chunk in gemini_response_stream if isinstance(chunk, str)]).strip()
            logger.debug(f"Full raw response from Gemini for ArXiv suggestions: {full_gemini_text_response}")

            if "NO_ARXIV_PAPERS_FOUND" in full_gemini_text_response or not full_gemini_text_response:
                logger.info("Gemini indicated no ArXiv suggestions found or returned an empty response.")
            else:
                # Enhanced parsing for ArXiv details. This is a best-effort parser.
                # Ideally, Gemini would return structured JSON if its API allows that reliably.
                papers_data_str = full_gemini_text_response.split("---") # Assuming "---" separates papers if not JSON
                current_paper_details = {}
                
                # Regex patterns for better extraction
                title_re = re.compile(r"^\s*(?:[0-9]+\.\s*)?Title:(.*)", re.IGNORECASE | re.MULTILINE)
                authors_re = re.compile(r"^\s*Authors:(.*)", re.IGNORECASE | re.MULTILINE)
                abstract_re = re.compile(r"^\s*Abstract:(.*)", re.IGNORECASE | re.MULTILINE)
                arxiv_id_re = re.compile(r"^\s*ArXiv ID:(.*)", re.IGNORECASE | re.MULTILINE)

                # Try to parse as JSON first, as it's more reliable
                try:
                    parsed_json_data = json.loads(full_gemini_text_response)
                    # Assuming parsed_json_data is a list of paper dicts or a dict containing them
                    if isinstance(parsed_json_data, list):
                        for paper_data_entry in parsed_json_data:
                            if isinstance(paper_data_entry, dict) and "title" in paper_data_entry and "arxiv_id" in paper_data_entry and "abstract" in paper_data_entry:
                                gemini_arxiv_papers_info.append({
                                    "title": paper_data_entry.get("title", "").strip(),
                                    "authors": paper_data_entry.get("authors", []), # Assuming authors is a list
                                    "abstract": paper_data_entry.get("abstract", "").strip(),
                                    "arxiv_id": paper_data_entry.get("arxiv_id", "").strip()
                                })
                        logger.info(f"Successfully parsed {len(gemini_arxiv_papers_info)} ArXiv papers from Gemini JSON response.")
                    # Add more specific JSON structure handling if needed
                except json.JSONDecodeError:
                    logger.info("Gemini response for ArXiv papers was not valid JSON. Falling back to text parsing.")
                    # Fallback to text parsing
                    for paper_block in re.split(r'\n\s*\d+\.\s*Title:|\n\s*Title:', '\n' + full_gemini_text_response): # Split by "1. Title:" or "Title:"
                        if not paper_block.strip():
                            continue
                        
                        paper_data = {}
                        # Re-prepend "Title:" because the split might remove it for the first block
                        # and for subsequent blocks, the split point is before "Title:"
                        search_block = "Title: " + paper_block 

                        title_match = title_re.search(search_block)
                        if title_match: paper_data["title"] = title_match.group(1).strip()

                        authors_match = authors_re.search(search_block)
                        if authors_match:
                            authors_list_str = authors_match.group(1).strip()
                            # Handle potential list-like string for authors e.g. "['Author A', 'Author B']" or "Author A, Author B"
                            if authors_list_str.startswith('[') and authors_list_str.endswith(']'):
                                try:
                                    paper_data["authors"] = eval(authors_list_str) # eval is risky, use ast.literal_eval if possible
                                except: # Fallback if eval fails
                                    paper_data["authors"] = [a.strip() for a in authors_list_str.strip('[]').split(',') if a.strip()]
                            else:
                                paper_data["authors"] = [a.strip() for a in authors_list_str.split(',') if a.strip()]
                        else:
                            paper_data["authors"] = []


                        abstract_match = abstract_re.search(search_block)
                        if abstract_match: paper_data["abstract"] = abstract_match.group(1).strip()
                        
                        arxiv_id_match = arxiv_id_re.search(search_block)
                        if arxiv_id_match: paper_data["arxiv_id"] = arxiv_id_match.group(1).strip()

                        if paper_data.get("title") and paper_data.get("arxiv_id") and paper_data.get("abstract"):
                            gemini_arxiv_papers_info.append(paper_data)
                        else:
                            logger.warning(f"Could not parse all required fields from paper block: {paper_block[:100]}...")
                    
                    if gemini_arxiv_papers_info:
                        logger.info(f"Parsed {len(gemini_arxiv_papers_info)} ArXiv paper suggestions from Gemini (text parsing).")
                    else:
                        logger.warning("Text parsing of Gemini ArXiv response yielded no structured papers.")


        except Exception as e_gemini_arxiv:
            logger.error(f"Error calling or parsing Gemini for ArXiv paper suggestions: {e_gemini_arxiv}", exc_info=True)
    
    elif is_paper_query and DEBUG_SKIP_GEMINI_SUGGESTIONS:
         logger.info("DEBUG_SKIP_GEMINI_SUGGESTIONS is true. Skipping Gemini ArXiv paper suggestions.")

    # Format Gemini ArXiv Paper Suggestions for LLM Instruction
    processed_titles_for_llm_instruction = ""
    contextual_info_for_llm_about_papers = "" # Initialize to empty string

    if gemini_arxiv_papers_info: # Check if list is not empty
        title_lines = []
        for paper_info in gemini_arxiv_papers_info:
            title_line = paper_info.get("title", "").strip()
            arxiv_id = paper_info.get("arxiv_id", "").strip()
            if arxiv_id:
                title_line += f" (arXiv:{arxiv_id})"
            if title_line: # Add only if title is not empty after stripping
                title_lines.append(title_line)
        
        if title_lines: # If we actually got some valid titles
            processed_titles_for_llm_instruction = "\\n".join(title_lines)
            contextual_info_for_llm_about_papers = processed_titles_for_llm_instruction
            logger.info(f"Prepared processed_titles_for_llm_instruction with {len(title_lines)} titles.")
        else:
            logger.info("Gemini suggested papers but all titles were empty after processing; processed_titles_for_llm_instruction remains empty.")
    else:
        logger.info("No Gemini ArXiv papers info to process for processed_titles_for_llm_instruction.")

    # --- Phase 2: Targeted RAG (FAISS Retrieval based on Gemini ArXiv Suggestions) ---
    targeted_faiss_documents_for_arxiv = [] # Stores dicts: {"gemini_suggestion": {}, "faiss_document": Doc}
    if systems_status["faiss_index_loaded"] and faiss_index and gemini_arxiv_papers_info and not DEBUG_SKIP_RAG_CONTEXT:
        logger.info(f"Performing targeted FAISS search for {len(gemini_arxiv_papers_info)} Gemini ArXiv suggestions.")
        seen_targeted_faiss_content_snippets = [] # For semantic deduplication of content from FAISS

        for paper_suggestion in gemini_arxiv_papers_info:
            search_query_for_faiss = paper_suggestion.get("title", "")
            arxiv_id_from_gemini = paper_suggestion.get("arxiv_id")
            if arxiv_id_from_gemini: # Add ArXiv ID to search query if present
                 search_query_for_faiss += f" arxiv:{arxiv_id_from_gemini}" 

            if not search_query_for_faiss.strip():
                logger.debug(f"Skipping targeted FAISS search for empty Gemini suggestion.")
                continue

            try:
                # Fetch a few candidates to check for best match. Using similarity_search_with_score.
                # k=3 means we get top 3 closest matches from FAISS for the `search_query_for_faiss`
                docs_with_scores = faiss_index.similarity_search_with_score(search_query_for_faiss, k=3)
                
                best_match_doc_for_this_suggestion = None
                # Score interpretation: For FAISS with default L2 distance, lower score is better.
                # If using cosine similarity directly (e.g. via `FAISS.similarity_search_with_score_by_vector`)
                # higher score would be better. Assuming default L2 distance here.
                lowest_score_for_this_suggestion = float('inf') 

                for doc_candidate, score in docs_with_scores:
                    candidate_title = doc_candidate.metadata.get("title", "").strip()
                    local_arxiv_id = doc_candidate.metadata.get("arxiv_id", "").strip() # Your FAISS stores this

                    # Criterion 1: Exact ArXiv ID match (highest priority)
                    if arxiv_id_from_gemini and local_arxiv_id and arxiv_id_from_gemini.lower() == local_arxiv_id.lower():
                        best_match_doc_for_this_suggestion = doc_candidate
                        logger.debug(f"Strong match by ArXiv ID for '{paper_suggestion.get('title')}': FAISS doc '{candidate_title}' (ID: {local_arxiv_id})")
                        break # Found definitive match for this suggestion, move to next suggestion

                    # Criterion 2: Semantic similarity of titles (if no ArXiv ID match yet)
                    # Check if title from Gemini suggestion is semantically similar to title from FAISS doc
                    # Threshold 0.80 for title similarity is reasonably strict.
                    if are_texts_semantically_similar(paper_suggestion.get("title",""), candidate_title, threshold=0.80):
                       if score < lowest_score_for_this_suggestion: # If titles are similar, prefer doc with lower distance score
                           best_match_doc_for_this_suggestion = doc_candidate
                           lowest_score_for_this_suggestion = score
                           logger.debug(f"Potential semantic title match for '{paper_suggestion.get('title')}': FAISS doc '{candidate_title}' (Score: {score})")
                
                # After checking all candidates for the current Gemini suggestion:
                if best_match_doc_for_this_suggestion:
                    page_content_str = str(best_match_doc_for_this_suggestion.page_content if best_match_doc_for_this_suggestion.page_content is not None else "")
                    content_snippet_for_check = page_content_str[:300] # First 300 chars for dupe check
                    
                    is_semantically_dupe_content = any(
                        are_texts_semantically_similar(content_snippet_for_check, existing_snippet, threshold=0.90)
                        for existing_snippet in seen_targeted_faiss_content_snippets
                    )

                    if not is_semantically_dupe_content:
                        targeted_faiss_documents_for_arxiv.append({
                            "gemini_suggestion": paper_suggestion, # Store the original Gemini suggestion dict
                            "faiss_document": best_match_doc_for_this_suggestion # Store the found Langchain Document object
                        })
                        seen_targeted_faiss_content_snippets.append(content_snippet_for_check)
                        logger.info(f"Added targeted FAISS match for Gemini suggestion (Title: '{paper_suggestion.get('title','N/A_TITLE')}').")
                    else:
                        logger.info(f"Targeted FAISS found doc for '{paper_suggestion.get('title','N/A_TITLE')}', but content was semantically duplicate of another *already added* targeted result.")
                else:
                    logger.info(f"No strong local match in FAISS found for Gemini ArXiv suggestion: '{paper_suggestion.get('title','N/A_TITLE')}'")

            except Exception as e_targeted_faiss:
                logger.error(f"Error during targeted FAISS search for suggestion '{paper_suggestion.get('title', 'N/A_TITLE')}': {e_targeted_faiss}", exc_info=True)
        
        logger.info(f"Finished targeted FAISS search. Found {len(targeted_faiss_documents_for_arxiv)} distinct local documents for Gemini ArXiv suggestions.")
    
    elif DEBUG_SKIP_RAG_CONTEXT:
        logger.info("DEBUG_SKIP_RAG_CONTEXT is true. Skipping targeted RAG based on Gemini ArXiv suggestions.")
    elif not gemini_arxiv_papers_info:
        logger.info("No Gemini ArXiv papers were suggested, so skipping targeted FAISS search.")
    else:
        logger.info("FAISS index not loaded. Skipping targeted RAG based on Gemini ArXiv suggestions.")


    # --- Combine and Deduplicate General RAG Context (if not a paper query or if paper RAG yielded little) ---
    # This section populates `final_general_rag_documents_for_prompt` for non-paper queries,
    # or as a fallback if paper-specific RAG (targeted_faiss_documents_for_arxiv) didn't find much.
    final_general_rag_documents_for_prompt = []
    if not DEBUG_SKIP_RAG_CONTEXT: # Check RAG skip flag
        # Only proceed with general RAG if it's not a paper query OR 
        # if it IS a paper query BUT the targeted search for ArXiv papers yielded no results.
        if not is_paper_query or (is_paper_query and not targeted_faiss_documents_for_arxiv):
            logger.info("Proceeding with general RAG context population (either not a paper query, or paper query with no targeted results).")
            final_seen_content_snippets_for_semantic_check_general = [] 
            final_seen_titles_normalized_general = set()

            for doc in general_retrieved_documents_from_faiss: # `general_retrieved_documents_from_faiss` from earlier general search
                if len(final_general_rag_documents_for_prompt) >= AppConfig.N_RETRIEVED_DOCS:
                    break # Stop if we have enough documents
                
            title = doc.metadata.get("title", "").strip()
            normalized_title = title.lower()
            page_content_str = str(doc.page_content if doc.page_content is not None else "")
            content_snippet_for_check = page_content_str[:300]

            is_semantically_dupe_content_general = any(
                are_texts_semantically_similar(content_snippet_for_check, existing_snippet, threshold=0.90)
                for existing_snippet in final_seen_content_snippets_for_semantic_check_general
            )
                                
            # Add if: (no title OR title not seen before) AND content not semantically duplicate
            if (not title or normalized_title not in final_seen_titles_normalized_general) and not is_semantically_dupe_content_general:
                final_general_rag_documents_for_prompt.append(doc)
                if title: final_seen_titles_normalized_general.add(normalized_title)
                final_seen_content_snippets_for_semantic_check_general.append(content_snippet_for_check)
            elif title and normalized_title in final_seen_titles_normalized_general:
                 logger.info(f"Skipping (title duplicate) general FAISS doc for general context: '{title}'")
            elif is_semantically_dupe_content_general:
                logger.info(f"Skipping (semantic duplicate content) general FAISS doc for general context: '{title}'")
            logger.info(f"Populated {len(final_general_rag_documents_for_prompt)} documents for general RAG context.")
        else:
            logger.info("Skipping general RAG context population because it is a paper query AND targeted ArXiv RAG yielded results.")
            
    elif DEBUG_SKIP_RAG_CONTEXT:
        logger.info("DEBUG_SKIP_RAG_CONTEXT is TRUE: Skipping general RAG context in Gemma prompt.")
    else:
        logger.info("final_general_rag_documents_for_prompt is empty. No general RAG context string will be built for Gemma.")

    # --- Construct retrieved_context_str_for_gemma_prompt (from General RAG documents) ---
    # This string is intended for general queries or as fallback context for paper queries where Gemma itself forms suggestions.
    retrieved_context_str_for_gemma_prompt = ""
    if final_general_rag_documents_for_prompt and not DEBUG_SKIP_RAG_CONTEXT: 
        context_parts = []
        for i, doc_to_add in enumerate(final_general_rag_documents_for_prompt[:AppConfig.N_RETRIEVED_DOCS]):
            content = str(doc_to_add.page_content).strip() if doc_to_add.page_content is not None else ""
            title = doc_to_add.metadata.get('title', f'Retrieved Document {i+1}')
            max_content_len = 700 # Characters per RAG document in prompt
            context_parts.append(f"Context Document {i+1} (Title: {title}):\\n{content[:max_content_len]}{'...' if len(content) > max_content_len else ''}")
            
        if context_parts:
            # Clearly delimit this general context for the LLM
            retrieved_context_str_for_gemma_prompt = (
                "\\n\\n--- Start of General Retrieved Context ---\\n"
                + "\\n\\n".join(context_parts) + 
                "\\n--- End of General Retrieved Context ---\\n"
            )
            logger.info(f"General RAG context string prepared for Gemma with {len(final_general_rag_documents_for_prompt)} documents.")
    elif DEBUG_SKIP_RAG_CONTEXT:
        logger.info("DEBUG_SKIP_RAG_CONTEXT is true. No general RAG context string will be built for Gemma.")
    else:
        logger.info("final_general_rag_documents_for_prompt is empty. No general RAG context string will be built for Gemma.")

    # --- Gemma Stop Sequences ---
    # These are common phrases the model might try to output that we want to stop.
    # This list can be expanded based on observed model behavior.
    stop_sequences_general = [
        "User:", "User Query:", "USER:", "ASSISTANT:", "Assistant:", "System:", 
        "\nUser:", "\nUSER:", "\nASSISTANT:", "\nAssistant:", "\nSystem:",
        "Context:", "CONTEXT:", "Answer:", "ANSWER:", "Note:", "Response:",
        "In this response,", "In summary,", "To summarize,", "Let's expand on",
        "The user is asking", "The user wants to know",
        "This text follows", "The following is a", "This is a text about",
        "My goal is to", "I am an AI assistant", "As an AI",
        "\n\nHuman:", "<|endoftext|>", "<|im_end|>", "\u202f",
        "STOP_ASSISTANT_PRIMING_PHRASE:" # Changed to avoid conflict with actual priming, just in case
    ]
    
    generation_params_general = {
        "temperature": 0.2, 
        "top_p": 0.8,       
        "top_k": 40,        
        "repeat_penalty": 1.15, 
        "max_tokens": AppConfig.LLAMA_MAX_TOKENS,   # Increased for larger output
        "stop": stop_sequences_general
    }

    # --- Role and Task Definition for General Knowledge ---
    # This prompt tries to make Gemma concise and avoid meta-commentary or conversational fluff.
    general_knowledge_instruction = (
        "You are a concise and factual AI assistant. Your primary goal is to provide direct answers or explanations based on your training data and the provided conversation log. "
        "Focus solely on the user's query. Do NOT include conversational introductions or closings like 'Hello!', 'Sure!', 'Certainly!', 'Okay, here is...', 'I hope this helps!', or 'Let me know if you have other questions.' "
        "Do NOT engage in meta-commentary (e.g., 'In this response, I will...', 'I will now explain...', 'Based on the context...'). "
        "Avoid any self-reference (e.g., 'As an AI assistant...', 'My purpose is to...'). "
        "Avoid section headers or titles in your response (e.g., 'Explanation:', 'Details:', 'Summary:'). "
        "If the conversation log is relevant, use it to understand the context of the current query. "
        "The response should be approximately 100-150 words for a general explanation, unless the query specifically asks for more detail or a list."
    )

    if is_paper_query:
        if processed_titles_for_llm_instruction and not DEBUG_SKIP_GEMINI_SUGGESTIONS: # Gemini provided titles
            paper_specific_instruction = (
                f"You are an AI research assistant. Your task is to process a list of paper titles provided under 'PAPER TITLES TO PROCESS'. "
                f"For EACH paper title in that list, you MUST provide a relevant abstract. Follow these steps meticulously for each title:\n\n"
                
                f"PAPER TITLES TO PROCESS (one title per line; includes arXiv ID if available from initial suggestion):\n{processed_titles_for_llm_instruction}\n\n"
                
                f"INSTRUCTIONS FOR PROCESSING EACH PAPER TITLE FROM THE LIST ABOVE:\n"
                f"1. Take the current paper title (and its arXiv ID, if present) from the 'PAPER TITLES TO PROCESS' list.\n"
                f"2. Search the 'ADDITIONAL CONTEXT FROM DOCUMENT DATABASE (curated RAG results):' section (if present later in this prompt) for this exact title or a highly similar one. \n"
                f"3. If a STRONG match is found in the DOCUMENT DATABASE context AND it provides relevant content/abstract:\n"
                f"   - Use the retrieved content as the abstract for this paper. The abstract should be unique to this paper. Do NOT use placeholder text.\n"
                f"   - Your output for this paper MUST be formatted STRICTLY as (example for a paper found in the database):\n"
                f"     1. Title: [Exact Paper Title from 'PAPER TITLES TO PROCESS' (including arXiv ID if it was there)]\n"
                f"        Abstract: [The relevant retrieved abstract/content snippet from the DOCUMENT DATABASE]\n"
                f"4. If NO strong match is found in the DOCUMENT DATABASE context, OR if a match is found but it has no usable abstract:\n"
                f"   - You MUST generate a concise, plausible, and UNIQUE abstract (1-3 sentences) for the paper based SOLELY on its title and your general knowledge. Do NOT use placeholder text. Do NOT repeat abstracts from other papers.\n"
                f"   - Your output for this paper MUST be formatted STRICTLY as (example for a generated abstract):\n"
                f"     2. Title: [Exact Paper Title from 'PAPER TITLES TO PROCESS' (including arXiv ID if it was there)]\n"
                f"        Abstract: [Your UNIQUE generated abstract based on the title]\n"
                f"5. If, after attempting step 4, you absolutely CANNOT generate a meaningful abstract even from the title:\n"
                f"   - Your output for this paper MUST be formatted STRICTLY as (example):\n"
                f"     3. Title: [Exact Paper Title from 'PAPER TITLES TO PROCESS' (including arXiv ID if it was there)]\n"
                f"        Abstract: Unable to provide an abstract for this title.\n\n"
                
                f"CRITICAL OVERALL OUTPUT FORMATTING RULES (Adhere strictly):\n"
                f"- Your entire response MUST begin *EXACTLY* with '1. Title: ' for the first paper processed, and continue numbering sequentially for ALL subsequent papers from the 'PAPER TITLES TO PROCESS' list.\n"
                f"- Process EVERY title from the 'PAPER TITLES TO PROCESS' list.\n"
                f"- NO conversational text, preamble, introduction, or conclusion. ONLY the numbered list of titles with their abstracts.\n"
                f"- Each paper's 'X. Title: ...' line MUST be on its own line.\n"
                f"- Each paper's 'Abstract: ...' line MUST start on the immediate next line, indented with exactly three spaces.\n"
                f"- Do NOT use any markdown list characters like '*', '-', or similar at the start of any line related to the paper list.\n"
                f"- Ensure the paper titles (and arXiv IDs, if present) in your output EXACTLY match those from the 'PAPER TITLES TO PROCESS' list.\n"
                f"- DO NOT include a 'Reasoning' section. Only Title and Abstract."
            )
        else: # Paper query, but no Gemini suggestions (or they are skipped), so LLM suggests its own
            paper_specific_instruction = (
                f"You are an AI research assistant. The user is asking for paper suggestions. "
                f"Your task is to suggest 3 to 5 relevant academic paper titles based on the user's query and your general knowledge. "
                f"For EACH of the 3 to 5 papers you suggest, present each paper using a strict two-line format:\n"
                f"1. The FIRST line for each paper: '[Number]. Title: [The Paper Title You Are Suggesting] '\n"
                f"2. The SECOND line for that SAME paper (immediately following the title line): '   Abstract: [Your concise 3-4 sentence plausible abstract for THIS specific title]'\n"
                f"Ensure the abstract is distinctly separate from the title, on its own indented line, and prefixed with 'Abstract: '. Do not merge the title and abstract onto a single line.\n\n"
                
                f"If general 'Retrieved Document' context is available later in this prompt, you can check if any of YOUR suggested titles coincidentally match any retrieved document titles. If a strong match exists AND the retrieved content is suitable as an abstract, you MAY use that content for the abstract, noting it by starting the abstract with '(From database): '. Otherwise, you MUST generate the abstract from your own knowledge based on the title you suggested.\n\n"

                f"CRITICAL OUTPUT FORMATTING RULES (Adhere strictly to ensure proper display and parsing):\n"
                f"1. Your entire response MUST begin *EXACTLY* with '1. Title: ' for the first paper you suggest. There must be NO text, preamble, or conversational phrases before this first '1. Title: ' line.\n"
                f"2. For EACH paper you suggest (from 1 to 5 papers):\n"
                f"   a. Start the line with the sequential paper number (e.g., 1, 2, 3), followed by a period, a single space, and then the phrase 'Title: ' followed by the paper title you are suggesting. (Example: '1. Title: Example Paper Title')\n"
                f"   b. The paper title itself should NOT contain any literal '\\n' characters. If a title is long, let it wrap naturally.\n"
                f"   c. On the VERY NEXT LINE (achieved by outputting a single newline character '\\n' immediately after the title line, and nothing else on that line), you MUST indent with EXACTLY three spaces, then write 'Abstract: ' followed by the concise (1-2 sentences) abstract for THIS SPECIFIC TITLE.\n"
                f"   d. The abstract content itself MUST be an actual abstract. It should NOT be conversational, nor should it be a comment about the list of papers or the suggestion process. It should NOT contain any literal '\\n' characters within a sentence. If the abstract is long, let it wrap naturally. Use a single newline character '\\n' ONLY to end the abstract line.\n"
                f"   e. After the abstract line for one paper, the next paper's 'X. Title:' line (if any) MUST start immediately on the next line. Do NOT insert any blank lines between suggested papers.\n"
                f"   f. Do NOT use any markdown list markers (like '* ', '- ', '+ ') or any other characters at the start of the 'X. Title:' or '   Abstract:' lines other than what is explicitly specified.\n\n"
                
                f"EXAMPLE OF PERFECTLY CORRECT FORMAT FOR TWO SUGGESTED PAPERS:\n"
                f"1. Title: Deep Learning for Molecules and Materials\n"
                f"   Abstract: This paper reviews the application of deep learning in chemistry and materials science, covering areas like property prediction and generative models.\n"
                f"2. Title: The Role of Catalysis in Green Chemistry\n"
                f"   Abstract: (From database): Explores various catalytic methods that contribute to environmentally friendly chemical processes, reducing waste and energy consumption.\n\n"

                f"MANDATORY OVERALL RESPONSE RULES:\n"
                f"- Your response MUST ONLY contain the numbered list of your suggested titles and their corresponding abstracts, strictly following ALL CRITICAL OUTPUT FORMATTING RULES above. Nothing else should appear in your response, either before or after this list.\n"
                f"- If, for a specific title YOU suggest, you cannot confidently generate a meaningful abstract (and no database content is usable), use the phrase 'General suggestion based on topic; specific abstract not available.' as the abstract content for that paper's abstract line.\n"
                f"- If, after careful consideration, you determine you cannot find or generate ANY relevant paper titles to suggest based on the user's query and your knowledge, your entire response MUST consist ONLY of the exact phrase: 'I am unable to suggest specific papers from my current knowledge based on your query.' (This exact phrase, and absolutely nothing else).\n"
                f"- Do NOT include any conversational introductions (e.g., 'Sure, here are some papers...'), closings (e.g., 'I hope this helps!'), meta-commentary about your process, or self-references (unless it's the specific 'unable to suggest' phrase above)."
            )
    
    # Determine which instruction to use
    query_lower = actual_user_question.lower()
    is_paper_query = any(keyword in query_lower for keyword in ["paper", "suggest", "recommend", "article", "publication", "study", "research"])

    if is_paper_query:
        instruction_to_use = paper_specific_instruction
        logger.info("Using paper-specific instruction for Gemma.")
    else:
        instruction_to_use = general_knowledge_instruction
        logger.info("Using general knowledge instruction for Gemma.")

    # --- Constructing the Final Prompt for Gemma ---
    prompt_for_gemma = f"{instruction_to_use}\\n\\n"
    if conversation_log_str_for_gemma and conversation_log_str_for_gemma != "No previous conversation.":
        prompt_for_gemma += f"CONVERSATION LOG:\\n{conversation_log_str_for_gemma}\\n\\n"
    
    if not DEBUG_SKIP_AUGMENTATION:
        if not DEBUG_SKIP_RAG_CONTEXT: # Check RAG skip flag
            # If it's a paper query with Gemini titles, the instruction tells Gemma
            # to process titles from 'PAPER TITLES TO PROCESS' (which is `processed_titles_for_llm_instruction`)
            # and look for abstracts in 'ADDITIONAL CONTEXT FROM DOCUMENT DATABASE'.
            if is_paper_query and processed_titles_for_llm_instruction and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
                # The `processed_titles_for_llm_instruction` is already part of `paper_specific_instruction` under 'PAPER TITLES TO PROCESS'
                # So, we only need to add the FAISS retrieved abstracts if available.
                
                # The `retrieved_context_str_for_gemma_prompt` was built from `final_general_rag_documents_for_prompt`
                # For paper queries where Gemini provided titles, we should instead build context from `targeted_faiss_documents_for_arxiv`.
                
                targeted_arxiv_context_parts = []
                if targeted_faiss_documents_for_arxiv: # This list contains {'gemini_suggestion': {}, 'faiss_document': Doc}
                    for i, item in enumerate(targeted_faiss_documents_for_arxiv):
                        faiss_doc_obj = item.get("faiss_document")
                        if faiss_doc_obj:
                            content = str(faiss_doc_obj.page_content).strip() if faiss_doc_obj.page_content is not None else ""
                            title = faiss_doc_obj.metadata.get('title', f'Retrieved ArXiv Document {i+1}')
                            arxiv_id_local = faiss_doc_obj.metadata.get('arxiv_id', 'N/A')
                            max_content_len = 700 # Characters per RAG document in prompt
                            targeted_arxiv_context_parts.append(f"Context for ArXiv Paper (Title: {title}, ID: {arxiv_id_local}):\\n{content[:max_content_len]}{'...' if len(content) > max_content_len else ''}")
                
                if targeted_arxiv_context_parts:
                    final_context_for_prompt = (
                        "\\n\\n--- ADDITIONAL CONTEXT FROM DOCUMENT DATABASE (curated RAG results for suggested ArXiv papers) ---\\n"
                        + "\\n\\n".join(targeted_arxiv_context_parts) + 
                        "\\n--- End of ArXiv Document Database Context ---\\n"
                    )
                    prompt_for_gemma += final_context_for_prompt
                    logger.info(f"Targeted ArXiv RAG context string prepared for Gemma with {len(targeted_arxiv_context_parts)} documents.")
                else:
                    logger.info("Paper query with Gemini titles, but no targeted RAG documents from FAISS to provide as context. Gemma will generate abstracts.")

            elif retrieved_context_str_for_gemma_prompt: # General RAG context for other cases
                prompt_for_gemma += f"ADDITIONAL CONTEXT FROM DOCUMENT DATABASE (curated RAG results):\\n{retrieved_context_str_for_gemma_prompt}\\n\\n"
            else:
                logger.info("DEBUG: No RAG context (neither specific paper context nor general) to add.")
        else:
            logger.warning("DEBUG_SKIP_RAG_CONTEXT is TRUE: Skipping RAG context in Gemma prompt.")

        if is_paper_query and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
            if processed_titles_for_llm_instruction:
                logger.info("DEBUG: Gemini paper titles are included in 'PAPER TITLES TO PROCESS' within the instruction.")
            else:
                logger.info("DEBUG: Paper query, but no Gemini paper titles were prepared for the instruction (LLM will suggest its own).")
        elif is_paper_query and DEBUG_SKIP_GEMINI_SUGGESTIONS:
             logger.warning("DEBUG_SKIP_GEMINI_SUGGESTIONS is TRUE: Skipping Gemini suggestions for instruction.")
    else:
        logger.warning("DEBUG_SKIP_AUGMENTATION is TRUE: Skipping ALL augmentation in Gemma prompt.")
    
    prompt_for_gemma += f"USER QUERY: {actual_user_question}\\n\\nASSISTANT'S FACTUAL ANSWER:"

    logger.info(f"Final Prompt for Gemma (first 300 chars): {prompt_for_gemma[:300]}...")
    logger.info(f"Full Prompt for Gemma:\\n{prompt_for_gemma}") 
    if gemini_arxiv_papers_info: # Log what Gemini originally suggested that led to targeted search
        logger.info(f"Gemini initially suggested papers for targeted search: {json.dumps(gemini_arxiv_papers_info, indent=2)}")
    logger.info(f"Generation parameters for Gemma: {generation_params_general}")

    cancel_event_for_llm = active_cancellation_events.get(chat_id)
    if not cancel_event_for_llm:
        logger.warning(f"No cancel_event found for chat_id {chat_id} in generate_local_rag_response. LLM generation will not be cancellable.")
        cancel_event_for_llm = threading.Event() # Dummy event

    try:
        stream = local_llm.create_chat_completion( 
            messages=[{"role": "user", "content": prompt_for_gemma}],
            **generation_params_general,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if cancel_event_for_llm.is_set():
                logger.info(f"Local LLM (Gemma) generation cancelled for chat_id {chat_id}.")
                yield CANCEL_MESSAGE
                # local_llm.close() or similar if the Llama object needs explicit stream closing
                break # Exit the loop

            delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta_content:
                yield delta_content
                full_response += delta_content
        
        logger.info(f"Gemma full response (after stream): {full_response.strip()}")

    except Exception as e:
        logger.error(f"Error during local LLM generation: {e}", exc_info=True)
        yield f"Error generating response from local model: {str(e)}"

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index2.html') 

@app.route('/api/sidebar-data', methods=['GET'])
def get_sidebar_data():
    conn = get_db_connection()
    try:
        folders_cursor = conn.execute("SELECT id, name FROM folders ORDER BY created_at DESC")
        folders_data = []
        for folder_row in folders_cursor.fetchall():
            folder = dict(folder_row)
            chats_cursor = conn.execute(
                "SELECT id, title, last_snippet FROM chats WHERE folder_id = ? ORDER BY updated_at DESC",
                (folder['id'],)
            )
            folder['chats'] = [dict(chat_row) for chat_row in chats_cursor.fetchall()]
            folders_data.append(folder)
        
        uncategorized_chats_cursor = conn.execute(
            "SELECT id, title, last_snippet FROM chats WHERE folder_id IS NULL ORDER BY updated_at DESC"
        )
        uncategorized_chats_data = [dict(chat_row) for chat_row in uncategorized_chats_cursor.fetchall()]
        
        return jsonify({"folders": folders_data, "uncategorized_chats": uncategorized_chats_data})
    except sqlite3.Error as e:
        logger.error(f"Error fetching sidebar data: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch sidebar data"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/folders', methods=['POST'])
def create_folder():
    data = request.get_json()
    folder_name = data.get('name')
    if not folder_name:
        return jsonify({"error": "Folder name is required"}), 400
    conn = get_db_connection()
    try:
        cursor = conn.execute("INSERT INTO folders (name) VALUES (?)", (folder_name,))
        conn.commit()
        return jsonify({"id": cursor.lastrowid, "name": folder_name}), 201
    except sqlite3.IntegrityError: # Example: if folder names were unique
        return jsonify({"error": "Folder with this name might already exist"}), 409
    except sqlite3.Error as e:
        logger.error(f"Error creating folder: {e}", exc_info=True)
        return jsonify({"error": "Failed to create folder"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chats', methods=['POST'])
def create_chat():
    data = request.get_json()
    title = data.get('title', 'New Chat')
    folder_id = data.get('folder_id') # Can be None

    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO chats (title, folder_id) VALUES (?, ?)",
            (title, folder_id)
        )
        conn.commit()
        chat_id = cursor.lastrowid
        # Optionally, add an initial message to the new chat
        # conn.execute("INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
        #              (chat_id, 'bot', 'Hi there! How can I help you today?'))
        # conn.commit()
        return jsonify({"id": chat_id, "title": title, "folder_id": folder_id}), 201
    except sqlite3.Error as e:
        logger.error(f"Error creating chat: {e}", exc_info=True)
        return jsonify({"error": "Failed to create chat"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chats/<int:chat_id>/messages', methods=['GET'])
def get_chat_messages(chat_id):
    conn = get_db_connection()
    try:
        # First, get chat title
        chat_info_cursor = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
        chat_info = chat_info_cursor.fetchone()
        if not chat_info:
            return jsonify({"error": "Chat not found"}), 404
        
        messages_cursor = conn.execute(
            "SELECT sender, content, timestamp FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,)
        )
        messages = [dict(row) for row in messages_cursor.fetchall()]
        return jsonify({"title": chat_info["title"], "messages": messages})
    except sqlite3.Error as e:
        logger.error(f"Error fetching messages for chat {chat_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch messages"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chats/<int:chat_id>/folder', methods=['PUT'])
def move_chat_to_folder(chat_id):
    data = request.get_json()
    new_folder_id = data.get('folder_id') # This can be None/null to move to uncategorized

    conn = get_db_connection()
    try:
        # Check if chat exists
        chat_exists_cursor = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        if not chat_exists_cursor.fetchone():
            return jsonify({"error": "Chat not found"}), 404

        # If moving to a specific folder, check if folder exists
        if new_folder_id is not None:
            folder_exists_cursor = conn.execute("SELECT id FROM folders WHERE id = ?", (new_folder_id,))
            if not folder_exists_cursor.fetchone():
                return jsonify({"error": "Target folder not found"}), 404
        
        conn.execute(
            "UPDATE chats SET folder_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_folder_id, chat_id)
        )
        conn.commit()
        logger.info(f"Moved chat {chat_id} to folder {new_folder_id}")
        return jsonify({"message": "Chat moved successfully", "chat_id": chat_id, "new_folder_id": new_folder_id}), 200
    except sqlite3.Error as e:
        logger.error(f"Error moving chat {chat_id} to folder {new_folder_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to move chat"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/folders/<int:folder_id>', methods=['PUT'])
def update_folder(folder_id):
    data = request.get_json()
    new_name = data.get('name')

    if not new_name or not new_name.strip():
        return jsonify({"error": "Folder name cannot be empty"}), 400

    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT id FROM folders WHERE id = ?", (folder_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Folder not found"}), 404

        conn.execute("UPDATE folders SET name = ? WHERE id = ?", (new_name.strip(), folder_id))
        conn.commit()
        logger.info(f"Renamed folder {folder_id} to '{new_name.strip()}'")
        return jsonify({"message": "Folder renamed successfully", "id": folder_id, "name": new_name.strip()}), 200
    except sqlite3.Error as e:
        logger.error(f"Error renaming folder {folder_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to rename folder"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chats/<int:chat_id>', methods=['PUT'])
def update_chat(chat_id):
    data = request.get_json()
    new_title = data.get('title')

    if not new_title or not new_title.strip():
        return jsonify({"error": "Chat title cannot be empty"}), 400

    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Chat not found"}), 404

        conn.execute("UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_title.strip(), chat_id))
        conn.commit()
        logger.info(f"Renamed chat {chat_id} to '{new_title.strip()}'")
        return jsonify({"message": "Chat renamed successfully", "id": chat_id, "title": new_title.strip()}), 200
    except sqlite3.Error as e:
        logger.error(f"Error renaming chat {chat_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to rename chat"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chat', methods=['POST'])
def chat_api(): # Renamed from chat to avoid conflict with function name chat
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_message_content = data.get('message')
    if not user_message_content:
        return jsonify({"error": "Message content is required"}), 400

    chat_id = data.get('chat_id')
    model_choice = data.get('model_type', 'gemma') # Default to gemma if not specified
    
    MAX_HISTORY_MESSAGES = 10 
    current_cancel_event = threading.Event()

    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed in chat_api at the beginning.")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        if chat_id:
            chat = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat:
                logger.info(f"Invalid chat_id {chat_id} provided. Treating as a new chat.")
                chat_id = None
                
        if not chat_id:
            new_chat_title = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}" 
            cursor = conn.execute("INSERT INTO chats (title) VALUES (?)", (new_chat_title,))
            conn.commit()
            chat_id = cursor.lastrowid
            logger.info(f"Created new chat with ID: {chat_id}, Title: {new_chat_title}")

        conn.execute(
            "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
            (chat_id, 'user', user_message_content)
        )
        conn.commit()
        logger.info(f"Stored user message for chat_id {chat_id}: '{user_message_content[:50]}...'")

        active_cancellation_events[chat_id] = current_cancel_event
        logger.info(f"Registered cancellation event for chat_id {chat_id}")

        history_context_for_llm = "No previous conversation history for this session."
        full_prompt_for_model = f"User: {user_message_content}"
        
        if chat_id: 
            actual_history_for_context = []
            history_cursor_for_prompt = conn.execute(
                "SELECT sender, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC LIMIT ?",
                (chat_id, MAX_HISTORY_MESSAGES)
            )
            fetched_history_rows = history_cursor_for_prompt.fetchall()

            if fetched_history_rows:
                history_rows_for_context = fetched_history_rows[:-1]
                if history_rows_for_context:
                    for row in history_rows_for_context:
                        sender_tag = "User" if row['sender'] == 'user' else "Assistant"
                        actual_history_for_context.append(f"{sender_tag}: {row['content']}")
                    history_context_for_llm = "\n".join(actual_history_for_context)
            
            if actual_history_for_context:
                full_prompt_for_model = f"{history_context_for_llm}\nUser: {user_message_content}"
            else:
                full_prompt_for_model = f"User: {user_message_content}"
            
            logger.info(f"Constructed prompt for chat_id {chat_id} (history rows for context: {len(actual_history_for_context)}). Prompt (first 100 chars): {full_prompt_for_model[:100]}...")

        response_generator = None
        if model_choice == 'gemma':
            logger.info(f"Using Gemma (local Llama) for chat_id {chat_id} with prompt: '{full_prompt_for_model[:100]}...'")
            response_generator = generate_local_rag_response(full_prompt_for_model, chat_id)
        elif model_choice == 'wikipedia':
            logger.info(f"Using Wikipedia + Gemini for chat_id {chat_id}. Original query: '{user_message_content[:100]}...'")
            wiki_summary = fetch_wikipedia_summary(user_message_content)

            if not wiki_summary or \
               wiki_summary.startswith("No Wikipedia article found") or \
               wiki_summary.startswith("Error:"):
                logger.warning(f"Wikipedia search for '{user_message_content[:50]}...' yielded: {wiki_summary}. Will stream this direct to user.")
                def direct_response_generator(message):
                    yield message
                response_generator = direct_response_generator(wiki_summary or "Sorry, I could not retrieve information from Wikipedia for your query.")
            else:
                logger.info(f"Wikipedia summary found for '{user_message_content[:50]}...'. Crafting prompt for Gemini.")
                gemini_prompt_with_wiki = (
                    f"You are a helpful AI assistant. You have been provided with a summary from Wikipedia and the user's original query.\n"
                    f"Please use the Wikipedia summary to construct a comprehensive and well-formatted answer to the user's original query.\n"
                    f"If conversation history is also provided, use it for additional context.\n\n"
                    f"Wikipedia Summary:\n---\n{wiki_summary}\n---\n\n"
                    f"User's Original Query:\n---\n{user_message_content}\n---\n\n"
                    f"Conversation History (if relevant):\n---\n{history_context_for_llm}\n---\n\n"
                    f"Please now answer the user's original query based all this information. Provide a direct answer."
                )
                logger.info(f"Sending combined Wikipedia/Query to Gemini for chat_id {chat_id}: '{gemini_prompt_with_wiki[:200]}...'" )
                response_generator = generate_gemini_response(gemini_prompt_with_wiki, chat_id)
        elif model_choice == 'gemini':
            logger.info(f"Using Gemini API directly for chat_id {chat_id} with prompt: '{full_prompt_for_model[:100]}...'" )
            response_generator = generate_gemini_response(full_prompt_for_model, chat_id)
        else:
            logger.warning(f"Invalid model choice '{model_choice}' for chat_id {chat_id}")
            active_cancellation_events.pop(chat_id, None)
            return jsonify({"error": "Invalid model choice"}), 400

        if response_generator is None:
            logger.error(f"Response generator was None for model {model_choice}, chat_id {chat_id}")
            active_cancellation_events.pop(chat_id, None)
            return jsonify({"error": "Failed to get response from model"}), 500
        
        full_bot_response_parts = []
        def stream_and_collect():
            nonlocal full_bot_response_parts
            db_conn_for_stream = get_db_connection()
            if not db_conn_for_stream:
                logger.error(f"Failed to get DB connection for stream_and_collect (chat_id: {chat_id}). Bot response will not be saved.")
                yield "Error: Could not save conversation history due to a database issue."
                return

            try:
                for chunk in response_generator:
                    full_bot_response_parts.append(chunk)
                    yield chunk 
            except Exception as e_stream:
                logger.error(f"Error during response streaming for chat_id {chat_id}: {e_stream}", exc_info=True)
                yield "Error: An error occurred while streaming the response."
            finally:
                logger.info(f"Stream to client finished for chat_id {chat_id}. Attempting to save full bot response to DB.")
                bot_response_content = "".join(full_bot_response_parts).strip()
                
                removed_event = active_cancellation_events.pop(chat_id, None)
                if removed_event:
                    logger.info(f"Cleaned up cancellation event for chat_id {chat_id}.")
                else:
                    logger.warning(f"No cancellation event found to clean up for chat_id {chat_id} during stream_and_collect finally block.")
                
                if bot_response_content:
                    try:
                        db_conn_for_stream.execute(
                            "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
                            (chat_id, 'bot', bot_response_content)
                        )
                        snippet = (user_message_content[:70] + "...") if len(user_message_content) > 70 else user_message_content
                        db_conn_for_stream.execute(
                            "UPDATE chats SET last_snippet = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (snippet, chat_id)
                        )
                        db_conn_for_stream.commit()
                        logger.info(f"Bot response and chat snippet for chat_id {chat_id} saved to DB (snippet: '{snippet}').")
                    except sqlite3.Error as e_db_stream:
                        logger.error(f"Database error while saving streamed bot response for chat_id {chat_id}: {e_db_stream}", exc_info=True)
                    except Exception as e_save_stream:
                        logger.error(f"Unexpected error while saving streamed bot response for chat_id {chat_id}: {e_save_stream}", exc_info=True)
                    finally:
                        if db_conn_for_stream:
                            db_conn_for_stream.close()
                            logger.debug(f"Closed DB connection for stream_and_collect (chat_id: {chat_id}).")
                else:
                    logger.info(f"No bot response content generated or collected for chat_id {chat_id}, not saving to DB.")
        
        return Response(stream_and_collect(), mimetype='text/plain') 

    except sqlite3.Error as e_sqlite:
        logger.error(f"Database error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e_sqlite}", exc_info=True)
        if 'chat_id' in locals() and chat_id is not None:
            active_cancellation_events.pop(chat_id, None)
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e_main:
        logger.error(f"Unexpected error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e_main}", exc_info=True)
        if 'chat_id' in locals() and chat_id is not None:
            active_cancellation_events.pop(chat_id, None)
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn:
            conn.close()
            logger.debug(f"Closed main DB connection for /api/chat request (chat_id: {chat_id if 'chat_id' in locals() else 'unknown'}).")

@app.route('/api/cancel_stream/<int:chat_id>', methods=['POST'])
def cancel_stream(chat_id):
    logger.info(f"Received cancellation request for chat_id: {chat_id}")
    event = active_cancellation_events.get(chat_id)
    if event:
        event.set() # Signal the event
        logger.info(f"Cancellation event set for chat_id: {chat_id}")
        return jsonify({"message": f"Cancellation signal sent for chat_id {chat_id}."}), 200
    else:
        logger.warning(f"No active stream found to cancel for chat_id: {chat_id}")
        return jsonify({"message": f"No active stream found for chat_id {chat_id} to cancel."}), 404

@app.route('/api/folders/<int:folder_id>', methods=['DELETE'])
def delete_folder(folder_id):
    conn = get_db_connection()
    try:
        # Check if folder exists
        cursor = conn.execute("SELECT id FROM folders WHERE id = ?", (folder_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Folder not found"}), 404

        # What to do with chats in the folder? For now, let's make them uncategorized.
        # Alternatively, you could prevent deletion if not empty, or delete chats too (cascade).
        # For simplicity: move chats to uncategorized.
        conn.execute("UPDATE chats SET folder_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE folder_id = ?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        conn.commit()
        logger.info(f"Deleted folder {folder_id} and moved its chats to uncategorized.")
        return jsonify({"message": "Folder deleted successfully"}), 200
    except sqlite3.Error as e:
        logger.error(f"Error deleting folder {folder_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete folder"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chats/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    conn = get_db_connection()
    try:
        # Check if chat exists
        cursor = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Chat not found"}), 404

        # Messages will be deleted due to CASCADE constraint on messages.chat_id FOREIGN KEY
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
        logger.info(f"Deleted chat {chat_id}.")
        return jsonify({"message": "Chat deleted successfully"}), 200
    except sqlite3.Error as e:
        logger.error(f"Error deleting chat {chat_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete chat"}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Initialize systems when running the script directly
    init_db()  # Initialize the database schema
    initialize_systems()  # Ensure systems are initialized
    app.run(debug=AppConfig.FLASK_DEBUG, host=AppConfig.FLASK_HOST, port=AppConfig.FLASK_PORT) 
