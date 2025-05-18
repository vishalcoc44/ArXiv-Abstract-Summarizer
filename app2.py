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
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = "gemini-2.0-flash" # Updated model
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    # Construct the API URL (ensure GEMINI_API_KEY is loaded before this class is defined if used directly here)
    # Alternatively, construct it in the function or ensure AppConfig.GEMINI_API_KEY is checked for None before use.
    
    # Model Paths & Parameters
    LOCAL_MODEL_PATH = "gemma_1b_finetuned_q4_0.gguf"
    # Path to the pre-built FAISS index directory (created by create_lc_faiss_index.py)
    FAISS_INDEX_PATH = "langchain_faiss_store_optimized" 
    
    # RAG Configuration
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    N_RETRIEVED_DOCS = 3
    LLAMA_MAX_TOKENS = 1024 # Example, adjust as needed
    LLAMA_TEMPERATURE = 0.3
    LLAMA_TOP_P = 0.9
    
    # Llama.cpp model parameters
    LLAMA_N_CTX = 4096
    LLAMA_N_GPU_LAYERS = 0 
    LLAMA_VERBOSE = False

    # Flask App settings
    FLASK_DEBUG = True
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5001
    DATABASE_URL = 'chat_app.db'

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

# --- Cleanup function for Llama model ---
def cleanup_local_llm():
    global local_llm
    if local_llm is not None:
        logger.info("Cleaning up local Llama model...")
        try:
            # Accessing internal _model and _ctx might be needed if no public close() method exists
            # However, llama-cpp-python's Llama object should handle this in its __del__ or a close method.
            # For now, let's assume __del__ should work if the object is valid.
            # Setting to None will help trigger garbage collection if it hasn't happened.
            # A more explicit model.close() or model.free() would be better if available.
            # The Llama object itself has a context manager, so if it were used like:
            # with Llama(...) as llm: # it would auto-cleanup.
            # Since it's global, we do this.
            if hasattr(local_llm, 'close'): # Check if a close method exists (newer versions might)
                local_llm.close()
            elif hasattr(local_llm, '_model') and local_llm._model is not None: # Attempt more direct cleanup if no close()
                 logger.info("Attempting to free model via internal attributes as no close() method found.")
                 # This is risky and depends on llama-cpp-python internals if they haven't exposed a clean .close()
                 # Forcing __del__ by removing reference
                 del local_llm
                 local_llm = None
            else:
                 del local_llm # Fallback to hoping __del__ handles it
                 local_llm = None
            logger.info("Local Llama model cleanup attempt finished.")
        except Exception as e:
            logger.error(f"Error during local_llm cleanup: {e}", exc_info=True)

# Register the cleanup function to be called at exit
atexit.register(cleanup_local_llm)

# --- System Initialization Status Flags ---
systems_status = {
    "local_llm_loaded": False,
    "faiss_index_loaded": False,
    "gemini_model_configured": bool(AppConfig.GEMINI_API_KEY) # True if API key exists for REST method
}

# --- Helper Functions ---
def initialize_systems():
    global local_llm, faiss_index, embeddings_model # removed gemini_model from globals for SDK
    logger.info("Initializing systems...")

    # 1. Initialize Llama.cpp model
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

    logger.info(f"Received in generate_local_rag_response: {user_query_with_history}")

    lines = user_query_with_history.strip().split('\\n')
    actual_user_question = lines[-1] if lines else ""
    conversation_log_for_prompt = "\\n".join(lines[:-1]) if len(lines) > 1 else "No previous conversation."

    # --- Phase 1: General RAG (FAISS Document Retrieval based on user query) ---
    general_retrieved_documents_from_faiss = [] # Stores Langchain Document objects
    if systems_status["faiss_index_loaded"] and faiss_index and actual_user_question:
        try:
            logger.info(f"Performing general FAISS similarity search for: '{actual_user_question[:100]}...'")
            retrieved_documents = faiss_index.similarity_search(actual_user_question, k=AppConfig.N_RETRIEVED_DOCS + 2) 
            if retrieved_documents:
                seen_titles = set()
                temp_unique_docs = []
                for doc in retrieved_documents:
                    title = doc.metadata.get('title', '').strip()
                    normalized_title = title.lower()
                    if title and normalized_title not in seen_titles:
                        temp_unique_docs.append(doc)
                        seen_titles.add(normalized_title)
                    elif not title:
                        temp_unique_docs.append(doc)
                general_retrieved_documents_from_faiss = temp_unique_docs # These are already somewhat deduplicated by title
                logger.info(f"General FAISS search initially retrieved {len(general_retrieved_documents_from_faiss)} documents (after basic title dedupe).")
            else:
                logger.info("No documents retrieved from general FAISS search for the query.")
        except Exception as e_faiss:
            logger.error(f"Error during general FAISS retrieval: {e_faiss}", exc_info=True)
    else:
        logger.info("FAISS index not loaded or query is empty. Skipping general RAG retrieval.")

    query_lower = actual_user_question.lower()
    is_paper_query = any(keyword in query_lower for keyword in ["paper", "suggest", "recommend", "article", "publication", "study", "research"])

    # --- Phase 1: Gemini Paper Candidate Generation (Task 2 - Part 1) ---
    unique_gemini_paper_suggestions = [] 
    processed_titles_for_llm_instruction = "" # New variable for LLM instruction list
    contextual_info_for_llm_about_papers = "" # New variable for context about papers for LLM

    if is_paper_query and systems_status["gemini_model_configured"] and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
        logger.info(f"Paper query detected. Attempting to get suggestions from Gemini for: '{actual_user_question[:100]}...'")
        try:
            gemini_papers_raw = get_gemini_paper_suggestions(actual_user_question)
            if gemini_papers_raw:
                seen_gemini_titles = set()
                for paper in gemini_papers_raw:
                    title = paper.get('title', '').strip()
                    normalized_title = title.lower()
                    if title and normalized_title not in seen_gemini_titles:
                        unique_gemini_paper_suggestions.append(paper) # Store the dicts
                        seen_gemini_titles.add(normalized_title)
                
                if unique_gemini_paper_suggestions:
                    logger.info(f"Gemini suggested {len(unique_gemini_paper_suggestions)} unique papers initially.")
                    # Create a simple list of titles (with arXiv if present) for the instruction
                    simple_title_list = []
                    for paper in unique_gemini_paper_suggestions:
                        title = paper.get('title', '')
                        arxiv_id = paper.get('arxiv_id')
                        full_title_str = f"{title}{f' (arXiv:{arxiv_id})' if arxiv_id else ''}"
                        simple_title_list.append(full_title_str.strip())
                    
                    # This list is for the LLM to iterate over in its instructions
                    processed_titles_for_llm_instruction = "\\n".join([f"{i+1}. {t}" for i, t in enumerate(simple_title_list)])
            else:
                logger.info("No paper suggestions returned from Gemini.")
        except Exception as e_gemini_sugg:
            logger.error(f"Error getting paper suggestions from Gemini: {e_gemini_sugg}", exc_info=True)
    elif is_paper_query:
        logger.info("Paper query, but Gemini is not configured. Skipping Gemini suggestions.")

    # --- Phase 2: Targeted RAG (FAISS Retrieval based on Gemini Suggestions) ---
    targeted_faiss_documents = []
    if systems_status["faiss_index_loaded"] and faiss_index and unique_gemini_paper_suggestions:
        logger.info(f"Performing targeted FAISS search for {len(unique_gemini_paper_suggestions)} Gemini suggestions.")
        seen_targeted_faiss_titles_normalized = set() # Deduplicate titles found via this targeted search
        for paper_suggestion in unique_gemini_paper_suggestions:
            title_to_search = paper_suggestion.get("title")
            if title_to_search:
                try:
                    docs = faiss_index.similarity_search(title_to_search, k=1) # Fetch top 1 for precision
                    if docs:
                        # Check if this doc is distinct enough before adding
                        doc_from_targeted_search = docs[0]
                        doc_title = doc_from_targeted_search.metadata.get("title", "").strip()
                        normalized_doc_title = doc_title.lower()

                        if normalized_doc_title and normalized_doc_title not in seen_targeted_faiss_titles_normalized:
                            # Further check: is the found FAISS title semantically similar to the Gemini title?
                            if are_texts_semantically_similar(title_to_search, doc_title, threshold=0.85): # Threshold for title matching
                                targeted_faiss_documents.append(doc_from_targeted_search)
                                seen_targeted_faiss_titles_normalized.add(normalized_doc_title)
                                logger.debug(f"Targeted FAISS found relevant doc for '{title_to_search}': '{doc_title}'")
                            else:
                                logger.debug(f"Targeted FAISS found doc '{doc_title}' for '{title_to_search}', but titles not semantically similar enough.")
                        elif doc_title:
                             logger.debug(f"Targeted FAISS found doc '{doc_title}' for '{title_to_search}', but it was a title-duplicate of another targeted result.")
                except Exception as e_targeted_faiss:
                    logger.error(f"Error during targeted FAISS search for title '{title_to_search}': {e_targeted_faiss}", exc_info=True)
        logger.info(f"Retrieved {len(targeted_faiss_documents)} documents from targeted FAISS search.")

    # --- Combine and Deduplicate RAG Context (Task 1 Enhancement) ---
    final_rag_documents_for_prompt = []
    final_seen_content_snippets_for_semantic_check = [] # Store snippets of content for semantic dedupe
    final_seen_titles_normalized = set() # Store normalized titles for title-based dedupe

    # Priority to targeted FAISS documents (these are from Gemini's suggestions)
    for doc in targeted_faiss_documents:
        if len(final_rag_documents_for_prompt) >= AppConfig.N_RETRIEVED_DOCS:
            break
        title = doc.metadata.get("title", "").strip()
        normalized_title = title.lower()
        page_content_str = str(doc.page_content if doc.page_content is not None else "")
        content_snippet_for_check = page_content_str[:300] # Use a larger snippet for semantic check

        is_semantically_dupe = False
        for existing_snippet in final_seen_content_snippets_for_semantic_check:
            if are_texts_semantically_similar(content_snippet_for_check, existing_snippet, threshold=0.90): # Stricter for pre-LLM
                is_semantically_dupe = True
                logger.info(f"Skipping (semantic duplicate) targeted FAISS doc: '{title}'")
                break
                        
        if normalized_title not in final_seen_titles_normalized and not is_semantically_dupe:
            final_rag_documents_for_prompt.append(doc)
            final_seen_titles_normalized.add(normalized_title)
            final_seen_content_snippets_for_semantic_check.append(content_snippet_for_check)
        elif title and normalized_title in final_seen_titles_normalized:
            logger.info(f"Skipping (title duplicate) targeted FAISS doc: '{title}'")

    # Add general RAG documents if space allows and they are distinct
    remaining_slots = AppConfig.N_RETRIEVED_DOCS - len(final_rag_documents_for_prompt)
    if remaining_slots > 0:
        for doc in general_retrieved_documents_from_faiss:
            if len(final_rag_documents_for_prompt) >= AppConfig.N_RETRIEVED_DOCS:
                break
            title = doc.metadata.get("title", "").strip()
            normalized_title = title.lower()
            page_content_str = str(doc.page_content if doc.page_content is not None else "")
            content_snippet_for_check = page_content_str[:300]

            is_semantically_dupe = False
            for existing_snippet in final_seen_content_snippets_for_semantic_check:
                if are_texts_semantically_similar(content_snippet_for_check, existing_snippet, threshold=0.90):
                    is_semantically_dupe = True
                    logger.info(f"Skipping (semantic duplicate) general FAISS doc: '{title}'")
                    break
                            
            if normalized_title not in final_seen_titles_normalized and not is_semantically_dupe:
                final_rag_documents_for_prompt.append(doc)
                final_seen_titles_normalized.add(normalized_title)
                final_seen_content_snippets_for_semantic_check.append(content_snippet_for_check)
            elif title and normalized_title in final_seen_titles_normalized:
                 logger.info(f"Skipping (title duplicate) general FAISS doc: '{title}'")

    # --- Construct retrieved_context_str_for_prompt from final_rag_documents_for_prompt ---
    retrieved_context_str_for_prompt = ""
    retrieved_docs_for_log = [] # <--- ADD THIS LINE EXACTLY HERE

    # Build the contextual_info_for_llm_about_papers if Gemini suggestions exist
    if is_paper_query and unique_gemini_paper_suggestions and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
        contextual_info_for_llm_about_papers = f"CONTEXT FOR PROVIDED PAPER TITLES:\\n{processed_titles_for_llm_instruction}\\n\\n"
        logger.info("Gemini suggestions exist and will be included in the context.")
    else:
        contextual_info_for_llm_about_papers = ""
        logger.info("No Gemini suggestions exist or they are being skipped.")

    # Build general RAG context if not a paper query with specific context, or if paper query but no gemini suggestions
    if not (is_paper_query and contextual_info_for_llm_about_papers) and final_rag_documents_for_prompt:
        context_parts = []
        # Current logic limits to 1 RAG doc for debugging, let's use AppConfig.N_RETRIEVED_DOCS
        # for doc_to_add in final_rag_documents_for_prompt[:1]: # Original DEBUG limit
        for i, doc_to_add in enumerate(final_rag_documents_for_prompt[:AppConfig.N_RETRIEVED_DOCS]):
            content = str(doc_to_add.page_content).strip() if doc_to_add.page_content is not None else ""
            title = doc_to_add.metadata.get('title', f'Retrieved Document {i+1}')
            # Truncate content to avoid overly long prompts
            max_content_len = 700 # Characters per RAG document in prompt
            context_parts.append(f"Retrieved Document {i+1} (Title: {title}):\\n{content[:max_content_len]}{'...' if len(content) > max_content_len else ''}")
            retrieved_docs_for_log.append({"title": title, "content_preview": content[:100] + "..."})
            
        if context_parts:
            retrieved_context_str_for_prompt = "\\n".join(context_parts)
            logger.info(f"General RAG context prepared with {len(retrieved_docs_for_log)} documents for LLM.")
        else:
            logger.info("No general RAG documents will be added to the LLM prompt.")

    # --- Stop sequences ---
    # These are common phrases the model might try to output that we want to stop.
    # This list can be expanded based on observed model behavior.
    stop_sequences_general = [
        "User:", "User Query:", "USER:", "ASSISTANT:", "Assistant:", "System:", 
        "\\nUser:", "\\nUSER:", "\\nASSISTANT:", "\\nAssistant:", "\\nSystem:",
        "Context:", "CONTEXT:", "Answer:", "ANSWER:", "Note:", "Response:",
        "In this response,", "In summary,", "To summarize,", "Let's expand on",
        "The user is asking", "The user wants to know",
        "This text follows", "The following is a", "This is a text about",
        "My goal is to", "I am an AI assistant", "As an AI",
        "\\n\\nHuman:", "<|endoftext|>", "<|im_end|>", "\\u202f",
        "STOP_ASSISTANT_PRIMING_PHRASE:" # Changed to avoid conflict with actual priming, just in case
    ]
    
    generation_params_general = {
        "temperature": 0.2, 
        "top_p": 0.8,       
        "top_k": 40,        
        "repeat_penalty": 1.15, 
        "max_tokens": 500,   # Temporarily increased for debugging empty response
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
                f"   - Your output for this paper MUST be formatted STRICTLY as (example):\n"
                f"     1. Title: [Exact Paper Title from 'PAPER TITLES TO PROCESS' (including arXiv ID if it was there)]\n"
                f"        Abstract (from local database): [The relevant retrieved abstract/content snippet from the DOCUMENT DATABASE]\n"
                f"4. If NO strong match is found in the DOCUMENT DATABASE context, OR if a match is found but it has no usable abstract:\n"
                f"   - You MUST generate a concise, plausible, and UNIQUE abstract (1-3 sentences) for the paper based SOLELY on its title and your general knowledge. Do NOT use placeholder text. Do NOT repeat abstracts from other papers.\n"
                f"   - Your output for this paper MUST be formatted STRICTLY as (example):\n"
                f"     2. Title: [Exact Paper Title from 'PAPER TITLES TO PROCESS' (including arXiv ID if it was there)]\n"
                f"        Abstract (generated): [Your UNIQUE generated abstract based on the title]\n"
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
                f"Your task is to suggest up to 3-5 relevant academic paper titles based on the user's query and your general knowledge. "
                f"For each paper you suggest, provide its title and then generate a brief, plausible-sounding abstract (1-2 sentences).\\n\\n"
                
                f"If general 'Retrieved Document' context is available in this prompt, you can check if any of your suggested titles coincidentally match any retrieved document titles. If so, you may use that content for the abstract, noting it as '(potentially related to retrieved context)'. Otherwise, generate the abstract from your knowledge.\\n\\n"

                f"CRITICAL OUTPUT FORMATTING RULES (Adhere strictly to ensure proper display):\\n"
                f"1. Your response MUST begin *EXACTLY* with '1. Title: ' (for the first paper) and contain NO text or preamble before it.\\n"
                f"2. For EACH paper you suggest:\\n"
                f"   a. Start the line with the paper number, a period, a single space, and then 'Title: ' followed by the paper title. (e.g., '1. Title: Example Paper Title')\\n"
                f"   b. The paper title itself should NOT contain any literal '\\\\n' characters. If a title is long, let it wrap naturally.\\n"
                f"   c. On the VERY NEXT LINE (achieved by outputting a single newline character after the title line), indent with EXACTLY three spaces, then write 'Abstract: ' followed by the concise (1-2 sentences) abstract.\\n"
                f"   d. The abstract itself should NOT contain any literal '\\\\n' characters within a sentence. If the abstract is long, let it wrap naturally. Use a single newline character ONLY to end the abstract line.\\n"
                f"   e. After the abstract line for one paper, the next paper's 'X. Title:' line MUST start immediately on the next line. Do NOT insert blank lines between papers.\\n"
                f"   f. Do NOT use any markdown list markers like '*', '-', or similar characters at the start of any line related to the paper list.\\n\\n"
                
                f"EXAMPLE OF CORRECT FORMAT FOR TWO PAPERS (this is how your output should look):\\n"
                f"1. Title: Deep Learning for Molecules and Materials\\n"
                f"   Abstract: This paper reviews the application of deep learning in chemistry and materials science, covering areas like property prediction and generative models.\\n"
                f"2. Title: The Role of Catalysis in Green Chemistry\\n"
                f"   Abstract: Explores various catalytic methods that contribute to environmentally friendly chemical processes, reducing waste and energy consumption.\\n\\n"

                f"FURTHER OVERALL RESPONSE RULES:\\n"
                f"- Your response MUST ONLY contain the numbered list of your suggested titles and their abstracts, following the CRITICAL OUTPUT FORMATTING RULES. Nothing else before or after this list.\\n"
                f"- If you cannot confidently generate a meaningful abstract for a title you suggest, use 'Abstract: General suggestion based on topic; specific abstract not available.' for that paper's abstract line.\\n"
                f"- If you cannot find any relevant papers to suggest from your knowledge, respond ONLY with the exact phrase: 'I am unable to suggest specific papers from my current knowledge based on your query.' (This exact phrase, and nothing else).\\n"
                f"- Do NOT include conversational introductions, closings (like 'I hope this helps!'), meta-commentary, or self-references (unless using the specific inability phrase above)."
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
    if conversation_log_for_prompt and conversation_log_for_prompt != "No previous conversation.":
        prompt_for_gemma += f"CONVERSATION LOG:\\n{conversation_log_for_prompt}\\n\\n"
    
    if not DEBUG_SKIP_AUGMENTATION:
        if not DEBUG_SKIP_RAG_CONTEXT: # Check RAG skip flag
            if is_paper_query and contextual_info_for_llm_about_papers and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
                prompt_for_gemma += f"{contextual_info_for_llm_about_papers}\\n\\n" # Add the specific paper context
            elif retrieved_context_str_for_prompt: # General RAG context
                prompt_for_gemma += f"ADDITIONAL CONTEXT FROM DOCUMENT DATABASE (curated RAG results):\\n{retrieved_context_str_for_prompt}\\n\\n"
            else:
                logger.info("DEBUG: No RAG context (neither specific paper context nor general) to add.")
        else:
            logger.warning("DEBUG_SKIP_RAG_CONTEXT is TRUE: Skipping RAG context in Gemma prompt.")

        # Gemini suggestions are now primarily handled by processed_titles_for_llm_instruction within paper_specific_instruction
        # No need to add a separate block for gemini_suggestions_list_for_instruction here
        if is_paper_query and not DEBUG_SKIP_GEMINI_SUGGESTIONS:
            if processed_titles_for_llm_instruction:
                logger.info("DEBUG: Gemini paper titles are included in 'PAPER TITLES TO PROCESS' within the instruction.")
            else:
                logger.info("DEBUG: Paper query, but no Gemini paper titles were prepared for the instruction.")
        elif is_paper_query and DEBUG_SKIP_GEMINI_SUGGESTIONS:
             logger.warning("DEBUG_SKIP_GEMINI_SUGGESTIONS is TRUE: Skipping Gemini suggestions for instruction.")
    else:
        logger.warning("DEBUG_SKIP_AUGMENTATION is TRUE: Skipping ALL augmentation in Gemma prompt.")
    
    prompt_for_gemma += f"USER QUERY: {actual_user_question}\\n\\nASSISTANT'S FACTUAL ANSWER:"

    logger.info(f"Final Prompt for Gemma (first 300 chars): {prompt_for_gemma[:300]}...")
    logger.info(f"Full Prompt for Gemma:\\n{prompt_for_gemma}") # Uncommented for debugging
    if retrieved_docs_for_log:
        logger.info(f"RAG documents added to prompt: {json.dumps(retrieved_docs_for_log, indent=2)}")
    if unique_gemini_paper_suggestions: # Log what Gemini originally suggested
        logger.info(f"Gemini initially suggested papers: {json.dumps(unique_gemini_paper_suggestions, indent=2)}")
    logger.info(f"Generation parameters for Gemma: {generation_params_general}")

    try:
        stream = local_llm.create_chat_completion( # Corrected method call
            messages=[{"role": "user", "content": prompt_for_gemma}],
            **generation_params_general,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta_content:
                yield delta_content
                full_response += delta_content
        
        logger.info(f"Gemma full response (simplified): {full_response.strip()}")

    except Exception as e:
        logger.error(f"Error during local LLM generation (simplified path): {e}", exc_info=True)
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
    
    # --- Configuration for Context History ---
    MAX_HISTORY_MESSAGES = 10 # Number of recent messages to include in context
    # ---

    # Create a cancellation event for this request
    current_cancel_event = threading.Event()
    # Ensure chat_id is determined before this line if it can be new

    conn = get_db_connection()
    if not conn:
        # This scenario should ideally be handled more gracefully, 
        # perhaps with a retry or a clearer error message to the user.
        logger.error("Database connection failed in chat_api at the beginning.")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        if chat_id:
            chat = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat:
                logger.info(f"Invalid chat_id {chat_id} provided. Treating as a new chat.")
                chat_id = None # If chat_id is invalid, treat as a new chat
                
        if not chat_id:
            # Create new chat. Consider making title more dynamic or based on first message later.
            new_chat_title = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}" 
            cursor = conn.execute("INSERT INTO chats (title) VALUES (?)", (new_chat_title,))
            conn.commit()
            chat_id = cursor.lastrowid
            logger.info(f"Created new chat with ID: {chat_id}, Title: {new_chat_title}")

        # Store user message
        conn.execute(
            "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
            (chat_id, 'user', user_message_content)
        )
        conn.commit()
        logger.info(f"Stored user message for chat_id {chat_id}: '{user_message_content[:50]}...'")

        # Register cancellation event for this chat_id *after* chat_id is confirmed/created
        active_cancellation_events[chat_id] = current_cancel_event
        logger.info(f"Registered cancellation event for chat_id {chat_id}")

        # --- Retrieve and Format Chat History ---
        full_prompt_for_model = f"User: {user_message_content}" # Default to current message if no history
        
        if chat_id: 
            prompt_history_parts = []
            # Fetch messages prior to the one just inserted for context
            # We order by timestamp ASC to build the history chronologically
            query = (
                "SELECT sender, content FROM messages "
                "WHERE chat_id = ? AND timestamp < ("
                "SELECT MAX(timestamp) FROM messages "
                "WHERE chat_id = ? AND sender = 'user'"
                ") ORDER BY timestamp ASC LIMIT ?"
            )
            history_cursor_for_prompt = conn.execute(
                query,
                (chat_id, chat_id, MAX_HISTORY_MESSAGES)
            )
            fetched_history = history_cursor_for_prompt.fetchall()

            if fetched_history:
                for row in fetched_history:
                    sender_tag = "User" if row['sender'] == 'user' else "Assistant"
                    prompt_history_parts.append(f"{sender_tag}: {row['content']}")
                
                history_string = "\n".join(prompt_history_parts) # Use literal newline for LLM
                full_prompt_for_model = f"{history_string}\nUser: {user_message_content}"
            
            logger.info(f"Constructed prompt for chat_id {chat_id} (history rows: {len(fetched_history)}). Prompt (first 100 chars): {full_prompt_for_model[:100]}...")
        # --- End Retrieve and Format Chat History ---

        response_generator = None
        if model_choice == 'gemma':
            logger.info(f"Using Gemma (local Llama) for chat_id {chat_id} with prompt: '{full_prompt_for_model[:100]}...'")
            response_generator = generate_local_rag_response(full_prompt_for_model, chat_id)
        elif model_choice == 'gemini':
            logger.info(f"Using Gemini API for chat_id {chat_id} with prompt: '{full_prompt_for_model[:100]}...'")
            response_generator = generate_gemini_response(full_prompt_for_model, chat_id)
        else:
            logger.warning(f"Invalid model choice '{model_choice}' for chat_id {chat_id}")
            active_cancellation_events.pop(chat_id, None) # Clean up event if returning early
            return jsonify({"error": "Invalid model choice"}), 400 # Return early

        if response_generator is None:
            logger.error(f"Response generator was None for model {model_choice}, chat_id {chat_id}")
            active_cancellation_events.pop(chat_id, None) # Clean up event
            return jsonify({"error": "Failed to get response from model"}), 500 # Return early
        
        full_bot_response_parts = []
        def stream_and_collect():
            nonlocal full_bot_response_parts
            # This db_conn_for_stream is crucial because the main `conn` will be closed 
            # by the time this generator's finally block might execute in some edge cases 
            # or if the main request handling finishes before the stream does.
            db_conn_for_stream = get_db_connection()
            if not db_conn_for_stream:
                logger.error(f"Failed to get DB connection for stream_and_collect (chat_id: {chat_id}). Bot response will not be saved.")
                # Yield an error message or handle as appropriate
                yield "Error: Could not save conversation history due to a database issue."
                return

            try:
                for chunk in response_generator:
                    full_bot_response_parts.append(chunk)
                    yield chunk 
            except Exception as e_stream:
                logger.error(f"Error during response streaming for chat_id {chat_id}: {e_stream}", exc_info=True)
                yield "Error: An error occurred while streaming the response." # Let client know
            finally:
                logger.info(f"Stream to client finished for chat_id {chat_id}. Attempting to save full bot response to DB.")
                bot_response_content = "".join(full_bot_response_parts).strip()
                
                # Clean up cancellation event for this chat_id
                removed_event = active_cancellation_events.pop(chat_id, None)
                if removed_event:
                    logger.info(f"Cleaned up cancellation event for chat_id {chat_id}.")
                else:
                    logger.warning(f"No cancellation event found to clean up for chat_id {chat_id} during stream_and_collect finally block.")
                
                if bot_response_content: # Only save if there's content
                    try:
                        # Store bot message
                        db_conn_for_stream.execute(
                            "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
                            (chat_id, 'bot', bot_response_content)
                        )
                        
                        # Update chat's last_snippet and updated_at
                        # Using the last user message as snippet, or first part of it.
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
        if 'chat_id' in locals() and chat_id is not None: # Ensure chat_id is defined
            active_cancellation_events.pop(chat_id, None) # Clean up on error
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e_main:
        logger.error(f"Unexpected error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e_main}", exc_info=True)
        if 'chat_id' in locals() and chat_id is not None: # Ensure chat_id is defined
            active_cancellation_events.pop(chat_id, None) # Clean up on error
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn: # This is the main connection for the request handling part
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
