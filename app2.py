import os
import pickle
from flask import Flask, request, jsonify, render_template, Response
from llama_cpp import Llama
# import google.generativeai as genai # No longer used for Gemini if using REST API
import requests # For direct HTTP calls to Gemini API
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# import numpy as np # No longer needed here as we load pre-built FAISS
import logging # For improved logging
import sqlite3
import datetime
import colorama
import json # Add this import at the top of the file if not already there
import re # For parsing Gemini response
import atexit # For controlled cleanup

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

# --- Helper function to get paper suggestions from Gemini ---
def get_gemini_paper_suggestions(user_query_for_gemini: str) -> list[dict[str, str | None]]:
    if not systems_status["gemini_model_configured"] or not AppConfig.GEMINI_API_KEY:
        logger.warning("Attempted to get paper suggestions, but Gemini API is not configured.")
        return []

    gemini_prompt = (
        f"Based on current general knowledge, please list up to 5-7 highly relevant paper titles "
        f"for the query: '{user_query_for_gemini}'. "
        f"If readily available and confident, include their arXiv IDs in the format 'Title (arXiv:xxxx.xxxxx)'. "
        f"If no arXiv ID is known, just provide the title. "
        f"Provide each paper on a new line. If no specific papers can be suggested, respond with 'NO_SUGGESTIONS_FOUND'."
    )
    logger.info(f"Sending prompt to Gemini for paper suggestions: '{gemini_prompt[:150]}...'")

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
                if title:
                    suggested_papers_info.append({"title": title, "arxiv_id": arxiv_id})
            elif line: 
                suggested_papers_info.append({"title": line, "arxiv_id": None})

        logger.info(f"Parsed {len(suggested_papers_info)} paper suggestions from Gemini: {suggested_papers_info}")
        return suggested_papers_info

    except Exception as e:
        logger.error(f"Error calling or parsing Gemini for paper suggestions: {e}", exc_info=True)
        return []

# --- End Helper --- 

# UNCOMMENTED generate_local_rag_response
def generate_local_rag_response(user_query_with_history: str, chat_id=None):
    if not systems_status["local_llm_loaded"]:
        logger.warning("Attempted to use local model for RAG, but it's not loaded.")
        yield "Local model is not loaded. Cannot generate RAG response."
        return
    if not systems_status["faiss_index_loaded"]:
        logger.warning("Attempted to use RAG, but FAISS index is not loaded.")
        yield "RAG components (FAISS index) are not available. Cannot generate RAG response."
        return

    try:
        actual_user_question = user_query_with_history
        if isinstance(user_query_with_history, str) and "\nUser: " in user_query_with_history:
            parts = user_query_with_history.split("\nUser: ")
            if parts:
                actual_user_question = parts[-1]
        logger.info(f"Processing RAG for actual user question: '{actual_user_question[:100]}...'")

        is_paper_suggestion_query = False
        keywords_for_paper_suggestion = [
            "top papers", "list papers", "recommend papers", "find papers on", 
            "literature on", "key papers in", "recent papers in", "suggest papers",
            "summarize papers on", "what papers discuss"
        ]
        if any(keyword in actual_user_question.lower() for keyword in keywords_for_paper_suggestion):
            is_paper_suggestion_query = True
            logger.info(f"Query identified as potentially needing external paper suggestions: '{actual_user_question}'")

        gemini_suggested_papers = [] 
        if is_paper_suggestion_query:
            gemini_suggested_papers = get_gemini_paper_suggestions(actual_user_question)
        
        final_context_docs = []
        verified_gemini_papers_metadata = []

        if gemini_suggested_papers:
            logger.info(f"Checking local FAISS for {len(gemini_suggested_papers)} suggestions from Gemini...")
            for paper_info in gemini_suggested_papers:
                query_text = paper_info['title']
                retrieved_docs_for_title = faiss_index.similarity_search_with_score(query_text, k=1)
                
                if retrieved_docs_for_title:
                    doc, score = retrieved_docs_for_title[0]
                    SIMILARITY_THRESHOLD_L2 = 1.0 
                    if score <= SIMILARITY_THRESHOLD_L2:
                        logger.info(f"Found potential local match for Gemini suggestion '{paper_info['title']}' with score {score:.4f}. Adding to context.")
                        final_context_docs.append(doc) 
                        if doc.metadata: # Ensure metadata exists
                           verified_gemini_papers_metadata.append(doc.metadata)
                        if len(final_context_docs) >= AppConfig.N_RETRIEVED_DOCS * 2:
                            break 
                    else:
                        logger.info(f"Local match for '{paper_info['title']}' found but score {score:.4f} too low (threshold {SIMILARITY_THRESHOLD_L2}).")
            
            if final_context_docs:
                logger.info(f"Found {len(final_context_docs)} documents in local FAISS corresponding to Gemini's suggestions.")
            else:
                logger.info("None of Gemini's suggestions were found with high confidence in local FAISS.")

        if not is_paper_suggestion_query or not final_context_docs:
            if not final_context_docs: 
                logger.info(f"Performing general FAISS similarity search with query: '{actual_user_question[:100]}...'")
                retrieved_docs_general = faiss_index.similarity_search(actual_user_question, k=AppConfig.N_RETRIEVED_DOCS)
                if retrieved_docs_general:
                    final_context_docs.extend(retrieved_docs_general)

        context_for_llm = "No relevant context found in the local knowledge base for your query."
        if final_context_docs:
            unique_docs_by_content = {}
            for doc in final_context_docs:
                if doc.page_content not in unique_docs_by_content:
                    unique_docs_by_content[doc.page_content] = doc
            final_context_docs = list(unique_docs_by_content.values())            
            context_for_llm = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])

        prompt_introduction = ""
        if verified_gemini_papers_metadata:
            titles_found_locally = [meta.get('title', '[Unknown Title]') for meta in verified_gemini_papers_metadata if meta]
            if titles_found_locally:
                prompt_introduction = (
                    f"An external knowledge source suggested the following papers related to your query: '{actual_user_question}'. "
                    f"I have found information for these in my local knowledge base: \n- " + "\n- ".join(titles_found_locally) +
                    f"\n\nPlease use the context below, which contains details for these locally found papers, to answer the user's question."
                )
            else:
                prompt_introduction = "I consulted an external knowledge source, but couldn't verify its suggestions in my local data. Using general local knowledge instead:"
        else:
            prompt_introduction = (
                "You are a factual assistant. Your primary goal is to answer the user's question based *only* on the provided \"Context\" below.\n"
                "- If the context contains the answer, provide it clearly and concisely.\n"
                "- If the context does *not* contain the information to answer the question, you MUST explicitly state \"The provided context does not have the information to answer this question.\"\n"
                "- Do NOT use any external knowledge or make assumptions beyond the provided context.\n"
                "- If the user asks a general knowledge question that is not answerable from the context (e.g., 'What are the top 5 math papers?'), and no relevant context is provided, state that you cannot answer this without specific context or access to a real-time database."
            )

        final_llm_prompt = f"""{prompt_introduction}

Context:
{context_for_llm}

User Question: {actual_user_question}

Assistant Answer:"""
        
        logger.debug(f"Final prompt for local LLM (GGUF) (first 500 chars):\n{final_llm_prompt[:500]}...")

        output_stream = local_llm(
            final_llm_prompt,
            max_tokens=AppConfig.LLAMA_MAX_TOKENS if hasattr(AppConfig, 'LLAMA_MAX_TOKENS') else 512, 
            stop=["User Question:", "User:", "\n\n", "<|im_end|>"], 
            temperature=AppConfig.LLAMA_TEMPERATURE if hasattr(AppConfig, 'LLAMA_TEMPERATURE') else 0.3, 
            top_p=AppConfig.LLAMA_TOP_P if hasattr(AppConfig, 'LLAMA_TOP_P') else 0.9,
            echo=False,
            stream=True
        )
        
        full_response_for_log = []
        for output_chunk in output_stream:
            if 'choices' in output_chunk and len(output_chunk['choices']) > 0:
                chunk_text = output_chunk['choices'][0].get('text', '')
                if chunk_text:
                    full_response_for_log.append(chunk_text)
                    yield chunk_text
        
        logger.info(f"Generated local RAG response (streamed, hybrid approach if used): '{''.join(full_response_for_log)[:100]}...'")

    except Exception as e:
        logger.error(f"Error during RAG generation with local model (hybrid approach): {e}", exc_info=True)
        yield "Sorry, I encountered an error generating a response with the local model and RAG."
        return

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

        # --- Retrieve and Format Chat History ---
        full_prompt_for_model = f"User: {user_message_content}" # Default to current message if no history
        
        if chat_id: 
            prompt_history_parts = []
            # Fetch messages prior to the one just inserted for context
            # We order by timestamp ASC to build the history chronologically
            history_cursor_for_prompt = conn.execute(
                """SELECT sender, content FROM messages 
                   WHERE chat_id = ? AND timestamp < (SELECT MAX(timestamp) FROM messages WHERE chat_id = ? AND sender = 'user')
                   ORDER BY timestamp ASC 
                   LIMIT ?""", 
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
            return jsonify({"error": "Invalid model choice"}), 400 # Return early

        if response_generator is None:
            logger.error(f"Response generator was None for model {model_choice}, chat_id {chat_id}")
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
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e_main:
        logger.error(f"Unexpected error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e_main}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn: # This is the main connection for the request handling part
            conn.close()
            logger.debug(f"Closed main DB connection for /api/chat request (chat_id: {chat_id if 'chat_id' in locals() else 'unknown'}).")

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
    init_db() # Initialize the database schema
    initialize_systems() # Ensure systems are initialized
    app.run(debug=AppConfig.FLASK_DEBUG, host=AppConfig.FLASK_HOST, port=AppConfig.FLASK_PORT) 