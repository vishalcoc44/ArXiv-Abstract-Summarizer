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

colorama.init()

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
    
    # Llama.cpp model parameters
    LLAMA_N_CTX = 2048
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

# UNCOMMENTED generate_local_rag_response
def generate_local_rag_response(user_query, chat_id=None):
    if not systems_status["local_llm_loaded"]:
        logger.warning("Attempted to use local model for RAG, but it's not loaded.")
        yield "Local model is not loaded. Cannot generate RAG response."
        return
    if not systems_status["faiss_index_loaded"]:
        logger.warning("Attempted to use RAG, but FAISS index is not loaded.")
        yield "RAG components (FAISS index) are not available. Cannot generate RAG response."
        return

    try:
        logger.info(f"Searching FAISS index for: '{user_query}'")
        retrieved_docs = faiss_index.similarity_search(user_query, k=AppConfig.N_RETRIEVED_DOCS)
        
        context_for_llm = "No relevant context found."
        if not retrieved_docs:
            logger.info("No relevant documents found for RAG context.")
        else:
            context_for_llm = "\n".join([doc.page_content for doc in retrieved_docs])
            logger.info(f"Retrieved {len(retrieved_docs)} documents for RAG context.")
        
        prompt = f"""Based on the following context, please provide a comprehensive answer to the user's question. If the context does not contain the answer, state that the context is insufficient or you don't know based on the provided information.

Context:
{context_for_llm}

User Question: {user_query}

Assistant Answer:"""
        
        logger.debug(f"Prompt for local LLM (GGUF):\n{prompt}")

        output_stream = local_llm(
            prompt,
            max_tokens=512, 
            stop=["User Question:", "User:", "\n\n"],
            echo=False,
            stream=True  # Enable streaming for llama-cpp-python
        )
        
        logger.info("Local LLM (GGUF) stream started...")
        full_response_for_log = []
        for output_chunk in output_stream:
            # logging.debug(f"Local LLM chunk: {output_chunk}") # For debugging
            if 'choices' in output_chunk and len(output_chunk['choices']) > 0:
                chunk_text = output_chunk['choices'][0].get('text', '')
                if chunk_text:
                    # logger.info(f"Yielding local LLM chunk: {chunk_text}")
                    full_response_for_log.append(chunk_text)
                    yield chunk_text
        
        logger.info(f"Generated local RAG response (streamed): '{''.join(full_response_for_log)[:100]}...'")
        # No single return value needed here as we are yielding

    except Exception as e:
        logger.error(f"Error during RAG generation with local model: {e}", exc_info=True)
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

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        if chat_id:
            # Verify chat_id exists and belongs to user (simplified: just check existence)
            chat = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat:
                # If chat_id is provided but not found, create a new one.
                # This could be an alternative to erroring out, depending on desired UX.
                # For now, let's assume we'd rather create a new chat if a bogus ID is passed.
                # Or, more strictly, return an error:
                # conn.close()
                # return jsonify({"error": "Chat not found"}), 404
                # Decided: for robustness, if an invalid chat_id comes, make a new chat.
                chat_id = None 
                
        if not chat_id:
            # Create new chat
            cursor = conn.execute("INSERT INTO chats (user_id, folder_id) VALUES (?, ?)", (1, None)) # Assuming user_id 1, no folder
            conn.commit()
            chat_id = cursor.lastrowid
            logger.info(f"Created new chat with ID: {chat_id}")

        # Store user message
        conn.execute(
            "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
            (chat_id, 'user', user_message_content)
        )
        conn.commit()
        logger.info(f"Stored user message for chat_id {chat_id}")

        response_generator = None
        if model_choice == 'gemma':
            logger.info(f"Using Gemma (local Llama) for chat_id {chat_id}")
            response_generator = generate_local_rag_response(user_message_content, chat_id)
        elif model_choice == 'gemini':
            logger.info(f"Using Gemini API for chat_id {chat_id}")
            response_generator = generate_gemini_response(user_message_content, chat_id)
        else:
            # conn.close() # Already in finally
            return jsonify({"error": "Invalid model choice"}), 400

        if response_generator is None:
            # This case should ideally be handled by the model functions returning an error yield
            logger.error(f"Response generator was None for model {model_choice}, chat_id {chat_id}")
            # conn.close() # Already in finally
            return jsonify({"error": "Failed to get response from model"}), 500
        
        full_bot_response_parts = []
        def stream_and_collect():
            nonlocal full_bot_response_parts
            db_conn_for_stream = None # Use a separate connection for this generator's scope
            try:
                for chunk in response_generator:
                    full_bot_response_parts.append(chunk)
                    yield chunk 
            finally:
                # This block executes after the generator is exhausted or closed.
                logger.info(f"Stream to client finished for chat_id {chat_id}. Saving full bot response to DB.")
                bot_response_content = "".join(full_bot_response_parts)
                if bot_response_content: # Only save if there's content
                    try:
                        db_conn_for_stream = get_db_connection()
                        # Store bot message
                        db_conn_for_stream.execute(
                            "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
                            (chat_id, 'bot', bot_response_content)
                        )
                        
                        # Update chat's last_snippet and updated_at
                        # Use user message for snippet, as per original logic.
                        snippet = (user_message_content[:30] + "...") if len(user_message_content) > 30 else user_message_content
                        db_conn_for_stream.execute(
                            "UPDATE chats SET last_snippet = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (snippet, chat_id)
                        )
                        db_conn_for_stream.commit()
                        logger.info(f"Bot response and chat snippet for chat_id {chat_id} saved to DB.")
                    except sqlite3.Error as e_db:
                        logger.error(f"Database error while saving streamed bot response for chat_id {chat_id}: {e_db}", exc_info=True)
                    finally:
                        if db_conn_for_stream:
                            db_conn_for_stream.close()
                else:
                    logger.info(f"No bot response content generated for chat_id {chat_id}, not saving to DB.")
        
        # Stream to client. The stream_and_collect generator will handle DB ops in its finally block.
        return Response(stream_and_collect(), mimetype='text/plain') 

    except sqlite3.Error as e:
        logger.error(f"Database error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e}", exc_info=True)
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn:
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