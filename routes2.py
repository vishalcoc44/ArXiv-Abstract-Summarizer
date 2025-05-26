from flask import Flask, request, jsonify, render_template, Response
import sqlite3
import datetime
import logging
import threading # For current_cancel_event in chat_api

from config import AppConfig
from db import get_db_connection
from utils import fetch_wikipedia_summary
from models2 import (
    generate_local_rag_response,
    register_cancellation_event,
    unregister_cancellation_event,
    get_cancellation_event,
    active_cancellation_events # For direct access in cancel_stream if needed, or use getters
)

logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config.from_object(AppConfig)

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
    except sqlite3.IntegrityError:
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
    folder_id = data.get('folder_id')

    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO chats (title, folder_id) VALUES (?, ?)",
            (title, folder_id)
        )
        conn.commit()
        chat_id = cursor.lastrowid
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
    new_folder_id = data.get('folder_id')

    conn = get_db_connection()
    try:
        chat_exists_cursor = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        if not chat_exists_cursor.fetchone():
            return jsonify({"error": "Chat not found"}), 404

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
def chat_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_message_content = data.get('message')
    if not user_message_content:
        return jsonify({"error": "Message content is required"}), 400

    chat_id = data.get('chat_id')
    model_choice = data.get('model_type', 'gemma')
    
    MAX_HISTORY_MESSAGES = 10 
    current_cancel_event = threading.Event()

    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed in chat_api at the beginning.")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        if chat_id:
            chat_db_check = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat_db_check:
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

        register_cancellation_event(chat_id, current_cancel_event)
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
            
            logger.info(f"Constructed prompt for chat_id {chat_id} (history rows for context: {len(actual_history_for_context)}). Prompt (first 100 chars): {full_prompt_for_model[:100]}...")

        response_generator = None
        if model_choice == 'gemma':
            logger.info(f"Using Gemma (local Llama) for chat_id {chat_id} with prompt: '{full_prompt_for_model[:100]}...'")
            response_generator = generate_local_rag_response(full_prompt_for_model, chat_id)
        elif model_choice == 'wikipedia':
            logger.info(f"Using Wikipedia + Local LLM for chat_id {chat_id}. Original query: '{user_message_content[:100]}...'")
            wiki_summary = fetch_wikipedia_summary(user_message_content)
            if not wiki_summary or \
               wiki_summary.startswith("No Wikipedia article found") or \
               wiki_summary.startswith("Error:"):
                logger.warning(f"Wikipedia search for '{user_message_content[:50]}...' yielded: {wiki_summary}. Will stream this direct to user.")
                def direct_response_generator(message):
                    yield message
                response_generator = direct_response_generator(wiki_summary or "Sorry, I could not retrieve information from Wikipedia for your query.")
            else:
                logger.info(f"Wikipedia summary found for '{user_message_content[:50]}...'. Crafting prompt for Local LLM.")
                local_llm_prompt_with_wiki = (
                    f"You are a helpful AI assistant. You have been provided with a summary from Wikipedia and the user's original query.\n"
                    f"Please use the Wikipedia summary to construct a comprehensive and well-formatted answer to the user's original query.\n"
                    f"If conversation history is also provided, use it for additional context.\n\n"
                    f"Wikipedia Summary:\n---\n{wiki_summary}\n---\n\n"
                    f"User's Original Query:\n---\n{user_message_content}\n---\n\n"
                    f"Conversation History (if relevant):\n---\n{history_context_for_llm}\n---\n\n"
                    f"Please now answer the user's original query based all this information. Provide a direct answer."
                )
                logger.info(f"Sending combined Wikipedia/Query to Local LLM for chat_id {chat_id}: '{local_llm_prompt_with_wiki[:200]}...'" )
                response_generator = generate_local_rag_response(local_llm_prompt_with_wiki, chat_id)
        elif model_choice == 'gemini':
            logger.info(f"Model choice was 'gemini', but now redirecting to Local LLM for chat_id {chat_id} with prompt: '{full_prompt_for_model[:100]}...'" )
            response_generator = generate_local_rag_response(full_prompt_for_model, chat_id)
        else:
            logger.warning(f"Invalid model choice '{model_choice}' for chat_id {chat_id}")
            unregister_cancellation_event(chat_id)
            return jsonify({"error": "Invalid model choice"}), 400

        if response_generator is None:
            logger.error(f"Response generator was None for model {model_choice}, chat_id {chat_id}")
            unregister_cancellation_event(chat_id)
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
                removed_event = unregister_cancellation_event(chat_id)
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
        if 'chat_id' in locals() and chat_id is not None: unregister_cancellation_event(chat_id)
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e_main:
        logger.error(f"Unexpected error in /api/chat for chat_id {chat_id if 'chat_id' in locals() else 'unknown'}: {e_main}", exc_info=True)
        if 'chat_id' in locals() and chat_id is not None: unregister_cancellation_event(chat_id)
        return jsonify({"error": "An unexpected server error occurred."}), 500
    finally:
        if conn:
            conn.close()
            logger.debug(f"Closed main DB connection for /api/chat request (chat_id: {chat_id if 'chat_id' in locals() else 'unknown'}).")

@app.route('/api/cancel_stream/<int:chat_id>', methods=['POST'])
def cancel_stream(chat_id):
    logger.info(f"Received cancellation request for chat_id: {chat_id}")
    event = get_cancellation_event(chat_id)
    if event:
        event.set()
        logger.info(f"Cancellation event set for chat_id: {chat_id}")
        return jsonify({"message": f"Cancellation signal sent for chat_id {chat_id}."}), 200
    else:
        logger.warning(f"No active stream found to cancel for chat_id: {chat_id}")
        return jsonify({"message": f"No active stream found for chat_id {chat_id} to cancel."}), 404

@app.route('/api/chats/clear-all', methods=['DELETE'])
def clear_all_chats():
    conn = get_db_connection()
    try:
        # First delete all messages from all chats
        conn.execute("DELETE FROM messages")
        logger.info("Deleted all messages from all chats.")

        # Then delete all chats
        conn.execute("DELETE FROM chats")
        logger.info("Deleted all chats.")
        
        # Optionally, also delete all folders if that's desired
        # conn.execute("DELETE FROM folders")
        # logger.info("Deleted all folders.")

        conn.commit()
        return jsonify({"message": "All chats, associated messages have been cleared."}), 200
    except sqlite3.Error as e:
        logger.error(f"Error clearing all chats and messages: {e}", exc_info=True)
        conn.rollback() # Rollback in case of error
        return jsonify({"error": "Failed to clear all chats"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/folders/<int:folder_id>', methods=['DELETE'])
def delete_folder(folder_id):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT id FROM folders WHERE id = ?", (folder_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Folder not found"}), 404
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
        cursor = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Chat not found"}), 404
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