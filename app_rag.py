import os
import numpy as np
import pickle # Restored if used by any non-RAG PyTorch logic, though likely not.
import torch
import time
import requests
import gc
import threading
import psutil
import concurrent.futures
from functools import lru_cache # Restored if used
# Restored if used
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging
# Removed PeftModel, PeftConfig as they were likely part of earlier RAG or not used in the target Pytorch version
from flask import Flask, request, render_template, jsonify
import logging
from torch.nn.attention import SDPBackend, sdpa_kernel # Keep for explicit Flash Attention context
from flask_socketio import SocketIO, emit

# --- CPU Specific Imports (Might be needed depending on how you use Int8/compile) ---
# Try uncommenting this if you are on Intel hardware and installed IPEX
# try:
#     import intel_extension_for_pytorch as ipex
#     # IPEX optimization specific settings can go here if needed before loading
#     ipex_available = True
#     logger = logging.getLogger(__name__) # Re-get logger after potential IPEX changes
#     logger.info("Intel Extension for PyTorch (IPEX) imported.")
# except ImportError:
#     ipex_available = False
#     logger = logging.getLogger(__name__)
#     logger.warning("Intel Extension for PyTorch (IPEX) not found. Install for potential CPU speedups on Intel hardware (`pip install intel-extension-for-pytorch`).")
# -------------------------------------------------------------------------------------


# Reduce transformers logging to avoid overhead
transformers_logging.set_verbosity_error()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            template_folder='templates_rag',
            static_folder='static_rag')

# Define paths for local development
base_dir = os.path.dirname(os.path.abspath(__file__))
# Set the path to your local fine-tuned model
local_model_dir = os.path.join(base_dir, "fine_tuned_model_gemma_colab_3_1b")

# Gemini API configuration
GEMINI_API_KEY = '' # User updated this
GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}'

# Global model and tokenizer variables
model = None
tokenizer = None
model_loaded = False
model_loading_lock = threading.Lock()
compiled_model = None

# Global thread pool for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

# ===== RESPONSE CACHE =====
response_cache = {}
CACHE_MAX_SIZE = 2000

# Precompiled KV cache (if it was used with PyTorch version)
precompiled_kv_cache = {} # Kept, though its usage might need review for PyTorch

# Performance tuning constants
MAX_STD_CONTEXT_LENGTH = 256
MAX_GENERATION_TOKENS = 300

socketio = SocketIO(app)

@socketio.on('query')
def handle_query(query_data):
    query = query_data['text']
    
    # Start generation in a background thread
    def generate_and_stream():
        for token in generate_streaming(query):
            emit('response_chunk', {'text': token})
        emit('response_complete')
    
    threading.Thread(target=generate_and_stream).start()

def load_model_in_background():
    """Load the PyTorch model in a background thread."""
    global model, tokenizer, model_loaded, compiled_model

    with model_loading_lock:
        if model_loaded:
            return

    try:
        logger.info(f"Starting background PyTorch model loading from {local_model_dir}...")
        start_time_load = time.time()

        try:
            cpu_ram_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"Available system RAM: {cpu_ram_gb:.2f} GB")
        except Exception as mem_error:
            logger.warning(f"Error detecting system memory: {mem_error}. Assuming 8GB.")
            cpu_ram_gb = 8

        tokenizer = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Adjusted RAM check - 8GB is usually minimum for even quantized models
        if cpu_ram_gb < 8:
            logger.warning("Not enough RAM to load model. Will use Gemini API exclusively.")
            model_loaded = False
            return

        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")

        model_kwargs = {
            "low_cpu_mem_usage": True,
        }

        if cuda_available:
            logger.info("Configuring PyTorch for GPU performance.")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "quantization_config": quantization_config,
                "attn_implementation": "flash_attention_2"
            })
            logger.info("Loading model with GPU acceleration and 4-bit quantization using Flash Attention 2.")
        else:
            logger.info("Configuring PyTorch for CPU performance.")
            import multiprocessing
            # Experiment with logical vs. physical cores for optimal thread count
            # num_threads = multiprocessing.cpu_count() # Logical cores
            num_threads = multiprocessing.cpu_count() or 1 # Physical cores
            torch.set_num_threads(num_threads)
            logger.info(f"Set torch threads to {num_threads}")

            # --- CPU Configuration (No Int8 with bitsandbytes without CUDA) ---
            model_kwargs.update({
                "device_map": "cpu",
                "torch_dtype": torch.float32, # Standard for CPU base loading
                # "load_in_8bit": True, # Removed: bitsandbytes 8-bit needs CUDA or specific non-GPU setup
            })
            logger.info("Loading model with optimized CPU config (Int8 quantization for CPU via bitsandbytes disabled as CUDA is not available).")
            # ----------------------------


        # Load the model
        try:
             model = AutoModelForCausalLM.from_pretrained(local_model_dir, **model_kwargs)
             logger.info("Model loaded successfully from pretrained.")
        except Exception as load_error:
             logger.error(f"Failed to load model with specified kwargs: {load_error}", exc_info=True)
             # Attempt to load without Int8 if it failed - this fallback might be less relevant now
             if "load_in_8bit" in model_kwargs: # This condition will likely be false now
                 logger.warning("Attempting to load model without Int8 quantization due to previous error.")
                 model_kwargs.pop("load_in_8bit")
                 try:
                     model = AutoModelForCausalLM.from_pretrained(local_model_dir, **model_kwargs)
                     logger.info("Model loaded successfully without Int8.")
                 except Exception as fallback_error:
                     logger.error(f"Fallback model loading also failed: {fallback_error}", exc_info=True)
                     raise fallback_error # Re-raise the error if fallback fails
             else: # If not related to load_in_8bit, re-raise directly
                 raise load_error


        model.eval()

        if hasattr(model, 'generation_config'):
            try:
                # Note: Static KV cache is often more beneficial on GPU,
                # but can still be enabled if the model supports it.
                # Using dynamic cache for CPU as it might be more robust with torch.compile and varying sequence lengths.
                model.generation_config.cache_implementation = "dynamic"
                logger.info("Enabled dynamic KV cache for generation.")
            except Exception as e:
                logger.warning(f"Could not enable dynamic KV cache: {e}")
        else:
            logger.warning("Model does not have generation_config, cannot set dynamic KV cache.")

        # --- Re-enable torch.compile for CPU ---
        compiled_model = None
        if not cuda_available: # Attempt torch.compile primarily for CPU
            try:
                logger.info("Attempting torch.compile for CPU performance... This might be slow initially.") # Updated log
                # Experiment with modes: "reduce-overhead", "max-autotune"
                # fullgraph=True can be ambitious and sometimes fails, fullgraph=False is more robust.
                # If using IPEX (imported above), uncomment backend="ipex"
                
                # DISABLED torch.compile for now to diagnose slowdown
                # compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False) # Reduced fullgraph for robustness
                # compile_mode_used = "reduce-overhead" 
                # # compiled_model = torch.compile(model, mode="max-autotune", fullgraph=False) # Alternative: Experiment with max-autotune
                # # compile_mode_used = "max-autotune"
                # # if ipex_available:
                # #    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False, backend="ipex") # Use IPEX backend

                if compiled_model is not None:
                    logger.info("torch.compile successful. Model compiled.")
                else:
                    logger.info("torch.compile was skipped/disabled. Using standard model.") # New log for clarity

            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Using standard model.")
                compiled_model = None # Ensure it's None if compilation fails
        else:
            logger.info("Skipping torch.compile on GPU for now, as benefits can vary and might conflict with Flash Attention / quantization.")

        # Warmup
        active_warmup_model = compiled_model if compiled_model is not None else model
        logger.info(f"Running warmup with {'compiled' if compiled_model else 'standard'} model...")
        warmup_input_text = "Hello, this is a warmup query for the model."

        if tokenizer is None: # Should not happen if first tokenizer load succeeded
             tokenizer = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True)
             if tokenizer.pad_token is None:
                 tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
                 tokenizer.pad_token_id = tokenizer.eos_token_id

        # Move warmup inputs to the model's actual device (should be 'cpu' here)
        warmup_inputs = tokenizer(warmup_input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(active_warmup_model.device if hasattr(active_warmup_model, 'device') else 'cpu')
        
        # Store the compile mode to check later if needed
        compile_mode_used = "reduce-overhead" # Default or actual mode used
        
        with torch.no_grad(), torch.inference_mode():
            # Run warmup twice for torch.compile(mode="max-autotune") if used
            _ = active_warmup_model.generate(**warmup_inputs, max_new_tokens=5, do_sample=False)
            # Check the compile_mode_used variable instead of compiled_model.mode
            if compiled_model and compile_mode_used == "max-autotune":
                 logger.info("Running second warmup pass for max-autotune...")
                 _ = active_warmup_model.generate(**warmup_inputs, max_new_tokens=5, do_sample=False)

        logger.info(f"Warmup with {'compiled' if compiled_model else 'standard'} model complete.")
        # --- End of torch.compile section ---

        model_loaded = True
        logger.info(f"PyTorch Model loaded successfully in {time.time() - start_time_load:.1f} seconds. Compiled: {compiled_model is not None}")

    except Exception as e:
        logger.error(f"Fatal error during PyTorch model loading: {e}", exc_info=True)
        model_loaded = False
        # Ensure resources are released on failure
        if model is not None: del model; model = None
        if tokenizer is not None: del tokenizer; tokenizer = None
        if compiled_model is not None: del compiled_model; compiled_model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.error("PyTorch model loading failed. Will rely on Gemini API.")


def _generate_response(query):
    global model, tokenizer, model_loaded, compiled_model

    # Check if the non-compiled model or compiled model is loaded
    is_local_model_ready = model_loaded and (model is not None or compiled_model is not None)

    if not is_local_model_ready:
        logger.warning("PyTorch Model (base or compiled) is not loaded. Falling back to Gemini API.")
        return generate_gemini_response(query) # Fallback if model loading failed

    cached_response = check_cache(query)
    if cached_response:
        return cached_response

    try:
        prompt = f"### Question: {query}\n### Answer: Provide a detailed response with at least 5-7 sentences explaining the topic thoroughly."
        # Indicate whether compile is active
        logger.info(f"Generating PyTorch response for: {query} (Compiled: {compiled_model is not None})")
        start_time = time.time()

        # Use the compiled model if available, otherwise the base model
        active_model = compiled_model if compiled_model is not None else model

        # Ensure tokenizer and model are available - redundant check due to initial model_loaded check, but safe
        if tokenizer is None or active_model is None:
             logger.error("Tokenizer or active model is None despite model_loaded being True.")
             # This indicates a critical failure state, fall back to Gemini
             model_loaded = False # Force future checks to fail faster
             return generate_gemini_response(query)


        # Ensure inputs are on the correct device (CPU in this case)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_STD_CONTEXT_LENGTH,  # Max length for the input prompt section.
            padding=True,                       # Pad to the actual length of the (truncated) prompt, not a fixed max_length.
            add_special_tokens=True
        ).to(active_model.device if hasattr(active_model, 'device') else 'cpu') # Ensure inputs are on CPU

        input_ids_length = inputs.input_ids.shape[1]
        logger.info(f"Tokenized input length for PyTorch: {input_ids_length} tokens. Tokenizer max_length setting: {MAX_STD_CONTEXT_LENGTH}")
        # The truncation/padding might cut off the prompt ending if it's very long,
        # this check isn't foolproof but indicates a potential issue.
        if input_ids_length == MAX_STD_CONTEXT_LENGTH and tokenizer.decode(inputs.input_ids[0]).strip().endswith(tokenizer.eos_token):
             logger.warning("Input prompt for PyTorch may have been truncated before padding.")


        generate_kwargs = {
            "max_new_tokens": MAX_GENERATION_TOKENS,
            "min_new_tokens": 50,
            "temperature": 0.6,
            "top_p": 0.92,
            "top_k": 15,
            "num_beams": 1, # Beam search can be slower on CPU, keeping 1 is standard for speed
            "do_sample": True,
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "early_stopping": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True, # <<< RE-ENABLED KV CACHING FOR PERFORMANCE
            # Removed output_attentions/hidden_states for performance unless needed
            # "output_attentions": False,
            # "output_hidden_states": False,
            # "return_dict_in_generate": False # Keep as False for simplicity
        }

        # No CUDA/Flash Attention specific calls needed for CPU path
        with torch.no_grad(), torch.inference_mode():
            outputs = active_model.generate(**inputs, **generate_kwargs)


        generation_time = time.time() - start_time
        logger.info(f"PyTorch Response generated in {generation_time:.2f} seconds (Compiled: {compiled_model is not None})")

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Improved answer extraction logic
        answer = full_response
        # Remove the prompt if it's still in the output
        prompt_text = prompt.replace("### Answer: Provide a detailed response with at least 5-7 sentences explaining the topic thoroughly.", "").strip()
        if answer.startswith(prompt_text):
             answer = answer[len(prompt_text):].strip()
        # Remove the specific "### Answer:" marker if it's present at the start
        if answer.startswith("### Answer:"):
             answer = answer[len("### Answer:"):].strip()

        # Basic check to ensure some content was generated
        if not answer or len(answer) < 20: # Check for minimal length
             logger.warning(f"Generated answer is short or empty. Full output: {full_response}")
             answer = "Could not generate a satisfactory response with the local model. Please try a different query or use the Gemini API."


        update_cache(query, answer)

        # Trigger memory cleanup if generation was slow
        if generation_time > 15.0: # Threshold might need tuning
             thread_pool.submit(clean_memory)

        return answer if answer else "I couldn't generate a response with PyTorch model. Please try again."

    except Exception as e:
        logger.error(f"Error generating PyTorch response (Compiled: {compiled_model is not None}): {e}", exc_info=True)
        # Fallback to Gemini on PyTorch generation error
        logger.warning("PyTorch generation failed, attempting fallback to Gemini API.")
        return generate_gemini_response(query)


def clean_memory():
    """Clean up memory (CPU specific)."""
    logger.info("Cleaning CPU memory...")
    gc.collect()
    # Removed torch.cuda.empty_cache() as we are focusing on CPU
    logger.info("CPU memory cleanup complete.")


def get_cache_key(query, model_type="local"):
    query_clean = query.strip().lower()
    return f"{model_type}:{hash(query_clean)}"

def check_cache(query, model_type="local"):
    cache_key = get_cache_key(query, model_type)
    if cache_key in response_cache:
        logger.info(f"Cache hit for query: {query} (Model: {model_type})")
        return response_cache[cache_key]
    return None

def update_cache(query, response, model_type="local"):
    cache_key = get_cache_key(query, model_type)
    if len(response_cache) >= CACHE_MAX_SIZE:
        try:
            # Use popitem(last=False) for LRU-like behavior (removes oldest)
            key_to_remove, _ = response_cache.popitem(last=False)
            logger.info(f"Cache full, removed oldest entry: {key_to_remove}")
        except KeyError:
             pass # Cache was empty
    response_cache[cache_key] = response
    logger.info(f"Cache updated for query: {query} (Model: {model_type})")


def generate_response(query):
    """Determines which model to use (local PyTorch or Gemini API) and generates a response."""
    global model_loaded, model, compiled_model
    # Check if *either* the base or compiled local model is successfully loaded
    is_local_model_ready = model_loaded and (model is not None or compiled_model is not None)

    if not is_local_model_ready:
        logger.info("Local PyTorch model not ready or failed to load. Using Gemini API.")
        return generate_gemini_response(query)

    # If local model is ready, try to use it first
    # _generate_response handles internal fallback to Gemini if generation fails
    return _generate_response(query)


def generate_gemini_response(query):
    """Generates a response using the Gemini API."""
    cached_response = check_cache(query, model_type="gemini")
    if cached_response: return cached_response

    try:
        logger.info(f"Sending query to Gemini API: {query}")
        payload = {
            "contents": [{"parts": [{"text": query}]}],
            "generationConfig": {"temperature": 0.7, "topP": 0.9, "maxOutputTokens": 800}
        }
        start_time = time.time()
        response = requests.post(GEMINI_API_URL, json=payload, timeout=30) # Increased timeout slightly

        if response.status_code != 200:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return f"Error with Gemini API: {response.status_code} - {response.text}"

        response_json = response.json()
        generation_time = time.time() - start_time
        logger.info(f"Gemini response received in {generation_time:.2f} seconds")

        if "candidates" in response_json and len(response_json["candidates"]) > 0 and "content" in response_json["candidates"][0]:
             # Extract text carefully, handling potential missing 'parts' or 'text'
             try:
                 text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                 update_cache(query, text, model_type="gemini")
                 return text
             except (KeyError, IndexError) as e:
                 logger.error(f"Unexpected structure in Gemini API response: {response_json} - {e}")
                 return "Received an unexpected response format from Gemini API. Please try again."
        elif "promptFeedback" in response_json:
            feedback = response_json["promptFeedback"]
            block_reason = feedback.get("blockReason", "unknown")
            logger.warning(f"Gemini API blocked prompt: {feedback}")
            return f"Gemini API blocked the query. Reason: {block_reason}. Safety ratings: {feedback.get('safetyRatings', 'N/A')}"
        else:
            logger.warning(f"No candidates or prompt feedback found in Gemini response: {response_json}")
            return "No response content from Gemini API. Please try again."

    except requests.exceptions.Timeout:
        logger.error("Gemini API request timed out.")
        return "Error: The request to Gemini API timed out."
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}", exc_info=True)
        return f"Error with Gemini API: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index_route():
    if request.method == 'POST':
        query = request.form['query'].strip()
        if query:
            response_text = generate_response(query)
            return render_template('index.html', query=query, response=response_text)
    return render_template('index.html', query="", response="")

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    if not data or 'query' not in data: return jsonify({'error': 'No query provided'}), 400
    query = data['query'].strip()
    if not query: return jsonify({'error': 'Empty query'}), 400

    logger.info(f"API request received with query: {query}")

    model_type_req = data.get('model_type', 'local') # Default to local
    # Check if *either* the base or compiled local model is loaded and ready
    is_local_model_ready = model_loaded and (model is not None or compiled_model is not None)
    force_gemini = model_type_req.lower() == 'gemini' or not is_local_model_ready

    response_text = ""
    final_model_type = "gemini" if force_gemini else f"local (PyTorch{', compiled' if compiled_model else ''})"

    try:
        start_time_api = time.time()
        if force_gemini:
            logger.info(f"Using Gemini API for query: {query}")
            response_text = generate_gemini_response(query)
        else:
            logger.info(f"Using local PyTorch model for query: {query}")
            # Call the main generation function which includes the internal fallback
            response_text = generate_response(query)

        generation_time_api = time.time() - start_time_api

        return jsonify({
            'response': response_text,
            'generation_time_seconds': generation_time_api,
            'model_type': final_model_type
        })

    except Exception as e:
        # This outer catch should theoretically only happen if generate_response itself raises an unexpected error
        error_message = str(e)
        logger.error(f"Unexpected error processing API request: {error_message}", exc_info=True)
        error_description = "An unexpected error occurred during response generation."

        # Attempt a final fallback to Gemini if anything went wrong with the local path
        try:
            logger.warning("API request failed with local model, attempting fallback to Gemini.")
            fallback_response = generate_gemini_response(query)
            return jsonify({
                'response': fallback_response,
                'warning': f"Primary model failed. Using Gemini API instead. Error details: {error_description}",
                'error_details': error_message, # Include original error for debug
                'model_type': 'gemini'
            }), 200 # Return 200 even with warning if fallback succeeded
        except Exception as fallback_error:
            logger.error(f"Gemini fallback also failed in API handler: {fallback_error}", exc_info=True)
            return jsonify({
                'error': 'Both local and Gemini models failed to generate a response',
                'error_details': f"Primary error: {error_message}. Fallback error: {fallback_error}"
            }), 500


@app.route('/api/model-status', methods=['GET'])
def model_status():
    global model_loaded, model, compiled_model
    device_str = 'none'
    if model_loaded:
        # Check compiled model device first if it exists
        if compiled_model is not None and hasattr(compiled_model, 'device'):
            device_str = str(compiled_model.device)
        # Then check base model device
        elif model is not None and hasattr(model, 'device'):
            device_str = str(model.device)
        # If model_loaded is true but device isn't directly available (unlikely but defensive)
        elif torch.cuda.is_available():
             device_str = 'cuda (assumed)'
        else:
             device_str = 'cpu (assumed)'


    status_details = {
        'loaded': model_loaded,
        'engine': 'PyTorch' if model_loaded else 'None',
        'compiled': compiled_model is not None if model_loaded else False,
        'device': device_str,
        'cpu_threads': torch.get_num_threads() if not torch.cuda.is_available() else None, # Show CPU threads only in CPU mode
        'cache_size': len(response_cache)
    }
    return jsonify(status_details)

if __name__ == '__main__':
    logger.info("Starting Flask PyTorch/Gemini hybrid app...")
    # Start loading the PyTorch model in a background thread immediately
    threading.Thread(target=load_model_in_background, daemon=True).start()
    # Run the Flask app
    # debug=True should be False in production
    app.run(debug=False, host='0.0.0.0', port=5001)