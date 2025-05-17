import faiss
import os
import numpy as np
import pickle
import torch
import time
import requests
import gc
import threading
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define paths for local development (no Google Drive)
base_dir = os.path.dirname(os.path.abspath(__file__))
# Set the path to your local fine-tuned model
local_model_dir = os.path.join(base_dir, "fine_tuned_model_gemma_colab_3_1b")

# Optional FAISS search data paths - uncomment if you need these
# faiss_index_path = os.path.join(base_dir, "data", "faiss_index.bin")  
# metadata_path = os.path.join(base_dir, "data", "abstract_metadata.pkl")
# embeddings_path = os.path.join(base_dir, "data", "abstract_embeddings.npy")

# Gemini API configuration
GEMINI_API_KEY = ''  # Replace with your actual API key
GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}'

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False
model_loading_lock = threading.Lock()

def load_model_in_background():
    """Load the model in a background thread to avoid blocking the app startup"""
    global model, tokenizer, model_loaded
    
    try:
        logger.info(f"Starting background model loading from {local_model_dir}...")
        
        # Load tokenizer first - this is fast
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Optimize CPU settings
        torch.set_num_threads(4)  # Limit threads to avoid overloading CPU
        
        if cuda_available:
            # GPU settings
            logger.info("Loading model with GPU acceleration")
            model = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Most efficient CPU settings - simple loading without quantization
            logger.info("Loading model on CPU with performance optimizations")
            model = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                torch_dtype=torch.float32,  # Standard precision for CPU
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        # Force model to evaluation mode
        model.eval()
        
        # Run a warmup inference to initialize everything
        logger.info("Running warmup inference...")
        warmup_input = tokenizer("Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**warmup_input, max_new_tokens=1)
        
        logger.info("Model loaded and ready for inference")
        model_loaded = True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False

# Start loading the model in the background on app startup
threading.Thread(target=load_model_in_background, daemon=True).start()

# LRU cache for response generation - great for repeated queries
@lru_cache(maxsize=100)
def cached_generate_response(query_text):
    """Generate and cache responses for faster repeated queries"""
    return _generate_response(query_text)

def _generate_response(query):
    """Internal function that actually generates the response"""
    global model, tokenizer, model_loaded
    
    try:
        if not model_loaded:
            return "Model is still loading. Please try again in a moment."
        
        # Simple prompt
        prompt = f"### Question: {query}\n### Answer:"
        
        logger.info(f"Generating response for: {query}")
        start_time = time.time()
        
        # Generate response with faster settings
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_STD_CONTEXT_LENGTH - 1, # Try 255 instead of 256
            padding="max_length",
            add_special_tokens=True
        ).to(model.device if hasattr(model, 'device') else 'cpu')
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,        # Reduced for speed
                temperature=0.5,           # Reduced for more consistent outputs
                top_p=0.9,
                num_beams=1,              # Greedy decoding for speed
                do_sample=False,          # Disable sampling for faster generation
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Log generation time
        generation_time = time.time() - start_time
        logger.info(f"Response generated in {generation_time:.2f} seconds")
        
        # Get full response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "### Answer:" in full_response:
            answer = full_response.split("### Answer:")[-1].strip()
        else:
            answer = full_response.replace(prompt, "").strip()
        
        # Clear GPU memory if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer if answer else "I couldn't generate a response. Please try again."
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred: {str(e)}"

def generate_response(query):
    """Public function that uses caching when possible"""
    # Clean query to improve cache hits
    cleaned_query = query.strip().lower()
    return cached_generate_response(cleaned_query)

def generate_gemini_response(query):
    """Generate a response using Gemini API."""
    try:
        logger.info(f"Sending query to Gemini API: {query}")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": query}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 800
            }
        }
        
        start_time = time.time()
        response = requests.post(
            GEMINI_API_URL,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return f"Error with Gemini API: {response.status_code}"
        
        response_json = response.json()
        generation_time = time.time() - start_time
        logger.info(f"Gemini response received in {generation_time:.2f} seconds")
        
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            return text
        else:
            return "No response from Gemini API. Please try again."
    
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}")
        return f"Error with Gemini API: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index_route():
    """Handle web requests."""
    if request.method == 'POST':
        # This route handles traditional form submissions
        query = request.form['query'].strip()
        if query:
            response = generate_response(query)
    return render_template('index.html', query=query, response=response)
    return render_template('index.html', query="", response="")

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for AJAX requests from JavaScript."""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    
    # Log the incoming request
    logger.info(f"API request received with query: {query}")
    
    # Check if request is for Gemini or local model
    model_type = data.get('model_type', 'local')
    
    # Generate response using the appropriate model
    try:
        if model_type == 'gemini':
            response = generate_gemini_response(query)
        else:
            response = generate_response(query)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing API request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check if the model is loaded and ready"""
    return jsonify({'loaded': model_loaded})

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)