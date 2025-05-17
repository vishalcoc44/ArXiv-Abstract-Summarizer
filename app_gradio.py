import os
import numpy as np
import torch
import time
import requests
import gc
import threading
import psutil
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging
import logging
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()

# Define paths and configs
base_dir = os.path.dirname(os.path.abspath(__file__))
local_model_dir = os.path.join(base_dir, "fine_tuned_model_gemma_colab_3_1b")

# Gemini API configuration
GEMINI_API_KEY = 'YOUR_API_KEY'  # Store this securely or use environment variables
GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}'

# Global variables
model = None
tokenizer = None
model_loaded = False
model_loading_lock = threading.Lock()
compiled_model = None
response_cache = {}
CACHE_MAX_SIZE = 2000

# Performance tuning constants
MAX_STD_CONTEXT_LENGTH = 256
MAX_GENERATION_TOKENS = 150  # Reduced for faster responses

# Keep your existing functions 
def load_model_in_background():
    """Load the PyTorch model in a background thread."""
    global model, tokenizer, model_loaded, compiled_model
    # ... (keep your existing implementation)

def generate_response(query):
    """Determines which model to use (local PyTorch or Gemini API) and generates a response."""
    # ... (keep your existing implementation)

def generate_gemini_response(query):
    """Generates a response using the Gemini API."""
    # ... (keep your existing implementation)

# Define streaming function for real-time output
def generate_streaming_response(query, history, progress=gr.Progress()):
    """Generate a streaming response token by token for real-time display."""
    is_local_model_ready = model_loaded and (model is not None or compiled_model is not None)
    cached = check_cache(query)
    
    if cached:
        return cached
    
    # Use Gemini fallback if local model not ready
    if not is_local_model_ready:
        logger.info("Local model not ready, using Gemini API")
        response = generate_gemini_response(query)
        return response
    
    # Setup variables and inputs for local model
    active_model = compiled_model if compiled_model is not None else model
    prompt = f"### Question: {query}\n### Answer:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_STD_CONTEXT_LENGTH,
        padding="max_length",
        add_special_tokens=True
    ).to(active_model.device if hasattr(active_model, 'device') else 'cpu')
    
    # Set up streaming generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": MAX_GENERATION_TOKENS,
        "temperature": 0.6,
        "top_p": 0.92,
        "top_k": 15,
        "streamer": streamer,
        "use_cache": True,
    }
    
    # Start generation in a separate thread
    generation_thread = threading.Thread(
        target=lambda: active_model.generate(**inputs, **generate_kwargs)
    )
    generation_thread.start()
    
    # Stream the output
    generated_text = ""
    progress(0, desc="Generating...")
    
    for i, new_text in enumerate(streamer):
        generated_text += new_text
        progress((i+1)/MAX_GENERATION_TOKENS)
        yield generated_text
    
    # Cache the result
    update_cache(query, generated_text)
    return generated_text

# Create Gradio Interface
def create_gradio_interface():
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# AI Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_status = gr.Markdown("Loading model...")
                model_type = gr.Radio(
                    ["Local Model", "Gemini API"], 
                    label="Model Type", 
                    value="Local Model"
                )
            
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="Ask me anything...", container=False)
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")
            
        # Add special animation effect
        gr.HTML("""
            <style>
                .hovered-message {
                    animation: fadeIn 0.3s;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
        """)
        
        def update_model_status():
            global model_loaded, compiled_model
            if not model_loaded:
                return "⏳ Model is still loading... Using Gemini API as fallback"
            device_info = "GPU" if hasattr(model, 'device') and 'cuda' in str(model.device) else "CPU"
            compiled_info = "(Compiled)" if compiled_model is not None else ""
            return f"✅ Model loaded on {device_info} {compiled_info}"
        
        def user_input(user_message, history):
            return "", history + [[user_message, None]]
            
        def bot_response(history, model_choice):
            query = history[-1][0]
            model_type = "gemini" if model_choice == "Gemini API" else "local"
            
            # For streaming display (gradio will handle it)
            for response in generate_streaming_response(query, history):
                history[-1][1] = response
                yield history
        
        submit_btn.click(
            user_input, 
            [msg, chatbot], 
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, model_type],
            chatbot
        )
        
        clear_btn.click(lambda: [], None, chatbot)
        
        # Setup periodic status update
        demo.load(update_model_status, None, model_status, every=5)
        
    return demo

# Start model loading and launch Gradio
if __name__ == "__main__":
    # Start model loading in background
    threading.Thread(target=load_model_in_background, daemon=True).start()
    
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.queue()
    demo.launch()
