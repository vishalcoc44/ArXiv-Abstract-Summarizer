import os

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
    
    # Model Paths & Parameters for TensorRT-LLM
    # LOCAL_MODEL_PATH now serves as the base model identifier, typically for tokenizer.
    LOCAL_MODEL_PATH = os.getenv('MODEL_PATH', 'gemma-3-4b-it-q4_0.gguf') # HF model ID for tokenizer
    TENSORRT_ENGINE_DIR = os.getenv('TENSORRT_ENGINE_DIR', 'tensorrt_llm_gemma_engine') # IMPORTANT: Set to your TRT engine path
    TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', LOCAL_MODEL_PATH) # Path for AutoTokenizer, can be same as LOCAL_MODEL_PATH
    
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'langchain_faiss_store_optimized1')
    
    # RAG Configuration (shared conceptual parameters)
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    N_RETRIEVED_DOCS = int(os.getenv('N_RETRIEVED_DOCS', '15'))
    # These "LLAMA_" prefixed vars are used by models3.py for generation, keep for now.
    LLAMA_MAX_TOKENS = int(os.getenv('LLAMA_MAX_TOKENS', '2048'))  # Max new tokens for generation
    LLAMA_TEMPERATURE = float(os.getenv('LLAMA_TEMPERATURE', '0.3'))
    LLAMA_TOP_P = float(os.getenv('LLAMA_TOP_P', '0.9'))
    
    # Llama.cpp model parameters (No longer used with TensorRT-LLM or handled during engine build)
    # LLAMA_N_CTX = int(os.getenv('LLAMA_N_CTX', '8192'))
    # LLAMA_N_GPU_LAYERS = int(os.getenv('LLAMA_N_GPU_LAYERS', '-1'))
    # LLAMA_N_BATCH = int(os.getenv('LLAMA_N_BATCH', '2048'))
    # LLAMA_VERBOSE = True

    # Flask App settings
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))
    DATABASE_URL = os.getenv('DATABASE_URL', 'chat_app.db')
    
    # Cloud specific settings
    REDIS_URL = os.getenv('REDIS_URL')  # For session management in cloud
    CLOUD_STORAGE_BUCKET = os.getenv('CLOUD_STORAGE_BUCKET')  # For storing models in cloud 