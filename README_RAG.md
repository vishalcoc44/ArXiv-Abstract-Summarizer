# Local LLM Chat Application

This application provides a web interface to interact with a locally hosted, fine-tuned Gemma 1.1B model. It also includes an option to use the Gemini API as an alternative or fallback.

## Features

-   **Local Model Inference**: Primarily uses a fine-tuned Gemma 1.1B model running locally.
-   **Gemini API Integration**: Can switch to or fallback to the Gemini API for responses.
-   **Web Interface**: User-friendly chat interface built with Flask, HTML, CSS, and JavaScript.
-   **Performance Optimizations**: Includes various optimizations for faster local model inference, such as:
    -   4-bit quantization (BitsAndBytesConfig)
    -   Flash Attention 2 (if CUDA is available)
    -   `torch.compile` (if CUDA is available and PyTorch 2.0+)
    -   Optimized tokenizer usage
    -   Response caching
    -   Background model loading
-   **Model Status Monitoring**: The UI indicates when the local model is loading and when it's ready.

## Core Components

-   **`app_rag.py`**: The main Flask application. It handles:
    -   Loading and managing the local Gemma model and tokenizer.
    -   Generating responses using either the local model or the Gemini API.
    -   Serving the web interface.
    -   API endpoints for chat and model status.
-   **`templates_rag/index.html`**: The HTML structure for the chat interface.
-   **`static_rag/js/script.js`**: Client-side JavaScript for handling user interactions, sending requests to the backend, and updating the UI.
-   **`static_rag/css/style.css`**: CSS for styling the web interface.
-   **`fine_tuned_model_gemma_colab_3_1b/` (Directory)**: This directory should contain your fine-tuned Gemma model files (tokenizer, model weights, config).

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.8+
    *   PyTorch (preferably with CUDA support for GPU acceleration)
    *   Transformers library by Hugging Face
    *   Flask
    *   Other dependencies (see `requirements.txt` if available, or install as needed: `psutil`, `requests`, `numpy`)

2.  **Local Model**:
    *   Ensure your fine-tuned Gemma model files are located in a directory named `fine_tuned_model_gemma_colab_3_1b` within the same directory as `app_rag.py`.
    *   The application expects to find the model and tokenizer in this specific sub-directory.

3.  **Gemini API Key**:
    *   If you plan to use the Gemini API, replace the placeholder API key in `app_rag.py` with your actual key:
        ```python
        GEMINI_API_KEY = 'YOUR_ACTUAL_GEMINI_API_KEY'
        ```

4.  **Install Dependencies**:
    *   If a `requirements.txt` file is present:
        ```bash
        pip install -r requirements.txt
        ```
    *   Otherwise, install the necessary packages manually:
        ```bash
        pip install torch torchvision torchaudio
        pip install transformers flask psutil requests numpy bitsandbytes accelerate sentencepiece
        ```
        *(Adjust `torch` installation according to your CUDA version if applicable. Visit pytorch.org for specific instructions.)*

5.  **Run the Application**:
    ```bash
    python app_rag.py
    ```

6.  **Access the Interface**:
    *   Open your web browser and navigate to `http://localhost:5001`.

## Usage

-   The interface provides buttons to select between the "Local Model" and "Search" (Gemini API).
-   The "Local Model" button will indicate if the model is still loading.
-   Type your query in the input box and press Enter or click the send button.
-   Responses will be displayed in the chat area, along with the model used and generation time.

## Performance Notes

-   The first query to the local model after startup might take longer as the model "warms up."
-   Performance heavily depends on your hardware (CPU, GPU, RAM).
-   GPU acceleration (via CUDA) significantly improves inference speed for the local model.
-   The application attempts to apply several optimizations. If `torch.compile` or Flash Attention 2 are successfully enabled, this will be logged by the application and can improve speed on compatible hardware. 