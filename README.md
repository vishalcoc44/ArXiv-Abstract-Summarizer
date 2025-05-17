# Ultra-Fast Gemma 3 1B Inference Optimized App

This Flask application provides an interface to interact with a fine-tuned Gemma 3 1B model with ultra-fast inference optimizations.

## Optimization Features

- **PyTorch Compile**: Utilizes `torch.compile` with the Inductor backend for dramatically faster inference
- **Inference Mode**: Uses `torch.inference_mode()` for maximum performance
- **CUDA Optimizations**: 
  - Enables TF32 precision on NVIDIA GPUs
  - Uses CuDNN benchmarking for optimal kernel selection
  - GPU memory management and caching
- **Quantization**: Supports 4-bit quantization for faster inference on applicable systems
- **Threading Optimization**: Automatically adjusts thread count based on available CPU cores
- **Model Caching**: Efficiently caches model outputs to avoid redundant computations
- **Progressive Response**: Shows real-time typing and response generation metrics
- **Hybrid Serving**: Falls back to Gemini API when the local model is unavailable
- **Reactive UI**: Adaptive UI that shows model state and optimization level

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and go to http://localhost:5000

## Model Loading

The application automatically loads the fine-tuned model in the background, optimizing it based on your hardware:

- On CUDA-enabled GPUs: Uses mixed precision, torch.compile, and CUDA optimizations
- On CPU: Uses optimized thread allocation and inference mode

## Performance Notes

- First response may be slower as the model warms up JIT compilation
- Subsequent responses will be significantly faster
- Response speed depends on your hardware (CPU/GPU)
- Compiled model performance improves over time as paths become more optimized

## Fallback to Gemini API

If local model loading fails or takes too long, the application automatically falls back to using Google's Gemini API. 