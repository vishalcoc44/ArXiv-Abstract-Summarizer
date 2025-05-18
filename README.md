# ArXiv RAG Chat Application

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project is a sophisticated chat application that combines local and cloud-based Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) capabilities. It focuses on providing accurate responses to academic and research-related queries by leveraging a local FAISS vector store containing arXiv paper abstracts.

The application offers two modes of operation:
1. **Local RAG Mode** (Gemma): Uses a quantized Gemma 1B model (GGUF format) for fast local inference with RAG capabilities.
2. **Cloud API Mode** (Gemini): Leverages Google's Gemini API for more powerful capabilities when needed.

Additionally, the system implements an innovative hybrid approach where the local model can delegate paper suggestion queries to Gemini API for improved accuracy.

## 🔑 Key Features

- **Dual Model Support**: Switch between local Gemma (quantized GGUF model) and cloud-based Gemini API
- **RAG Implementation**: Vector similarity search using FAISS index of arXiv abstracts
- **Context Preservation**: Maintains chat history for continuous conversation
- **Folder Organization**: Organize chats into custom folders
- **Responsive UI**: Modern and intuitive user interface with animations
- **Hybrid Model Approach**: Delegates specific queries to cloud model when appropriate
- **Streaming Responses**: Real-time streaming of responses for better user experience
- **Database Persistence**: Stores conversations using SQLite
- **Markdown Support**: Renders responses with Markdown formatting

## 📋 Prerequisites

- Python 3.9+ (recommended: 3.10+)
- CUDA-enabled GPU (optional, for faster inference)
- At least 8GB RAM (16GB+ recommended)
- Google Gemini API key (optional, for Gemini model access)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv arxiv_env
   source arxiv_env/bin/activate  # Linux/macOS
   # OR
   arxiv_env\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create a `.env` file in the project root):
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Prepare the model** (Option A - Download pre-quantized model):
   - Download the quantized Gemma model (`gemma_1b_finetuned_q4_0.gguf`) to the project directory

   **OR**

   **Prepare the model** (Option B - Quantize your own model):
   ```bash
   python merge_and_save_gemma.py  # Merges the model checkpoints
   # Then use a tool like llama.cpp to quantize the model
   ```

## 🚀 Data Preparation

### 1. Collect arXiv Abstracts
```bash
python collect_arxiv_abstracts.py
```
This script fetches recent paper abstracts from arXiv and saves them to `./arxiv_abstracts/abstracts.json`.

### 2. Generate Embeddings
```bash
python generate_embeddings.py
```
This creates vector embeddings for the abstracts using the Sentence Transformers model.

### 3. Build FAISS Index
```bash
python build_faiss_index.py
```
Creates a raw FAISS index from the embeddings.

### 4. Create LangChain-compatible FAISS Index
```bash
python create_lc_faiss_index.py
```
Creates a LangChain-compatible FAISS index in the `langchain_faiss_store_optimized` directory.

## 🌟 Running the Application

1. **Initialize the database** (happens automatically on first run):
   ```bash
   # The database schema in static/schema.sql will be applied automatically
   ```

2. **Start the Flask server**:
   ```bash
   python app2.py
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:5001`

## 💻 System Architecture

### Components

1. **Frontend**:
   - HTML/CSS/JavaScript interface
   - GSAP for animations
   - Marked.js for Markdown rendering
   - Lottie for advanced animations

2. **Backend**:
   - Flask web server
   - SQLite database
   - Llama.cpp for local model inference
   - FAISS vector store for similarity search
   - LangChain for RAG implementation

3. **Models**:
   - Local: Quantized Gemma 1B model (GGUF format)
   - Cloud: Google Gemini API

4. **Data Flow**:
   - User query → Context retrieval → Model inference → Streaming response

### Database Schema

- **Folders Table**: Organizes chats into categories
- **Chats Table**: Stores chat metadata and references to folders
- **Messages Table**: Contains individual chat messages

## 🗄️ Project Structure

```
├── app2.py                            # Main Flask application
├── static/                            # Static assets 
│   ├── style2.css                     # CSS styles
│   ├── script2.js                     # Frontend JavaScript
│   └── schema.sql                     # Database schema
├── templates/                         # HTML templates
│   └── index2.html                    # Main interface template
├── collect_arxiv_abstracts.py         # Script to collect arXiv data
├── generate_embeddings.py             # Script to create embeddings
├── build_faiss_index.py               # Script to build raw FAISS index
├── create_lc_faiss_index.py           # Script for LangChain FAISS index
├── gemma_1b_finetuned_q4_0.gguf       # Quantized Gemma model
├── langchain_faiss_store_optimized/   # FAISS vector store directory
│   ├── index.faiss                    # FAISS index file
│   └── index.pkl                      # Python pickle file with metadata
└── chat_app.db                        # SQLite database
```

## 🔍 Advanced Features

### 1. Hybrid RAG Implementation

The system implements a sophisticated approach for paper recommendations:
- When users ask for paper recommendations, the query is first sent to the Gemini API
- Gemini suggests relevant papers and potential arXiv IDs
- The local FAISS index verifies these suggestions against the local knowledge base
- Verified papers are presented to the user with detailed information

### 2. Context Management

The application intelligently manages conversation context:
- Maintains a sliding window of previous messages (configured as MAX_HISTORY_MESSAGES)
- Formats the conversation history appropriately for the model
- Ensures coherent multi-turn conversations

### 3. Streaming Response Handling

The system implements efficient response streaming:
- Manages concurrent database operations during streaming
- Handles cleanup and resource management properly
- Provides real-time feedback to the user

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Gemma](https://blog.google/technology/developers/gemma-open-models/) - Google's open model
- [ArXiv API](https://arxiv.org/help/api/index) - For providing research paper data
- [LangChain](https://www.langchain.com/) - For RAG implementation utilities
- [FAISS](https://github.com/facebookresearch/faiss) - For efficient similarity search
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) - For model quantization and inference 