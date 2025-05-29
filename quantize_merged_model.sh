#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Path to the merged Hugging Face model directory (output of merge_and_save_gemma.py)
MERGED_MODEL_PATH="./merged_gemma_1b_finetuned_hf"

# Path to your llama.cpp checkout directory
# IMPORTANT: Adjust this path to where your llama.cpp repository is cloned.
LLAMA_CPP_PATH="../llama.cpp" 

# Directory to save the GGUF files
GGUF_OUTPUT_DIR="./gguf_models"

# Name for the output GGUF file (without quantization suffix)
MODEL_NAME="gemma_1b_finetuned_merged"

# --- Script ---

echo "--- GGUF Conversion and Quantization Script ---"

# Ensure output directory exists
mkdir -p "${GGUF_OUTPUT_DIR}"
echo "Output directory for GGUF files: ${GGUF_OUTPUT_DIR}"

# Path to llama.cpp conversion and quantization tools
CONVERT_PY="${LLAMA_CPP_PATH}/convert.py"
QUANTIZE_EXEC="${LLAMA_CPP_PATH}/quantize" # Assumes quantize is built (e.g., by running 'make' in llama.cpp dir)

# Check if MERGED_MODEL_PATH exists
if [ ! -d "${MERGED_MODEL_PATH}" ]; then
    echo "Error: Merged model path '${MERGED_MODEL_PATH}' not found."
    echo "Please ensure the path is correct and the model has been merged."
    exit 1
fi
echo "Using merged Hugging Face model from: ${MERGED_MODEL_PATH}"

# Check if LLAMA_CPP_PATH and necessary tools exist
if [ ! -d "${LLAMA_CPP_PATH}" ]; then
    echo "Error: llama.cpp directory '${LLAMA_CPP_PATH}' not found."
    echo "Please update LLAMA_CPP_PATH to your correct llama.cpp directory."
    exit 1
fi
if [ ! -f "${CONVERT_PY}" ]; then
    echo "Error: llama.cpp convert.py not found at '${CONVERT_PY}'."
    echo "Please update LLAMA_CPP_PATH."
    exit 1
fi
if [ ! -x "${QUANTIZE_EXEC}" ]; then
    echo "Error: llama.cpp quantize executable not found at '${QUANTIZE_EXEC}' or not executable."
    echo "Please ensure llama.cpp is built (run 'make' in '${LLAMA_CPP_PATH}') and LLAMA_CPP_PATH is correct."
    exit 1
fi
echo "Using llama.cpp tools from: ${LLAMA_CPP_PATH}"


# Output GGUF file (fp16 base)
GGUF_FP16_FILE="${GGUF_OUTPUT_DIR}/${MODEL_NAME}.fp16.gguf"

echo ""
echo "Step 1: Converting Hugging Face model at '${MERGED_MODEL_PATH}' to GGUF (fp16)..."
# It's generally recommended to convert to f16 first, then quantize from the f16 GGUF.
# Some models might benefit from f32, but f16 is common.
python3 "${CONVERT_PY}" "${MERGED_MODEL_PATH}" \
    --outfile "${GGUF_FP16_FILE}" \
    --outtype f16 # Output type f16

if [ $? -ne 0 ]; then
    echo "Error during GGUF fp16 conversion. Aborting."
    exit 1
fi
echo "GGUF fp16 conversion successful: ${GGUF_FP16_FILE}"


# Quantization methods to apply
# Common methods: Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0
# Q4_K_M and Q5_K_M are often good balances of size and quality.
# Add or remove types as needed.
QUANTIZATION_TYPES=("Q4_K_M" "Q5_K_M") 

echo ""
echo "Step 2: Quantizing GGUF model to various formats..."

for QTYPE in "${QUANTIZATION_TYPES[@]}"; do
    QUANTIZED_GGUF_FILE="${GGUF_OUTPUT_DIR}/${MODEL_NAME}.${QTYPE}.gguf"
    echo "  Quantizing to ${QTYPE} -> ${QUANTIZED_GGUF_FILE}"
    
    # Ensure the source fp16 file exists before attempting quantization
    if [ ! -f "${GGUF_FP16_FILE}" ]; then
        echo "  Error: Base fp16 GGUF file '${GGUF_FP16_FILE}' not found. Cannot quantize."
        continue # Skip to next type
    fi
    
    "${QUANTIZE_EXEC}" "${GGUF_FP16_FILE}" "${QUANTIZED_GGUF_FILE}" "${QTYPE}"
    
    if [ $? -ne 0 ]; then
        echo "  Error quantizing to ${QTYPE}. Output may not have been created."
    else
        echo "  Successfully quantized to ${QTYPE}: ${QUANTIZED_GGUF_FILE}"
    fi
done

echo ""
echo "--- Quantization Process Completed ---"
echo "Original fp16 GGUF model: ${GGUF_FP16_FILE}"
echo "Quantized models are in: ${GGUF_OUTPUT_DIR}"
echo "You can now use these GGUF files with llama.cpp based applications." 