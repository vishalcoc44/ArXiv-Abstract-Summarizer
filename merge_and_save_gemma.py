import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # Import PeftModel
import os

# --- Configuration ---
base_model_name = "google/gemma-3-1b-it" # Or your specific Gemma 1B variant
adapter_path = "./fine_tuned_model_gemma_colab_3_1b" # Path to your LoRA adapters
merged_model_output_path = "./merged_gemma_1b_finetuned_hf" # Directory to save the merged model

# --- Ensure output directory exists ---
os.makedirs(merged_model_output_path, exist_ok=True)

print(f"Loading base model: {base_model_name}...")
# Load base model in a precision that's easy to convert (e.g., float16 or bfloat16 if supported)
# For CPU, float32 is fine for this step, conversion script will handle it.
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32, # Using float32 for broadest compatibility in this step
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
print("Base model and tokenizer loaded.")

print(f"Loading LoRA adapters from: {adapter_path} using PeftModel...")
try:
    # Explicitly load the base model WITH the adapter config using PeftModel
    # This ensures the model is PEFT-aware from the start for these adapters
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    print("PEFT model with adapters loaded.")

    print("Merging adapters into the base model...")
    # The merge_and_unload() method is called on the PeftModel instance
    # It returns the base model with weights merged, and unloads PEFT layers
    merged_model = peft_model.merge_and_unload()
    print("Adapters merged successfully.")

    # After merging, 'merged_model' is your base model (e.g., Gemma3ForCausalLM)
    # with the LoRA weights incorporated.

except Exception as e:
    print(f"Error during adapter loading/merging with PeftModel: {e}")
    print("Please ensure your adapter_path is correct and contains a PEFT-compatible adapter (e.g., adapter_config.json, adapter_model.bin).")
    exit()

print(f"Saving merged model and tokenizer to: {merged_model_output_path}...")
merged_model.save_pretrained(merged_model_output_path) # Save the merged_model
tokenizer.save_pretrained(merged_model_output_path)

print("Merged model and tokenizer saved successfully.")
print(f"Next steps: Convert '{merged_model_output_path}' to GGUF using llama.cpp.")
