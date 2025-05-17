# -- coding: utf-8 --
import torch
import numpy as np
import os
import json
import random
import logging
import gc
from datetime import datetime, timedelta
import traceback
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb # Import for 8-bit optimizer
from google.colab import drive

# --- Set TOKENIZERS_PARALLELISM to false to avoid warning ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
print(f"Using device: {device}")

# --- START: MODIFICATIONS FOR COLAB & HUGGING FACE TOKEN ---

# --- 0. Mount Google Drive ---
drive.mount('/content/drive')
print("Google Drive mounted.")

# --- 1. Hugging Face Token ---
# Directly paste your Hugging Face token here
# For example: HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_TOKEN = ""  # <<<< PASTE YOUR TOKEN HERE

if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE" or not HF_TOKEN:
    print("Warning: Hugging Face token not set. Please paste your token in the HF_TOKEN variable.")
    # Optionally, you might want to exit or raise an error if the token is critical
    # exit()
else:
    print("Hugging Face token provided.")

# --- 2. Configure File Paths for Google Colab ---
# Ensure your files are uploaded to your Colab session's root directory
# or adjust the paths accordingly (e.g., "/content/drive/MyDrive/data/")

# Path to your abstracts file in Colab
abstracts_path = "/content/drive/MyDrive/arxiv/abstracts.json"  # <<<< ADJUST IF YOUR FILE IS IN A SUBFOLDER OF MyDrive

# Path to your FAISS index file in Colab (if you plan to use one later)
# This script currently doesn't use a FAISS file, but the path is ready.
faiss_index_path = "/content/drive/MyDrive/arxiv/faiss_index.faiss" # <<<< ADJUST IF YOUR FILE IS IN A SUBFOLDER OF MyDrive

# Base path for outputs in Colab
output_base_path = "/content/drive/MyDrive/output" # Can also be changed to /content/drive/MyDrive/output if you want outputs on Drive
output_dir = os.path.join(output_base_path, "fine_tuned_model_gemma_colab")
log_dir = os.path.join(output_base_path, "logs")

# --- END: MODIFICATIONS FOR COLAB & HUGGING FACE TOKEN ---

# Validate input paths
if not os.path.exists(abstracts_path):
    print(f"Error: Abstracts file not found at {abstracts_path}. Please upload it to Colab and check the path.")
    exit()
# If you were using the faiss_index_path, you would add a check here:
# if not os.path.exists(faiss_index_path):
#     print(f"Error: FAISS index file not found at {faiss_index_path}.")
#     exit()
print("Input paths checked.")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
print(f"Output directory will be: {output_dir}")


# Load abstracts
print("\nLoading abstracts...")
num_total_abstracts = 0
try:
    with open(abstracts_path, "r", encoding="utf-8") as f:
        all_abstracts_data_full = json.load(f)
    num_total_abstracts = len(all_abstracts_data_full)
    print(f"Loaded {num_total_abstracts} abstracts.")
except Exception as e:
    print(f"Error loading abstracts: {e}")
    exit()

# Parameters & Subsampling
num_samples_to_use = 500 # Changed to be less than 1000
max_context_len = 192;
max_answer_len = 50;
max_seq_length = 256

if num_total_abstracts > num_samples_to_use:
    print(f"Shuffling and selecting {num_samples_to_use} of {num_total_abstracts} abstracts...")
    random.seed(42)
    random.shuffle(all_abstracts_data_full)
    all_abstracts_data = all_abstracts_data_full[:num_samples_to_use]
    del all_abstracts_data_full
    gc.collect()
else:
    all_abstracts_data = all_abstracts_data_full
sample_size = len(all_abstracts_data)
print(f"Using {sample_size} abstracts for training.")
print(f"Params: max_seq_len={max_seq_length}, max_context={max_context_len}, max_answer={max_answer_len}")

# --- Data Generation ---
# !!! --- WARNING --- !!!
# This section generates simplistic data ("Summarize" -> "Start of abstract").
# This WILL likely lead to ZERO LOSS and a model that CANNOT answer general questions.
# For a useful Q&A model, you MUST replace this logic with a method that
# generates realistic Questions and Answers based on the abstract content.
# Options: Manual creation, using another LLM (via API), finding existing SciQA datasets.
# !!! --- /WARNING --- !!!
print("\nGenerating training data (Simple Summarization Task - REQUIRES REPLACEMENT FOR Q&A)...")
train_data_text = []
num_skipped_empty = 0
fixed_question = "Please summarize this abstract."
for abstract_data in all_abstracts_data:
    abstract_text = abstract_data.get("abstract", "")
    if not abstract_text:
        num_skipped_empty += 1
        continue
    context_tokens = abstract_text.split()
    # Ensure context and answer do not exceed their max lengths
    context = " ".join(context_tokens[:max_context_len])
    answer = " ".join(context_tokens[:max_answer_len]) # Simplistic answer
    formatted_string = f"### Question: {fixed_question}\n### Context: {context}\n### Answer: {answer}"
    train_data_text.append(formatted_string)

if num_skipped_empty > 0:
    print(f"Warning: Skipped {num_skipped_empty} empty abstracts.")
num_generated = len(train_data_text)
print(f"Generated {num_generated} training examples.")

if not train_data_text:
    print("Error: No training data generated. Check your abstracts and data generation logic.")
    exit()

dataset = Dataset.from_dict({"text": train_data_text}).shuffle(seed=42)
del train_data_text, all_abstracts_data
gc.collect()
torch.cuda.empty_cache()
# --- End Data Generation ---

# --- Model and Tokenizer Loading ---
print("\nLoading model and tokenizer...")
model_name = "google/gemma-3-4b-it" # Changed to 2B for potentially faster/lighter Colab use
                                  # You can change this back to "google/gemma-2-9b-it" if resources allow
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
except Exception as e:
    print(f"Error loading tokenizer (check HF_TOKEN and model_name): {e}")
    exit()

tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    print("Set pad_token to eos_token.")

# Tokenize function
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    labels = tokenized_output["input_ids"].clone()
    # Mask padding tokens for loss calculation
    labels[tokenized_output["attention_mask"] == 0] = -100
    tokenized_output["labels"] = labels
    return tokenized_output

print("\nTokenizing dataset...")
try:
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=2 # Adjust based on Colab CPU cores, 2 is usually safe
    )
    print("Tokenization complete.")
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

del dataset # Free up memory
gc.collect()
torch.cuda.empty_cache()

# Quantization and Model Loading
# quantization_config = BitsAndBytesConfig( # COMMENTED OUT
#     load_in_4bit=True, # COMMENTED OUT
#     bnb_4bit_use_double_quant=True, # COMMENTED OUT
#     bnb_4bit_quant_type="nf4", # COMMENTED OUT
#     bnb_4bit_compute_dtype=torch.float16 # COMMENTED OUT
# ) # COMMENTED OUT
print("\nLoading base model...") # MODIFIED: Removed "with 4-bit quantization"
try:
    # Determine device explicitly for loading
    effective_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Attempting to load model on: {effective_device}")

    model_kwargs = {
        "token": HF_TOKEN,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True  # Always enable this to reduce initial RAM spike
    }

    if effective_device.type == 'cuda':
        logger.info("GPU available. Using device_map='auto' with low_cpu_mem_usage=True.")
        model_kwargs["device_map"] = "auto"
    else:
        logger.info("CPU only. Using device_map='cpu' with low_cpu_mem_usage=True.")
        model_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # If on CPU, explicitly move model to CPU if not already there (Trainer also does this but good practice)
    # if effective_device.type == 'cpu':
    #     model.to(effective_device)

    print("Base model loaded successfully.")
except Exception as e:
    print(f"Error loading base model (check HF_TOKEN, model_name, and GPU availability/memory): {e}")
    traceback.print_exc()
    # exit() # COMMENTED OUT to allow further error analysis if model doesn't load

# Embedding Resize Check
print("\nChecking model embeddings...")
try:
    current_embedding_size = model.get_input_embeddings().weight.size(0)
    tokenizer_size = len(tokenizer)
    if tokenizer_size > current_embedding_size: # Check if tokenizer has more tokens (e.g. new pad_token)
        print(f"Resizing token embeddings from {current_embedding_size} to {tokenizer_size} to match tokenizer.")
        model.resize_token_embeddings(tokenizer_size)
        # Verify new embedding size matches after resize
        new_embedding_size = model.get_input_embeddings().weight.size(0)
        if new_embedding_size == tokenizer_size:
            print(f"Embedding resize successful. New size: {new_embedding_size}")
        else:
            print(f"Warning: Embedding resize attempted, but new size {new_embedding_size} does not match tokenizer size {tokenizer_size}.")
    else:
        print(f"No embedding resizing needed. Tokenizer size: {tokenizer_size}, Model embedding size: {current_embedding_size}")
except Exception as resize_err:
    print(f"Warning: Embedding check/resize failed: {resize_err}. Training will continue, but this might cause issues if new tokens were added.")

# LoRA Config and Application
print("\nConfiguring LoRA...")
logger.info("Configuring LoRA...")
# Gemma specific target modules might differ slightly, check model card if issues arise.
# Common modules for many decoder models:
target_modules_gemma = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# For some Gemma versions, only attention blocks might be targeted by default in some peft examples:
# target_modules_gemma_alt = ["q_proj", "k_proj", "v_proj", "o_proj"]

peft_config = LoraConfig(
    r=16, # Rank of LoRA matrices
    lora_alpha=32, # Alpha scaling factor
    target_modules=target_modules_gemma,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
try:
    model = get_peft_model(model, peft_config)
    print("LoRA adapter applied.")
    model.print_trainable_parameters()
except Exception as e:
    print(f"Error applying LoRA: {e}")
    exit()
# --- End Model and Tokenizer Loading ---

# --- Trainer Setup ---
class TimeEstimatorCallback(TrainerCallback):
    def _init(self): # Corrected __init_
        super()._init_() # Call parent constructor
        self.total_steps = 0
        self.start_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = datetime.now()
        self.total_steps = state.max_steps
        print(f"\n***** Train Start ***** (Total Steps: {self.total_steps})")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            elapsed = datetime.now() - self.start_time
            steps_per_second = state.global_step / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            eta_str = "N/A"
            if steps_per_second > 0:
                remaining_steps = self.total_steps - state.global_step
                eta_seconds = remaining_steps / steps_per_second if remaining_steps > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            loss_str, lr_str = "N/A", "N/A"
            if state.log_history:
                try:
                    last_log = state.log_history[-1]
                    loss = last_log.get('loss')
                    lr = last_log.get('learning_rate')
                    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
                    lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else "N/A"
                except (IndexError, TypeError): # Handle empty log or unexpected type
                    pass
            print(f"Step {state.global_step}/{self.total_steps} | Loss: {loss_str} | LR: {lr_str} | ETA: {eta_str}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.start_time is None:
            print("\n***** Train End (Error in callback start time) *****")
            return
        total_time = datetime.now() - self.start_time
        print(f"\n***** Train End ***** (Total Time: {str(total_time).split('.')[0]})")
        avg_loss_str = "N/A"
        if state.log_history:
            losses = [log.get('loss') for log in state.log_history if log and isinstance(log.get('loss'), (int, float))]
            if losses:
                avg_loss = sum(losses) / len(losses)
                avg_loss_str = f"{avg_loss:.4f}"
        print(f"Average training loss: {avg_loss_str}")


# Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1, # Keep low for initial testing
    per_device_train_batch_size=2, # Adjust based on Colab GPU VRAM (T4/P100 often need 1 or 2 with 4-bit)
    gradient_accumulation_steps=8, # Effective batch size = 2 * 8 = 16
    gradient_checkpointing=False, # Set to True to save VRAM, but slows down training.
                                  # If True, model.enable_input_require_grads() might be needed before PEFT
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir=log_dir,
    logging_strategy="steps",
    logging_steps=25, # Log more frequently for smaller datasets/epochs
    learning_rate=5e-5, # AdamW8bit often works well with slightly higher LRs
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    # optim="adamw_8bit", # COMMENTED OUT
    optim="adamw_torch",  # MODIFIED: Fallback to standard AdamW
    weight_decay=0.01,
    # fp16=True, # COMMENTED OUT
    fp16=False, # MODIFIED: Disabled mixed precision for broader compatibility (CPU/non-ideal GPU)
    bf16=False, # Set to True if using Ampere GPUs and want to try bfloat16
    report_to="none", # "tensorboard" or "wandb" if you have them set up
    dataloader_num_workers=2, # Usually 2 is fine for Colab
    remove_unused_columns=True,
    disable_tqdm=False,
)

# Custom optimizer and scheduler (as in your original script, slightly simplified)
# Note: The Trainer usually handles optimizer creation based on optim arg.
# This custom setup is for more fine-grained control if needed.
# For adamw_8bit, the Trainer should handle it correctly.
# However, if you face issues, this explicit setup can be useful.

def create_custom_optimizer_and_scheduler(model, args, num_training_steps):
    from transformers.trainer_pt_utils import get_parameter_names
    from torch.optim import AdamW # Fallback
    
    print(f"Optimizer type from args: {args.optim}")
    print(f"Configured LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = None
    # if args.optim == "adamw_8bit": # MODIFIED: Conditional logic for optimizer type
    if args.optim == "adamw_8bit" and device == "cuda": # Only attempt 8bit if explicitly requested AND on CUDA
        try:
            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), # Use TrainingArguments defaults
                eps=args.adam_epsilon,
                # weight_decay is handled by parameter groups
            )
            print(f"Using bnb.optim.AdamW8bit with LR: {args.learning_rate}")
        except Exception as e:
            print(f"Failed to init AdamW8bit: {e}. Falling back to AdamW.")
            args.optim = "adamw_torch" # Fallback

    if optimizer is None: # Fallback if adamw_8bit failed or was not specified
         optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
         print(f"Using torch.optim.AdamW with LR: {args.learning_rate}")


    from transformers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps,
    )
    # The min_lr wrapping from your script can be added here if desired
    # original_get_lr = lr_scheduler.get_lr
    # def get_lr_with_min():
    #     lrs = original_get_lr()
    #     return [max(lr, 1e-7) for lr in lrs] # Slightly lower min_lr
    # lr_scheduler.get_lr = get_lr_with_min
    print(f"Created {args.lr_scheduler_type} scheduler with {args.get_warmup_steps(num_training_steps)} warmup steps.")
    return optimizer, lr_scheduler

# Calculate number of training steps
# Ensure model is defined before this step, or handle NameError if model loading failed
train_dataset_size = len(tokenized_dataset)
total_train_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
if train_dataset_size % total_train_batch_size == 0:
    num_update_steps_per_epoch = train_dataset_size // total_train_batch_size
else:
    num_update_steps_per_epoch = (train_dataset_size // total_train_batch_size) + 1
num_training_steps = num_update_steps_per_epoch * training_args.num_train_epochs
print(f"Dataset size: {train_dataset_size}, Effective batch size: {total_train_batch_size}")
print(f"Calculated training steps: {num_training_steps} (steps_per_epoch: {num_update_steps_per_epoch})")


# Initialize Trainer
# The Trainer should ideally create the optimizer and scheduler if optim is set.
# If you still want to use the custom one:
try:
    # Ensure model is defined before attempting to create optimizers with it
    if 'model' not in locals() or model is None:
        raise NameError("Model was not loaded successfully. Cannot create optimizers.")
    custom_optimizers = create_custom_optimizer_and_scheduler(model, training_args, num_training_steps)
except NameError as ne:
    print(f"Error during optimizer creation: {ne}")
    # Handle the case where model isn't defined, perhaps by exiting or skipping training
    custom_optimizers = (None, None) # Set to None to avoid further errors if Trainer can handle it
    print("Skipping custom optimizer creation due to model loading failure.")

trainer = Trainer(
    model=model if 'model' in locals() else None, # Pass model only if defined
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    optimizers=custom_optimizers, # Pass the custom optimizer and scheduler
    callbacks=[TimeEstimatorCallback()]
)
# --- End Trainer Setup ---

# --- Training ---
print("\n--- Starting Fine-Tuning ---")
logger.info(f"Starting fine-tuning: {training_args.num_train_epochs} epoch(s), {num_generated} samples.")
try:
    if hasattr(model, 'enable_input_require_grads') and training_args.gradient_checkpointing:
        model.enable_input_require_grads() # Needed for gradient checkpointing with PEFT

    train_result = trainer.train()
    logger.info("Training finished successfully.")
    print("\nTraining finished. Saving final model adapter and metrics...")
    
    # Save the LoRA adapter
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_adapter_path) # Saves only the adapter
    tokenizer.save_pretrained(final_adapter_path) # Save tokenizer with adapter for easy loading
    print(f"LoRA adapter saved to {final_adapter_path}")

    # Save full model if needed (will be larger)
    # final_model_path = os.path.join(output_dir, "final_full_model")
    # trainer.save_model(final_model_path) # This saves the merged model if possible, or base + adapter
    # print(f"Full model (or base + adapter) saved to {final_model_path}")


    metrics = train_result.metrics
    metrics["train_samples"] = num_generated
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(f"Metrics and trainer state saved to {output_dir}")

except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    print("\n--- Training Failed ---")
    print(traceback.format_exc())
    print(f"Error: {e}")
    print("Attempting to save partial state...")
    logger.info("Attempting error state save...")
    try:
        error_save_dir = os.path.join(output_dir, "error_state_adapter")
        os.makedirs(error_save_dir, exist_ok=True)
        if hasattr(model, 'save_pretrained'): # PEFT model
            model.save_pretrained(error_save_dir)
            tokenizer.save_pretrained(error_save_dir)
        else: # Full model
            trainer.save_model(error_save_dir)
        trainer.save_state() # Save trainer state regardless
        print(f"Partial model adapter/state saved to {error_save_dir} after error.")
    except Exception as save_e:
        print(f"Could not save model/state after error: {save_e}")
finally:
    print("\n--- Cleaning Up ---")
    logger.info("Cleaning up resources...")
    try:
        del model
        del trainer
        del tokenized_dataset
        # custom_optimizers might hold references, clear them
        if 'custom_optimizers' in locals(): del custom_optimizers

    except NameError: # In case any variable was not defined due to an earlier exit
        pass
    except Exception as e:
        logger.warning(f"Exception during variable deletion: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")
    print("Cleanup complete.")
    print("Script finished.")
