"""
This script performs quantization on the Qwen1.5-4B-Chat model using LLMCompressor.
It involves model initialization, preparation of calibration data, and application of 
quantization techniques (SmoothQuant and GPTQ). The quantized model is saved at the end.
"""

# Import necessary libraries
from vllm import LLM, SamplingParams
from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Model configuration
MODEL_ID = "./Qwen1.5-4B-Chat"

"""
Step 1: Create the model and tokenizer
"""
# Load the sparse version of the model with automatic device mapping and data type
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""
Step 2: Prepare calibration data
"""
# Configuration for calibration data
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 7000

# Load the dataset and preprocess
# Using the UltraChat dataset for calibration
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Shuffle and select the required number of samples for calibration
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# Preprocess the dataset: apply chat templates
def preprocess(example):
    """
    Applies the chat template to the input messages and prepares the text for tokenization.
    """
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

# Apply the preprocessing function to the dataset
ds = ds.map(preprocess)

# Tokenize the processed text
def tokenize(sample):
    """
    Tokenizes the input text with a maximum sequence length and without padding.
    """
    return tokenizer(
        sample["text"], 
        padding=False, 
        max_length=MAX_SEQUENCE_LENGTH, 
        truncation=True, 
        add_special_tokens=False
    )

# Tokenize the dataset and remove original columns
ds = ds.map(tokenize, remove_columns=ds.column_names)

"""
Step 3: Apply quantization
"""
# Define the quantization recipe
# SmoothQuant modifies the weight matrices with a smoothing factor
# GPTQ quantizes linear layers to W8A8 format, ignoring certain layers
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Perform one-shot quantization
# The quantization process involves passing the calibration dataset through the model
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the quantized model and tokenizer
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Quantized model saved to: {SAVE_DIR}")
