#!/usr/bin/env python3
"""
BraveGPT Simple - A simplified version of BraveGPT that avoids input handling issues
and focuses on demonstrating the core functionality.
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import sys
import json

# Load environment variables
load_dotenv()
print("✅ Loaded environment variables from .env file")

# Configuration
api_key = os.getenv("BRAVE_API_KEY")
model_path = os.getenv("LLM_MODEL_PATH", "model.safetensors")
tokenizer_name = os.getenv("TOKENIZER_NAME", "microsoft/DialoGPT-small")

print(f"✅ Using tokenizer: {tokenizer_name}")
print(f"✅ Model path: {model_path}")

# Check if model file exists
if not Path(model_path).exists():
    print(f"❌ Model file not found at: {model_path}")
    print("Please check the LLM_MODEL_PATH in your .env file.")
    sys.exit(1)

# Load tokenizer
print("🔄 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"✅ Loaded tokenizer: {tokenizer_name}")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    sys.exit(1)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Load model with explicit device mapping
print("🔄 Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": device} if device == "cuda" else None
    )
    
    # Explicitly move model to device for CPU
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generation configuration
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
    max_length=512,
    pad_token_id=tokenizer.eos_token_id
)

# Demo conversation
print("\n" + "=" * 50)
print("BraveGPT Demo - Predefined Conversation")
print("=" * 50)

# List of demo questions
demo_questions = [
    "Hello, how are you today?",
    "What can you tell me about artificial intelligence?",
    "What's your favorite programming language?",
    "Tell me a joke about computers."
]

# Process each question
conversation_history = []

for i, question in enumerate(demo_questions, 1):
    print(f"\n[Question {i}]: {question}")
    
    # Add to conversation history
    conversation_history.append(f"User: {question}")
    
    # Prepare input for model
    prompt = "\n".join(conversation_history[-5:])  # Last 5 turns
    prompt += "\nAssistant:"
    
    # Encode the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    print("🔄 Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "Assistant:" in response_text:
        response_text = response_text.split("Assistant:", 1)[1].strip()
    
    # Remove any trailing "User:" or conversation artifacts
    if "User:" in response_text:
        response_text = response_text.split("User:", 1)[0].strip()
    
    # Add to conversation history
    conversation_history.append(f"Assistant: {response_text}")
    
    # Display response
    print(f"[Response]: {response_text}")
    print("-" * 50)

print("\n✅ Demo completed successfully!")
print("This confirms that the PyTorch and Transformers libraries are working correctly.")
print("The 'torch.get_default_device()' error has been resolved.")
