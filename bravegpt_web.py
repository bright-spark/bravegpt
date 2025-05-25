#!/usr/bin/env python3
"""
BraveGPT Web - A web interface for BraveGPT with improved prompting,
conversation saving, and a Gradio UI.
"""

import requests
import json
import os
import torch
import sys
import time
import datetime
import re
import gradio as gr
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pickle
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)
from safetensors.torch import load_file
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Constants
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant that provides helpful, accurate, and thoughtful responses.
You have access to real-time web search results to provide up-to-date information.
When answering, consider the search results provided but use your own knowledge when appropriate.
Be concise but thorough, and always prioritize accuracy over speculation.
If you don't know something or if the search results don't provide relevant information, be honest about it.
"""

@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    snippet: str
    relevance_score: float = 0.0

@dataclass
class Message:
    role: str  # "system", "user", "assistant", or "search"
    content: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class Conversation:
    id: str
    title: str
    messages: List[Message]
    created_at: float
    updated_at: float
    
    @classmethod
    def new(cls, title: str = None):
        """Create a new conversation"""
        conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not title:
            title = f"Conversation {conv_id}"
        
        current_time = time.time()
        
        # Add system message
        messages = [Message(role="system", content=DEFAULT_SYSTEM_PROMPT, timestamp=current_time)]
        
        return cls(
            id=conv_id,
            title=title,
            messages=messages,
            created_at=current_time,
            updated_at=current_time
        )
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the conversation"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = message.timestamp
        return message
    
    def save(self):
        """Save conversation to disk"""
        file_path = CONVERSATIONS_DIR / f"{self.id}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        return file_path
    
    @classmethod
    def load(cls, conv_id: str):
        """Load conversation from disk"""
        file_path = CONVERSATIONS_DIR / f"{conv_id}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Conversation {conv_id} not found")
        
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    @classmethod
    def list_all(cls):
        """List all saved conversations"""
        conversations = []
        for file_path in CONVERSATIONS_DIR.glob("*.pkl"):
            try:
                with open(file_path, "rb") as f:
                    conv = pickle.load(f)
                    conversations.append(conv)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Sort by updated_at (most recent first)
        conversations.sort(key=lambda x: x.updated_at, reverse=True)
        return conversations

class BraveSearchAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
    
    def search(self, query: str, count: int = 10, freshness: str = "pd") -> List[SearchResult]:
        """Enhanced search with better result processing"""
        params = {
            "q": query,
            "count": min(count, 20),
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate",
            "freshness": freshness,
            "text_decorations": False,
            "spellcheck": True
        }
        
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"]:
                    # Calculate relevance score
                    relevance = self._calculate_relevance(query, result)
                    
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("description", ""),
                        snippet=result.get("snippet", ""),
                        relevance_score=relevance
                    )
                    results.append(search_result)
            
            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _calculate_relevance(self, query: str, result: Dict) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        query_words = set(query.lower().split())
        
        # Title relevance (higher weight)
        title_words = set(result.get("title", "").lower().split())
        title_overlap = len(query_words.intersection(title_words))
        score += title_overlap * 3
        
        # Description relevance
        desc_words = set(result.get("description", "").lower().split())
        desc_overlap = len(query_words.intersection(desc_words))
        score += desc_overlap * 2
        
        # URL relevance (domain quality)
        url = result.get("url", "").lower()
        if any(domain in url for domain in ["wikipedia", "github", "stackoverflow", "docs."]):
            score += 2
        
        return score

class ConversationalLLM:
    def __init__(self, model_path: str, tokenizer_name: str = None):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self._load_model_and_tokenizer(tokenizer_name)
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            max_length=2048,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
    def _load_model_and_tokenizer(self, tokenizer_name: str = None):
        """Load model from safetensors and tokenizer"""
        print("🤖 Loading conversational model...")
        
        try:
            # Try to determine model type from path or use common models
            possible_models = [
                tokenizer_name,
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small", 
                "facebook/blenderbot-400M-distill",
                "microsoft/DialoGPT-large"
            ]
            
            tokenizer_loaded = False
            for model_name in possible_models:
                if model_name:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        print(f"✅ Loaded tokenizer: {model_name}")
                        tokenizer_loaded = True
                        break
                    except Exception as e:
                        continue
            
            if not tokenizer_loaded:
                raise Exception("Could not load any compatible tokenizer")
            
            # Load model architecture (try common architectures)
            try:
                # Try DialoGPT first (most common for conversation)
                # Set device_map to avoid using get_default_device
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/DialoGPT-medium", 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map={"":self.device} if self.device == "cuda" else None
                )
                
                # Load your custom weights
                if self.model_path.exists():
                    custom_weights = load_file(str(self.model_path))
                    # Filter weights that match model architecture
                    filtered_weights = {}
                    model_keys = set(self.model.state_dict().keys())
                    
                    for key, value in custom_weights.items():
                        if key in model_keys:
                            if self.model.state_dict()[key].shape == value.shape:
                                filtered_weights[key] = value
                    
                    # Load compatible weights
                    if filtered_weights:
                        self.model.load_state_dict(filtered_weights, strict=False)
                        print(f"✅ Loaded {len(filtered_weights)} custom weights")
                    else:
                        print("⚠️ No compatible weights found, using base model")
                else:
                    print("⚠️ Model file not found, using base model")
                
            except Exception as e:
                print(f"⚠️ Error loading custom model: {e}")
                # Fallback to base model with explicit device mapping
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/DialoGPT-small",
                    device_map={"":self.device} if self.device == "cuda" else None
                )
            
            # Explicitly move model to device instead of relying on device_map for CPU
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def generate_response(self, conversation: Conversation, max_length: int = 512) -> str:
        """Generate response based on conversation history with improved prompting"""
        try:
            # Format conversation history for the model
            prompt = self._format_conversation_for_model(conversation)
            
            # Encode the input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_length,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant's response
            response = self._extract_assistant_response(response, prompt)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error: {e}"
    
    def _format_conversation_for_model(self, conversation: Conversation) -> str:
        """Format conversation history for improved prompting"""
        # Start with system prompt
        system_message = next((msg for msg in conversation.messages if msg.role == "system"), None)
        prompt = f"""System: {system_message.content if system_message else DEFAULT_SYSTEM_PROMPT}

"""
        
        # Add conversation history (excluding system messages)
        for msg in conversation.messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
            elif msg.role == "search":
                prompt += f"Search Results: {msg.content}\n"
        
        # Add final assistant prompt
        prompt += "Assistant:"
        
        return prompt
    
    def _extract_assistant_response(self, full_response: str, prompt: str) -> str:
        """Extract only the assistant's response from the model output"""
        # Remove the prompt portion
        if prompt in full_response:
            response = full_response[len(prompt):].strip()
        else:
            # If prompt isn't found exactly, try to extract after the last "Assistant:" occurrence
            parts = full_response.split("Assistant:")
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response
        
        # Remove any trailing "User:" or conversation artifacts
        if "User:" in response:
            response = response.split("User:", 1)[0].strip()
        
        # Clean up any other artifacts
        response = re.sub(r'<[^>]+>', '', response)  # Remove HTML-like tags
        
        return response.strip()

# Add this code to the end of your bravegpt_web.py file

class BraveGPT:
    def __init__(self, brave_api_key: str, model_path: str, tokenizer_name: str = None):
        self.brave_search = BraveSearchAPI(brave_api_key)
        self.llm = ConversationalLLM(model_path, tokenizer_name)
        self.search_cache = {}
    
    def _should_search(self, user_input: str) -> bool:
        """Determine if we should search for current information"""
        # Keywords that suggest need for current information
        current_info_keywords = [
            "latest", "current", "recent", "today", "news", 
            "update", "now", "happening", "trend", "weather",
            "price", "stock", "market", "event", "release"
        ]
        
        # Check for time-related questions
        time_patterns = [
            r"what.*time", r"current.*time", r"today", r"yesterday",
            r"this week", r"this month", r"this year", r"202[0-9]"
        ]
        
        # Check for keywords
        if any(keyword in user_input.lower() for keyword in current_info_keywords):
            return True
        
        # Check for time patterns
        if any(re.search(pattern, user_input.lower()) for pattern in time_patterns):
            return True
            
        return False
    
    def _extract_search_query(self, user_input: str) -> str:
        """Extract and optimize search query from user input"""
        # Remove question words and common phrases
        query = user_input.lower()
        query = re.sub(r'^(what|who|when|where|why|how|can you|could you|please|tell me|find|search for|look up)', '', query, flags=re.IGNORECASE).strip()
        query = re.sub(r'\?|\.|!|,|;|:|"|\(|\)|\[|\]|\'', '', query).strip()
        
        # Remove filler words
        filler_words = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "and", "or", "but", "in", "on", "at"]
        query_words = [word for word in query.split() if word.lower() not in filler_words]
        
        # Limit to first 7 words for more focused search
        return " ".join(query_words[:7])
    
    def _format_search_context(self, search_results: List[SearchResult], query: str) -> str:
        """Format search results into context for the LLM"""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        context = f"Current date: {current_date}\n\n"
        context += f"Search results for: '{query}'\n\n"
        
        # Add top search results
        for i, result in enumerate(search_results[:5], 1):
            context += f"{i}. {result.title}\n"
            context += f"   URL: {result.url}\n"
            context += f"   {result.snippet}\n\n"
        
        return context
    
    def process_message(self, conversation: Conversation, user_input: str) -> Tuple[str, bool, str, List[Dict]]:
        """Process user message and generate response with search if needed"""
        # Add user message to conversation
        conversation.add_message("user", user_input)
        
        # Check if we should search for current information
        should_search = self._should_search(user_input)
        search_results = []
        search_query = ""
        
        if should_search:
            # Extract search query
            search_query = self._extract_search_query(user_input)
            
            # Check cache first
            if search_query in self.search_cache:
                search_results = self.search_cache[search_query]
                print(f"Using cached search results for: '{search_query}'")
            else:
                # Perform search
                print(f"Searching for: '{search_query}'...")
                search_results = self.brave_search.search(search_query, count=7)
                
                # Cache results
                self.search_cache[search_query] = search_results
                
                # Limit cache size (simple LRU)
                if len(self.search_cache) > 10:
                    self.search_cache.pop(next(iter(self.search_cache)))
        
        # Format context from search results if available
        if search_results:
            search_context = self._format_search_context(search_results, search_query)
            # Add search results to conversation
            conversation.add_message("search", search_context)
        
        # Generate response
        response = self.llm.generate_response(conversation, max_length=512)
        
        # Add assistant response to conversation
        conversation.add_message("assistant", response)
        
        # Save conversation
        conversation.save()
        
        # Format sources for display
        sources = []
        if search_results:
            for result in search_results[:5]:
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet
                })
        
        return response, should_search, search_query, sources

# Gradio Web UI Implementation
def create_web_ui():
    # Load environment variables
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not found in environment variables")
    
    model_path = os.getenv("LLM_MODEL_PATH", "model.safetensors")
    if not Path(model_path).exists():
        print(f"Warning: Model file not found at: {model_path}. Using default model.")
    
    tokenizer_name = os.getenv("TOKENIZER_NAME", "microsoft/DialoGPT-small")
    
    # Initialize BraveGPT
    bravegpt = BraveGPT(api_key, model_path, tokenizer_name)
    
    # Current conversation
    current_conversation = Conversation.new("New Conversation")
    
    # List saved conversations
    saved_conversations = Conversation.list_all()
    
    def chat(message, history, conversation_id):
        nonlocal current_conversation
        
        # If conversation_id changed, load that conversation
        if conversation_id and conversation_id != current_conversation.id:
            try:
                current_conversation = Conversation.load(conversation_id)
            except Exception as e:
                return f"Error loading conversation: {e}", history
        
        # Process message
        response, used_search, search_query, sources = bravegpt.process_message(current_conversation, message)
        
        # Format sources if available
        if used_search and sources:
            source_text = "\n\n**Sources:**\n"
            for i, source in enumerate(sources, 1):
                source_text += f"{i}. [{source['title']}]({source['url']})\n"
            response += source_text
        
        return response, history + [(message, response)]
    
    def create_new_conversation():
        nonlocal current_conversation
        current_conversation = Conversation.new("New Conversation")
        return current_conversation.id, gr.Dropdown(choices=get_conversation_choices(), value=current_conversation.id), []
    
    def get_conversation_choices():
        conversations = Conversation.list_all()
        return [gr.Dropdown.update(choices=[(c.id, c.title) for c in conversations], value=current_conversation.id)]
    
    def load_conversation(conversation_id):
        nonlocal current_conversation
        if not conversation_id:
            return [], current_conversation.id
        
        try:
            current_conversation = Conversation.load(conversation_id)
            # Convert conversation to chat history format
            history = []
            user_msg = None
            
            for msg in current_conversation.messages:
                if msg.role == "user":
                    user_msg = msg.content
                elif msg.role == "assistant" and user_msg:
                    history.append((user_msg, msg.content))
                    user_msg = None
            
            return history, current_conversation.id
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return [], current_conversation.id
    
    def refresh_conversations():
        return gr.Dropdown(choices=[(c.id, c.title) for c in Conversation.list_all()], value=current_conversation.id)
    
    def update_conversation_title(new_title):
        nonlocal current_conversation
        if not new_title.strip():
            return gr.Textbox(value=current_conversation.title)
        
        current_conversation.title = new_title
        current_conversation.save()
        return gr.Dropdown(choices=[(c.id, c.title) for c in Conversation.list_all()], value=current_conversation.id)
    
    # Create Gradio interface
    with gr.Blocks(title="BraveGPT Web") as demo:
        gr.Markdown("# 🤖 BraveGPT Web")
        gr.Markdown("A conversational AI with real-time web search capabilities")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
                    submit_btn = gr.Button("Send")
            
            with gr.Column(scale=1):
                gr.Markdown("### Conversation")
                conversation_title = gr.Textbox(label="Title", value=current_conversation.title)
                update_title_btn = gr.Button("Update Title")
                
                gr.Markdown("### Saved Conversations")
                conversation_dropdown = gr.Dropdown(
                    [(c.id, c.title) for c in saved_conversations],
                    label="Select Conversation",
                    value=current_conversation.id if saved_conversations else None
                )
                
                with gr.Row():
                    load_btn = gr.Button("Load")
                    new_btn = gr.Button("New")
                    refresh_btn = gr.Button("Refresh")
        
        # Hidden state
        conversation_id = gr.State(current_conversation.id)
        
        # Set up event handlers
        submit_btn.click(chat, [msg, chatbot, conversation_id], [msg, chatbot])
        msg.submit(chat, [msg, chatbot, conversation_id], [msg, chatbot])
        new_btn.click(create_new_conversation, [], [conversation_id, conversation_dropdown, chatbot])
        load_btn.click(load_conversation, [conversation_dropdown], [chatbot, conversation_id])
        refresh_btn.click(refresh_conversations, [], [conversation_dropdown])
        update_title_btn.click(update_conversation_title, [conversation_title], [conversation_dropdown])
    
    return demo

def main():
    # Create and launch the web UI
    demo = create_web_ui()
    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    main()