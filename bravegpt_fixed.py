#!/usr/bin/env python3
"""
BraveGPT Fixed - A fixed version of BraveGPT with resolved PyTorch compatibility
and improved input handling.
"""

import requests
import json
import os
import torch
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)
from safetensors.torch import load_file
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# ANSI colors for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    snippet: str
    relevance_score: float = 0.0

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
            print(f"{Colors.RED}Search error: {e}{Colors.END}")
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
        
        # Conversation history
        self.conversation_history = []
        
    def _load_model_and_tokenizer(self, tokenizer_name: str = None):
        """Load model from safetensors and tokenizer"""
        print(f"{Colors.BLUE}🤖 Loading conversational model...{Colors.END}")
        
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
                        print(f"{Colors.GREEN}✅ Loaded tokenizer: {model_name}{Colors.END}")
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
                        print(f"{Colors.GREEN}✅ Loaded {len(filtered_weights)} custom weights{Colors.END}")
                    else:
                        print(f"{Colors.YELLOW}⚠️ No compatible weights found, using base model{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}⚠️ Model file not found, using base model{Colors.END}")
                
            except Exception as e:
                print(f"{Colors.YELLOW}⚠️ Error loading custom model: {e}{Colors.END}")
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
            
            print(f"{Colors.GREEN}✅ Model loaded on {self.device}{Colors.END}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def generate_response(self, user_input: str, context: str = None, max_length: int = 512):
        """Generate conversational response"""
        try:
            # Add user input to conversation history
            self.conversation_history.append(f"User: {user_input}")
            
            # Prepare input for model
            if context:
                prompt = f"Context: {context}\n\nConversation:\n"
                prompt += "\n".join(self.conversation_history[-5:])  # Last 5 turns
                prompt += "\nAssistant:"
            else:
                prompt = "\n".join(self.conversation_history[-5:])  # Last 5 turns
                prompt += "\nAssistant:"
            
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
            response = self._clean_response(response)
            
            # Add to conversation history
            self.conversation_history.append(f"Assistant: {response}")
            
            return response
            
        except Exception as e:
            print(f"{Colors.RED}Error generating response: {e}{Colors.END}")
            return f"I'm sorry, I encountered an error: {e}"
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Extract only the assistant's part of the response
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        
        # Remove any trailing "User:" or conversation artifacts
        if "User:" in response:
            response = response.split("User:", 1)[0].strip()
        
        # Clean up any other artifacts
        response = re.sub(r'<[^>]+>', '', response)  # Remove HTML-like tags
        
        return response.strip()
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

class ChatGPTLikeBraveGPT:
    def __init__(self, brave_api_key: str, model_path: str, tokenizer_name: str = None):
        self.brave_search = BraveSearchAPI(brave_api_key)
        
        # Load conversational model
        self.llm = ConversationalLLM(model_path, tokenizer_name)
        
        # Conversation context and search cache
        self.conversation_context = []
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
    
    def chat_response(self, user_input: str):
        """Generate ChatGPT-like response with real-time data"""
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
                print(f"{Colors.BLUE}Using cached search results for: '{search_query}'{Colors.END}")
            else:
                # Perform search
                print(f"{Colors.BLUE}Searching for: '{search_query}'...{Colors.END}")
                search_results = self.brave_search.search(search_query, count=7)
                
                # Cache results
                self.search_cache[search_query] = search_results
                
                # Limit cache size (simple LRU)
                if len(self.search_cache) > 10:
                    self.search_cache.pop(next(iter(self.search_cache)))
        
        # Format context from search results if available
        context = None
        if search_results:
            context = self._format_search_context(search_results, search_query)
        
        # Generate response
        response = self.llm.generate_response(user_input, context=context)
        
        # Determine confidence level based on search results
        confidence = "high"
        if should_search and not search_results:
            confidence = "low"
        elif should_search and len(search_results) < 3:
            confidence = "medium"
        
        # Format sources for display
        sources = []
        if search_results:
            for result in search_results[:5]:
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet
                })
        
        return {
            "response": response,
            "used_search": should_search,
            "search_query": search_query if should_search else None,
            "sources": sources,
            "confidence": confidence
        }
    
    def interactive_chat(self):
        """ChatGPT-like interactive interface with simplified input handling"""
        
        # Welcome message
        print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}🤖 ChatGPT-like BraveGPT{Colors.END}")
        print(f"{Colors.BLUE}Conversational AI with real-time web data{Colors.END}")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")
        print(f"\n{Colors.BOLD}Commands:{Colors.END}")
        print(f"• Type naturally - I'll search for current info when needed")
        print(f"• 'reset' - Clear conversation history")
        print(f"• 'quit' - Exit chat")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.END}\n")
        
        # Simple input loop with robust handling
        while True:
            try:
                # Print prompt and flush to ensure it's displayed
                print(f"\n{Colors.BOLD}{Colors.GREEN}You:{Colors.END} ", end="")
                sys.stdout.flush()
                
                # Read input directly from stdin
                user_input = sys.stdin.readline().strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"\n{Colors.BLUE}👋 Goodbye! Thanks for chatting!{Colors.END}")
                    break
                
                elif user_input.lower() == 'reset':
                    self.llm.reset_conversation()
                    self.conversation_context = []
                    self.search_cache = {}
                    print(f"{Colors.GREEN}✅ Conversation reset!{Colors.END}")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate response with error handling
                try:
                    print(f"{Colors.BLUE}🔄 Generating response...{Colors.END}")
                    result = self.chat_response(user_input)
                    
                    # Display response
                    response_style = Colors.BLUE if result["confidence"] == "high" else Colors.YELLOW if result["confidence"] == "medium" else Colors.RED
                    
                    print(f"\n{response_style}🤖 Assistant ({result['confidence']} confidence):{Colors.END}")
                    print(f"{response_style}{result['response']}{Colors.END}")
                    
                    # Show sources if search was used
                    if result["used_search"] and result["sources"]:
                        print(f"\n{Colors.BLUE}📚 Sources (searched: '{result['search_query']}'):{Colors.END}")
                        for i, source in enumerate(result["sources"], 1):
                            print(f"{Colors.BLUE}{i}. {source['title']}{Colors.END}")
                            print(f"   {source['snippet']}")
                            print(f"   {Colors.UNDERLINE}🔗 {source['url']}{Colors.END}\n")
                
                except Exception as e:
                    print(f"{Colors.RED}❌ Error generating response: {e}{Colors.END}")
                    import traceback
                    print(f"{Colors.RED}{traceback.format_exc()}{Colors.END}")
            
            except KeyboardInterrupt:
                print(f"\n\n{Colors.BLUE}👋 Goodbye! Thanks for chatting!{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.END}")
                import traceback
                print(f"{Colors.RED}{traceback.format_exc()}{Colors.END}")
                print(f"{Colors.YELLOW}Continuing to next interaction...{Colors.END}")

def main():
    """Main function to start ChatGPT-like BraveGPT"""
    
    # Load environment variables from .env file
    load_dotenv()
    print(f"{Colors.GREEN}✅ Loaded environment variables from .env file{Colors.END}")
    
    # Get configuration
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        print(f"{Colors.RED}❌ Brave Search API Key not found in environment.{Colors.END}")
        print(f"{Colors.YELLOW}Please add BRAVE_API_KEY to your .env file and try again.{Colors.END}")
        return
    
    # Get model path
    model_path = os.getenv("LLM_MODEL_PATH", "model.safetensors")
    if not Path(model_path).exists():
        print(f"{Colors.RED}❌ Model file not found at: {model_path}{Colors.END}")
        print(f"{Colors.YELLOW}Please check the LLM_MODEL_PATH in your .env file.{Colors.END}")
        return
    
    # Get tokenizer name from environment
    tokenizer_name = os.getenv("TOKENIZER_NAME")
    print(f"{Colors.GREEN}✅ Using tokenizer: {tokenizer_name or 'auto-detection'}{Colors.END}")
    
    # Initialize and start chat
    try:
        chat_gpt = ChatGPTLikeBraveGPT(api_key, model_path, tokenizer_name)
        chat_gpt.interactive_chat()
    except Exception as e:
        print(f"{Colors.RED}❌ Error initializing ChatGPT-like BraveGPT: {e}{Colors.END}")
        # Print more detailed error information
        import traceback
        print(f"{Colors.RED}{traceback.format_exc()}{Colors.END}")

if __name__ == "__main__":
    main()
