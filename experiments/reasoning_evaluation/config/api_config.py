"""
API configuration and constants for the reasoning evaluation framework.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")

class APIConfig:
    """Configuration for API clients and settings."""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    
    # Available Models
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant", 
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "qwen-qwq-32b",
        "deepseek-r1-distill-llama-70b"
    ]
    
    MISTRAL_MODELS = [
        "mistral-large-latest"
    ]
    
    # Default Settings
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_RETRY_DELAY = 5
    DEFAULT_MAX_RETRIES = 3
    
    # Gemini-specific settings
    GEMINI_SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    
    GEMINI_GENERATION_CONFIG = {
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": DEFAULT_MAX_TOKENS,
    }
    
    @classmethod
    def validate_api_key(cls, api_type: str, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
            
        if api_type == "groq" and not api_key.startswith("gsk_"):
            return False
            
        # Add other API key validations as needed
        return True
    
    @classmethod
    def get_available_models(cls, api_type: str) -> List[str]:
        """Get available models for a specific API."""
        if api_type == "groq":
            return cls.GROQ_MODELS
        elif api_type == "mistral":
            return cls.MISTRAL_MODELS
        elif api_type == "gemini":
            # Gemini models are dynamic, return empty list
            return []
        else:
            return []
    
    @classmethod
    def validate_model(cls, api_type: str, model: str) -> bool:
        """Validate if a model is available for a specific API."""
        available_models = cls.get_available_models(api_type)
        return model in available_models
