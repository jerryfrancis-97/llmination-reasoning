"""
Factory for creating API clients.
"""

from typing import Dict, Any
from .base_client import BaseAPIClient
from .groq_client import GroqClient
from .gemini_client import GeminiClient
from .mistral_client import MistralClient
from ..config.api_config import APIConfig

def create_api_client(api_type: str, api_key: str = None, **kwargs) -> BaseAPIClient:
    """
    Create an API client based on the specified type.
    
    Args:
        api_type: Type of API ('groq', 'gemini', 'mistral')
        api_key: API key for the service
        **kwargs: Additional arguments to pass to the client
        
    Returns:
        Configured API client instance
        
    Raises:
        ValueError: If api_type is not supported or api_key is invalid
    """
    # Get API key from config if not provided
    if not api_key:
        if api_type == "groq":
            api_key = APIConfig.GROQ_API_KEY
        elif api_type == "gemini":
            api_key = APIConfig.GEMINI_API_KEY
        elif api_type == "mistral":
            api_key = APIConfig.MISTRAL_API_KEY
        else:
            raise ValueError(f"Unknown API type: {api_type}")
    
    # Validate API key
    if not api_key:
        raise ValueError(f"No API key provided for {api_type}")
    
    if not APIConfig.validate_api_key(api_type, api_key):
        raise ValueError(f"Invalid API key for {api_type}")
    
    # Create appropriate client
    if api_type.lower() == "groq":
        return GroqClient(api_key, **kwargs)
    elif api_type.lower() == "gemini":
        return GeminiClient(api_key, **kwargs)
    elif api_type.lower() == "mistral":
        return MistralClient(api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def get_available_apis() -> list[str]:
    """Get list of available API types."""
    return ["groq", "gemini", "mistral"]

def get_api_info(api_type: str) -> Dict[str, Any]:
    """
    Get information about a specific API.
    
    Args:
        api_type: Type of API
        
    Returns:
        Dictionary with API information
    """
    if api_type == "groq":
        return {
            "name": "Groq",
            "models": APIConfig.GROQ_MODELS,
            "key_format": "gsk_...",
            "description": "Fast inference API for open models"
        }
    elif api_type == "gemini":
        return {
            "name": "Google Gemini",
            "models": "Dynamic (API call required)",
            "key_format": "AIza...",
            "description": "Google's multimodal AI model"
        }
    elif api_type == "mistral":
        return {
            "name": "Mistral AI",
            "models": APIConfig.MISTRAL_MODELS,
            "key_format": "Any format",
            "description": "Mistral's large language models"
        }
    else:
        return {"error": f"Unknown API type: {api_type}"}
