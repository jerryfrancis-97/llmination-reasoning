"""
Groq API client implementation.
"""

from typing import Dict, Any
from groq import Groq
from .base_client import BaseAPIClient
from ..config.api_config import APIConfig

class GroqClient(BaseAPIClient):
    """Client for Groq API."""
    
    def _validate_api_key(self) -> bool:
        """Validate Groq API key format."""
        return APIConfig.validate_api_key("groq", self.api_key)
    
    def _make_request(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Make request to Groq API.
        
        Args:
            prompt: The prompt to send
            model: Model name to use
            temperature: Temperature setting
            
        Returns:
            Response dictionary
        """
        if not self.validate_model(model):
            raise ValueError(f"Invalid model: {model}. Available models: {', '.join(self.get_available_models())}")
        
        # Ensure temperature is within valid range
        temperature = max(0.0, min(1.0, temperature))
        
        client = Groq(api_key=self.api_key)
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model=model,
            temperature=temperature,
        )
        
        return {
            "text": chat_completion.choices[0].message.content
        }
    
    def get_available_models(self) -> list[str]:
        """Get list of available Groq models."""
        return APIConfig.GROQ_MODELS
    
    def validate_model(self, model: str) -> bool:
        """Validate if a model is available in Groq."""
        return APIConfig.validate_model("groq", model)
