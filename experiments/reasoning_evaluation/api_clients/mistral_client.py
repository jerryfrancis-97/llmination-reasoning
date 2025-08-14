"""
Mistral API client implementation.
"""

from typing import Dict, Any
from mistralai import Mistral
from .base_client import BaseAPIClient
from ..config.api_config import APIConfig

class MistralClient(BaseAPIClient):
    """Client for Mistral API."""
    
    def _validate_api_key(self) -> bool:
        """Validate Mistral API key."""
        return bool(self.api_key and len(self.api_key) > 0)
    
    def _make_request(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Make request to Mistral API.
        
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
        
        try:
            client = Mistral(api_key=self.api_key)
            chat_response = client.chat.complete(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature
            )
            
            return {
                "text": chat_response.choices[0].message.content
            }
            
        except Exception as e:
            return {"text": f"Error: {str(e)}"}
    
    def get_available_models(self) -> list[str]:
        """Get list of available Mistral models."""
        return APIConfig.MISTRAL_MODELS
    
    def validate_model(self, model: str) -> bool:
        """Validate if a model is available in Mistral."""
        return APIConfig.validate_model("mistral", model)
