"""
Gemini API client implementation.
"""

from typing import Dict, Any
import google.generativeai as genai
from .base_client import BaseAPIClient
from ..config.api_config import APIConfig

class GeminiClient(BaseAPIClient):
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: str, retry_delay: int = 5, max_retries: int = 3):
        """Initialize Gemini client and configure API."""
        super().__init__(api_key, retry_delay, max_retries)
        
        # Configure Gemini API
        try:
            genai.configure(api_key=self.api_key)
            # Test configuration by listing models
            genai.configure(api_key=self.api_key)
            # Configuration is complete; defer API connectivity checks to later calls if needed.
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {e}")
    
    def _validate_api_key(self) -> bool:
        """Validate Gemini API key."""
        return bool(self.api_key and len(self.api_key) > 0)
    
    def _make_request(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Make request to Gemini API.
        
        Args:
            prompt: The prompt to send
            model: Model name to use
            temperature: Temperature setting
            
        Returns:
            Response dictionary
        """
        # Add model name prefix if not present
        if not model.startswith("models/"):
            model = f"models/{model}"
        
        # Validate model name
        try:
            models = genai.list_models()
            available_models = [m.name for m in models]
            
            if model not in available_models:
                suggested_model = "models/gemini-1.5-pro"  # Default to a stable model
                print(f"\nWarning: Model {model} not found. Using {suggested_model} instead.")
                model = suggested_model
        except Exception as e:
            print(f"Error listing models: {e}")
            return {"text": f"Error: Unable to validate model name - {str(e)}"}
        
        # Ensure temperature is within valid range
        temperature = max(0.0, min(1.0, temperature))
        
        try:
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    **APIConfig.GEMINI_GENERATION_CONFIG,
                    "temperature": temperature
                },
                safety_settings=APIConfig.GEMINI_SAFETY_SETTINGS
            )
            
            response = model_instance.generate_content(prompt)
            
            # Extract text from response
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    response_text = response.candidates[0].content.text
                elif hasattr(response.candidates[0], 'text'):
                    response_text = response.candidates[0].text
            
            if not response_text:
                raise ValueError("Unable to extract text from Gemini response")
            
            return {"text": response_text}
            
        except Exception as e:
            return {"text": f"Error: {str(e)}"}
    
    def get_available_models(self) -> list[str]:
        """Get list of available Gemini models."""
        try:
            models = genai.list_models()
            return [m.name for m in models]
        except Exception:
            return []
    
    def validate_model(self, model: str) -> bool:
        """Validate if a model is available in Gemini."""
        try:
            if not model.startswith("models/"):
                model = f"models/{model}"
            models = genai.list_models()
            available_models = [m.name for m in models]
            return model in available_models
        except Exception:
            return False
