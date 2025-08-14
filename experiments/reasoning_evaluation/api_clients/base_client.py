"""
Abstract base class for API clients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time

class BaseAPIClient(ABC):
    """Abstract base class for API clients."""
    
    def __init__(self, api_key: str, retry_delay: int = 5, max_retries: int = 3):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for the service
            retry_delay: Seconds to wait between retries
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        
        if not self._validate_api_key():
            raise ValueError(f"Invalid API key for {self.__class__.__name__}")
    
    @abstractmethod
    def _validate_api_key(self) -> bool:
        """Validate the API key format."""
        pass
    
    @abstractmethod
    def _make_request(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Make the actual API request."""
        pass
    
    def query(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Query the API with retry logic.
        
        Args:
            prompt: The prompt to send
            model: Model name to use
            temperature: Temperature setting
            
        Returns:
            Response dictionary with text and response_time
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = self._make_request(prompt, model, temperature)
                response["response_time"] = time.time() - start_time
                return response
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error querying {self.__class__.__name__} (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "text": f"Error: {str(e)}",
                        "response_time": time.time() - start_time
                    }
    
    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this API."""
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Validate if a model is available."""
        pass
