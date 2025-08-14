from abc import ABC, abstractmethod
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from openai import OpenAI
import os
import time
from config import Config
from response_parser import ResponseParser

class BaseAPIClient(ABC):
    """Abstract base class for API clients"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> dict:
        """Generate response from the API"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API client is properly configured"""
        pass

class GeminiClient(BaseAPIClient):
    """Client for Google Gemini API"""
    
    def __init__(self, api_key: str, rate_limiter=None):
        if not api_key:
            raise ValueError("Gemini API key is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        self.rate_limiter = rate_limiter
    
    def is_available(self) -> bool:
        return True  # If we can create the client, it's available
    
    def generate_response(self, prompt: str) -> dict:
        """Generate response from Gemini API"""
        try:
            # Wait for rate limit if rate limiter is provided
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            
            generation_config = {
                "temperature": Config.GEMINI_TEMPERATURE,
                "max_output_tokens": Config.GEMINI_MAX_OUTPUT_TOKENS,
            }
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            # Extract text from response
            if ResponseParser.is_max_tokens_reached(str(response)):
                return {
                    "success": False,
                    "response": None,
                    "error": "Max output tokens reached"
                }
            
            if isinstance(response, GenerateContentResponse) and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
            else:
                response_text = str(response)
            
            return {
                "success": True,
                "response": response_text,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

class OpenAIClient(BaseAPIClient):
    """Client for OpenAI API"""
    
    def __init__(self, api_key: str, rate_limiter=None):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key)
        self.rate_limiter = rate_limiter
    
    def is_available(self) -> bool:
        return True  # If we can create the client, it's available
    
    def generate_response(self, prompt: str) -> dict:
        """Generate response from OpenAI API with retry logic"""
        for attempt in range(Config.MAX_RETRIES):
            try:
                # Wait for rate limit if rate limiter is provided
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()
                
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=Config.OPENAI_TEMPERATURE,
                    max_tokens=Config.OPENAI_MAX_TOKENS
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                return {
                    "success": True,
                    "response": response_text,
                    "error": None
                }
                
            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    print(f"OpenAI attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "success": False,
                        "response": None,
                        "error": str(e)
                    }

def create_api_client(model_type: str, api_key: str, rate_limiter=None) -> BaseAPIClient:
    """Factory function to create the appropriate API client"""
    if model_type == "openai":
        return OpenAIClient(api_key, rate_limiter)
    elif model_type == "gemini":
        return GeminiClient(api_key, rate_limiter)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
