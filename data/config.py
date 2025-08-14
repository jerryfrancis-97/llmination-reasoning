import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Rate limiting configuration
    REQUESTS_PER_MINUTE = 15
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # File paths
    INPUT_FILE = "perplexmath-dataset.json"
    OUTPUT_FILE = "perplexmath-generated-answers-openai.json"
    
    # Model configuration
    MODEL_TYPE = "openai"  # or "gemini"
    
    # Processing configuration
    NUM_PROCESSES = 5
    MAX_RETRIES = 3
    MAX_ATTEMPTS = 2
    
    # API configuration
    OPENAI_MODEL = "gpt-4.1-nano"
    OPENAI_TEMPERATURE = 0
    OPENAI_MAX_TOKENS = 100000
    GEMINI_TEMPERATURE = 0
    GEMINI_MAX_OUTPUT_TOKENS = 100000
