# Reasoning Evaluation Framework

This directory contains the reasoning evaluation code, organized into multiple modules for better maintainability and separation of concerns.

## New Structure

### Core Modules

1. **`config/`** - Configuration and constants
   - `api_config.py`: API settings, keys, and model configurations
   - `evaluation_config.py`: Evaluation prompts, thresholds, and settings

2. **`api_clients/`** - API client implementations
   - `base_client.py`: Abstract base class for all API clients
   - `groq_client.py`: Groq API client implementation
   - `gemini_client.py`: Google Gemini API client implementation
   - `mistral_client.py`: Mistral AI API client implementation
   - `client_factory.py`: Factory for creating appropriate API clients

3. **`analysis/`** - Reasoning analysis logic
   - `reasoning_analyzer.py`: Keyword-based reasoning analysis
   - `self_evaluation.py`: Self-evaluation using the same model

4. **`utils/`** - Utility functions
   - `response_parser.py`: Response parsing and extraction utilities
   - `data_utils.py`: Data handling and processing utilities

5. **`framework/`** - Main framework orchestration
   - `llm_reasoning_framework.py`: Main framework class coordinating all components

6. **`main.py`** - Entry point for the refactored system

### Legacy File

- **`../reasoning_evaluation.py`** - Original file (kept for backward compatibility)
  - Contains all original functionality
  - Points users to the new structure

## Usage

### Running the new Framework

To run the complete reasoning evaluation, use the new main entry point:

```bash
cd experiments
python reasoning_evaluation/main.py
```

### Custom Usage

You can also use individual components for custom analysis:

```python
from reasoning_evaluation.framework import LLMReasoningFramework
from reasoning_evaluation.api_clients import create_api_client
from reasoning_evaluation.analysis import ReasoningAnalyzer

# Create framework with models
models = [
    {"api": "gemini", "model": "gemini-2.0-flash", "temperature": 0.0}
]
framework = LLMReasoningFramework(models)

# Run evaluation
framework.run("path/to/data.json", is_self_reason=True)
```

If you were using the old script:

1. **Replace imports**: Use the new modular imports
2. **Update function calls**: Functions are now methods of classes
3. **Use framework**: For complete evaluation, use `LLMReasoningFramework`
4. **Custom analysis**: Import specific classes for targeted analysis

## Configuration

### Environment Variables

Create a `.env` file in the parent directory:

```bash
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=your_gemini_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

### Model Configuration

Update the models list in `main.py`:

```python
models = [
    {"api": "groq", "model": "llama3-70b-8192", "temperature": 0.0},
    {"api": "gemini", "model": "gemini-2.0-flash", "temperature": 0.0},
    {"api": "mistral", "model": "mistral-large-latest", "temperature": 0.0}
]
```

### Adding New API Providers

1. Create new client class inheriting from `BaseAPIClient`
2. Implement required abstract methods
3. Add to `client_factory.py`
4. Update configuration in `api_config.py`

### Adding New Analysis Methods

1. Add new analysis class in `analysis/` directory
2. Integrate into `LLMReasoningFramework`
3. Update configuration as needed

### Adding New Evaluation Types

1. Extend `EvaluationConfig` with new prompts/settings
2. Add corresponding analysis logic
3. Update the main framework to handle new evaluation types

## Testing

Each module can be tested independently:

```python
# Test API client
from reasoning_evaluation.api_clients import GroqClient
client = GroqClient("test_key")
assert client.validate_model("llama3-70b-8192")

# Test reasoning analyzer
from reasoning_evaluation.analysis import ReasoningAnalyzer
analyzer = ReasoningAnalyzer()
annotations, answer, metrics = analyzer.annotate_reasoning_chain("test text")
```

### Debug Mode

Enable debug logging by setting environment variable:

```bash
export DEBUG=1
python reasoning_evaluation/main.py
```

This modular structure makes it easy to extend and maintain the reasoning evaluation capabilities while preserving all original functionality.
