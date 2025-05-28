#!/usr/bin/env python3
import os
import time
import json
import pandas as pd
import requests
import datetime
import getpass
import sys
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Constants
REASONING_META_PROMPT = "\n\nFor the answer above, classify your reasoning as one of [Recall, Reasoning, Hallucination, Uncertain]. State the label alone."
CONFIDENCE_META_PROMPT = "\n\nOn a scale of 0-100%, how confident are you in your answer? State only the percentage."

# Available models
GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "claude-3-opus-20240229",
    "deepseek-r1-distill-llama-70b"
]

GEMINI_MODELS = [
    "gemini-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]

class LLMReasoningFramework:
    def __init__(self, models: List[Dict[str, Any]] = None):
        """
        Initialize the framework with specified models.
        
        Args:
            models: List of model configurations
        """
        self.models = models or []
        self.results_df = pd.DataFrame(columns=[
            "prompt_id", "prompt_text", "api", "model", 
            "answer", "reasoning_type", "confidence", 
            "response_time", "timestamp",
            "problem_type", "subject", "level", "original_answer"  # New columns for the specific dataset
        ])
        self.retry_delay = 5  # seconds to wait between retries
        self.max_retries = 3  # maximum number of retries per API call
        
        # API keys - will be set during runtime
        self.groq_api_key = None
        self.gemini_api_key = None
        
    def set_api_keys(self):
        """Get API keys from environment or prompt the user"""
        # Try to get from environment first
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # If not found, prompt the user
        if not self.groq_api_key and any(model["api"] == "groq" for model in self.models):
            print("\nGroq API key not found in environment variables.")
            self.groq_api_key = getpass.getpass("Enter your Groq API key: ")
        
        if not self.gemini_api_key and any(model["api"] == "gemini" for model in self.models):
            print("\nGemini API key not found in environment variables.")
            self.gemini_api_key = getpass.getpass("Enter your Gemini API key: ")
            
        # Configure Gemini if key is available
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        
    def load_prompts(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load prompts from the specific JSON file format for math problems.
        
        Args:
            file_path: Path to the JSON file containing math problems
            
        Returns:
            List of dictionaries with prompt IDs and text
        """
        prompts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
                
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "modified_problem" in item:
                    # Using unique_id or index as ID
                    prompt_id = item.get("unique_id", i)
                    prompts.append({
                        "id": prompt_id,
                        "text": item["modified_problem"],
                        "problem_type": item.get("problem_type", ""),
                        "subject": item.get("subject", ""),
                        "level": item.get("level", ""),
                        "original_answer": item.get("original_answer", ""),
                        "original_problem": item.get("original_problem", "")
                    })
                    
        return prompts

    # rate limit
    # def query_groq(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
    #     """
    #     Query the Groq API with improved rate limit handling.
    #     """
    #     if not self.groq_api_key:
    #         return {
    #             "text": "Error: Groq API key not provided",
    #             "response_time": 0
    #         }
            
    #     start_time = time.time()
    #     headers = {
    #         "Authorization": f"Bearer {self.groq_api_key}",
    #         "Content-Type": "application/json"
    #     }
        
    #     data = {
    #         "model": model,
    #         "messages": [{"role": "user", "content": prompt}],
    #         "temperature": temperature
    #     }
        
    #     # Exponential backoff strategy
    #     base_delay = self.retry_delay  # Start with your configured delay (5 seconds)
        
    #     for attempt in range(self.max_retries):
    #         try:
    #             response = requests.post(
    #                 "https://api.groq.com/openai/v1/chat/completions",
    #                 headers=headers,
    #                 json=data
    #             )
                
    #             # Check specifically for rate limit errors (usually 429)
    #             if response.status_code == 429:
    #                 retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
    #                 print(f"Rate limit exceeded. Waiting for {retry_after} seconds before retry.")
    #                 time.sleep(retry_after)
    #                 continue
                    
    #             response.raise_for_status()
                
    #             result = response.json()
    #             end_time = time.time()
                
    #             print('response: ',result["choices"][0]["message"]["content"])
    #             return {
    #                 "text": result["choices"][0]["message"]["content"],
    #                 "response_time": end_time - start_time
    #             }
                
    #         except requests.exceptions.HTTPError as e:
    #             print(f"HTTP Error: {e}")
                
    #             # Handle 429 errors that might come through as exceptions
    #             if e.response.status_code == 429:
    #                 retry_after = int(e.response.headers.get('Retry-After', base_delay * (2 ** attempt)))
    #                 print(f"Rate limit exceeded. Waiting for {retry_after} seconds before retry.")
    #                 time.sleep(retry_after)
    #                 continue
                    
    #             if attempt < self.max_retries - 1:
    #                 delay = base_delay * (2 ** attempt)  # Exponential backoff
    #                 print(f"Retrying in {delay} seconds...")
    #                 time.sleep(delay)
    #             else:
    #                 return {
    #                     "text": f"Error: {str(e)}",
    #                     "response_time": time.time() - start_time
    #                 }
    #         except Exception as e:
    #             print(f"Error querying Groq (attempt {attempt+1}/{self.max_retries}): {e}")
    #             if attempt < self.max_retries - 1:
    #                 delay = base_delay * (2 ** attempt)  # Exponential backoff
    #                 print(f"Retrying in {delay} seconds...")
    #                 time.sleep(delay)
    #             else:
    #                 return {
    #                     "text": f"Error: {str(e)}",
    #                     "response_time": time.time() - start_time
    #                 }
    def query_groq(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Query the Groq API with improved handling for math problems.
        Returns both reasoning and final answer.
        """
        if not self.groq_api_key:
            return {
                "text": "Error: Groq API key not provided",
                "response_time": 0,
                "reasoning": "",
                "answer": ""
            }
            
        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Modified system message to ensure both reasoning and answer are provided in a structured format
        system_message = """You are an expert mathematician solving math problems. 
        
        Please provide your response in the following format:
        
        <reasoning>
        [Provide detailed step-by-step reasoning here using LaTeX notation for mathematical expressions where appropriate]
        </reasoning>
        
        <answer>
        [Provide the final answer as a number or expression]
        </answer>
        
        Make sure to include all your work in the reasoning section and only the final result in the answer section."""
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        # Exponential backoff strategy
        base_delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                
                # Check specifically for rate limit errors
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                    print(f"Rate limit exceeded. Waiting for {retry_after} seconds before retry.")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                
                result = response.json()
                end_time = time.time()
                
                # Get the full content
                content = result["choices"][0]["message"]["content"]
                
                # Extract reasoning and answer using regex
                import re
                reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                answer = answer_match.group(1).strip() if answer_match else ""
                
                # If parsing fails, use the full response but still attempt to separate
                if not reasoning and not answer:
                    # Try to make a best guess - assume everything before the last paragraph is reasoning
                    # and the last paragraph is the answer
                    paragraphs = content.split('\n\n')
                    if len(paragraphs) > 1:
                        answer = paragraphs[-1].strip()
                        reasoning = '\n\n'.join(paragraphs[:-1]).strip()
                    else:
                        # If no clear separation, set both to the full content
                        reasoning = content
                        answer = content
                
                # For debugging
                print(f"Response from {model} (truncated):")
                print(content[:300] + "..." if len(content) > 300 else content)
                
                return {
                    "text": content,  # Full text response
                    "reasoning": reasoning,  # Extracted reasoning
                    "answer": answer,  # Extracted answer
                    "response_time": end_time - start_time
                }
                
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error: {e}")
                
                # Handle rate limit errors
                if hasattr(e, 'response') and e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                    print(f"Rate limit exceeded. Waiting for {retry_after} seconds before retry.")
                    time.sleep(retry_after)
                    continue
                    
                if attempt < self.max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return {
                        "text": f"Error: {str(e)}",
                        "reasoning": "",
                        "answer": "",
                        "response_time": time.time() - start_time
                    }
            except Exception as e:
                print(f"Error querying Groq (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return {
                        "text": f"Error: {str(e)}",
                        "reasoning": "",
                        "answer": "",
                        "response_time": time.time() - start_time
                    }
        
    def query_gemini(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Query the Gemini API.
        
        Args:
            prompt: The prompt to send to the model
            model: Name of the Gemini model
            temperature: Temperature setting for generation
            
        Returns:
            Dictionary containing response information
        """
        if not self.gemini_api_key:
            return {
                "text": "Error: Gemini API key not provided",
                "response_time": 0
            }
            
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Configure the model
                generation_config = {
                    "temperature": temperature,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }
                
                # Create the model instance
                model_instance = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config
                )
                
                # Generate content
                response = model_instance.generate_content(prompt)
                end_time = time.time()
                
                return {
                    "text": response.text,
                    "response_time": end_time - start_time
                }
                
            except Exception as e:
                print(f"Error querying Gemini (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "text": f"Error: {str(e)}",
                        "response_time": time.time() - start_time
                    }
    
    def query_model(self, prompt: str, api: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Query appropriate API based on api parameter.
        
        Args:
            prompt: The prompt to send to the model
            api: API type ('groq' or 'gemini')
            model: Model name
            temperature: Temperature setting
            
        Returns:
            API response
        """
        full_prompt = prompt + REASONING_META_PROMPT + CONFIDENCE_META_PROMPT
        
        if api.lower() == "groq":
            return self.query_groq(full_prompt, model, temperature)
        elif api.lower() == "gemini":
            return self.query_gemini(full_prompt, model, temperature)
        else:
            raise ValueError(f"Unsupported API: {api}")
    
    def parse_response(self, response_text: str) -> Tuple[str, str, Optional[float], str]:
        """
        Enhanced parser that extracts the final numeric answer and reasoning from math problems.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Tuple of (answer, reasoning_type, confidence, reasoning)
        """
        # First, try to find reasoning type
        reasoning_labels = ["Recall", "Reasoning", "Hallucination", "Uncertain"]
        
        # Default values
        full_response = response_text
        numerical_answer = None  # This will hold just the numeric answer
        reasoning_type = None
        confidence = None
        
        # Extract the reasoning type (usually at the end)
        for label in reasoning_labels:
            if label in response_text:
                parts = response_text.split(label)
                if len(parts) > 1:
                    reasoning_type = label
                    break
        
        # If we didn't find a clear label, look for the last line containing any label
        if not reasoning_type:
            lines = response_text.split('\n')
            for line in reversed(lines):
                line = line.strip()
                for label in reasoning_labels:
                    if label in line:
                        reasoning_type = label
                        break
                if reasoning_type:
                    break
        
        # Extract ONLY the final numerical answer from the solution
        import re
        
        # Common patterns for final answers in math solutions
        answer_patterns = [
            # "The answer is X" patterns
            r"(?:the|my|our|final)?\s*answer(?:\s+is)?(?:\s*:)?\s*([-+]?[0-9,\.\s]+)",
            r"(?:the|my|our)?\s*result(?:\s+is)?(?:\s*:)?\s*([-+]?[0-9,\.\s]+)",
            
            # "Therefore X" patterns
            r"(?:thus|therefore|so|hence),?(?:\s+the\s+answer\s+is)?\s*([-+]?[0-9,\.\s]+)",
            
            # Equation ending with = X
            r"(?:=\s*)([-+]?[0-9,\.\s]+)(?:\s*$|\s*\.|$)",
            
            # Just a number at the end of text/line
            r"(?:^|\n\s*)([-+]?[0-9,\.\s]+)(?:\s*$|\s*\.)",
            
            # LaTeX format numbers
            r"\$\s*([-+]?[0-9,\.\s]+)\s*\$"
        ]
        
        # Try to extract a numerical answer using the patterns
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                # Use the last match, which is likely the final answer
                # Clean up the extracted number
                raw_answer = matches[-1].strip()
                # Remove any commas in the number
                cleaned_answer = raw_answer.replace(',', '').strip()
                numerical_answer = cleaned_answer
                break
        
        # If we couldn't extract using patterns, try a more general approach
        # Look for the last number in the text
        if not numerical_answer:
            # Find all numbers in the text
            number_matches = re.findall(r'([-+]?[0-9,\.\s]+)', response_text)
            if number_matches:
                # Use the last number as the answer
                numerical_answer = number_matches[-1].replace(',', '').strip()
        
        # Look for confidence percentage
        confidence_pattern = r'(\d{1,3})%'
        confidence_match = re.search(confidence_pattern, response_text)
        if confidence_match:
            try:
                confidence_value = int(confidence_match.group(1))
                if 0 <= confidence_value <= 100:
                    confidence = confidence_value
            except:
                pass
        
        # If still not found, set defaults
        if not reasoning_type:
            # For math problems, usually it's reasoning unless very basic
            reasoning_type = "Reasoning"
        
        # ONLY store the numerical answer, not the full response
        answer = numerical_answer if numerical_answer else "No numerical answer found"
        
        # Extract reasoning - this is the full model response minus any concluding tags/labels
        reasoning = full_response
        
        # Remove the reasoning label and anything after it from the reasoning text
        if reasoning_type in reasoning:
            reasoning = reasoning.split(reasoning_type)[0].strip()
        
        # Remove any final answer statements to isolate just the reasoning part
        for pattern in [
            r"(?:the|my|our|final)?\s*answer(?:\s+is)?(?:\s*:)?\s*[-+]?[0-9,\.\s]+",
            r"(?:the|my|our)?\s*result(?:\s+is)?(?:\s*:)?\s*[-+]?[0-9,\.\s]+",
            r"(?:thus|therefore|so|hence),?(?:\s+the\s+answer\s+is)?\s*[-+]?[0-9,\.\s]+"
        ]:
            reasoning = re.sub(pattern, "", reasoning, flags=re.IGNORECASE)
        
        # Clean up the reasoning text
        reasoning = reasoning.strip()
        
        # If reasoning extraction failed, use the original text
        if not reasoning:
            reasoning = full_response
        
        # Log the extraction for debugging
        print(f"Extracted answer: {answer}")
        print(f"Extracted reasoning type: {reasoning_type}")
        
        return answer, reasoning_type, confidence, reasoning
    
    def log_result(self, prompt_dict, api, model, answer, reasoning_type, confidence, response_time, reasoning=""):
        """
        Log results to the DataFrame.
        
        Args:
            prompt_dict: Dictionary containing prompt information
            api: API provider (e.g., 'openai', 'anthropic', 'groq')
            model: Model name
            answer: Parsed answer
            reasoning_type: Type of reasoning used
            confidence: Confidence level
            response_time: Time taken for response
            reasoning: Detailed reasoning provided by the model
        """
        new_row = pd.DataFrame({
            "prompt_id": [prompt_dict["id"]],
            "prompt_text": [prompt_dict["text"]],
            "api": [api],
            "model": [model],
            "answer": [answer],
            "reasoning": [reasoning],  # Added reasoning field
            "reasoning_type": [reasoning_type],
            "confidence": [confidence],
            "response_time": [response_time],
            "timestamp": [datetime.datetime.now()],
            "problem_type": [prompt_dict.get("problem_type", "")],
            "subject": [prompt_dict.get("subject", "")],
            "level": [prompt_dict.get("level", "")],
            "original_answer": [prompt_dict.get("original_answer", "")]
        })
        
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
    
    def save_results(self, file_path: str, format: str = "pkl") -> None:
        """
        Save results DataFrame to file.
        
        Args:
            file_path: Path for the output file
            format: Output format ('pkl', 'csv', or 'json')
        """
        if format.lower() == "csv":
            self.results_df.to_csv(file_path, index=False)
        elif format.lower() == "json":
            # Convert DataFrame to JSON
            json_data = self.results_df.to_json(orient="records", indent=2)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
        else:
            self.results_df.to_pickle(file_path)
    

# add time between each call
    def run(self, prompts: List[Dict[str, Any]], output_file: str = "results", 
        output_format: str = "pkl", limit: Optional[int] = None) -> pd.DataFrame:
        """
        Run the full analysis pipeline with rate limit handling.
        
        Args:
            prompts: List of prompt dictionaries
            output_file: Base name for output file
            output_format: Output format ('pkl', 'csv', or 'json')
            limit: Maximum number of prompts to process
            
        Returns:
            Results DataFrame
        """
        # Make sure API keys are set
        self.set_api_keys()
        
        if limit and limit > 0:
            prompts = prompts[:limit]
        
        print(f"Processing {len(prompts)} prompts across {len(self.models)} models...")
        
        # Process one model at a time to better manage rate limits
        for model_config in self.models:
            api = model_config["api"]
            model = model_config["model"]
            temp = model_config.get("temp", 0.0)
            
            print(f"\n=== Processing all prompts for {api}/{model} ===")
            
            # Track rate limits
            api_call_times = []  # Keep track of timestamps for recent API calls
            rate_window = 60  # 60 seconds window for rate limiting
            max_calls_per_window = 10  # Adjust based on your API tier (example: 10 calls per minute)
            
            for prompt_dict in tqdm(prompts, desc=f"Processing {api}/{model}"):
                prompt_id = prompt_dict["id"]
                prompt_text = prompt_dict["text"]
                
                print(f"\nQuerying {api}/{model} for prompt #{prompt_id}")
                
                # Rate limit management
                current_time = time.time()
                
                # Remove timestamps older than the rate window
                api_call_times = [t for t in api_call_times if current_time - t < rate_window]
                
                # Check if we need to wait due to rate limits
                if len(api_call_times) >= max_calls_per_window:
                    # Calculate how long to wait
                    oldest_call = min(api_call_times)
                    wait_time = rate_window - (current_time - oldest_call) + 1  # Add 1 second buffer
                    print(f"Rate limit approaching. Waiting {wait_time:.1f} seconds before next request...")
                    time.sleep(wait_time)
                    # Reset our tracking after waiting
                    current_time = time.time()
                    api_call_times = [t for t in api_call_times if current_time - t < rate_window]
                
                try:
                    # Record this API call time
                    api_call_times.append(time.time())
                    
                    response = self.query_model(
                        prompt=prompt_text,
                        api=api,
                        model=model,
                        temperature=temp
                    )
                    
                    answer, reasoning_type, confidence, reasoning = self.parse_response(response["text"])
                    
                    self.log_result(
                        prompt_dict=prompt_dict,
                        api=api,
                        model=model,
                        answer=answer,
                        reasoning_type=reasoning_type,
                        confidence=confidence,
                        response_time=response["response_time"],
                        reasoning=reasoning  # Added reasoning parameter
                    )
                    
                    # Add a small delay between successful requests to be cautious
                    if api.lower() == "groq":
                        delay = 2  # 2 second delay between Groq requests
                        print(f"Adding {delay} second delay between requests...")
                        time.sleep(delay)
                    
                except Exception as e:
                    print(f"Error processing prompt {prompt_id} with {api}/{model}: {e}")
                    
                    # Check if it's a rate limit error
                    if "429" in str(e) or "too many" in str(e).lower() or "rate limit" in str(e).lower():
                        # Wait longer for rate limit errors
                        wait_time = 20  # Wait 20 seconds on rate limit error
                        print(f"Rate limit exceeded. Waiting {wait_time} seconds before continuing...")
                        time.sleep(wait_time)
                        
                        # Try again with this prompt (optional)
                        print("Retrying prompt after waiting...")
                        try:
                            # Record this API call time
                            api_call_times.append(time.time())
                            
                            response = self.query_model(
                                prompt=prompt_text,
                                api=api,
                                model=model,
                                temperature=temp
                            )
                            
                            answer, reasoning_type, confidence, reasoning = self.parse_response(response["text"])
                            
                            self.log_result(
                                prompt_dict=prompt_dict,
                                api=api,
                                model=model,
                                answer=answer,
                                reasoning_type=reasoning_type,
                                confidence=confidence,
                                response_time=response["response_time"],
                                reasoning=reasoning  # Added reasoning parameter
                            )
                            
                            continue  # Skip the error logging if retry succeeds
                        except Exception as retry_e:
                            print(f"Retry also failed: {retry_e}")
                    
                    # Log the error in the DataFrame
                    self.log_result(
                        prompt_dict=prompt_dict,
                        api=api,
                        model=model,
                        answer=f"Error: {str(e)}",
                        reasoning_type="Error",
                        confidence=None,
                        response_time=-1,
                        reasoning="Error occurred during processing"  # Added reasoning parameter for errors
                    )
            
            # Wait a bit longer between switching models
            if api.lower() == "groq":
                print(f"Finished processing model {model}. Waiting 15 seconds before starting next model...")
                time.sleep(15)
        
        # Save results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{output_file}_{timestamp}.{output_format}"
        self.save_results(file_path, output_format)
        
        # Save intermediate results after each model to prevent data loss
        intermediate_file = f"{output_file}_interim_{timestamp}.{output_format}"
        self.save_results(intermediate_file, output_format)
        print(f"Intermediate results saved to {intermediate_file}")
        
        print(f"All processing complete! Final results saved to {file_path}")
        return self.results_df


def select_models():
    """
    Interactive model selection.
    
    Returns:
        List of selected model configurations
    """
    selected_models = []
    
    print("\n=== Model Selection ===")
    
    # Groq model selection
    print("\nAvailable Groq models:")
    for i, model in enumerate(GROQ_MODELS):
        print(f"{i+1}. {model}")
    
    print("\nSelect Groq models (comma-separated numbers, or 0 to skip Groq):")
    groq_selection = input("> ")
    
    if groq_selection and groq_selection != "0":
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in groq_selection.split(",")]
            for idx in selected_indices:
                if 0 <= idx < len(GROQ_MODELS):
                    selected_models.append({
                        "api": "groq",
                        "model": GROQ_MODELS[idx],
                        "temp": 0.0
                    })
        except ValueError:
            print("Invalid selection. Skipping Groq models.")
    
    # Gemini model selection
    print("\nAvailable Gemini models:")
    for i, model in enumerate(GEMINI_MODELS):
        print(f"{i+1}. {model}")
    
    print("\nSelect Gemini models (comma-separated numbers, or 0 to skip Gemini):")
    gemini_selection = input("> ")
    
    if gemini_selection and gemini_selection != "0":
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in gemini_selection.split(",")]
            for idx in selected_indices:
                if 0 <= idx < len(GEMINI_MODELS):
                    selected_models.append({
                        "api": "gemini",
                        "model": GEMINI_MODELS[idx],
                        "temp": 0.0
                    })
        except ValueError:
            print("Invalid selection. Skipping Gemini models.")
    
    # Set temperature
    print("\nSet temperature for all models (0.0-1.0, default 0.0):")
    temp_str = input("> ")
    
    if temp_str:
        try:
            temp = float(temp_str)
            if 0.0 <= temp <= 1.0:
                for model in selected_models:
                    model["temp"] = temp
            else:
                print("Temperature should be between 0.0 and 1.0. Using default 0.0.")
        except ValueError:
            print("Invalid temperature. Using default 0.0.")
    
    if not selected_models:
        print("\nNo models selected. Using default models:")
        selected_models = [
            {"api": "groq", "model": "llama3-8b-8192", "temp": 0.0},
            {"api": "gemini", "model": "gemini-pro", "temp": 0.0}
        ]
        for model in selected_models:
            print(f"- {model['api']}/{model['model']}")
    
    print(f"\nSelected {len(selected_models)} models.")
    return selected_models


def main():
    """Main function for the math problem analysis framework."""
    print("\nMath Problem Reasoning Analysis Framework")
    print("====================================================")
    
    # Select models
    models = select_models()
    
    # Create framework instance with selected models
    framework = LLMReasoningFramework(models)
    
    # Get input file
    input_file = input("\nEnter path to math problem JSON file: ")
    
    # Get output preferences
    output_file = input("Enter base name for output file (default 'math_results'): ") or "math_results"
    
    # Output format selection
    print("\nSelect output format:")
    print("1. pickle (.pkl) - Best for later Python processing")
    print("2. CSV (.csv) - Good for spreadsheet analysis")
    print("3. JSON (.json) - Flexible format for various tools")
    format_choice = input("Enter choice (1-3, default 1): ") or "1"
    
    output_format = {
        "1": "pkl",
        "2": "csv",
        "3": "json"
    }.get(format_choice, "pkl")
    
    # Check if we should limit the number of prompts
    limit_str = input("\nMaximum number of problems to process (leave empty for all): ")
    limit = int(limit_str) if limit_str.strip() else None
    
    try:
        # Load prompts
        prompts = framework.load_prompts(input_file)
        print(f"\nLoaded {len(prompts)} math problems from {input_file}")
        
        if len(prompts) == 0:
            print("No problems found. Exiting.")
            return
            
        # Preview first prompt
        print("\nPreview of first problem:")
        preview = prompts[0]["text"][:150] + "..." if len(prompts[0]["text"]) > 150 else prompts[0]["text"]
        print(f"ID: {prompts[0]['id']}")
        print(f"Text: {preview}")
        print(f"Type: {prompts[0].get('problem_type', 'N/A')}")
        print(f"Subject: {prompts[0].get('subject', 'N/A')}")
        print(f"Level: {prompts[0].get('level', 'N/A')}")
        print(f"Original Answer: {prompts[0].get('original_answer', 'N/A')}")
        
        # Confirm run
        proceed = input("\nProceed with analysis? (y/n, default y): ").lower() or "y"
        if proceed != "y":
            print("Exiting without running analysis.")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Run analysis
        results = framework.run(prompts, output_file="results/" + output_file, output_format=output_format, limit=limit)
        
        # Generate accuracy metrics if original answers are available
        if "original_answer" in results.columns and results["original_answer"].notna().any():
            # Simple exact match accuracy
            results["correct"] = results.apply(lambda row: str(row["original_answer"]).strip() in str(row["answer"]), axis=1)
            accuracy = results["correct"].mean() * 100
            
            print("\nAccuracy Metrics:")
            print(f"Exact match accuracy: {accuracy:.2f}%")
            
            # Accuracy by model
            model_accuracy = results.groupby(["api", "model"])["correct"].mean() * 100
            print("\nAccuracy by Model:")
            for (api, model), acc in model_accuracy.items():
                print(f"  {api}/{model}: {acc:.2f}%")
            
            # Accuracy by problem type
            if "problem_type" in results.columns:
                type_accuracy = results.groupby("problem_type")["correct"].mean() * 100
                print("\nAccuracy by Problem Type:")
                for problem_type, acc in type_accuracy.items():
                    print(f"  {problem_type}: {acc:.2f}%")
            
            # Accuracy by reasoning type
            reasoning_accuracy = results.groupby("reasoning_type")["correct"].mean() * 100
            print("\nAccuracy by Reasoning Type:")
            for reasoning, acc in reasoning_accuracy.items():
                count = results[results["reasoning_type"] == reasoning].shape[0]
                print(f"  {reasoning}: {acc:.2f}% (Count: {count})")
        
        print("\nAnalysis complete!")
        print(f"Results saved in 'results/{output_file}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("An unexpected error occurred. Please check your input file and settings.")


if __name__ == "__main__":
    main()