#!/usr/bin/env python3
import os
import time
import json
import pandas as pd
import requests
import datetime
import getpass
import sys
import re
import random
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv
import traceback
import google.generativeai as genai
from tqdm import tqdm
from groq import Groq  # Add Groq client import
from mistralai import Mistral  # Updated Mistral import

# Load environment variables
load_dotenv("../.env")

# Constants
BASE_META_PROMPT = "\n\nAnswer the question above to the best of your ability."
FORMAT_PROMPT = "\n\n Output your final answer in the format {'FINAL_ANSWER': <final_answer>, 'LABEL': <label>, 'CONFIDENCE': <confidence>} in the last line separately."
REASONING_META_PROMPT = "\n\nFor the answer above, classify your reasoning as one of [Recall, Reasoning, Hallucination, Uncertain]. State the label alone."
CONFIDENCE_META_PROMPT = "\n\nOn a scale of 0-100%, how confident are you in your answer? State only the percentage."
CHAIN_OF_THOUGHT_PROMPT = "\n\nBefore answering, walk through your reasoning step by step."

# Available models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", 
    "llama3-70b-8192", #llama3-70b-instruct HF
    "llama3-8b-8192",
    "gemma2-9b-it",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b"
]

MISTRAL_MODELS = [
    "mistral-large-latest"
]


class LLMReasoningFramework:
    def __init__(self, models: List[Dict[str, Any]] = None):
        """
        Initialize the framework with specified models.
        
        Args:
            models: List of model configurations
        """
        self.models = models or []
        # Updated results DataFrame columns
        self.results_df = pd.DataFrame()
        self.retry_delay = 5  # seconds to wait between retries
        self.max_retries = 3  # maximum number of retries per API call
        
        # API keys - will be set during runtime
        self.groq_api_key = None
        self.gemini_api_key = None
        self.mistral_api_key = None
        
    def set_api_keys(self):
        """Get API keys from environment or prompt the user"""
        # Try to get from environment first
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        print("self.groq_api_key", self.groq_api_key, not self.groq_api_key)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        # Validate Groq API key if needed
        if any(model["api"] == "groq" for model in self.models):
            if not self.groq_api_key:
                print("\nGroq API key not found in environment variables.")
                self.groq_api_key = getpass.getpass("Enter your Groq API key: ")
            # Validate Groq API key format
            if not self.groq_api_key.startswith("gsk_"):
                print("\nWarning: Groq API key appears invalid. It should start with 'gsk_'")
                retry = input("Would you like to enter the key again? (y/n): ")
                if retry.lower() == 'y':
                    self.groq_api_key = getpass.getpass("Enter your Groq API key: ")
            print("Groq API key validated")
        
        # Handle Gemini key similarly
        if any(model["api"] == "gemini" for model in self.models):
            if not self.gemini_api_key:
                print("\nGemini API key not found in environment variables.")
                self.gemini_api_key = getpass.getpass("Enter your Gemini API key: ")
            
        # Handle Mistral key similarly
        if any(model["api"] == "mistral" for model in self.models):
            if not self.mistral_api_key:
                print("\nMistral API key not found in environment variables.")
                self.mistral_api_key = getpass.getpass("Enter your Mistral API key: ")
            # Validate Mistral API key format
            if not self.mistral_api_key.startswith(""):  # Add validation if needed
                print("\nWarning: Mistral API key appears invalid.")
                retry = input("Would you like to enter the key again? (y/n): ")
                if retry.lower() == 'y':
                    self.mistral_api_key = getpass.getpass("Enter your Mistral API key: ")
            print("Mistral API key validated")
            
        # Configure Gemini if key is available
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("Gemini API configured successfully")
            except Exception as e:
                print(f"Error configuring Gemini API: {e}")
        
    def load_prompts(self, input_data: Union[str, List[str], List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """
        Load prompts from text, CSV, JSON file, or direct input.
        
        Args:
            input_data: Can be:
                - File path (str)
                - Direct question (str)
                - List of questions (List[str])
                - List of prompt dictionaries (List[Dict[str, str]])
                
        Returns:
            List of dictionaries with prompt IDs and text
        """
        prompts = []
        
        try:

            # Check if it's a file path
            if os.path.exists(input_data):
                # Handle JSON file
                if input_data.endswith('.json'):
                    with open(input_data, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Handle list of prompts
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                if isinstance(item, dict) and "modified_problem" in item:
                                    prompts.append({
                                        "question_id": item["question_id"],
                                        "modified_problem": item["modified_problem"],
                                        "problem_type": item["problem_type"],
                                        "subject": item["subject"],
                                        "level": item["level"],
                                    })
                                   
    
        except Exception as e:
            print(f"Error loading prompts: {e}")
            traceback.print_exc()
            return []
            
            
        return prompts
    
    def query_mistral(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Query the Mistral API using official client.
        
        Args:
            prompt: The prompt text to send
            model: Name of the Mistral model to use
            temperature: Temperature parameter for generation (0.0 to 1.0)
            
        Returns:
            Dict containing response text and timing information
        """
        if not self.mistral_api_key:
            return {
                "text": "Error: Mistral API key not provided",
                "response_time": 0
            }
        
        if model not in MISTRAL_MODELS:
            return {
                "text": f"Error: Invalid model name. Available models: {', '.join(MISTRAL_MODELS)}",
                "response_time": 0
            }
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Initialize Mistral client with new syntax
                client = Mistral(api_key=self.mistral_api_key)
                
                # Make API call with new syntax
                chat_response = client.chat.complete(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=min(max(temperature, 0.0), 1.0)
                )
                
                return {
                    "text": chat_response.choices[0].message.content,
                    "response_time": time.time() - start_time
                }
                
            except Exception as e:
                print(f"Error querying Mistral (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "text": f"Error: {str(e)}",
                        "response_time": time.time() - start_time
                    }
                
    def query_groq(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Query the Groq API using official client."""
        if not self.groq_api_key:
            return {
                "text": "Error: Groq API key not provided",
                "response_time": 0
            }
        
        # Clean up model name
        # model = model.lower().replace("-8192", "").replace("-32768", "")
        if model not in GROQ_MODELS:
            return {
                "text": f"Error: Invalid model name. Available models: {', '.join(GROQ_MODELS)}",
                "response_time": 0
            }
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Initialize Groq client
                client = Groq(api_key=self.groq_api_key)
                
                # Make API call
                chat_completion = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    model=model,
                    temperature=min(max(temperature, 0.0), 1.0),
                )
                
                return {
                    "text": chat_completion.choices[0].message.content,
                    "response_time": time.time() - start_time
                }
                
            except Exception as e:
                print(f"Error querying Groq (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "text": f"Error: {str(e)}",
                        "response_time": time.time() - start_time
                    }
    
    def query_gemini(self, prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Query the Gemini API."""
        if not self.gemini_api_key:
            return {
                "text": "Error: Gemini API key not provided",
                "response_time": 0
            }
            
        start_time = time.time()
        
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
            return {
                "text": f"Error: Unable to validate model name - {str(e)}",
                "response_time": time.time() - start_time
            }
    
        for attempt in range(self.max_retries):
            try:
                # Configure the model
                generation_config = {
                    "temperature": temperature,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }
                
                safety_settings = [
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
                
                try:
                    # List available models first
                    models = genai.list_models()
                    available_models = [m.name for m in models]
                    
                    if model not in available_models:
                        raise ValueError(f"Model {model} not found. Available models: {available_models}")
                    
                    # Create the model instance
                    model_instance = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Generate content
                    response = model_instance.generate_content(prompt)
                    end_time = time.time()
                    
                    # Handle response format
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
                
                    return {
                        "text": response_text,
                        "response_time": end_time - start_time
                    }
                
                except Exception as model_error:
                    print(f"Error with model configuration: {model_error}")
                    raise
                
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
            api: API type ('groq', 'gemini', or 'mistral')
            model: Model name
            temperature: Temperature setting
            
        Returns:
            API response
        """
        # Create the chain-of-thought prompt first
        cot_prompt = prompt + CHAIN_OF_THOUGHT_PROMPT + BASE_META_PROMPT
        
        # Then add the meta-prompts for reasoning type and confidence
        full_prompt = cot_prompt + REASONING_META_PROMPT + CONFIDENCE_META_PROMPT + FORMAT_PROMPT
        
        if api.lower() == "groq":
            return self.query_groq(full_prompt, model, temperature)
        elif api.lower() == "gemini":
            return self.query_gemini(full_prompt, model, temperature)
        elif api.lower() == "mistral":
            return self.query_mistral(full_prompt, model, temperature)
        else:
            raise ValueError(f"Unsupported API: {api}")
    
    def parse_response(self, response_text: str) -> Tuple[str, str, str, Optional[float]]:
        """
        Parse model response to extract answer, chain of thought, reasoning type, and confidence.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Tuple of (answer, chain_of_thought, reasoning_type, confidence)
        """
        # Reasoning labels
        reasoning_labels = ["Recall", "Reasoning", "Hallucination", "Uncertain"]
        
        # Default values
        full_answer = response_text
        chain_of_thought = ""
        final_answer = response_text
        reasoning_type = None
        confidence = None
        
        # Extract the chain of thought part
        # Look for common indicators of a step-by-step process
        step_indicators = ["Step 1", "First,", "To solve this", "Let's break this down", 
                          "I'll approach this", "Let me think", "Let's think", 
                          "To determine", "Let's calculate"]
        
        # Look for a reasoning process followed by an answer
        for indicator in step_indicators:
            if indicator in response_text:
                # Find where the final answer likely begins
                answer_indicators = ["Therefore,", "So,", "In conclusion,", "Thus,", 
                                     "The answer is", "This means", "To summarize", 
                                     "Finally,", "In summary"]
                
                for ans_ind in answer_indicators:
                    ans_parts = response_text.split(ans_ind)
                    if len(ans_parts) > 1:
                        chain_of_thought = ans_parts[0].strip()
                        final_answer = ans_ind + ans_parts[1].strip()
                        break
                
                if chain_of_thought:  # If we found a clean separation
                    break
        
        # If no clear structure was found, use heuristic 
        if not chain_of_thought:
            # Try to split by paragraph
            paragraphs = response_text.split("\n\n")
            if len(paragraphs) > 1:
                # Assume earlier paragraphs are reasoning and the last is the answer
                chain_of_thought = "\n\n".join(paragraphs[:-1])
                # final_answer = paragraphs[-1]
            else:
                pass
                # No clear way to separate, leave as is
                # final_answer = response_text
        
        # Parse the structured output format from the last line
        lines = response_text.split('\n')
        for line in reversed(lines):
            if line.strip():  # Find last non-empty line
                try:
                    # Try to parse JSON-like structure from the line
                    match = re.search(r"{'FINAL_ANSWER':\s*(.+?),\s*'LABEL':\s*(.+?),\s*'CONFIDENCE':\s*(.+?)}", line)
                    if match:
                        final_answer = match.group(1).strip()
                        reasoning_type = match.group(2).strip()
                        try:
                            confidence = float(match.group(3).strip())
                        except ValueError:
                            confidence = None
                        break
                except:
                    continue
        
        # Final cleanup
        final_answer = final_answer.strip()
        chain_of_thought = chain_of_thought.strip()
        print("output", final_answer, chain_of_thought, reasoning_type, confidence)

        return final_answer, chain_of_thought, reasoning_type, confidence
    
    def log_result(self, question_id: Any, modified_problem: str, api: str, model: str, 
                  final_answer: str, chain_of_thought: str, reasoning_type: str, 
                  confidence: Optional[float], response_time: float,
                  annotations: List[str], reasoning_count_metrics: Dict[str, int], reasoning_pct_metrics: Dict[str, float],
                  problem_type: str, subject: str, level: str,
                  verification_check: str = "Not Verified") -> None:
        """
        Add a result to the DataFrame.
        
        Args:
            question_id: Unique identifier for the question
            modified_problem: The modified problem text
            api: API used ('groq' or 'gemini')
            model: Model name
            answer: Model's answer
            chain_of_thought: Step-by-step reasoning process
            reasoning_type: Extracted reasoning type
            confidence: Confidence score (if available)
            response_time: Time taken for API response
            annotations: Annotations for the reasoning chain
            reasoning_count_metrics: Metrics for the reasoning chain
            reasoning_pct_metrics: Percentage metrics for the reasoning chain
            problem_type: Type of problem
            subject: Subject of the problem
            level: Level of the problem
            verification_check: Result of verification checks
        """
        # Perform external evaluation
        evaluation = "Pending"
        if reasoning_type == "Reasoning" and verification_check not in ["Verified Reasoning", "Novel Reasoning Verified", "Strong Reasoning Evidence"]:
            evaluation = "Potential False Reasoning Claim"
        elif reasoning_type == "Recall" and verification_check in ["Verified Reasoning", "Novel Reasoning Verified"]:
            evaluation = "Understated Reasoning Capability"
        elif verification_check in ["Incorrect", "Misled by Context", "Distracted by Irrelevant Details"]:
            evaluation = "Poor Reasoning"
        else:
            evaluation = "Consistent"
        
        # Create new row
        # Convert metrics dicts to dataframes
        count_metrics_df = pd.DataFrame([reasoning_count_metrics])
        pct_metrics_df = pd.DataFrame([reasoning_pct_metrics])
        
        # Create base dataframe
        new_row = pd.DataFrame({
            "question_id": [question_id],
            "modified_problem": [modified_problem],
            "api": [api],
            "model": [model],
            "final_answer": [final_answer],
            "chain_of_thought": [chain_of_thought],
            "reasoning_type": [reasoning_type],
            "confidence": [confidence],
            "response_time": [response_time],
            "problem_type": [problem_type],
            "subject": [subject],
            "level": [level],
            "timestamp": [datetime.datetime.now()],
            "annotations": [annotations],
            # "evaluation": [evaluation],
            # "verification_check": [verification_check]
        })
        
        # Add metrics columns
        for col in count_metrics_df.columns:
            new_row[col] = count_metrics_df[col]
        for col in pct_metrics_df.columns:
            new_row[col] = pct_metrics_df[col]
        # Add to DataFrame
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True, sort=False)

    def save_results(self, results_df: pd.DataFrame, file_name: str, format: str = "pkl") -> None:
        """
        Save results DataFrame to file.
        
        Args:
            results_df: DataFrame containing results to save
            file_path: Path for the output file
            format: Output format ('pkl', 'csv', or 'json')
        """
        try:
            # Create results directory if it doesn't exist
            
            # Update file path to be in results directory
            file_path = f"{file_name}.{format}"
            
            if format.lower() == "csv":
                # Handle potential encoding issues with CSV
                results_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"CSV saved: {file_path}")
            elif format.lower() == "json":
                # Convert DataFrame to JSON
                json_data = results_df.to_json(orient="records", indent=2)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                print(f"JSON saved: {file_path}")
            else:
                # Default to pickle format
                results_df.to_pickle(file_path)
                print(f"Pickle saved: {file_path}")
                
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()

    def annotate_reasoning_chain(self, text: str) -> tuple[list[str], str, dict[str, int]]:
        """
        Annotate each sentence in the reasoning chain with appropriate "handcrafted" labels and patterns.
        Returns annotated sentences and the final answer if found.
        """
        if not text:
            return [], ""

        annotations = []
        final_answer = ""

        # Enhanced labels for detecting reasoning types
        labels = {
            "initializing": r"(first|to begin|let's start|initial thought|to solve|approach)",
            "deduction": r"(therefore|so|thus|since|it follows that|hence|as a result|conclude|calculate|checking divisibility)",
            "adding-knowledge": r"(known fact|it is known|recall|by definition|according to|remember that)",
            "example-testing": r"(for example|for instance|consider|let's test|suppose|try)",
            "uncertainty-estimation": r"(might|could be|possibly|likely|uncertain|I'm not sure|I wonder)",
            "backtracking": r"(wait|on second thought|however|instead|better approach|let's change)",
            "checking": r"(verify|check|ensure|confirm|validate)"
        }

        # Stronger pattern matching for memorization vs. reasoning detection
        pattern_memorization = r"(recall|known fact|remember|by definition|it is known|formula|always true|rule|identity|theorem|lemma|law of|principle)"
        pattern_computation = r"(calculate|compute|evaluate|find|solve|determine)"
        pattern_reasoning = r"(deduce|therefore|hence|thus|conclude|derive|as a result|because|since|follows from|implies|means that)"
        pattern_exploration = r"(let's try|attempt|experiment|test|consider|what if|suppose)"
        pattern_uncertainty = r"(unsure|not certain|don't know|haven't seen|new to me|unfamiliar|never encountered|unusual)"

        # Split by sentences while handling potential edge cases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Metrics for analysis
        metrics = {
            "memorization_count": 0,
            "reasoning_count": 0,
            "exploration_count": 0,
            "uncertainty_count": 0,
            "computation_count": 0,
            "total_sentences": len(sentences)
        }

        for sentence in sentences:
            labeled = False
            for idx, (label, pattern) in enumerate(labels.items()):
                if re.search(pattern, sentence, re.IGNORECASE):
                    keywords = []

                    # Check for memorization indicators
                    if re.search(pattern_memorization, sentence, re.IGNORECASE):
                        keywords.append("M")
                        metrics["memorization_count"] += 1

                    # Check for reasoning indicators
                    if re.search(pattern_reasoning, sentence, re.IGNORECASE):
                        keywords.append("AcR")
                        metrics["reasoning_count"] += 1

                    # Check for exploration indicators
                    if re.search(pattern_exploration, sentence, re.IGNORECASE):
                        keywords.append("Exp")
                        metrics["exploration_count"] += 1

                    # Check for uncertainty indicators
                    if re.search(pattern_uncertainty, sentence, re.IGNORECASE):
                        keywords.append("Unc")
                        metrics["uncertainty_count"] += 1

                    # Check for computation indicators
                    if re.search(pattern_computation, sentence, re.IGNORECASE):
                        keywords.append("Comp")
                        metrics["computation_count"] += 1

                    keyword_str = f" {{{', '.join(keywords)}}}" if keywords else ""
                    annotation = f'["{idx}. {label}"] {sentence} {keyword_str} ["end-section"]'
                    annotations.append(annotation)
                    labeled = True
                    break

            if not labeled:
                # Even for unlabeled sentences, check for our special patterns
                keywords = []

                if re.search(pattern_memorization, sentence, re.IGNORECASE):
                    keywords.append("M")
                    metrics["memorization_count"] += 1

                if re.search(pattern_reasoning, sentence, re.IGNORECASE):
                    keywords.append("AcR")
                    metrics["reasoning_count"] += 1

                if re.search(pattern_exploration, sentence, re.IGNORECASE):
                    keywords.append("Exp")
                    metrics["exploration_count"] += 1

                if re.search(pattern_uncertainty, sentence, re.IGNORECASE):
                    keywords.append("Unc")
                    metrics["uncertainty_count"] += 1

                if re.search(pattern_computation, sentence, re.IGNORECASE):
                    keywords.append("Comp")
                    metrics["computation_count"] += 1

                keyword_str = f" {{{', '.join(keywords)}}}" if keywords else ""
                annotations.append(f'["7. separate"] {sentence} {keyword_str} ["end-section"]')

        # Extract final answer
        final_answer_match = re.search(r'final answer\s*[:\-]?\s*(.*)', text, re.IGNORECASE)
        if final_answer_match:
            # print("final_answer_match")
            final_answer = final_answer_match.group(1).strip()
        else:
            # Try to extract answer from the last few sentences if no explicit final answer
            last_sentences = " ".join(sentences[-3:]) if len(sentences) >= 3 else " ".join(sentences)
            answer_patterns = [
                r'answer\s*[:\-]?\s*(.*)',
                r'result\s*[:\-]?\s*(.*)',
                r'solution\s*[:\-]?\s*(.*)',
                r'coordinates\s*[:\-]?\s*(.*)',
                r'\(([\d\.\-]+),\s*([\d\.\-π\/\s]+)\)',  # For coordinates like (r,θ)
                r'boxed\{([^}]+)\}',  # For LaTeX boxed answers
                r'final answer\s*[:\-]?\s*(.*)',
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, last_sentences, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip() if len(match.groups()) == 1 else match.group(0).strip()
                    break
        return annotations, final_answer, metrics

    def compute_reasoning_metrics(self, annotations, final_answer, original_text, metrics):
        """
        Compute the reasoning metrics.
        """

        # Calculate reasoning pattern metrics and indicators
        total_indicators = sum(metrics.values()) - metrics["total_sentences"]

        # Initialize indicator metrics dict
        indicator_metrics_dict = {
            "memorization_pct": 0,
            "reasoning_pct": 0, 
            "exploration_pct": 0,
            "uncertainty_pct": 0,
            "computation_pct": 0,
            "primary_approach": "Balanced",
            "secondary_approach": ""
        }

        # Avoid division by zero
        if total_indicators > 0:
            indicator_metrics_dict["memorization_pct"] = (metrics["memorization_count"] / total_indicators) * 100
            indicator_metrics_dict["reasoning_pct"] = (metrics["reasoning_count"] / total_indicators) * 100
            indicator_metrics_dict["exploration_pct"] = (metrics["exploration_count"] / total_indicators) * 100
            indicator_metrics_dict["uncertainty_pct"] = (metrics["uncertainty_count"] / total_indicators) * 100
            indicator_metrics_dict["computation_pct"] = (metrics["computation_count"] / total_indicators) * 100

        # Categorize the approach based on the metrics
        if indicator_metrics_dict["memorization_pct"] > 40:
            indicator_metrics_dict["primary_approach"] = "Memorization"
        elif indicator_metrics_dict["reasoning_pct"] > 40:
            indicator_metrics_dict["primary_approach"] = "Reasoning"
        elif indicator_metrics_dict["computation_pct"] > 40:
            indicator_metrics_dict["primary_approach"] = "Computation"

        if indicator_metrics_dict["uncertainty_pct"] > 15:
            indicator_metrics_dict["secondary_approach"] = " with Uncertainty"
        elif indicator_metrics_dict["exploration_pct"] > 15:
            indicator_metrics_dict["secondary_approach"] = " with Exploration"

        # Store interpretation in dict
        # Build up interpretation string
        interpretation = ""
        
        # Determine primary approach
        if indicator_metrics_dict["memorization_pct"] > indicator_metrics_dict["reasoning_pct"] and indicator_metrics_dict["memorization_pct"] > indicator_metrics_dict["exploration_pct"]:
            interpretation = "The model appears to be RECALLING knowledge or formulas from its training data. This indicates the problem or a similar one may have been seen during training."
        elif indicator_metrics_dict["reasoning_pct"] > indicator_metrics_dict["memorization_pct"] and indicator_metrics_dict["reasoning_pct"] > indicator_metrics_dict["exploration_pct"]:
            interpretation = "The model is primarily using REASONING to derive the answer step by step. This indicates the model is applying general principles rather than recalling specific solutions."
        elif indicator_metrics_dict["exploration_pct"] > indicator_metrics_dict["memorization_pct"] and indicator_metrics_dict["exploration_pct"] > indicator_metrics_dict["reasoning_pct"]:
            interpretation = "The model is EXPLORING different approaches or testing hypotheses. This suggests it's applying problem-solving techniques rather than recalling solutions."

        # Add uncertainty assessment if present
        if indicator_metrics_dict["uncertainty_pct"] > 15:
            interpretation += " The model expresses UNCERTAINTY, suggesting it may not have seen this exact problem before and is working through unfamiliar territory."

        # Add computation assessment if present  
        if indicator_metrics_dict["computation_pct"] > 30:
            interpretation += " The response contains significant COMPUTATION, showing the model is calculating a solution rather than simply recalling it."
            
        # Store final interpretation
        indicator_metrics_dict["interpretation"] = interpretation

        # Generate and store overall assessment
        if indicator_metrics_dict["memorization_pct"] > 30 and indicator_metrics_dict["uncertainty_pct"] < 10:
            indicator_metrics_dict["likelihood_assessment"] = "HIGH likelihood the model has seen similar problems during training and is recalling patterns."
        elif indicator_metrics_dict["reasoning_pct"] > 30 and (indicator_metrics_dict["exploration_pct"] > 10 or indicator_metrics_dict["uncertainty_pct"] > 10):
            indicator_metrics_dict["likelihood_assessment"] = "HIGH likelihood the model is reasoning through a new problem rather than recalling a solution."
        else:
            indicator_metrics_dict["likelihood_assessment"] = "MEDIUM likelihood of either approach - model is using a mix of recall and reasoning."

        # result += indicator_metrics_dict["likelihood_assessment"] + "\n"

        return indicator_metrics_dict


    def analyze_reasoning(self, response):
        """
        Analyze reasoning for a given model response.
        """
        if response:
            annotations, final_answer, reasoning_count_metrics = self.annotate_reasoning_chain(response)
            reasoning_pct_metrics = self.compute_reasoning_metrics(annotations, final_answer, response, reasoning_count_metrics)
            return annotations, final_answer, reasoning_count_metrics, reasoning_pct_metrics
        else:
            return {}, "", {}, {}


    def run(self, input_data: Union[str, List[str], List[Dict[str, str]]], folder_name: str = "run_other",
        output_file: str = "results.csv", output_format: str = "csv") -> None:
        """
        Run the framework on input data.
        
        Args:
            input_data: File path, direct question, or list of questions/prompts
            output_file: Path for saving results (default: results.csv)
            output_format: Format to save results in (default: csv)
        """
        # Get API keys
        self.set_api_keys()
        
        # Load prompts with enhanced input support
        prompts = self.load_prompts(input_data)
        if not prompts:
            print("No prompts loaded. Exiting.")
            return
            
        print(f"\nLoaded {len(prompts)} prompts.")
        
        print("\nStarting evaluation...")
        
        # Process each prompt with each model
        with tqdm(total=len(prompts) * len(self.models)) as pbar:
            for prompt in prompts:
                
                if isinstance(prompt, dict) and "modified_problem" in prompt:
                    prompt_id = prompt.get("question_id", "unknown")
                    prompt_text = prompt["modified_problem"]
                    problem_type = prompt["problem_type"]
                    subject = prompt["subject"]
                    level = prompt["level"]
                    
                    
                    for model_config in self.models:
                        try:
                            # Query model
                            while True:
                                response = self.query_model(
                                    prompt_text,
                                    model_config["api"],
                                    model_config["model"],
                                    model_config.get("temperature", 0.0)
                                )
                                print("response", response)
                                
                                if "Error" in response["text"]:
                                    if "rate_limit_exceeded" in response["text"]:
                                        error_msg = json.loads(response["text"])["error"]["message"]
                                        wait_time_match = re.search(r"try again in (\d+)m(\d+\.\d+)s", error_msg)
                                        if wait_time_match:
                                            minutes = int(wait_time_match.group(1))
                                            seconds = float(wait_time_match.group(2))
                                            print(f"\nRate limit reached for prompt {prompt_id}. Need to wait {minutes}m {seconds:.2f}s before retrying.")
                                            time.sleep(minutes * 60 + seconds)
                                            continue
                                    print(f"\nError with prompt {prompt_id}: {response['text']}")
                                    break
                                break
                            
                            prepared_response = response["text"].replace("\\","\\\\")
                            # Parse response as a whole
                            answer, chain_of_thought, reasoning_type, confidence = self.parse_response(prepared_response)
                            # analyze with reasoning indicators for annotations
                            annotations, final_answer, reasoning_count_metrics, reasoning_pct_metrics = self.analyze_reasoning(prepared_response)
                            
                            # Log result
                            self.log_result(
                                question_id=prompt_id,
                                modified_problem=prompt_text,
                                api=model_config["api"],
                                model=model_config["model"],
                                final_answer=answer,
                                chain_of_thought=chain_of_thought,
                                reasoning_type=reasoning_type,
                                confidence=confidence,
                                response_time=response["response_time"],
                                annotations=annotations,
                                reasoning_count_metrics=reasoning_count_metrics,
                                reasoning_pct_metrics=reasoning_pct_metrics,
                                problem_type=problem_type,
                                subject=subject,
                                level=level,
                            )
                            
                        except Exception as e:
                            print(f"\nError processing prompt {prompt_id}: {e}")
                            traceback.print_exc()
                            
                        finally:
                            pbar.update(1)
        
        # Save results for each model separately
        print("\nSaving results...")
        # Create timestamped results directory
        
        results_dir = os.path.join("results", folder_name)
        os.makedirs(results_dir, exist_ok=True)
        
        for model_config in self.models:
            model_results = self.results_df[self.results_df['model'] == model_config['model']]
            if not model_results.empty:
                # Create model-specific output file path
                model_output_file = f"{output_file.rsplit('.', 1)[0]}_{model_config['model']}"
                extension = output_file.rsplit('.', 1)[1]
                # Save using the save_results method
                self.save_results(model_results, os.path.join(results_dir, model_output_file), extension)
                self.save_results(model_results, os.path.join(results_dir, model_output_file), "json")

        self.save_results(self.results_df, os.path.join(results_dir, "all_results_combined"), "csv")
        self.save_results(self.results_df, os.path.join(results_dir, "all_results_combined"), "json")
        print("\nEvaluation complete!")

if __name__ == "__main__":
    # Example usage
    models = [
        # {"api": "groq", "model": "llama3-70b-8192", "temperature": 0.0},
        # {"api": "groq", "model": "deepseek-r1-distill-llama-70b", "temperature": 0.0},
        {"api": "gemini", "model": "gemini-2.0-flash", "temperature": 0.0},
        # {"api": "gemini", "model": "gemini-1.5-flash", "temperature": 0.0},
        # # {"api": "gemini", "model": "gemini-1.5-pro", "temperature": 0.0},
        # {"api": "gemini", "model": "gemini-2.0-flash-thinking-exp", "temperature": 0.0},
        # {"api": "mistral", "model": "mistral-large-latest", "temperature": 0.0},
    ]
    
    framework = LLMReasoningFramework(models)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    framework.run("C:/llmination-reasoning/data/Kamesh_assigned.json", folder_name=f"run_{timestamp}_final_testing_gemini")
    