"""
Main framework for LLM reasoning evaluation.
"""

import os
import json
import datetime
import getpass
import traceback
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from tqdm import tqdm

from ..api_clients.client_factory import create_api_client
from ..analysis.reasoning_analyzer import ReasoningAnalyzer
from ..analysis.self_evaluation import SelfEvaluation
from ..utils.response_parser import ResponseParser

from ..config.api_config import APIConfig

class LLMReasoningFramework:
    """Main framework for evaluating LLM reasoning capabilities."""
    
    def __init__(self, models: List[Dict[str, Any]] = None):
        """
        Initialize the framework with specified models.
        
        Args:
            models: List of model configurations
        """
        self.models = models or []
        self.results_df = pd.DataFrame()
        self.api_clients = {}
        
        # Initialize analysis components
        self.reasoning_analyzer = ReasoningAnalyzer()
        self.self_evaluation = SelfEvaluation()
        
        # Initialize API clients for each model
        self._initialize_api_clients()
    
    def _initialize_api_clients(self):
        """Initialize API clients for each model."""
        for model_config in self.models:
            api_type = model_config["api"]
            if api_type not in self.api_clients:
                try:
                    self.api_clients[api_type] = create_api_client(api_type)
                except Exception as e:
                    print(f"Warning: Failed to initialize {api_type} client: {e}")
    
    def set_api_keys(self):
        """Get API keys from environment or prompt the user."""
        # Try to get from environment first
        groq_key = APIConfig.GROQ_API_KEY
        gemini_key = APIConfig.GEMINI_API_KEY
        mistral_key = APIConfig.MISTRAL_API_KEY
        
        # Validate and prompt for missing keys
        for model_config in self.models:
            api_type = model_config["api"]
            if api_type == "groq" and not groq_key:
                print("\nGroq API key not found in environment variables.")
                groq_key = getpass.getpass("Enter your Groq API key: ")
                if not APIConfig.validate_api_key("groq", groq_key):
                    print("\nWarning: Groq API key appears invalid. It should start with 'gsk_'")
                    retry = input("Would you like to enter the key again? (y/n): ")
                    if retry.lower() == 'y':
                        groq_key = getpass.getpass("Enter your Groq API key: ")
                print("Groq API key validated")
            
            elif api_type == "gemini" and not gemini_key:
                print("\nGemini API key not found in environment variables.")
                gemini_key = getpass.getpass("Enter your Gemini API key: ")
            
            elif api_type == "mistral" and not mistral_key:
                print("\nMistral API key not found in environment variables.")
                mistral_key = getpass.getpass("Enter your Mistral API key: ")
                print("Mistral API key validated")
        
        # Update API clients with new keys
        if groq_key:
            try:
                self.api_clients["groq"] = create_api_client("groq", groq_key)
            except Exception as e:
                print(f"Warning: Failed to update Groq client: {e}")
        
        if gemini_key:
            try:
                self.api_clients["gemini"] = create_api_client("gemini", gemini_key)
            except Exception as e:
                print(f"Warning: Failed to update Gemini client: {e}")
        
        if mistral_key:
            try:
                self.api_clients["mistral"] = create_api_client("mistral", mistral_key)
            except Exception as e:
                print(f"Warning: Failed to update Mistral client: {e}")
    
    def load_prompts(self, input_data: Union[str, List[str], List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """
        Load prompts from text, CSV, JSON file, or direct input.
        
        Args:
            input_data: Can be file path, direct question, or list of questions/prompts
                
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
        if api not in self.api_clients:
            raise ValueError(f"API client for {api} not initialized")
        
        return self.api_clients[api].query(prompt, model, temperature)
    
    def parse_response(self, response_text: str) -> tuple[str, str, str, Optional[float]]:
        """
        Parse model response to extract answer, chain of thought, reasoning type, and confidence.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Tuple of (answer, chain_of_thought, reasoning_type, confidence)
        """
        return ResponseParser.parse_response(response_text)
    
    def log_result(self, question_id: Any, modified_problem: str, api: str, model: str, 
                  final_answer: str, chain_of_thought: str, reasoning_type: str, 
                  confidence: Optional[float], response_time: float,
                  annotations: List[str], reasoning_count_metrics: Dict[str, int], 
                  reasoning_pct_metrics: Dict[str, float],
                  problem_type: str, subject: str, level: str,
                  verification_check: str = "Not Verified") -> None:
        """
        Add a result to the DataFrame.
        
        Args:
            question_id: Unique identifier for the question
            modified_problem: The modified problem text
            api: API used ('groq' or 'gemini')
            model: Model name
            final_answer: Model's answer
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
            file_name: Base name for the output file
            format: Output format ('pkl', 'csv', or 'json')
        """
        try:
            file_path = f"{file_name}.{format}"
            
            if format.lower() == "csv":
                results_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"CSV saved: {file_path}")
            elif format.lower() == "json":
                json_data = results_df.to_json(orient="records", indent=2)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                print(f"JSON saved: {file_path}")
            else:
                results_df.to_pickle(file_path)
                print(f"Pickle saved: {file_path}")
                
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()
    
    def analyze_reasoning_keyword(self, response: str) -> tuple[List[str], str, Dict[str, int], Dict[str, float]]:
        """Analyze reasoning for a given model response using keyword analysis."""
        if not response:
            return [], "", {}, {}
        
        annotations, final_answer, reasoning_count_metrics = self.reasoning_analyzer.annotate_reasoning_chain(response)
        reasoning_pct_metrics = self.reasoning_analyzer.compute_reasoning_metrics(
            annotations, final_answer, response, reasoning_count_metrics
        )
        
        return annotations, final_answer, reasoning_count_metrics, reasoning_pct_metrics
    
    def analyze_reasoning_self(self, response: str, model_config: Dict[str, Any]) -> tuple[List[str], Dict[str, int], Dict[str, Any]]:
        """Analyze reasoning for a given model response using self-evaluation."""
        if not response:
            return {}, {}, {}
        
        # Set API client for self-evaluation
        api_type = model_config["api"]
        if api_type in self.api_clients:
            self.self_evaluation.set_api_client(self.api_clients[api_type])
        
        annotations = self.self_evaluation.annotate_reasoning_chain_with_prompt(response, model_config)
        annotations_self_parsed, keyword_counts = self.self_evaluation.parse_reasoning_indicators(annotations)
        reasoning_indicator_metrics = self.self_evaluation.compute_annotation_metrics(
            annotations_self_parsed, keyword_counts
        )
        
        return annotations, keyword_counts, reasoning_indicator_metrics
    
    def run(self, input_data: Union[str, List[str], List[Dict[str, str]]], 
            is_self_reason: bool = False, folder_name: str = "run_other",
            output_file: str = "results.csv", output_format: str = "csv") -> None:
        """
        Run the framework on input data.
        
        Args:
            input_data: File path, direct question, or list of questions/prompts
            is_self_reason: Whether to use self-evaluation
            folder_name: Name for the results folder
            output_file: Path for saving results
            output_format: Format to save results in
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
                            response = self._query_model_with_retry(prompt_text, model_config, prompt_id)
                            
                            if "Error" in response["text"]:
                                print(f"\nError with prompt {prompt_id}: {response['text']}")
                                continue
                            
                            prepared_response = ResponseParser.clean_response_text(response["text"])
                            
                            # Parse response
                            answer, chain_of_thought, reasoning_type, confidence = self.parse_response(prepared_response)
                            
                            # Analyze reasoning
                            if is_self_reason:
                                annotations, keyword_counts, reasoning_indicator_metrics = self.analyze_reasoning_self(
                                    prepared_response, model_config
                                )
                                print(f"\nkeyword_counts: {keyword_counts}")
                                print(f"annotations: {annotations}")
                                print(f"reasoning_indicator_metrics: {reasoning_indicator_metrics}")
                            else:
                                annotations, final_answer, reasoning_count_metrics, reasoning_pct_metrics = self.analyze_reasoning_keyword(
                                    prepared_response
                                )
                                reasoning_indicator_metrics = reasoning_count_metrics
                            
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
                                problem_type=problem_type,
                                subject=subject,
                                level=level,
                                reasoning_count_metrics=reasoning_indicator_metrics if is_self_reason else reasoning_count_metrics,
                                reasoning_pct_metrics=reasoning_pct_metrics if not is_self_reason else {},
                            )
                            
                        except Exception as e:
                            print(f"\nError processing prompt {prompt_id}: {e}")
                            traceback.print_exc()
                            
                        finally:
                            pbar.update(1)
        
        # Save results
        print("\nSaving results...")
        results_dir = os.path.join("results", folder_name)
        os.makedirs(results_dir, exist_ok=True)
        
        for model_config in self.models:
            model_results = self.results_df[self.results_df['model'] == model_config['model']]
            if not model_results.empty:
                model_output_file = f"{output_file.rsplit('.', 1)[0]}_{model_config['model']}"
                extension = output_file.rsplit('.', 1)[1]
                self.save_results(model_results, os.path.join(results_dir, model_output_file), extension)
                self.save_results(model_results, os.path.join(results_dir, model_output_file), "json")
        
        self.save_results(self.results_df, os.path.join(results_dir, "all_results_combined"), "csv")
        self.save_results(self.results_df, os.path.join(results_dir, "all_results_combined"), "json")
        print("\nEvaluation complete!")
    
    def _query_model_with_retry(self, prompt_text: str, model_config: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
        """Query model with retry logic for rate limiting."""
        while True:
            response = self.query_model(
                prompt_text,
                model_config["api"],
                model_config["model"],
                model_config.get("temperature", 0.0)
            )
            
            if "Error" in response["text"]:
                if "rate_limit_exceeded" in response["text"]:
                    # Handle rate limiting
                    try:
                        error_msg = json.loads(response["text"])
                        if "error" in error_msg and "message" in error_msg["error"]:
                            wait_time_match = re.search(r"try again in (\d+)m(\d+\.\d+)s", error_msg["error"]["message"])
                            if wait_time_match:
                                minutes = int(wait_time_match.group(1))
                                seconds = float(wait_time_match.group(2))
                                print(f"\nRate limit reached for prompt {prompt_id}. Need to wait {minutes}m {seconds:.2f}s before retrying.")
                                import time
                                time.sleep(minutes * 60 + seconds)
                                continue
                    except:
                        pass
                
                print(f"\nError with prompt {prompt_id}: {response['text']}")
                break
            
            break
        
        return response
