import json
import openai
from typing import Dict, List, Any
import os
from datetime import datetime
import dotenv
import time

dotenv.load_dotenv("../.env")

class OpenAIPromptProcessor:
    def __init__(self, api_key: str = None):
        """Initialize the OpenAI client with API key."""
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get API key from environment variable
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Rate limiting: track last API call time
        self.last_api_call_time = 0
        self.min_interval = 20  # 20 seconds between calls (3 calls per minute)
    
    def load_context_data(self, context_file_path: str, csv_file_path: str = "missing_answer_or_code.csv") -> List[Dict[str, Any]]:
        """Load context data from JSON file filtered by question IDs from CSV file."""
        import csv
        
        # Load question IDs from CSV file
        question_ids = set()
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    question_ids.add(row['question_id'])
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        except KeyError:
            raise KeyError("CSV file must contain 'question_id' column")
        print(f"Loaded {len(question_ids)} question IDs from CSV file")
        # Load JSON data and filter by question IDs
        try:
            with open(context_file_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Context file not found: {context_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {context_file_path}")
        print(f"Loaded {len(all_data)} problems from perplexmath-dataset-answers.json file")
        # Filter data to only include items with question_ids from CSV
        filtered_data = []
        for item in all_data:
            if item.get('question_id') in question_ids:
                filtered_data.append(item)
        print(f"Filtered {len(filtered_data)} problems from perplexmath-dataset-answers.json file")
        return filtered_data
    def create_prompt(self, base_template: str, context_data: Dict[str, Any]) -> str:
        """Create a prompt by inserting context data into the base template."""
        try:
            # Replace placeholders in the template with context data
            # Supports {key} format placeholders
            return base_template.format(**context_data)
        except KeyError as e:
            raise KeyError(f"Missing key in context data: {e}")
    
    def call_openai_api(self, prompt: str, model: str = "gpt-3.5-turbo", 
                       max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Make API call to OpenAI and return the response."""
        import time
        import random
        
        # Rate limiting: ensure minimum interval between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            print(f"Rate limiting: waiting {sleep_time:.2f} seconds before API call")
            time.sleep(sleep_time)
        
        max_retries = 2
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.last_api_call_time = time.time()  # Update last call time
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except openai.error.RateLimitError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit, waiting {delay:.2f} seconds before retry {attempt + 2}/{max_retries}")
                    time.sleep(delay)
                else:
                    raise Exception(f"OpenAI API call failed after {max_retries} attempts due to rate limiting: {str(e)}")
            except openai.error.APIError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"API error, waiting {delay} seconds before retry {attempt + 2}/{max_retries}")
                    time.sleep(delay)
                else:
                    raise Exception(f"OpenAI API call failed after {max_retries} attempts due to API error: {str(e)}")
            except Exception as e:
                raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def process_prompts(self, base_template: str, context_file_path: str, 
                       output_file_path: str, model: str = "gpt-3.5-turbo",
                       max_tokens: int = 1000, temperature: float = 0.7) -> None:
        """Process all prompts and save results to JSON file."""
        # Load context data
        context_data_list = self.load_context_data(context_file_path)
        
        results, internal_results = [], []
        
        for i, context_data in enumerate(context_data_list[:2]):
            try:
                # Create prompt with context
                prompt = self.create_prompt(base_template, context_data)
                
                # Call OpenAI API
                response = self.call_openai_api(prompt, model, max_tokens, temperature)
                response = json.loads(response)
                # Store result
                internal_result = {
                    "index": i,
                    "question_id": context_data["question_id"],
                    "context_data": context_data,
                    "prompt": prompt,
                    "response": response,
                    # "timestamp": datetime.now().isoformat(),
                    "model": model
                }
                results.append(response)
                internal_results.append(internal_result)
                
                print(f"Processed item {i + 1}/{len(context_data_list)}")
                
            except Exception as e:
                # Store error information
                error_result = {
                    "index": i,
                    "context_data": context_data,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                internal_results.append(error_result)
                print(f"Error processing item {i + 1}: {str(e)}")
        
        # Save results to JSON file
        self.save_results(results, output_file_path)
        print(f"Results saved to {output_file_path}")
        self.save_results(internal_results, output_file_path.replace(".json", "_internal.json"))
        print(f"Internal results saved to {output_file_path.replace(".json", "_internal.json")}")
    
    def save_results(self, results: List[Dict[str, Any]], output_file_path: str) -> None:
        """Save results to JSON file."""
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Failed to save results: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = OpenAIPromptProcessor()
    
    # Define base prompt template
    base_template = """
    Given this json dictionary, for each problem_type "{problem_type}", replace it using the following process:
    given this input:
    {{
    "problem_type": "{problem_type}",
    "modified_problem": "{modified_problem}",
    "explanation": "{explanation}",
    "original_problem": "{original_problem}",
    "original_answer": "{original_answer}",
    "subject": "{subject}",
    "level": "{level}",
    "unique_id": "{unique_id}",
    "question_id": "{question_id}"
    }},

    Output the same thing except with two fields added: “correct_answer”, which contains the correct answer to the “modified_problem” and "solution_code", where the corresponding value is a string containing code to solve the problem in the "modified_problem" field. The correct_answer of impossible context problems should be “no solution”.
    """
    
    # Process prompts
    try:
        processor.process_prompts(
            base_template=base_template,
            context_file_path="perplexmath-dataset.json",
            output_file_path="openai_gpt3.5_turbo_results.json",
            model="gpt-3.5-turbo",
            max_tokens=1500,
            temperature=0
        )
    except Exception as e:
        print(f"Error: {str(e)}")
