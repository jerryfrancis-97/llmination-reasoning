import json
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
import time
import re
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from threading import Lock
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Rate limiting configuration
REQUESTS_PER_MINUTE = 15
request_times = []
request_lock = Lock()

# Prompt template for getting answers
ANSWER_PROMPT_TEMPLATE = """
Please solve the following math problem. Provide the final answer, reasoning (thinking process), and Python code using libraries like numpy, scipy, sympy, etc. to solve it programmatically, in the format specified below.

Problem:
{problem}

Important:
- Keep LaTeX formatting intact for all the content in the response
- Maintain the format for the response exactly as shown below
- If code cannot be generated for a problem, leave it blank.
- If the problem is not solvable, return "No solution found" as the final answer and leave the code sectionblank.
- There should always be a final answer.

Format your response exactly like this:
{{
    "final_answer": "The numerical or text answer",
    "reasoning": "<think> ... </think>", # this should contain the thinking process
    "solution_code": "# def solve_problem():\n    # Your solution code here\n    return final_answer"
}}
"""

def extract_fields(text):
    """Extract fields from response text using string find"""
    # Find final_answer
    final_start = text.find('"final_answer\": \"') + len('"final_answer\": \"')
    final_end = text.find('\",\n', final_start)
    final_answer = text[final_start:final_end] if final_start > -1 and final_end > -1 else None
    if final_answer:
        final_answer = final_answer.replace("\\\\", "\\")

    # Find reasoning 
    reason_start = text.find('\"reasoning\": \"') + len('\"reasoning\": \"')
    reason_end = text.find('\",\n ', reason_start)
    reasoning = text[reason_start:reason_end] if reason_start > -1 and reason_end > -1 else None

    # Find solution code
    code_start = text.find('\"solution_code\": \"') + len('\"solution_code\": \"')
    code_end = text.rfind('\"\n}\n')
    solution_code = text[code_start:code_end].replace('\\n', '\n') if code_start > -1 and code_end > -1 else None

    return {
        "final_answer": final_answer,
        "reasoning": reasoning, 
        "solution_code": solution_code
    }

def load_problems(file_path):
    """Load problems from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading problems: {str(e)}")
        return []

def wait_for_rate_limit():
    """Wait if necessary to stay within rate limits"""
    with request_lock:
        now = datetime.now()
        # Remove requests older than 1 minute
        while request_times and now - request_times[0] > timedelta(minutes=1):
            request_times.pop(0)
        
        # If at rate limit, wait until oldest request expires
        if len(request_times) >= REQUESTS_PER_MINUTE:
            wait_time = (request_times[0] + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                time.sleep(wait_time)
                
        # Add current request time
        request_times.append(now)

def get_openai_response(prompt, model="gpt-4-turbo-preview", temperature=0, max_tokens=1000):
    """
    Get response from OpenAI API for a given prompt.
    
    Args:
        prompt (str): The input prompt to send to the model
        model (str): The OpenAI model to use
        temperature (float): Controls randomness in the response
        max_tokens (int): Maximum number of tokens in the response
        
    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the request was successful
            - response (str): The model's response text
            - error (str): Error message if any
    """
    try:
        # Wait for rate limit before making request
        wait_for_rate_limit()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Make API request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
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

def get_openai_response_with_retry(prompt, max_retries=3, **kwargs):
    """
    Get response from OpenAI API with retry logic.
    
    Args:
        prompt (str): The input prompt to send to the model
        max_retries (int): Maximum number of retry attempts
        **kwargs: Additional arguments to pass to get_openai_response
        
    Returns:
        dict: Response from get_openai_response
    """
    for attempt in range(max_retries):
        result = get_openai_response(prompt, **kwargs)
        
        if result["success"]:
            return result
            
        print(f"Attempt {attempt + 1} failed: {result['error']}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
            
    return result

def get_problem_solution(problem_text, api_key, model_type="gemini"):
    """Get solution for a problem using either Gemini or OpenAI API"""
    # Prepare the prompt
    problem_text = problem_text.replace("\\", "\\\\")
    prompt = ANSWER_PROMPT_TEMPLATE.format(problem=problem_text)

    try:
        attempts = 0
        max_attempts = 2
        
        while attempts < max_attempts:
            try:
                # Wait for rate limit before making request
                wait_for_rate_limit()
                
                if model_type == "gemini":
                    # Configure API for this process
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
                    generation_config = {
                        "temperature": 0,
                        "max_output_tokens": 100000,
                    }
                    response = model.generate_content(prompt, generation_config=generation_config)
                    
                    # Extract text from response
                    if "GenerateContentResponse" in response and "MAX_TOKENS" in response:
                        print("Max output tokens reached")
                        return None
                    
                    if isinstance(response, GenerateContentResponse) and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text
                    else:
                        response_text = str(response)
                        
                else:  # OpenAI
                    result = get_openai_response_with_retry(
                        prompt,
                        model="gpt-4.1-nano",
                        temperature=0,
                        max_tokens=100000
                    )
                    
                    if not result["success"]:
                        print(f"OpenAI API error: {result['error']}")
                        return None
                        
                    response_text = result["response"]
                
                # Clean up and parse JSON response
                response_text = response_text.strip()
                
                if "GenerateContentResponse" in response_text and "MAX_TOKENS" in response_text:
                    print("Max output tokens reached")
                    return None
                    
                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '')
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '')
                response_text = response_text.replace("\\", "\\\\")
                
                # Try JSON parsing
                try:
                    solution_data = extract_fields(response_text)
                    print("extracted-------------------------------")
                    print(solution_data)
                except Exception as e:
                    print(f"Error extracting fields: {str(e)}")
                    solution_data = None
                
                # Verify we have all required fields
                if all(solution_data.get(field) for field in ["final_answer", "reasoning", "solution_code"]):
                    return solution_data
                else:
                    # Print which fields are missing
                    missing_fields = [field for field in ["final_answer", "reasoning", "solution_code"] 
                                   if not solution_data.get(field)]
                    print(f"Missing required fields in solution: {', '.join(missing_fields)}")
                    return solution_data
                    
            except Exception as e:
                print(f"Attempt {attempts+1} failed: {str(e)}")
                attempts += 1
                time.sleep(2)
        
        print(f"Failed to get complete solution after {max_attempts} attempts")
        return solution_data
        
    except Exception as e:
        print(f"Error getting solution: {str(e)}")
        return None

def process_problem(args):
    """Process a single problem and return results"""
    problem, api_key, model_type = args
    problem_type = problem.get('problem_type', '')
    
    # Handle irrelevant info problems directly
    if problem_type in ['irrelevant_info']:
        problem['reasoning'] = "Similar to original problem"
        problem['correct_answer'] = problem['original_answer']
        problem['solution_code'] = ""
        return problem, True
        
    # For other problem types, use API
    modified_problem = problem.get('modified_problem', '')
    if modified_problem:
        solution = get_problem_solution(modified_problem, api_key, model_type)
        if solution:
            if 'reasoning' in solution:
                problem['reasoning'] = solution['reasoning']
            if 'final_answer' in solution:
                problem['correct_answer'] = solution['final_answer']
            if 'solution_code' in solution:
                problem['solution_code'] = solution['solution_code']
            return problem, True
    return problem, False

def update_output_file(problem, output_file):
    """Update output file with new problem solution"""
    try:
        # Load existing data if file exists
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
            
        # Add new problem
        data.append(problem)
        
        # Write updated data back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error updating output file: {str(e)}")

def main():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INPUT_FILE = "perplexmath-dataset.json"
    OUTPUT_FILE = "perplexmath-generated-answers-openai.json"
    MODEL_TYPE = "openai"  # or "gemini"

    # Delete output file if it exists
    try:
        os.remove(OUTPUT_FILE)
        print(f"Deleted existing {OUTPUT_FILE} to start clean")
    except FileNotFoundError:
        pass

    # Load problems 
    print(f"Loading problems from {INPUT_FILE}...")
    problems = load_problems(INPUT_FILE)
    print(f"Loaded {len(problems)} problems")

    if len(problems) == 0:
        print("No problems loaded. Exiting.")
        return

    # Create a pool of workers
    num_processes = 5
    pool = mp.Pool(processes=num_processes)

    # Prepare arguments for each problem
    api_key = OPENAI_API_KEY if MODEL_TYPE == "openai" else GEMINI_API_KEY
    problem_args = [(problem, api_key, MODEL_TYPE) for problem in problems]

    # Process problems in parallel with progress bar
    results = []
    for result in tqdm(pool.imap(process_problem, problem_args), total=len(problems), desc="Processing problems"):
        problem, success = result
        if success:
            print("Successfully added solution")
            # Update output file immediately after each successful solution
            update_output_file(problem, OUTPUT_FILE)
            results.append(problem)
        else:
            print("Failed to get solution")

    # Close the pool
    pool.close()
    pool.join()

    # Save failed question IDs
    # Load problems from both files
    with open('perplexmath-generated-answers-openai.json', 'r') as f:
        completed_problems = json.load(f)
    with open('perplexmath-dataset.json', 'r') as f:
        all_problems = json.load(f)
        
    # Get question IDs of completed problems
    completed_question_ids = set(p['question_id'] for p in completed_problems)
    
    # Find questions that are in all_problems but not in completed_problems
    missing_questions = [p for p in all_problems 
                          if p['question_id'] not in completed_question_ids]
    
    # Save missing question IDs to file
    if missing_questions:
        with open('missing_questions.json', 'w') as f:
            json.dump(missing_questions, f, indent=2)
        print(f"\nSaved {len(missing_questions)} missing questions to missing_questions.json")

if __name__ == "__main__":
    main()
