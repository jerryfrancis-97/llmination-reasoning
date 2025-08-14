import json
import time
import re
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from datetime import datetime, timedelta
from config import Config
from api_clients import create_api_client
from rate_limiter import RateLimiter
from response_parser import ResponseParser

# Rate limiting configuration
rate_limiter = RateLimiter()

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

# extract_fields function removed - now handled by ResponseParser class

def load_problems(file_path):
    """Load problems from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading problems: {str(e)}")
        return []

# Rate limiting function removed - now handled by RateLimiter class

# OpenAI response function removed - now handled by OpenAIClient class

# OpenAI retry function removed - now handled by OpenAIClient class

def get_problem_solution(problem_text, api_key, model_type="gemini"):
    """Get solution for a problem using either Gemini or OpenAI API"""
    # Prepare the prompt
    problem_text = problem_text.replace("\\", "\\\\")
    prompt = ANSWER_PROMPT_TEMPLATE.format(problem=problem_text)

    try:
        attempts = 0
        max_attempts = Config.MAX_ATTEMPTS
        
        # Create API client
        try:
            api_client = create_api_client(model_type, api_key, rate_limiter)
        except ValueError as e:
            print(f"Error creating API client: {str(e)}")
            return None
        
        while attempts < max_attempts:
            try:
                # Get response from API client (rate limiting handled internally)
                result = api_client.generate_response(prompt)
                
                if not result["success"]:
                    print(f"API error: {result['error']}")
                    attempts += 1
                    time.sleep(2)
                    continue
                
                response_text = result["response"]
                
                # Clean up and parse JSON response using ResponseParser
                response_text = ResponseParser.clean_response_text(response_text)
                
                if ResponseParser.is_max_tokens_reached(response_text):
                    print("Max output tokens reached")
                    return None
                
                # Try JSON parsing
                try:
                    solution_data = ResponseParser.extract_fields(response_text)
                    print("extracted-------------------------------")
                    print(solution_data)
                except Exception as e:
                    print(f"Error extracting fields: {str(e)}")
                    solution_data = None
                
                # Verify we have all required fields
                is_valid, missing_fields = ResponseParser.validate_solution_data(solution_data)
                if is_valid:
                    return solution_data
                else:
                    # Print which fields are missing
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
    # Configuration is now centralized in Config class

    # Delete output file if it exists
    try:
        os.remove(Config.OUTPUT_FILE)
        print(f"Deleted existing {Config.OUTPUT_FILE} to start clean")
    except FileNotFoundError:
        pass

    # Load problems 
    print(f"Loading problems from {Config.INPUT_FILE}...")
    problems = load_problems(Config.INPUT_FILE)
    print(f"Loaded {len(problems)} problems")

    if len(problems) == 0:
        print("No problems loaded. Exiting.")
        return

    # Create a pool of workers
    pool = mp.Pool(processes=Config.NUM_PROCESSES)

    # Prepare arguments for each problem
    api_key = Config.OPENAI_API_KEY if Config.MODEL_TYPE == "openai" else Config.GEMINI_API_KEY
    problem_args = [(problem, api_key, Config.MODEL_TYPE) for problem in problems]

    # Process problems in parallel with progress bar
    results = []
    for result in tqdm(pool.imap(process_problem, problem_args), total=len(problems), desc="Processing problems"):
        problem, success = result
        if success:
            print("Successfully added solution")
            # Update output file immediately after each successful solution
            update_output_file(problem, Config.OUTPUT_FILE)
            results.append(problem)
        else:
            print("Failed to get solution")

    # Close the pool
    pool.close()
    pool.join()

    # Save failed question IDs
    # Load problems from both files
    with open(Config.OUTPUT_FILE, 'r') as f:
        completed_problems = json.load(f)
    with open(Config.INPUT_FILE, 'r') as f:
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
