import json
import google.generativeai as genai
import time

# Configuration
INPUT_FILE = "new_modified_math_problems.json"
OUTPUT_FILE = "answers_data.json"

# Prompt template for getting answers
ANSWER_PROMPT_TEMPLATE = """
Please solve the following math problem. Provide a brief and concise summary of thinking process/ approach used to solve the problem in <think></think> tags, followed by the final answer and Python code to solve it programmatically. If code cannot be written or the final answer is not found, leave it blank.

Problem:
{problem}

Format your response exactly like this:
{{
    "final_answer": "The numerical or text answer",
    "reasoning": "<think> ... </think>",
    "solution_code": "# def solve_problem():\n    # Your solution code here\n    return final_answer"
}}
"""

def load_problems(file_path):
    """Load problems from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading problems: {str(e)}")
        return []

def get_problem_solution(problem_text, api_key):
    """Get solution for a problem using Gemini API"""
    # Set up the Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare the prompt
    prompt = ANSWER_PROMPT_TEMPLATE.format(problem=problem_text)
    print("prompt: ", prompt)
    try:
        attempts = 0
        max_attempts = 3
        generation_config = {
            "temperature": 0,  # Lower temperature for more focused responses
            "max_output_tokens": 10000,
        }
        
        while attempts < max_attempts:
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                
                # Extract text from response
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                    response_text = response.parts[0].text
                else:
                    response_text = str(response)
                
                # Clean up and parse JSON response
                response_text = response_text.strip()
                response_text = response_text.replace('```json', '').replace('```', '')
                print("response_text: ", response_text)
            
                solution_data = json.loads(response_text)
                if "reasoning" in solution_data and "final_answer" in solution_data and "solution_code" in solution_data:
                    return solution_data
                else:
                    print("Invalid solution format received")
                    attempts += 1
                    
            except Exception as e:
                print(f"Attempt {attempts+1} failed: {str(e)}")
                attempts += 1
                time.sleep(2)
        
        print(f"Failed to get solution after {max_attempts} attempts")
        return None
        
    except Exception as e:
        print(f"Error getting solution: {str(e)}")
        return None

def save_results(problems_with_solutions, output_file):
    """Save problems with solutions to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problems_with_solutions, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    API_KEY = "AIzaSyDsFFc9Oboi1vBchAgFTZzvlTuPUwtmTVo"
    INPUT_FILE = "new_modified_math_problems.json"
    OUTPUT_FILE = "modified_math_problems_with_answers.json"

    # Load problems
    print(f"Loading problems from {INPUT_FILE}...")
    problems = load_problems(INPUT_FILE)
    print(f"Loaded {len(problems)} problems")

    if len(problems) == 0:
        print("No problems loaded. Exiting.")
        return
    
    # Process each problem
    for i, problem in enumerate(problems):
        if i==5:
            break
        print(f"\nProcessing problem {i+1}/{len(problems)}")
        
        modified_problem = problem.get('modified_problem', '')
        if modified_problem:
            solution = get_problem_solution(modified_problem, API_KEY)
            if solution:
                # Add solution data to problem dictionary
                problem['reasoning'] = solution['reasoning']
                problem['correct_answer'] = solution['final_answer']
                problem['solution_code'] = solution['solution_code']
                print("Successfully added solution")
            else:
                print("Failed to get solution")
    
    # Save updated problems
    save_results(problems, OUTPUT_FILE)

if __name__ == "__main__":
    main()
