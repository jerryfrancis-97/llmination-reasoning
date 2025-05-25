import json
import random
import os
import google.generativeai as genai
from google.api_core import retry
import time

# Configuration
JSONL_FILE_PATH = "math_problems.jsonl"  # Update with your actual JSONL file path
CONCEPT = "Number Theory"  # Target concept
DIFFICULTY_LEVEL = 5       # Target difficulty level
OUTPUT_FILE = "new_modified_math_problems.json"

# Prompt template for Gemini
PROMPT_TEMPLATE = """
Generate 5 different versions of the following math problem according to these specific categories:

1. Large numbers: Change values to extremely large numbers (e.g., billions or scientific notation)
2. Impossible context: Create a mathematically impossible or inconsistent version of the problem
3. Ambiguous: Remove key assumptions or make the problem underspecified in a way that is not immediately obvious
4. Paradox: Create a counter-intuitive or paradoxical version
5. Irrelevant info: Add numerical extraneous details related to the problem context but unrelated to the solution

Important: 
- Preserve the core mathematical concept
- Keep LaTeX formatting intact for mathematical expressions
- For each version, maintain the same subject and difficulty level

Original Problem:
{problem}

Output must be valid JSON (without code blocks or extra text) formatted exactly like this:
[
  {{
    "problem_type": "large_numbers",
    "modified_problem": "problem text with LaTeX formatting...",
    "explanation": "brief explanation of what was changed"
  }},
  {{
    "problem_type": "impossible_context",
    "modified_problem": "problem text with LaTeX formatting...",
    "explanation": "brief explanation of what was changed"
  }},
  {{
    "problem_type": "ambiguous",
    "modified_problem": "problem text with LaTeX formatting...",
    "explanation": "brief explanation of what was changed"
  }},
  {{
    "problem_type": "paradox",
    "modified_problem": "problem text with LaTeX formatting...",
    "explanation": "brief explanation of what was changed"
  }},
  {{
    "problem_type": "irrelevant_info",
    "modified_problem": "problem text with LaTeX formatting...",
    "explanation": "brief explanation of what was changed"
  }}
]
"""

def load_jsonl(file_path):
    """Load JSONL file line by line"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    print(f"Problematic line: {line[:100]}...")
    return data

def filter_problems(data, subject, level):
    """Filter problems by subject and level"""
    return [item for item in data 
            if item.get('subject', '').lower() == subject.lower() 
            and item.get('level', 0) == level]

def generate_modified_versions(problem_data, api_key):
    """Generate modified versions of a problem using Gemini API"""
    # Set up the Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    problem_text = problem_data.get('problem', '')
    
    # Prepare the prompt with the problem
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)
    
    try:
        # Generate response with retry mechanism
        attempts = 0
        max_attempts = 3

        generation_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                }
        while attempts < max_attempts:
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                
                # Extract text from response based on Gemini API version
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                    response_text = response.parts[0].text
                else:
                    response_text = str(response)
                
                # Clean up the response to get just the JSON part
                response_text = response_text.strip()
                
                # Find JSON array in the text
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_part = response_text[start_idx:end_idx]
                    modified_versions = json.loads(json_part)
                    
                    # Verify we have 5 versions
                    if isinstance(modified_versions, list) and len(modified_versions) == 5:
                        # Add metadata to each version
                        for version in modified_versions:
                            version.update({
                                "original_problem": problem_data.get("problem", ""),
                                "original_answer": problem_data.get("answer", ""),
                                "subject": problem_data.get("subject", ""),
                                "level": problem_data.get("level", 0),
                                "unique_id": problem_data.get("unique_id", "")
                            })
                        return modified_versions
                    else:
                        print(f"Expected 5 versions, got {len(modified_versions) if isinstance(modified_versions, list) else 'not a list'}")
                        attempts += 1
                else:
                    print("No valid JSON array found in response")
                    attempts += 1
                
            except Exception as e:
                print(f"Attempt {attempts+1} failed: {str(e)}")
                attempts += 1
                time.sleep(2)  # Wait before retrying
        
        print(f"Failed to generate versions after {max_attempts} attempts")
        return []
        
    except Exception as e:
        print(f"Error generating versions: {str(e)}")
        return []

def save_results(modified_problems, output_file):
    """Save modified problems to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_problems, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(modified_problems)} modified problems to {output_file}")

def main():

    API_KEY = "AIzaSyDsFFc9Oboi1vBchAgFTZzvlTuPUwtmTVo"

    # Configure with your API key
    genai.configure(api_key=API_KEY)

    # Check available models
    models = genai.list_models()
    print("Available models:")
    for model in models:
        print(f" - {model.name}: {model.display_name}")

    # Check if file exists
    if not os.path.exists(JSONL_FILE_PATH):
        print(f"Error: File not found at {JSONL_FILE_PATH}")
        return
    
    # Load and filter data
    print(f"Loading JSONL file from {JSONL_FILE_PATH}...")
    all_problems = load_jsonl(JSONL_FILE_PATH)
    print(f"Loaded {len(all_problems)} problems")
    
    # Filter for number theory problems at level 5
    filtered_problems = filter_problems(all_problems, CONCEPT, DIFFICULTY_LEVEL)
    print(f"Found {len(filtered_problems)} problems in {CONCEPT} at level {DIFFICULTY_LEVEL}")
    
    if not filtered_problems:
        print("No matching problems found. Try different criteria.")
        return
    
    # Process each problem
    all_modified_problems = []
    for i, problem in enumerate(filtered_problems):

        if i==2:
            break
        
        print(f"\nProcessing problem {i+1}/{len(filtered_problems)}:")
        print(f"ID: {problem.get('unique_id', 'Unknown')}")
        print("Problem excerpt:", problem.get('problem', '')[:100] + "..." if len(problem.get('problem', '')) > 100 else problem.get('problem', ''))
        
        # Generate modified versions
        modified_versions = generate_modified_versions(problem, api_key=API_KEY)
        
        if modified_versions:
            all_modified_problems.extend(modified_versions)
            print(f"Successfully generated {len(modified_versions)} versions")
        else:
            print("Failed to generate modified versions for this problem")

    
    # Save results
    if all_modified_problems:
        save_results(all_modified_problems, OUTPUT_FILE)
    else:
        print("No modified problems were generated")


if __name__ == "__main__":
    main()
