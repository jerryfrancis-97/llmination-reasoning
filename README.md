# llmination-reasoning
Repo for LLM reasoning work

fixing_problems.ipynb
- calls gemini API to fix the problem using 5 different types of modifying prompts
1. Change of values to large numbers
2. Mathematically impossible context/ incorrect math problem
3. Ambiguous problems with missing assumptions
4. Counter intuitive problems like paradoxes
5. Problems with extra irrelevant information

math_probolems.jsonl
- contains the dataset from huggingface

modified_path_problems.json
- contains the modified dataset output from fixing_problems.ipynb
