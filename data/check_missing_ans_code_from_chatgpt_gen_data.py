import json
import csv

# Load both JSON files
with open('perplexmath-dataset-answers.json', 'r') as f: 
    file_a = json.load(f)

with open('perplexmath-dataset-answers-serena.json', 'r') as f:
    file_b = json.load(f)

# Extract dicts from file B with impossible context problem_type and new keys
updates = {}
missing_data = []

for item in file_b:
    if item.get('problem_type') == 'impossible_context':

        question_id = item.get('question_id')
        has_correct_answer = 'correct_answer' in item
        has_solution_code = 'solution_code' in item
        
        # Only add to updates if it has at least one of the new keys
        if has_correct_answer or has_solution_code:
            if question_id:
                updates[question_id] = item
print(f"Updated {len(updates)} impossible context problems in file B")

# Update corresponding dicts in file A and check for missing data
for item in file_a:
    if (item.get('problem_type') == 'impossible_context' and 
        item.get('question_id') in updates):
        question_id = item['question_id']
        update_item = updates[question_id]
        
        # Update with new keys from file B
        if 'correct_answer' in update_item:
            item['correct_answer'] = update_item['correct_answer']
        if 'solution_code' in update_item:
            item['solution_code'] = update_item['solution_code']

# Check for missing data in file A 
for item in file_a:
    question_id = item.get('question_id')
    has_correct_answer = 'correct_answer' in item
    has_solution_code = 'solution_code' in item
    
    if not has_correct_answer or not has_solution_code:
        missing_data.append({
            'question_id': question_id,
            'missing_ans': not has_correct_answer,
            'missing_code': not has_solution_code
        })
print(f"Found {len(missing_data)} problems with missing answer or code data")

with open('perplexmath-datatset-answers_updated.json', 'w') as f:
    json.dump(file_a, f, indent=2)

# Save missing data to CSV
with open('missing_answer_or_code.csv', 'w', newline='') as csvfile:
    fieldnames = ['question_id', 'missing_ans', 'missing_code']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in missing_data:
        writer.writerow(row)

print(f"Updated {len(updates)} impossible context problems in file A")
print(f"Found {len(missing_data)} problems with missing answer or code data")
print(f"Missing data saved to missing_answer_or_code.csv")

