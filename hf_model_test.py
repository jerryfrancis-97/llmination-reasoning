import torch
import csv
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


import dotenv
dotenv.load_dotenv(".env")

# List of model IDs to compare
MODEL_IDS = [
    # "tiiuae/falcon-7b-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai/deepseek-llm-7b-base"
]

# System prompt for chat models
SYSTEM_PROMPT = "You are a helpful and concise assistant."

# Prompts (each is a new, single-turn interaction)
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the benefits of exercise on mental health?",
    "Tell me a joke about programming.",
]

# Optional HF token for gated models
HF_TOKEN = os.getenv("HF_TOKEN") # or 'hf_...'

# Output files
CSV_FILE = "llm_responses.csv"
JSON_FILE = "llm_responses.json"

def generate_chat_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_base_response(model, tokenizer, prompt):
    formatted_prompt = f"### Question:\n{prompt}\n\n### Answer:\n"
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    all_results = []

    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "prompt", "response"])

        for model_id in MODEL_IDS:
            print(f"\n🔷 Loading model: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=HF_TOKEN
            )

            # Detect if model supports chat format
            is_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

            for prompt in PROMPTS:
                print(f"\n🟢 Prompt: {prompt}")
                try:
                    response = (
                        generate_chat_response(model, tokenizer, prompt)
                        if is_chat else
                        generate_base_response(model, tokenizer, prompt)
                    )
                    print(f"\n📢 Response:\n{response}\n")

                    # Save to CSV
                    writer.writerow([model_id, prompt, response])

                    # Save to JSON list
                    all_results.append({
                        "model": model_id,
                        "prompt": prompt,
                        "response": response
                    })

                except Exception as e:
                    print(f"[ERROR] {model_id} failed on prompt '{prompt}': {e}")

    # Write JSON file
    with open(JSON_FILE, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
