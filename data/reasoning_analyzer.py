import requests
import re
import json

# Groq API Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq_api(prompt, model="llama3-70b-8192", max_tokens=1024):
    """
    Call the Groq API with the given prompt and return the response.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    try:
        print(f"Sending request to: {GROQ_API_URL}")
        print(f"Using model: {model}")
        response = requests.post(GROQ_API_URL, headers=headers, json=data)

        if not response.ok:
            print(f"Error status code: {response.status_code}")
            print(f"Error response: {response.text}")
            return None

        response_data = response.json()
        print(f"API response received. Status: {response.status_code}")
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        print(f"Response: {response.json() if response.ok else response.text}")
        return None

def annotate_reasoning_chain(text):
    """
    Annotate each sentence in the reasoning chain with appropriate labels.
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
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, last_sentences, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip() if len(match.groups()) == 1 else match.group(0).strip()
                break

    return annotations, final_answer, metrics

def format_output(annotations, final_answer, original_text, metrics):
    """
    Format the output with annotations, final answer, and reasoning metrics.
    """
    result = "\n==== ORIGINAL REASONING ====\n"
    result += original_text

    result += "\n\n==== ANNOTATED REASONING ====\n"
    for ann in annotations:
        result += ann + "\n"

    if final_answer:
        result += "\n==== FINAL ANSWER ====\n"
        result += final_answer

    # Calculate reasoning pattern metrics and indicators
    total_indicators = sum(metrics.values()) - metrics["total_sentences"]

    # Avoid division by zero
    if total_indicators > 0:
        memorization_pct = (metrics["memorization_count"] / total_indicators) * 100
        reasoning_pct = (metrics["reasoning_count"] / total_indicators) * 100
        exploration_pct = (metrics["exploration_count"] / total_indicators) * 100
        uncertainty_pct = (metrics["uncertainty_count"] / total_indicators) * 100
        computation_pct = (metrics["computation_count"] / total_indicators) * 100
    else:
        memorization_pct = reasoning_pct = exploration_pct = uncertainty_pct = computation_pct = 0

    # Categorize the approach based on the metrics
    primary_approach = "Balanced"
    secondary_approach = ""

    if memorization_pct > 40:
        primary_approach = "Memorization"
    elif reasoning_pct > 40:
        primary_approach = "Reasoning"
    elif computation_pct > 40:
        primary_approach = "Computation"

    if uncertainty_pct > 15:
        secondary_approach = " with Uncertainty"
    elif exploration_pct > 15:
        secondary_approach = " with Exploration"

    result += "\n\n==== REASONING ANALYSIS ====\n"
    result += f"Memorization indicators: {metrics['memorization_count']} ({memorization_pct:.1f}%)\n"
    result += f"Reasoning indicators: {metrics['reasoning_count']} ({reasoning_pct:.1f}%)\n"
    result += f"Exploration indicators: {metrics['exploration_count']} ({exploration_pct:.1f}%)\n"
    result += f"Uncertainty indicators: {metrics['uncertainty_count']} ({uncertainty_pct:.1f}%)\n"
    result += f"Computation indicators: {metrics['computation_count']} ({computation_pct:.1f}%)\n"
    result += f"Total sentences: {metrics['total_sentences']}\n\n"

    result += f"Primary approach: {primary_approach}{secondary_approach}\n\n"

    # Add detailed interpretation
    result += "==== INTERPRETATION ====\n"
    if memorization_pct > reasoning_pct and memorization_pct > exploration_pct:
        result += ("The model appears to be RECALLING knowledge or formulas from its training data. "
                  "This indicates the problem or a similar one may have been seen during training.\n")
    elif reasoning_pct > memorization_pct and reasoning_pct > exploration_pct:
        result += ("The model is primarily using REASONING to derive the answer step by step. "
                  "This indicates the model is applying general principles rather than recalling specific solutions.\n")
    elif exploration_pct > 15:
        result += ("The model is EXPLORING different approaches or testing hypotheses. "
                  "This suggests it's applying problem-solving techniques rather than recalling solutions.\n")

    if uncertainty_pct > 15:
        result += ("The model expresses UNCERTAINTY, suggesting it may not have seen this exact problem before "
                  "and is working through unfamiliar territory.\n")

    if computation_pct > 30:
        result += ("The response contains significant COMPUTATION, showing the model is calculating a solution "
                  "rather than simply recalling it.\n")

    # Generate overall assessment
    result += "\n==== LIKELIHOOD OF MEMORIZATION VS. REASONING ====\n"

    if memorization_pct > 30 and uncertainty_pct < 10:
        result += "HIGH likelihood the model has seen similar problems during training and is recalling patterns.\n"
    elif reasoning_pct > 30 and (exploration_pct > 10 or uncertainty_pct > 10):
        result += "HIGH likelihood the model is reasoning through a new problem rather than recalling a solution.\n"
    else:
        result += "MEDIUM likelihood of either approach - model is using a mix of recall and reasoning.\n"

    return result

def analyze_reasoning(prompt, use_api=True, model="llama3-70b-8192"):
    """
    Analyze reasoning for a given prompt.
    If use_api is True, it calls the Groq API; otherwise, it uses the prompt as the reasoning.
    """
    print(f"\n=== ANALYZING REASONING ===")
    print(f"Prompt: {prompt}")

    # Get reasoning output
    if use_api:
        print(f"\nCalling Groq API with model {model}...")
        reasoning_output = call_groq_api(prompt, model=model)
        if reasoning_output is None:
            return "Error: Failed to get response from Groq API."
    else:
        # For testing with a pre-defined reasoning chain
        reasoning_output = """
        To convert the point (0,3) from rectangular coordinates to polar coordinates, I need to find r and θ.

        First, I'll recall the formulas for converting from rectangular to polar coordinates:
        r = √(x² + y²)
        θ = atan2(y, x)

        Given the point (0,3), I have x = 0 and y = 3.

        Calculate r:
        r = √(0² + 3²)
        r = √(0 + 9)
        r = √9
        r = 3

        Calculate θ:
        Since x = 0 and y = 3, I need to find θ = atan2(3, 0).

        When x = 0 and y > 0, the angle is π/2 radians or 90 degrees.
        Therefore, θ = π/2

        To verify this is within the required range (0 ≤ θ < 2π), I confirm that π/2 is indeed between 0 and 2π.

        Therefore, the point (0,3) in polar coordinates is (3, π/2).

        Final answer: (3, π/2)
        """

    if reasoning_output:
        annotations, final_answer, metrics = annotate_reasoning_chain(reasoning_output)
        return format_output(annotations, final_answer, reasoning_output, metrics)
    else:
        return "No reasoning output to analyze."

# Test reasoning prompts
test_prompts = [
    "The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and $(x, y)$ and $x > 7$. What is the value of $x + y$?"
]

    # Run analysis on each test prompt
if __name__ == "__main__":
    for i, reasoning_prompt in enumerate(test_prompts):
        print(f"\n{'='*50}")
        print(f"TEST PROMPT {i+1}: {reasoning_prompt}")
        print(f"{'='*50}")

        # Try different models if the default one fails
        models_to_try = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
        success = False

        for model in models_to_try:
            print(f"\nAttempting with model: {model}")
            # Set use_api=True to use the actual API, False to use the test reasoning
            result = analyze_reasoning(reasoning_prompt, use_api=True, model=model)

            # If we got a successful result, break out of the loop
            if "Error: Failed to get response from Groq API" not in result:
                success = True
                print(result)
                break

        if not success:
            print("\nAll model attempts failed. Using test reasoning instead:")
            result = analyze_reasoning(reasoning_prompt, use_api=False)
            print(result)