"""
Evaluation configuration and prompts for the reasoning evaluation framework.
"""

class EvaluationConfig:
    """Configuration for evaluation prompts and settings."""
    
    # Base Prompts
    BASE_META_PROMPT = "\n\nAnswer the question above to the best of your ability."
    FORMAT_PROMPT = "\n\n Output your final answer in the format {'FINAL_ANSWER': <final_answer>, 'LABEL': <label>, 'CONFIDENCE': <confidence>} in the last line separately."
    REASONING_META_PROMPT = "\n\nFor the answer above, classify your reasoning as one of [Recall, Reasoning, Hallucination, Uncertain]. State the label alone."
    CONFIDENCE_META_PROMPT = "\n\nOn a scale of 0-100%, how confident are you in your answer? State only the percentage."
    CHAIN_OF_THOUGHT_PROMPT = "\n\nBefore answering, walk through your reasoning step by step."
    
    # Annotation Prompts
    ANNOTATION_PROMPT1 = """\n\nThe reasoning chain to analyze: \n\n"""
    ANNOTATION_PROMPT2 = """\n\n
Please split the following reasoning chain of an LLM into annotated parts using labels and the following format. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

Available labels:
["initializing"] -> The model is rephrasing the given task and states initial thoughts.
["deduction"] -> The model is performing a deduction step based on its current approach and assumptions.
["adding-knowledge"] -> The model is enriching the current approach with recalled facts.
["example-testing"] -> The model generates examples to test its current approach.
["uncertainty-estimation"] -> The model is stating its own uncertainty.
["backtracking"] -> The model decides to change its approach.
["checking"] -> The model is checking the correctness of its current approach. Provide reason why it wants to check the current approach.
["separate"] -> If there is a tail that has no annotation using the above labels.

Also include a keyword in the following format <keyword> for each label before the ["end-section"] for the following cases,
<Memorization> -> Memory/ Fact recall, the model is explicitly recalling or restating known facts from memory, internal knownledge base.
<Reasoning> -> Actual Reasoning, the model is trying to solving the problem at this step using first principles and NOT referring/ recalling anything from its knowledge base. 
<Computation> -> The model is performing a computation or a calculation step.
<Exploration> -> The model is exploring the problem space, trying to find a new approach or finding a solution.
<Uncertainty> -> The model is stating its own uncertainty to an approach or about the answer/solution.

Answer in the following format:
["label"]... <keyword> ["end-section"]
Only use the labels and keywords outlined above. 
"""
    
    # Reasoning Labels
    REASONING_LABELS = ["Recall", "Reasoning", "Hallucination", "Uncertain"]
    
    # Step Indicators for Chain of Thought Detection
    STEP_INDICATORS = [
        "Step 1", "First,", "To solve this", "Let's break this down", 
        "I'll approach this", "Let me think", "Let's think", 
        "To determine", "Let's calculate"
    ]
    
    # Answer Indicators
    ANSWER_INDICATORS = [
        "Therefore,", "So,", "In conclusion,", "Thus,", 
        "The answer is", "This means", "To summarize", 
        "Finally,", "In summary"
    ]
    
    # Final Answer Patterns
    FINAL_ANSWER_PATTERNS = [
        r'final answer\s*[:\-]?\s*(.*)',
        r'answer\s*[:\-]?\s*(.*)',
        r'result\s*[:\-]?\s*(.*)',
        r'solution\s*[:\-]?\s*(.*)',
        r'coordinates\s*[:\-]?\s*(.*)',
        r'\(([\d\.\-]+),\s*([\d\.\-π\/\s]+)\)',  # For coordinates like (r,θ)
        r'boxed\{([^}]+)\}',  # For LaTeX boxed answers
    ]
    
    # Analysis Thresholds
    MEMORIZATION_THRESHOLD = 40.0
    REASONING_THRESHOLD = 40.0
    COMPUTATION_THRESHOLD = 40.0
    UNCERTAINTY_THRESHOLD = 15.0
    EXPLORATION_THRESHOLD = 15.0
    COMPUTATION_PERCENTAGE_THRESHOLD = 30.0
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 80.0
    MEDIUM_CONFIDENCE_THRESHOLD = 50.0
    
    @classmethod
    def get_full_prompt(cls, base_prompt: str, include_chain_of_thought: bool = True, 
                        include_format: bool = True, include_reasoning: bool = True,
                        include_confidence: bool = True) -> str:
        """Build a complete prompt with optional components."""
        prompt = base_prompt
        
        if include_chain_of_thought:
            prompt += cls.CHAIN_OF_THOUGHT_PROMPT
            
        if include_format:
            prompt += cls.FORMAT_PROMPT
            
        if include_reasoning:
            prompt += cls.REASONING_META_PROMPT
            
        if include_confidence:
            prompt += cls.CONFIDENCE_META_PROMPT
            
        return prompt
