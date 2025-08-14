"""
Reasoning analyzer using keyword-based pattern matching.
"""

import re
from typing import List, Dict, Any, Tuple
from ..config.evaluation_config import EvaluationConfig

class ReasoningAnalyzer:
    """Analyzes reasoning using keyword-based pattern matching."""
    
    def __init__(self):
        """Initialize the reasoning analyzer."""
        # Enhanced labels for detecting reasoning types
        self.labels = {
            "initializing": r"(first|to begin|let's start|initial thought|to solve|approach)",
            "deduction": r"(therefore|so|thus|since|it follows that|hence|as a result|conclude|calculate|checking divisibility)",
            "adding-knowledge": r"(known fact|it is known|recall|by definition|according to|remember that)",
            "example-testing": r"(for example|for instance|consider|let's test|suppose|try)",
            "uncertainty-estimation": r"(might|could be|possibly|likely|uncertain|I'm not sure|I wonder)",
            "backtracking": r"(wait|on second thought|however|instead|better approach|let's change)",
            "checking": r"(verify|check|ensure|confirm|validate)"
        }
        
        # Pattern matching for different reasoning types
        self.patterns = {
            "memorization": r"(recall|known fact|remember|by definition|it is known|formula|always true|rule|identity|theorem|lemma|law of|principle)",
            "computation": r"(calculate|compute|evaluate|find|solve|determine)",
            "reasoning": r"(deduce|therefore|hence|thus|conclude|derive|as a result|because|since|follows from|implies|means that)",
            "exploration": r"(let's try|attempt|experiment|test|consider|what if|suppose)",
            "uncertainty": r"(unsure|not certain|don't know|haven't seen|new to me|unfamiliar|never encountered|unusual)"
        }
    
    def annotate_reasoning_chain(self, text: str) -> Tuple[List[str], str, Dict[str, int]]:
        """
        Annotate each sentence in the reasoning chain with appropriate labels.
        
        Args:
            text: The text to analyze
            
        Returns:
            Tuple of (annotations, final_answer, metrics)
        """
        if not text:
            return [], "", {}
        
        annotations = []
        final_answer = ""
        
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
            for idx, (label, pattern) in enumerate(self.labels.items()):
                if re.search(pattern, sentence, re.IGNORECASE):
                    keywords = self._extract_keywords(sentence, metrics)
                    keyword_str = f" {{{', '.join(keywords)}}}" if keywords else ""
                    annotation = f'["{idx}. {label}"] {sentence} {keyword_str} ["end-section"]'
                    annotations.append(annotation)
                    labeled = True
                    break
            
            if not labeled:
                # Even for unlabeled sentences, check for our special patterns
                keywords = self._extract_keywords(sentence, metrics)
                keyword_str = f" {{{', '.join(keywords)}}}" if keywords else ""
                annotations.append(f'["7. separate"] {sentence} {keyword_str} ["end-section"]')
        
        # Extract final answer
        final_answer = self._extract_final_answer(text, sentences)
        
        return annotations, final_answer, metrics
    
    def _extract_keywords(self, sentence: str, metrics: Dict[str, int]) -> List[str]:
        """Extract keywords from a sentence and update metrics."""
        keywords = []
        
        for keyword_type, pattern in self.patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                if keyword_type == "memorization":
                    keywords.append("M")
                    metrics["memorization_count"] += 1
                elif keyword_type == "reasoning":
                    keywords.append("AcR")
                    metrics["reasoning_count"] += 1
                elif keyword_type == "exploration":
                    keywords.append("Exp")
                    metrics["exploration_count"] += 1
                elif keyword_type == "uncertainty":
                    keywords.append("Unc")
                    metrics["uncertainty_count"] += 1
                elif keyword_type == "computation":
                    keywords.append("Comp")
                    metrics["computation_count"] += 1
        
        return keywords
    
    def _extract_final_answer(self, text: str, sentences: List[str]) -> str:
        """Extract the final answer from the text."""
        # Try explicit final answer patterns first
        for pattern in EvaluationConfig.FINAL_ANSWER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    return match.group(1).strip()
                else:
                    return match.group(0).strip()
        
        # Try to extract answer from the last few sentences if no explicit final answer
        last_sentences = " ".join(sentences[-3:]) if len(sentences) >= 3 else " ".join(sentences)
        
        for pattern in EvaluationConfig.FINAL_ANSWER_PATTERNS:
            match = re.search(pattern, last_sentences, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    return match.group(1).strip()
                else:
                    return match.group(0).strip()
        
        return ""
    
    def compute_reasoning_metrics(self, annotations: List[str], final_answer: str, 
                                 original_text: str, metrics: Dict[str, int]) -> Dict[str, Any]:
        """
        Compute reasoning metrics from the analysis.
        
        Args:
            annotations: List of annotated sentences
            final_answer: Extracted final answer
            original_text: Original text
            metrics: Raw metrics from annotation
            
        Returns:
            Dictionary containing computed metrics and interpretations
        """
        # Calculate reasoning pattern metrics and indicators
        total_indicators = sum(metrics.values()) - metrics["total_sentences"]
        
        # Initialize indicator metrics dict
        indicator_metrics = {
            "memorization_pct": 0,
            "reasoning_pct": 0, 
            "exploration_pct": 0,
            "uncertainty_pct": 0,
            "computation_pct": 0,
            "primary_approach": "Balanced",
            "secondary_approach": "",
            "interpretation": "",
            "likelihood_assessment": ""
        }
        
        # Avoid division by zero
        if total_indicators > 0:
            indicator_metrics["memorization_pct"] = (metrics["memorization_count"] / total_indicators) * 100
            indicator_metrics["reasoning_pct"] = (metrics["reasoning_count"] / total_indicators) * 100
            indicator_metrics["exploration_pct"] = (metrics["exploration_count"] / total_indicators) * 100
            indicator_metrics["uncertainty_pct"] = (metrics["uncertainty_count"] / total_indicators) * 100
            indicator_metrics["computation_pct"] = (metrics["computation_count"] / total_indicators) * 100
        
        # Categorize the approach based on the metrics
        if indicator_metrics["memorization_pct"] > EvaluationConfig.MEMORIZATION_THRESHOLD:
            indicator_metrics["primary_approach"] = "Memorization"
        elif indicator_metrics["reasoning_pct"] > EvaluationConfig.REASONING_THRESHOLD:
            indicator_metrics["primary_approach"] = "Reasoning"
        elif indicator_metrics["computation_pct"] > EvaluationConfig.COMPUTATION_THRESHOLD:
            indicator_metrics["primary_approach"] = "Computation"
        
        # Determine secondary characteristics
        if indicator_metrics["uncertainty_pct"] > EvaluationConfig.UNCERTAINTY_THRESHOLD:
            indicator_metrics["secondary_approach"] = " with Uncertainty"
        elif indicator_metrics["exploration_pct"] > EvaluationConfig.EXPLORATION_THRESHOLD:
            indicator_metrics["secondary_approach"] = " with Exploration"
        
        # Build interpretation
        indicator_metrics["interpretation"] = self._build_interpretation(indicator_metrics)
        
        # Generate likelihood assessment
        indicator_metrics["likelihood_assessment"] = self._generate_likelihood_assessment(indicator_metrics)
        
        return indicator_metrics
    
    def _build_interpretation(self, metrics: Dict[str, Any]) -> str:
        """Build interpretation string based on metrics."""
        interpretation = ""
        
        # Determine primary approach
        if (metrics["memorization_pct"] > metrics["reasoning_pct"] and 
            metrics["memorization_pct"] > metrics["exploration_pct"]):
            interpretation = "The model appears to be RECALLING knowledge or formulas from its training data. This indicates the problem or a similar one may have been seen during training."
        elif (metrics["reasoning_pct"] > metrics["memorization_pct"] and 
              metrics["reasoning_pct"] > metrics["exploration_pct"]):
            interpretation = "The model is primarily using REASONING to derive the answer step by step. This indicates the model is applying general principles rather than recalling specific solutions."
        elif (metrics["exploration_pct"] > metrics["memorization_pct"] and 
              metrics["exploration_pct"] > metrics["reasoning_pct"]):
            interpretation = "The model is EXPLORING different approaches or testing hypotheses. This suggests it's applying problem-solving techniques rather than recalling solutions."
        
        # Add uncertainty assessment if present
        if metrics["uncertainty_pct"] > EvaluationConfig.UNCERTAINTY_THRESHOLD:
            interpretation += " The model expresses UNCERTAINTY, suggesting it may not have seen this exact problem before and is working through unfamiliar territory."
        
        # Add computation assessment if present  
        if metrics["computation_pct"] > EvaluationConfig.COMPUTATION_PERCENTAGE_THRESHOLD:
            interpretation += " The response contains significant COMPUTATION, showing the model is calculating a solution rather than simply recalling it."
            
        return interpretation
    
    def _generate_likelihood_assessment(self, metrics: Dict[str, Any]) -> str:
        """Generate likelihood assessment based on metrics."""
        if (metrics["memorization_pct"] > 30 and 
            metrics["uncertainty_pct"] < 10):
            return "HIGH likelihood the model has seen similar problems during training and is recalling patterns."
        elif (metrics["reasoning_pct"] > 30 and 
              (metrics["exploration_pct"] > 10 or metrics["uncertainty_pct"] > 10)):
            return "HIGH likelihood the model is reasoning through a new problem rather than recalling a solution."
        else:
            return "MEDIUM likelihood of either approach - model is using a mix of recall and reasoning."
