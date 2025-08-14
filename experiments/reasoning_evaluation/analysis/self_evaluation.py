"""
Self-evaluation analysis using the same model to analyze its own reasoning.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from ..config.evaluation_config import EvaluationConfig
from ..api_clients.base_client import BaseAPIClient

class SelfEvaluation:
    """Handles self-evaluation of reasoning using the same model."""
    
    def __init__(self, api_client: Optional[BaseAPIClient] = None):
        """
        Initialize the self-evaluation analyzer.
        
        Args:
            api_client: API client to use for evaluation
        """
        self.api_client = api_client
    
    def set_api_client(self, api_client: BaseAPIClient):
        """Set the API client for evaluation."""
        self.api_client = api_client
    
    def annotate_reasoning_chain_with_prompt(self, response: str, model_config: Dict[str, Any]) -> str:
        """
        Annotate reasoning chain with a prompt using the same model.
        
        Args:
            response: The response to analyze
            model_config: Configuration for the model
            
        Returns:
            Annotated response
        """
        if not response or not self.api_client:
            return ""
        
        annotate_prompt = (EvaluationConfig.ANNOTATION_PROMPT1 + 
                          response + 
                          EvaluationConfig.ANNOTATION_PROMPT2)
        
        try:
            annotate_response = self.api_client.query(
                annotate_prompt,
                model_config["model"],
                model_config.get("temperature", 0.0)
            )
            return annotate_response["text"]
        except Exception as e:
            print(f"Error in self-evaluation: {e}")
            return ""
    
    def parse_reasoning_indicators(self, annotations: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Analyze reasoning for a given model response using the same eval model.
        
        Args:
            annotations: The annotated response
            
        Returns:
            Tuple of (annotation_keyword_counts, keyword_counts)
        """
        if not annotations:
            return {}, {}
        
        # Initialize counters
        annotation_keyword_counts = {}  # For "annotation--keyword" pairs
        keyword_counts = {}  # For just keywords
        
        # Regex patterns to extract annotations and keywords
        annotation_pattern = r'\["([^"]+)"\]'
        keyword_pattern = r'<([^>]+)>'
        section_end = r'\["end-section"\]'
        
        # Split response into sections (each section ends with ["end-section"])
        sections = re.split(section_end, annotations)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Find all annotations and keywords in this section
            annotations_found = re.findall(annotation_pattern, section)
            keywords_found = re.findall(keyword_pattern, section)
            
            # For each annotation-keyword pair in the section
            for annotation in annotations_found:
                for keyword in keywords_found:
                    pair_key = f"{annotation}--{keyword}"
                    annotation_keyword_counts[pair_key] = annotation_keyword_counts.get(pair_key, 0) + 1
            
            # Count keywords separately
            for keyword in keywords_found:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        keyword_counts['total_annotations'] = len(sections)
        
        return annotation_keyword_counts, keyword_counts
    
    def compute_annotation_metrics(self, annotation_keyword_counts: Dict[str, int], 
                                  keyword_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Compute metrics from annotation and keyword counts.
        
        Args:
            annotation_keyword_counts: Dict of annotation-keyword pair counts
            keyword_counts: Dict of keyword counts
            
        Returns:
            Dict containing computed metrics and interpretations
        """
        # Initialize metrics dictionary
        metrics = {
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
        
        # Add individual annotation-keyword pair counts to metrics
        for pair_key, count in annotation_keyword_counts.items():
            metrics[f"pair_{pair_key}"] = count
        
        # Add keyword counts to metrics
        for key, value in keyword_counts.items():
            metrics[key] = value
        
        # Calculate percentages if we have annotations
        total_indicators = sum(keyword_counts.values()) - keyword_counts.get('total_annotations', 0)
        
        if total_indicators > 0:
            memorization_count = keyword_counts.get('Memorization', 0)
            reasoning_count = keyword_counts.get('Reasoning', 0)
            exploration_count = keyword_counts.get('Exploration', 0)
            uncertainty_count = keyword_counts.get('Uncertainty', 0)
            computation_count = keyword_counts.get('Computation', 0)
            
            metrics["memorization_pct"] = (memorization_count / total_indicators) * 100
            metrics["reasoning_pct"] = (reasoning_count / total_indicators) * 100
            metrics["exploration_pct"] = (exploration_count / total_indicators) * 100
            metrics["uncertainty_pct"] = (uncertainty_count / total_indicators) * 100
            metrics["computation_pct"] = (computation_count / total_indicators) * 100
            
            # Determine primary approach
            if metrics["memorization_pct"] > EvaluationConfig.MEMORIZATION_THRESHOLD:
                metrics["primary_approach"] = "Memorization"
            elif metrics["reasoning_pct"] > EvaluationConfig.REASONING_THRESHOLD:
                metrics["primary_approach"] = "Reasoning"
            elif metrics["computation_pct"] > EvaluationConfig.COMPUTATION_THRESHOLD:
                metrics["primary_approach"] = "Computation"
            
            # Determine secondary characteristics
            if metrics["uncertainty_pct"] > EvaluationConfig.UNCERTAINTY_THRESHOLD:
                metrics["secondary_approach"] = " with Uncertainty"
            elif metrics["exploration_pct"] > EvaluationConfig.EXPLORATION_THRESHOLD:
                metrics["secondary_approach"] = " with Exploration"
            
            # Build interpretation
            metrics["interpretation"] = self._build_interpretation(metrics)
            
            # Generate likelihood assessment
            metrics["likelihood_assessment"] = self._generate_likelihood_assessment(metrics)
        
        return metrics
    
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
        
        # Add uncertainty assessment
        if metrics["uncertainty_pct"] > EvaluationConfig.UNCERTAINTY_THRESHOLD:
            interpretation += " The model expresses UNCERTAINTY, suggesting it may not have seen this exact problem before and is working through unfamiliar territory."
        
        # Add computation assessment
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
