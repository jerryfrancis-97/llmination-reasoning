"""
Response parsing utilities for the reasoning evaluation framework.
"""

import re
from typing import Tuple, Optional
from ..config.evaluation_config import EvaluationConfig

class ResponseParser:
    """Handles parsing of model responses."""
    
    @staticmethod
    def parse_response(response_text: str) -> Tuple[str, str, str, Optional[float]]:
        """
        Parse model response to extract answer, chain of thought, reasoning type, and confidence.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Tuple of (answer, chain_of_thought, reasoning_type, confidence)
        """
        # Default values
        full_answer = response_text
        chain_of_thought = ""
        final_answer = response_text
        reasoning_type = None
        confidence = None
        
        # Extract the chain of thought part
        chain_of_thought = ResponseParser._extract_chain_of_thought(response_text)
        
        # Parse the structured output format from the last line
        final_answer, reasoning_type, confidence = ResponseParser._parse_structured_output(response_text)
        
        # Final cleanup
        final_answer = final_answer.strip()
        chain_of_thought = chain_of_thought.strip()
        
        return final_answer, chain_of_thought, reasoning_type, confidence
    
    @staticmethod
    def _extract_chain_of_thought(text: str) -> str:
        """Extract the chain of thought from the response."""
        # Look for common indicators of a step-by-step process
        for indicator in EvaluationConfig.STEP_INDICATORS:
            if indicator in text:
                # Find where the final answer likely begins
                for ans_ind in EvaluationConfig.ANSWER_INDICATORS:
                    ans_parts = text.split(ans_ind)
                    if len(ans_parts) > 1:
                        return ans_parts[0].strip()
                
                if indicator in text:  # If we found a clean separation
                    break
        
        # If no clear structure was found, use heuristic 
        # Try to split by paragraph
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            # Assume earlier paragraphs are reasoning and the last is the answer
            return "\n\n".join(paragraphs[:-1])
        
        return ""
    
    @staticmethod
    def _parse_structured_output(text: str) -> Tuple[str, str, Optional[float]]:
        """Parse structured output format from the response."""
        lines = text.split('\n')
        
        for line in reversed(lines):
            if line.strip():  # Find last non-empty line
                try:
                    # Try to parse JSON-like structure from the line
                    match = re.search(r"{'FINAL_ANSWER':\s*(.+?),\s*'LABEL':\s*(.+?),\s*'CONFIDENCE':\s*(.+?)}", line)
                    if match:
                        final_answer = match.group(1).strip()
                        reasoning_type = match.group(2).strip()
                        try:
                            confidence = float(match.group(3).strip())
                        except ValueError:
                            confidence = None
                        return final_answer, reasoning_type, confidence
                except:
                    continue
        
        return text, None, None
    
    @staticmethod
    def clean_response_text(text: str) -> str:
        """Clean up response text by removing markdown and fixing escapes."""
        if not text:
            return text
        
        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text.replace('```json', '').replace('```', '')
        elif text.startswith('```'):
            text = text.replace('```', '')
        
        # Fix escaped characters
        text = text.replace("\\", "\\\\")
        
        return text.strip()
    
    @staticmethod
    def extract_final_answer(text: str) -> str:
        """Extract final answer using multiple patterns."""
        for pattern in EvaluationConfig.FINAL_ANSWER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    return match.group(1).strip()
                else:
                    return match.group(0).strip()
        return ""
