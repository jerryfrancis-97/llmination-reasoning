import re
from typing import Dict, Optional

class ResponseParser:
    """Handles parsing of API responses into structured data"""
    
    @staticmethod
    def extract_fields(text: str) -> Dict[str, Optional[str]]:
        """
        Extract fields from response text using string find operations
        
        Args:
            text (str): The raw response text from the API
            
        Returns:
            Dict containing extracted fields: final_answer, reasoning, solution_code
        """
        if not text:
            return {"final_answer": None, "reasoning": None, "solution_code": None}
            
        parser = ResponseParser()
        try:
            return {
                "final_answer": parser._extract_final_answer(text),
                "reasoning": parser._extract_reasoning(text),
                "solution_code": parser._extract_solution_code(text)
            }
        except Exception as e:
            print(f"Error in extract_fields: {str(e)}")
            return {"final_answer": None, "reasoning": None, "solution_code": None}
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final answer field from response text"""
        try:
            # Find final_answer field
            final_start = text.find('"final_answer\": \"') + len('"final_answer\": \"')
            if final_start <= len('"final_answer\": \"') - 1:  # Not found
                return None
                
            final_end = text.find('\",\n', final_start)
            if final_end == -1:  # Try alternative ending
                final_end = text.find('\",', final_start)
            
            if final_end == -1:  # Still not found
                return None
                
            final_answer = text[final_start:final_end]
            if final_answer:
                # Clean up escaped characters
                final_answer = final_answer.replace("\\\\", "\\")
                return final_answer
                
        except Exception as e:
            print(f"Error extracting final_answer: {str(e)}")
            
        return None
    
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract the reasoning field from response text"""
        try:
            # Find reasoning field
            reason_start = text.find('\"reasoning\": \"') + len('\"reasoning\": \"')
            if reason_start <= len('\"reasoning\": \"') - 1:  # Not found
                return None
                
            reason_end = text.find('\",\n ', reason_start)
            if reason_end == -1:  # Try alternative ending
                reason_end = text.find('\",', reason_start)
            
            if reason_end == -1:  # Still not found
                return None
                
            reasoning = text[reason_start:reason_end]
            return reasoning if reasoning else None
            
        except Exception as e:
            print(f"Error extracting reasoning: {str(e)}")
            
        return None
    
    def _extract_solution_code(self, text: str) -> Optional[str]:
        """Extract the solution code field from response text"""
        try:
            # Find solution_code field
            code_start = text.find('\"solution_code\": \"') + len('\"solution_code\": \"')
            if code_start <= len('\"solution_code\": \"') - 1:  # Not found
                return None
                
            code_end = text.rfind('\"\n}\n')
            if code_end == -1:  # Try alternative ending
                code_end = text.rfind('\"\n}')
            if code_end == -1:  # Try another alternative
                code_end = text.rfind('\"')
            
            if code_end == -1:  # Still not found
                return None
                
            solution_code = text[code_start:code_end]
            if solution_code:
                # Clean up escaped newlines
                solution_code = solution_code.replace('\\n', '\n')
                return solution_code
                
        except Exception as e:
            print(f"Error extracting solution_code: {str(e)}")
            
        return None
    
    @staticmethod
    def clean_response_text(text: str) -> str:
        """
        Clean up response text by removing markdown and fixing escapes
        
        Args:
            text (str): Raw response text
            
        Returns:
            Cleaned response text
        """
        if not text:
            return text
            
        original_length = len(text)
        
        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text.replace('```json', '').replace('```', '')
        elif text.startswith('```'):
            text = text.replace('```', '')
        
        # Fix escaped characters
        text = text.replace("\\", "\\\\")
        
        cleaned_text = text.strip()
        
        # Debug logging
        if len(cleaned_text) != original_length:
            print(f"Cleaned response text: {original_length} -> {len(cleaned_text)} characters")
        
        return cleaned_text
    
    @staticmethod
    def validate_solution_data(solution_data: Dict[str, Optional[str]]) -> tuple[bool, list]:
        """
        Validate that all required fields are present
        
        Args:
            solution_data (Dict): The parsed solution data
            
        Returns:
            Tuple of (is_valid, missing_fields)
        """
        required_fields = ["final_answer", "reasoning", "solution_code"]
        missing_fields = [field for field in required_fields if not solution_data.get(field)]
        
        is_valid = len(missing_fields) == 0
        return is_valid, missing_fields
    
    @staticmethod
    def is_max_tokens_reached(text: str) -> bool:
        """
        Check if the response indicates max tokens were reached
        
        Args:
            text (str): Response text to check
            
        Returns:
            True if max tokens reached, False otherwise
        """
        return "GenerateContentResponse" in text and "MAX_TOKENS" in text
    
    @staticmethod
    def get_parsing_stats(text: str) -> Dict[str, any]:
        """
        Get statistics about the response text for debugging
        
        Args:
            text (str): Response text to analyze
            
        Returns:
            Dict containing parsing statistics
        """
        if not text:
            return {"length": 0, "has_json_structure": False, "field_count": 0}
        
        stats = {
            "length": len(text),
            "has_json_structure": "{" in text and "}" in text,
            "field_count": 0,
            "contains_final_answer": '"final_answer"' in text,
            "contains_reasoning": '"reasoning"' in text,
            "contains_solution_code": '"solution_code"' in text
        }
        
        # Count actual fields found
        if stats["contains_final_answer"]:
            stats["field_count"] += 1
        if stats["contains_reasoning"]:
            stats["field_count"] += 1
        if stats["contains_solution_code"]:
            stats["field_count"] += 1
            
        return stats
