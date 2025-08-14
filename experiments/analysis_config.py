import os
from typing import List

class AnalysisConfig:
    """Configuration for model response data analysis"""
    
    # File paths
    DEFAULT_ANALYSIS_DIR = "analysis"
    DEFAULT_OUTPUT_PREFIX = "analysis_results_"
    
    # Required columns for different analysis tasks
    REQUIRED_COLUMNS = {
        'self_interpretation': ['reasoning_type', 'primary_approach', 'secondary_approach'],
        'task1': ['reasoning_type', 'subject', 'primary_approach', 'secondary_approach'],
        'task2': ['subject', 'primary_approach', 'secondary_approach', 'likelihood_assessment'],
        'task3': ['problem_type', 'primary_approach'],
        'task4': ['problem_type', 'secondary_approach']
    }
    
    # Analysis settings
    EXPLORATION_KEYWORDS = ['Exploration', 'exploration']
    REASONING_KEYWORDS = ['Reasoning', 'reasoning']
    RECALL_KEYWORDS = ['Recall', 'recall']
    
    # Output formatting
    SEPARATOR_LINE = "-" * 70
    SECTION_HEADER = "=" * 50
    
    # File naming patterns
    @staticmethod
    def extract_model_name(csv_file_path: str) -> str:
        """Extract model name from CSV file path"""
        filename = os.path.basename(csv_file_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]
        return "unknown_model"
    
    @staticmethod
    def get_output_file_path(analysis_dir: str, model_name: str) -> str:
        """Generate output file path for analysis results"""
        return os.path.join(analysis_dir, f"{AnalysisConfig.DEFAULT_OUTPUT_PREFIX}{model_name}.txt")
