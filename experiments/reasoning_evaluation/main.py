#!/usr/bin/env python3
"""
Main entry point for the refactored reasoning evaluation framework.
"""

import datetime
import os
from framework.llm_reasoning_framework import LLMReasoningFramework
from ..analysis_orchestrator import AnalysisOrchestrator

def main():
    """Main entry point for the analysis script."""
    # Configuration - update this path as needed
    csv_file_paths = [
        "results/run_20250529_045043_final_testing/results_deepseek-r1-distill-llama-70b.csv",
        "results/run_20250529_152151_final_testing/results_gemini-1.5-flash.csv",
        "results/run_20250529_152151_final_testing/results_gemini-2.0-flash.csv",
        "results/run_20250530_234358_final_llama3_70b_8192/results_llama3-70b-8192.csv"
    ]
    
    # Use the last one as default (most recent)
    csv_file = csv_file_paths[-1]
    
    print(f"Starting analysis for: {csv_file}")
    
    # Create and run orchestrator
    orchestrator = AnalysisOrchestrator(csv_file)
    success = orchestrator.run_analysis()
    
    if success:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed. Check the error messages above.")
        return 1
    
    return 0

def run_reasoning_evaluation():
    """Run the reasoning evaluation framework."""
    # Example usage with the new refactored structure
    models = [
        # {"api": "groq", "model": "llama3-70b-8192", "temperature": 0.0},
        # {"api": "groq", "model": "deepseek-r1-distill-llama-70b", "temperature": 0.0},
        {"api": "gemini", "model": "gemini-2.0-flash", "temperature": 0.0},
        # {"api": "gemini", "model": "gemini-1.5-flash", "temperature": 0.0},
        # {"api": "gemini", "model": "gemini-1.5-pro", "temperature": 0.0},
        # {"api": "gemini", "model": "gemini-2.0-flash-thinking-exp", "temperature": 0.0},
        # {"api": "mistral", "model": "mistral-large-latest", "temperature": 0.0},
    ]
    
    framework = LLMReasoningFramework(models)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the framework
    framework.run(
        "../data/perplexmath-dataset_test.json", 
        is_self_reason=True, 
        folder_name=f"run_{timestamp}_refactored"
    )

if __name__ == "__main__":
    # Run the reasoning evaluation
    run_reasoning_evaluation()
