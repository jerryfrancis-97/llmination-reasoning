import os
from typing import Dict, List, TextIO
from analysis_config import AnalysisConfig
from data_validator import DataValidator

class ReportGenerator:
    """Handles report generation and output formatting"""
    
    def __init__(self, output_file: TextIO):
        self.output_file = output_file
        self.validator = DataValidator()
    
    def write_header(self, title: str):
        """Write a section header"""
        self.output_file.write(f"\n{AnalysisConfig.SECTION_HEADER}\n")
        self.output_file.write(f"{title}\n")
        self.output_file.write(f"{AnalysisConfig.SECTION_HEADER}\n\n")
    
    def write_section(self, title: str):
        """Write a subsection header"""
        self.output_file.write(f"\n--- {title} ---\n")
    
    def write_separator(self):
        """Write a separator line"""
        self.output_file.write(f"{AnalysisConfig.SEPARATOR_LINE}\n\n")
    
    def write_data_overview(self, df, csv_file_path: str):
        """Write data overview section"""
        self.write_header("Data Overview")
        
        self.output_file.write(f"Data loaded successfully from {csv_file_path}\n")
        self.output_file.write(f"Total rows: {len(df)}\n")
        self.output_file.write(f"Total columns: {len(df.columns)}\n")
        
        # Write column information
        self.output_file.write("\nColumns in dataset:\n")
        for col in df.columns:
            self.output_file.write(f"- {col}\n")
        
        # Write data quality stats
        quality_stats = self.validator.check_data_quality(df)
        self.output_file.write(f"\nData quality:\n")
        self.output_file.write(f"- Missing data points: {quality_stats['missing_data']}\n")
        self.output_file.write(f"- Duplicate rows: {quality_stats['duplicate_rows']}\n")
        
        self.write_separator()
    
    def write_unique_values_section(self, df):
        """Write unique values section for filtering"""
        self.write_section("Unique Values for Filtering")
        
        key_columns = ['reasoning_type', 'problem_type', 'subject']
        for col in key_columns:
            if col in df.columns:
                unique_vals = self.validator.get_unique_values(df, col)
                self.output_file.write(f"Unique {col}s: {unique_vals}\n")
        
        self.write_separator()
    
    def write_self_interpretation_analysis(self, analysis_result: Dict):
        """Write self-interpretation vs actual analysis results"""
        self.write_header("Self-Interpretation vs Actual Analysis")
        
        if "error" in analysis_result:
            self.output_file.write(f"Error: {analysis_result['error']}\n")
            return
        
        # Write comparison table
        self.output_file.write("Comparison of Model's Claim vs. Observed Approach (Frequency):\n")
        self.output_file.write(analysis_result['comparison'].to_string() + "\n")
        
        # Write discrepancies
        self.output_file.write("\nDiscrepancy Analysis:\n")
        self.output_file.write(analysis_result['summary'] + "\n")
        
        # Write detailed discrepancies if any
        if analysis_result['discrepancies']:
            for key, data in analysis_result['discrepancies'].items():
                self.output_file.write(f"\n{key}:\n")
                self.output_file.write(data.to_string() + "\n")
        
        self.write_separator()
    
    def write_task1_results(self, reasoning_types: List[str], analysis_engine):
        """Write Task 1 results"""
        self.write_header("Task 1 Results")
        
        for reasoning_type in reasoning_types:
            self.write_section(f"Task 1: Approach by Subject for Reasoning Type: '{reasoning_type}'")
            
            result = analysis_engine.analyze_approach_by_reasoning_type_and_subject(reasoning_type)
            
            if "error" in result:
                self.output_file.write(f"Error: {result['error']}\n")
                continue
            
            self.output_file.write(f"Summary: {result['summary']}\n")
            self.output_file.write("\nDetailed results:\n")
            self.output_file.write(result['distribution'].to_string() + "\n")
            
            self.write_separator()
    
    def write_task2_results(self, analysis_engine):
        """Write Task 2 results"""
        self.write_header("Task 2 Results")
        
        result = analysis_engine.analyze_approach_and_likelihood_by_subject()
        
        if "error" in result:
            self.output_file.write(f"Error: {result['error']}\n")
            return
        
        self.output_file.write("Approach (Primary + Secondary) and Likelihood Assessment Counts by Subject:\n")
        self.output_file.write(result['distribution'].to_string() + "\n")
        self.output_file.write(f"\nSummary: {result['summary']}\n")
        
        self.write_separator()
    
    def write_task3_results(self, problem_types: List[str], analysis_engine):
        """Write Task 3 results"""
        self.write_header("Task 3 Results")
        
        for problem_type in problem_types:
            self.write_section(f"Task 3: Primary Approach Distribution for Problem Type: '{problem_type}'")
            
            result = analysis_engine.analyze_primary_approach_distribution_by_problem_type(problem_type)
            
            if "error" in result:
                self.output_file.write(f"Error: {result['error']}\n")
                continue
            
            self.output_file.write(f"Summary: {result['summary']}\n")
            self.output_file.write("\nDetailed results:\n")
            self.output_file.write(result['distribution'].to_string() + "\n")
            
            self.write_separator()
    
    def write_task4_results(self, analysis_engine):
        """Write Task 4 results"""
        self.write_header("Task 4 Results")
        
        result = analysis_engine.analyze_problem_type_causing_exploration()
        
        if "error" in result:
            self.output_file.write(f"Error: {result['error']}\n")
            return
        
        # Write secondary approach analysis
        if 'secondary_approach_analysis' in result:
            self.output_file.write("Problem types by frequency of 'Exploration' as Secondary Approach:\n")
            self.output_file.write(result['secondary_approach_analysis'].to_string() + "\n")
        
        # Write exploration indicator analysis
        if 'exploration_indicator_analysis' in result:
            self.output_file.write("\nProblem types by average 'exploration_indicator' (higher means more exploration):\n")
            self.output_file.write(result['exploration_indicator_analysis'].to_string() + "\n")
        
        self.write_separator()
    
    def write_footer(self):
        """Write report footer"""
        self.output_file.write("\nAnalysis script finished.\n")
