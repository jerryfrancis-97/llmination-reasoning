import pandas as pd
from typing import List
from analysis_config import AnalysisConfig
from analysis_engine import AnalysisEngine
from report_generator import ReportGenerator
from analysis_utils import load_data, setup_analysis_environment, validate_analysis_prerequisites, print_data_preview

class AnalysisOrchestrator:
    """Main orchestrator for the analysis process"""
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.df = None
        self.analysis_engine = None
        self.report_generator = None
        
    def run_analysis(self) -> bool:
        """
        Run the complete analysis process
        
        Returns:
            True if analysis completed successfully, False otherwise
        """
        try:
            # Step 1: Load data
            if not self._load_data():
                return False
            
            # Step 2: Validate prerequisites
            if not validate_analysis_prerequisites(self.df):
                return False
            
            # Step 3: Setup analysis environment
            analysis_dir, output_file_path = setup_analysis_environment(self.csv_file_path)
            
            # Step 4: Run analysis and generate report
            with open(output_file_path, "w") as output_file:
                if not self._run_analysis_with_report(output_file):
                    return False
            
            print(f"Analysis completed successfully. Results written to {output_file_path}")
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False
    
    def _load_data(self) -> bool:
        """Load and validate the input data"""
        self.df = load_data(self.csv_file_path)
        if self.df is None:
            return False
        
        # Print data preview
        print_data_preview(self.df)
        
        # Initialize analysis engine
        self.analysis_engine = AnalysisEngine(self.df)
        
        return True
    
    def _run_analysis_with_report(self, output_file) -> bool:
        """Run all analysis tasks and generate the report"""
        try:
            # Initialize report generator
            self.report_generator = ReportGenerator(output_file)
            
            # Write data overview
            self.report_generator.write_data_overview(self.df, self.csv_file_path)
            
            # Write unique values section
            self.report_generator.write_unique_values_section(self.df)
            
            # Run Task 1: Self-interpretation vs actual analysis
            self._run_task1_analysis()
            
            # Run Task 2: Approach and likelihood by subject
            self._run_task2_analysis()
            
            # Run Task 3: Primary approach distribution by problem type
            self._run_task3_analysis()
            
            # Run Task 4: Problem types causing exploration
            self._run_task4_analysis()
            
            # Write footer
            self.report_generator.write_footer()
            
            return True
            
        except Exception as e:
            output_file.write(f"Error during analysis: {e}\n")
            print(f"Error during analysis: {e}")
            return False
    
    def _run_task1_analysis(self):
        """Run Task 1 analysis"""
        # Self-interpretation vs actual analysis
        analysis_result = self.analysis_engine.analyze_self_interpretation_vs_actual()
        self.report_generator.write_self_interpretation_analysis(analysis_result)
        
        # Approach by reasoning type and subject
        if 'reasoning_type' in self.df.columns:
            reasoning_types = self.df['reasoning_type'].dropna().unique()
            if len(reasoning_types) > 0:
                self.report_generator.write_task1_results(reasoning_types, self.analysis_engine)
    
    def _run_task2_analysis(self):
        """Run Task 2 analysis"""
        self.report_generator.write_task2_results(self.analysis_engine)
    
    def _run_task3_analysis(self):
        """Run Task 3 analysis"""
        if 'problem_type' in self.df.columns:
            problem_types = self.df['problem_type'].dropna().unique()
            if len(problem_types) > 0:
                self.report_generator.write_task3_results(problem_types, self.analysis_engine)
    
    def _run_task4_analysis(self):
        """Run Task 4 analysis"""
        self.report_generator.write_task4_results(self.analysis_engine)

def main():
    """Main entry point for the analysis script"""
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

if __name__ == "__main__":
    exit(main())
