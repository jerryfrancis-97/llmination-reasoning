import pandas as pd
from typing import Dict, List, Optional, Tuple
from analysis_config import AnalysisConfig
from data_validator import DataValidator

class AnalysisEngine:
    """Handles all data analysis operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validator = DataValidator()
    
    def analyze_self_interpretation_vs_actual(self) -> Dict:
        """Analyze model's self-interpretation vs actual approach"""
        is_valid, missing_cols = self.validator.validate_dataframe(self.df, 'self_interpretation')
        if not is_valid:
            return {"error": f"Missing columns: {missing_cols}"}
        
        # Group by reasoning_type and then by primary and secondary approaches
        comparison = self.df.groupby(['reasoning_type', 'primary_approach', 'secondary_approach']).size().reset_index(name='count')
        
        # Find potential discrepancies
        discrepancies = self._find_discrepancies(comparison)
        
        return {
            "comparison": comparison,
            "discrepancies": discrepancies,
            "summary": self._summarize_discrepancies(discrepancies)
        }
    
    def analyze_approach_by_reasoning_type_and_subject(self, reasoning_type: str) -> Dict:
        """Analyze approach distribution by reasoning type and subject"""
        is_valid, missing_cols = self.validator.validate_dataframe(self.df, 'task1')
        if not is_valid:
            return {"error": f"Missing columns: {missing_cols}"}
        
        filtered_df = self.validator.filter_dataframe(self.df, 'reasoning_type', reasoning_type)
        if filtered_df.empty:
            return {"error": f"No data found for reasoning_type: {reasoning_type}"}
        
        # Combine primary and secondary approach
        filtered_df['combined_approach'] = filtered_df['primary_approach'] + " + " + filtered_df['secondary_approach'].fillna('None')
        
        # Analyze distribution
        result = filtered_df.groupby('subject')['combined_approach'].value_counts().rename('count').reset_index()
        
        return {
            "reasoning_type": reasoning_type,
            "filtered_data": filtered_df,
            "distribution": result,
            "summary": self._summarize_approach_distribution(result)
        }
    
    def analyze_approach_and_likelihood_by_subject(self) -> Dict:
        """Analyze approach and likelihood distribution by subject"""
        is_valid, missing_cols = self.validator.validate_dataframe(self.df, 'task2')
        if not is_valid:
            return {"error": f"Missing columns: {missing_cols}"}
        
        # Combine primary and secondary approach
        df_copy = self.df.copy()
        df_copy['combined_approach'] = df_copy['primary_approach'] + " + " + df_copy['secondary_approach'].fillna('None')
        
        # Group by subject, combined_approach, and likelihood_assessment
        result = df_copy.groupby(['subject', 'combined_approach', 'likelihood_assessment']).size().reset_index(name='count')
        result = result.sort_values(by=['subject', 'count'], ascending=[True, False])
        
        return {
            "distribution": result,
            "summary": self._summarize_likelihood_distribution(result)
        }
    
    def analyze_primary_approach_distribution_by_problem_type(self, problem_type: str) -> Dict:
        """Analyze primary approach distribution for a specific problem type"""
        is_valid, missing_cols = self.validator.validate_dataframe(self.df, 'task3')
        if not is_valid:
            return {"error": f"Missing columns: {missing_cols}"}
        
        filtered_df = self.validator.filter_dataframe(self.df, 'problem_type', problem_type)
        if filtered_df.empty:
            return {"error": f"No data found for problem_type: {problem_type}"}
        
        distribution = filtered_df['primary_approach'].value_counts(normalize=True) * 100
        
        return {
            "problem_type": problem_type,
            "distribution": distribution,
            "summary": self._summarize_problem_type_distribution(distribution)
        }
    
    def analyze_problem_type_causing_exploration(self) -> Dict:
        """Analyze which problem types cause more exploration"""
        is_valid, missing_cols = self.validator.validate_dataframe(self.df, 'task4')
        if not is_valid:
            return {"error": f"Missing columns: {missing_cols}"}
        
        results = {}
        
        # Analysis based on secondary_approach
        if 'secondary_approach' in self.df.columns:
            exploration_df = self.df[self.df['secondary_approach'].str.contains("Exploration", case=False, na=False)]
            if not exploration_df.empty:
                exploration_by_problem_type = exploration_df['problem_type'].value_counts().reset_index()
                exploration_by_problem_type.columns = ['problem_type', 'exploration_count']
                results['secondary_approach_analysis'] = exploration_by_problem_type.sort_values(by='exploration_count', ascending=False)
        
        # Analysis based on exploration_indicator (if numeric)
        if 'exploration_indicator' in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df['exploration_indicator']):
                avg_exploration = self.df.groupby('problem_type')['exploration_indicator'].mean().sort_values(ascending=False)
                results['exploration_indicator_analysis'] = avg_exploration
        
        return results
    
    def _find_discrepancies(self, comparison_df: pd.DataFrame) -> Dict:
        """Find discrepancies between claimed and actual approaches"""
        discrepancies = {}
        
        # Check for reasoning claims vs recall approaches
        reasoning_vs_recall = comparison_df[
            (comparison_df['reasoning_type'].str.contains("Reasoning", case=False, na=False)) &
            (comparison_df['primary_approach'].str.contains("Recall", case=False, na=False))
        ]
        if not reasoning_vs_recall.empty:
            discrepancies['reasoning_claim_vs_recall_approach'] = reasoning_vs_recall
        
        # Check for recall claims vs reasoning approaches
        recall_vs_reasoning = comparison_df[
            (comparison_df['reasoning_type'].str.contains("Recall", case=False, na=False)) &
            (comparison_df['primary_approach'].str.contains("Reasoning", case=False, na=False))
        ]
        if not recall_vs_reasoning.empty:
            discrepancies['recall_claim_vs_reasoning_approach'] = recall_vs_reasoning
        
        return discrepancies
    
    def _summarize_discrepancies(self, discrepancies: Dict) -> str:
        """Create a summary of discrepancies found"""
        if not discrepancies:
            return "No discrepancies found between claimed and actual approaches."
        
        summary = f"Found {len(discrepancies)} types of discrepancies:\n"
        for key, data in discrepancies.items():
            summary += f"- {key}: {len(data)} instances\n"
        
        return summary
    
    def _summarize_approach_distribution(self, distribution_df: pd.DataFrame) -> str:
        """Create a summary of approach distribution"""
        if distribution_df.empty:
            return "No distribution data available."
        
        total_instances = distribution_df['count'].sum()
        unique_subjects = distribution_df['subject'].nunique()
        
        return f"Analysis covers {unique_subjects} subjects with {total_instances} total instances."
    
    def _summarize_likelihood_distribution(self, distribution_df: pd.DataFrame) -> str:
        """Create a summary of likelihood distribution"""
        if distribution_df.empty:
            return "No likelihood distribution data available."
        
        total_instances = distribution_df['count'].sum()
        unique_subjects = distribution_df['subject'].nunique()
        unique_approaches = distribution_df['combined_approach'].nunique()
        
        return f"Analysis covers {unique_subjects} subjects, {unique_approaches} approaches, with {total_instances} total instances."
    
    def _summarize_problem_type_distribution(self, distribution: pd.Series) -> str:
        """Create a summary of problem type distribution"""
        if distribution.empty:
            return "No distribution data available."
        
        dominant_approach = distribution.index[0]
        dominant_percentage = distribution.iloc[0]
        
        return f"Dominant approach: {dominant_approach} ({dominant_percentage:.1f}%)"
