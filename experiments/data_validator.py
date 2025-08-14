import pandas as pd
from typing import List, Tuple, Optional
from analysis_config import AnalysisConfig

class DataValidator:
    """Handles data validation and integrity checks"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, task_name: str) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has required columns for a specific task
        
        Args:
            df: DataFrame to validate
            task_name: Name of the task to validate for
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        if df is None or df.empty:
            return False, ["DataFrame is None or empty"]
        
        if task_name not in AnalysisConfig.REQUIRED_COLUMNS:
            return False, [f"Unknown task: {task_name}"]
        
        required_cols = AnalysisConfig.REQUIRED_COLUMNS[task_name]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        return len(missing_cols) == 0, missing_cols
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column_name: str) -> List[str]:
        """Get unique values from a column, handling missing values gracefully"""
        if df is None or column_name not in df.columns:
            return []
        
        unique_vals = df[column_name].dropna().unique()
        return [str(val) for val in unique_vals if pd.notna(val)]
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> dict:
        """Check overall data quality and return statistics"""
        if df is None:
            return {"total_rows": 0, "total_columns": 0, "missing_data": 0}
        
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_data": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Add column-specific stats
        for col in df.columns:
            if col in df.columns:
                stats[f"{col}_unique_values"] = df[col].nunique()
                stats[f"{col}_missing"] = df[col].isnull().sum()
        
        return stats
    
    @staticmethod
    def filter_dataframe(df: pd.DataFrame, column: str, value: str, case_sensitive: bool = False) -> pd.DataFrame:
        """Filter DataFrame by column value with case-insensitive option"""
        if df is None or column not in df.columns:
            return pd.DataFrame()
        
        if case_sensitive:
            return df[df[column] == value]
        else:
            return df[df[column].str.contains(value, case=False, na=False)]
