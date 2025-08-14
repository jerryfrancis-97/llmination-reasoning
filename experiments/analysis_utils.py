import pandas as pd
import os
from typing import Optional
from analysis_config import AnalysisConfig

def load_data(csv_file_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV data into a pandas DataFrame with error handling
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        DataFrame if successful, None if failed
    """
    try:
        if not os.path.exists(csv_file_path):
            print(f"Error: The file {csv_file_path} was not found.")
            return None
        
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print(f"Warning: The file {csv_file_path} is empty.")
            return None
        
        print(f"Data loaded successfully from {csv_file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file_path} is empty or corrupted.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file {csv_file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        return None

def setup_analysis_environment(csv_file_path: str) -> tuple[str, str]:
    """
    Set up the analysis environment (create directories, generate file paths)
    
    Args:
        csv_file_path: Path to the input CSV file
        
    Returns:
        Tuple of (analysis_dir, output_file_path)
    """
    # Create analysis directory if it doesn't exist
    analysis_dir = AnalysisConfig.DEFAULT_ANALYSIS_DIR
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Extract model name and generate output file path
    model_name = AnalysisConfig.extract_model_name(csv_file_path)
    output_file_path = AnalysisConfig.get_output_file_path(analysis_dir, model_name)
    
    print(f"Analysis directory: {analysis_dir}")
    print(f"Output file: {output_file_path}")
    
    return analysis_dir, output_file_path

def print_data_preview(df: pd.DataFrame, num_rows: int = 5):
    """Print a preview of the loaded data"""
    if df is None or df.empty:
        print("No data to preview.")
        return
    
    print(f"\nFirst {num_rows} rows of the data:")
    print(df.head(num_rows))
    
    print("\nDataframe Info:")
    print(df.info())
    
    print(f"\nData shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def validate_analysis_prerequisites(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has the minimum required structure for analysis
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if prerequisites are met, False otherwise
    """
    if df is None or df.empty:
        print("Error: DataFrame is None or empty.")
        return False
    
    # Check for minimum required columns
    min_required_cols = ['primary_approach']
    missing_min_cols = [col for col in min_required_cols if col not in df.columns]
    
    if missing_min_cols:
        print(f"Error: Missing minimum required columns: {missing_min_cols}")
        return False
    
    # Check if there's any data to analyze
    if len(df) < 1:
        print("Error: DataFrame has no rows to analyze.")
        return False
    
    print("Analysis prerequisites validated successfully.")
    return True
