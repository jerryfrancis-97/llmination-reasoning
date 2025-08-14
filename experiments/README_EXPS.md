This directory contains the model response data analysis code, organized into multiple modules for better maintainability and separation of concerns.

## New Structure

### Core Modules

1. **`analysis_config.py`** - Configuration and constants
   - File paths and naming patterns
   - Required columns for different analysis tasks
   - Analysis settings and keywords
   - Output formatting constants

2. **`data_validator.py`** - Data validation and integrity checks
   - DataFrame validation for specific tasks
   - Column existence checks
   - Data quality statistics
   - Filtering utilities

3. **`analysis_engine.py`** - Core analysis logic
   - Self-interpretation vs actual approach analysis
   - Approach distribution analysis by reasoning type and subject
   - Likelihood assessment analysis
   - Problem type analysis
   - Exploration pattern analysis

4. **`report_generator.py`** - Report formatting and output
   - Section headers and separators
   - Data overview formatting
   - Task-specific result formatting
   - Error handling in output

5. **`analysis_utils.py`** - Utility functions
   - Data loading with error handling
   - Environment setup
   - Data preview and validation
   - Prerequisites checking

6. **`analysis_orchestrator.py`** - Main orchestration logic
   - Coordinates the entire analysis process
   - Error handling and recovery
   - Main entry point for the application

### Legacy File

- **`model_response_data_analyis.py`** - Original file (kept for backward compatibility)
  - Contains only the basic `load_data` function
  - Points users to the new structure

## Usage

### Running the Analysis

To run the complete analysis, use the new orchestrator:

```bash
python analysis_orchestrator.py
```

### Custom Analysis

You can also use individual components for custom analysis:

```python
from analysis_engine import AnalysisEngine
from data_validator import DataValidator
from report_generator import ReportGenerator

# Load your data
df = pd.read_csv("your_data.csv")

# Create analysis engine
engine = AnalysisEngine(df)

# Run specific analysis
result = engine.analyze_self_interpretation_vs_actual()
print(result)
```


## Configuration

Update the CSV file path in `analysis_orchestrator.py`:

```python
csv_file_paths = [
    "path/to/your/results_file.csv"
]
```

## Extending the Analysis

To add new analysis tasks:

1. Add required columns to `AnalysisConfig.REQUIRED_COLUMNS`
2. Add new method to `AnalysisEngine`
3. Add corresponding report method to `ReportGenerator`
4. Integrate into the orchestrator

This modular structure makes it easy to extend and maintain the analysis capabilities.
