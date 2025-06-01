import pandas as pd
import os
def load_data(csv_file_path):
    """Loads the CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Data loaded successfully from {csv_file_path}")
        # Basic data overview
        print("\nFirst 5 rows of the data:")
        print(df.head())
        print("\nDataframe Info:")
        df.info()
        # Assuming 'model_name' is not a column if the CSV is specific to one model
        # If 'model_name' column exists, it can be used for filtering.
        # For this script, we'll assume the analysis is for the model in the given CSV.
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def analyze_self_interpretation_vs_actual(df, file=None):
    """
    Analyzes how the model's self-interpretation (reasoning_type)
    compares with its actual approach (primary_approach, secondary_approach).
    """
    if df is None:
        return

    file.write("\n--- Analysis: Model's Self-Interpretation vs. Actual Approach ---\n")
    
    # Check if necessary columns exist
    required_cols = ['reasoning_type', 'primary_approach', 'secondary_approach']
    if not all(col in df.columns for col in required_cols):
        file.write(f"Error: One or more required columns ({', '.join(required_cols)}) not found in the DataFrame.\n")
        missing = [col for col in required_cols if col not in df.columns]
        file.write(f"Missing columns: {missing}\n")
        return

    # Group by reasoning_type and then by primary and secondary approaches
    comparison = df.groupby(['reasoning_type', 'primary_approach', 'secondary_approach']).size().reset_index(name='count')
    
    file.write("\nComparison of Model's Claim vs. Observed Approach (Frequency):\n")
    file.write(comparison.to_string() + "\n")

    file.write("\nHighlighting potential discrepancies (where reasoning_type might not align with primary_approach):\n")
    # Example: Model claims "Purely Reasoning" but primary approach is "Recall"
    discrepancy_example_1 = comparison[
        (comparison['reasoning_type'].str.contains("Reasoning", case=False, na=False)) &
        (comparison['primary_approach'].str.contains("Recall", case=False, na=False))
    ]
    if not discrepancy_example_1.empty:
        file.write("\nCases where model claims reasoning, but primary approach is recall:\n")
        file.write(discrepancy_example_1.to_string() + "\n")
    else:
        file.write("\nNo direct 'Reasoning' claim vs. 'Recall' primary approach found with current simple check.\n")

    # Example: Model claims "Recall" but primary approach is "Reasoning"
    discrepancy_example_2 = comparison[
        (comparison['reasoning_type'].str.contains("Recall", case=False, na=False)) &
        (comparison['primary_approach'].str.contains("Reasoning", case=False, na=False))
    ]
    if not discrepancy_example_2.empty:
        file.write("\nCases where model claims recall, but primary approach is reasoning:\n")
        file.write(discrepancy_example_2.to_string() + "\n")
    else:
        file.write("\nNo direct 'Recall' claim vs. 'Reasoning' primary approach found with current simple check.\n")
    file.write("-" * 70 + "\n\n")


def task1_approach_by_reasoning_type_and_subject(df, reasoning_type_filter, file=None):
    """
    For a (given model, reasoning_type), see what approach (primary + secondary)
    the model takes for different subjects in general.
    """
    if df is None:
        return
    
    required_cols = ['reasoning_type', 'subject', 'primary_approach', 'secondary_approach']
    if not all(col in df.columns for col in required_cols):
        file.write(f"Error: Task 1 requires columns: {', '.join(required_cols)}.\n")
        return

    file.write(f"\n--- Task 1: Approach by Subject for Reasoning Type: '{reasoning_type_filter}' ---\n")
    filtered_df = df[df['reasoning_type'] == reasoning_type_filter]

    if filtered_df.empty:
        file.write(f"No data found for reasoning_type: {reasoning_type_filter}\n")
        return

    # Combine primary and secondary approach for a clearer view
    filtered_df['combined_approach'] = filtered_df['primary_approach'] + " + " + filtered_df['secondary_approach'].fillna('None')
    
    # Show the distribution of combined approaches for each subject
    # Using crosstab for a clearer pivot-table like view
    # Or using groupby and then value_counts
    result = filtered_df.groupby('subject')['combined_approach'].value_counts().rename('count').reset_index()
    
    # To make it more readable, pivot
    try:
        pivot_result = result.pivot_table(index='subject', columns='combined_approach', values='count', fill_value=0)
        file.write(f"\nApproach (Primary + Secondary) distribution for subjects where model's reasoning_type was '{reasoning_type_filter}':\n")
        # file.write(pivot_result.to_string() + "\n")
    except Exception as e:
        file.write("Could not pivot the result, showing grouped list instead:\n")
        file.write(result.to_string() + "\n")
        
    file.write("-" * 70 + "\n\n")
    return result


def task2_approach_and_likelihood_by_subject(df, file=None):
    """
    For a given model, which approach it chooses (primary+secondary)
    with what likelihood (likelihood_assessment) for different subjects.
    """
    if df is None:
        return
        
    required_cols = ['subject', 'primary_approach', 'secondary_approach', 'likelihood_assessment']
    if not all(col in df.columns for col in required_cols):
        file.write(f"Error: Task 2 requires columns: {', '.join(required_cols)}.\n")
        return

    file.write("\n--- Task 2: Approach (Primary + Secondary) and Likelihood by Subject ---\n")
    
    # Combine primary and secondary approach
    df_copy = df.copy() # To avoid SettingWithCopyWarning
    df_copy['combined_approach'] = df_copy['primary_approach'] + " + " + df_copy['secondary_approach'].fillna('None')
    
    # Group by subject, combined_approach, and likelihood_interpretation
    result = df_copy.groupby(['subject', 'combined_approach', 'likelihood_assessment']).size().reset_index(name='count')
    
    file.write("\nApproach (Primary + Secondary) and Likelihood Assessment Counts by Subject:\n")
    # For better readability, one might want to pivot or further process this.
    # For now, printing the grouped result.
    # Sort for consistency
    result = result.sort_values(by=['subject', 'count'], ascending=[True, False])
    file.write(result.to_string() + "\n")

    return result

def task3_primary_approach_distribution_by_problem_type(df, problem_type_filter, file=None):
    """
    For a (given model, problem_type), what is the distribution (primary_approach) we see.
    """
    if df is None:
        return
        
    required_cols = ['problem_type', 'primary_approach']
    if not all(col in df.columns for col in required_cols):
        file.write(f"Error: Task 3 requires columns: {', '.join(required_cols)}.\n")
        return

    file.write(f"\n--- Task 3: Primary Approach Distribution for Problem Type: '{problem_type_filter}' ---\n")
    filtered_df = df[df['problem_type'] == problem_type_filter]

    if filtered_df.empty:
        file.write(f"No data found for problem_type: {problem_type_filter}\n")
        return

    distribution = filtered_df['primary_approach'].value_counts(normalize=True) * 100 # as percentage
    
    file.write(f"\nDistribution of Primary Approach for problem_type '{problem_type_filter}':\n")
    # file.write(distribution.to_string() + "\n")

    return distribution

def task4_problem_type_causing_exploration(df, file=None):
    """
    For a given model, which problem_type causes the model to explore more
    (based on secondary_approach or exploration_indicator).
    """
    if df is None:
        return

    file.write("\n--- Task 4: Problem Types Causing More Exploration ---\n")

    # Option 1: Based on 'secondary_approach' being 'Exploration' (or similar term)
    if 'secondary_approach' in df.columns and 'problem_type' in df.columns:
        exploration_df = df[df['secondary_approach'].str.contains("Exploration", case=False, na=False)]
        if not exploration_df.empty:
            exploration_by_problem_type = exploration_df['problem_type'].value_counts().reset_index()
            exploration_by_problem_type.columns = ['problem_type', 'exploration_count']
            file.write("\nProblem types by frequency of 'Exploration' as Secondary Approach:\n")
            file.write(exploration_by_problem_type.sort_values(by='exploration_count', ascending=False).to_string() + "\n")
        else:
            file.write("No instances found where 'secondary_approach' is 'Exploration'.\n")
    else:
        file.write("Skipping exploration analysis by 'secondary_approach' (columns missing).\n")


    # Option 2: Based on 'exploration_indicator' (if it exists and is numeric)
    if 'exploration_indicator' in df.columns and 'problem_type' in df.columns:
        if pd.api.types.is_numeric_dtype(df['exploration_indicator']):
            avg_exploration_indicator = df.groupby('problem_type')['exploration_indicator'].mean().sort_values(ascending=False)
            file.write("\nProblem types by average 'exploration_indicator' (higher means more exploration):\n")
            file.write(avg_exploration_indicator.to_string() + "\n")
        else:
            file.write("'exploration_indicator' column is not numeric, cannot calculate mean.\n")
    else:
        file.write("Skipping exploration analysis by 'exploration_indicator' (columns missing).\n")
        
    file.write("-" * 70 + "\n\n")
    # The function could return one or both of these results if needed for further processing
    # For now, it prints them.


if __name__ == '__main__':
    # Replace with the actual path to your CSV file
    # The user uploaded: "results_deepseek-r1-distill-llama-70b.csv"
    # csv_file = "results/run_20250529_045043_final_testing/results_deepseek-r1-distill-llama-70b.csv"
    # csv_file = "results/run_20250529_152151_final_testing/results_gemini-1.5-flash.csv"
    # csv_file = "results/run_20250529_152151_final_testing/results_gemini-2.0-flash.csv"
    csv_file = "results/run_20250530_234358_final_llama3_70b_8192/results_llama3-70b-8192.csv"
    
    # Create analysis directory if it doesn't exist
    analysis_dir = "analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Open output file for writing analysis results
    model_name = os.path.basename(csv_file).split('_')[1]  # Extract model name from csv filename
    output_file = os.path.join(analysis_dir, f"analysis_results_{model_name}.txt")
    with open(output_file, "w") as f:
        main_df = load_data(csv_file)

        if main_df is not None:
            # General analysis comparing how the model interprets its own reasoning vs its actual approach
            f.write("=== Self-Interpretation vs Actual Analysis ===\n\n")
            analyze_self_interpretation_vs_actual(main_df, file=f)

            # Print unique values from key columns to understand data distribution
            f.write("\nUnique values for filtering:\n")
            if 'reasoning_type' in main_df.columns:
                f.write(f"Unique reasoning_types: {main_df['reasoning_type'].unique()}\n")
            if 'problem_type' in main_df.columns:
                f.write(f"Unique problem_types: {main_df['problem_type'].unique()}\n") 
            if 'subject' in main_df.columns:
                f.write(f"Unique subjects: {main_df['subject'].unique()}\n")
            f.write("-" * 70 + "\n\n")

            # Task 1: For each reasoning type (e.g. deductive, inductive), analyze how the model's 
            # approach changes across different academic subjects (e.g. algebra vs geometry)
            f.write("\n=== Task 1 Results ===\n")
            if 'reasoning_type' in main_df.columns and not main_df['reasoning_type'].empty:
                reasoning_types = main_df['reasoning_type'].unique()
                for rt in reasoning_types:
                    f.write(f"\nRunning Task 1 with reasoning_type: {rt}\n")
                    result_df = task1_approach_by_reasoning_type_and_subject(main_df, rt, file=f)
                    if result_df is not None:
                        f.write("\nDetailed results:\n")
                        f.write(result_df.to_string() + "\n")
                        f.write("-" * 70 + "\n\n")
            else:
                f.write("\nSkipping Task 1 as 'reasoning_type' column is missing or empty.\n")

            # Task 2: For each academic subject, analyze what approaches (e.g. computation, recall) 
            # the model tends to use and how confident it is in those approaches
            f.write("\n=== Task 2 Results ===\n")
            task2_approach_and_likelihood_by_subject(main_df, file=f)
            
            # Task 3: For each problem type (e.g. word problems, proofs), analyze the distribution
            # of primary solution approaches to identify patterns in how the model tackles different problems
            f.write("\n=== Task 3 Results ===\n")
            if 'problem_type' in main_df.columns and not main_df['problem_type'].empty:
                problem_types = main_df['problem_type'].unique()
                for problem_type in problem_types:
                    f.write(f"\nRunning Task 3 with problem_type: {problem_type}\n")
                    result_df = task3_primary_approach_distribution_by_problem_type(main_df, problem_type, file=f)
                    if result_df is not None:
                        f.write("\nDetailed results:\n")
                        f.write(result_df.to_string() + "\n")
                        f.write("-" * 70 + "\n\n")
            else:
                f.write("\nSkipping Task 3 as 'problem_type' column is missing or empty.\n")

            # Task 4: Identify which types of problems cause the model to explore multiple solution paths
            # rather than immediately converging on a single approach, based on exploration indicators
            f.write("\n=== Task 4 Results ===\n")
            task4_problem_type_causing_exploration(main_df, file=f)

            f.write("\nAnalysis script finished.")
        else:
            f.write("Could not perform analysis as data loading failed.")

        print(f"Analysis results have been written to {output_file}")
