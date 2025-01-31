import pandas as pd
import glob
import os
import sys

# Define file paths for all result files across both configurations
config_a_files = glob.glob(os.getenv("CONFIG_A_PATH", "config_a/*.csv")) # Replace with your file paths
config_b_files = glob.glob(os.getenv("CONFIG_B_PATH", "config_b/*.csv")) # Replace with your file paths

# Function to process and calculate correctness for all files
def process_files(file_list):
    if not file_list:
        print("No files provided for processing.")
        return None
    
    all_data = []
    
    for file in file_list:
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        # Standardizing column names
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        
        # Ensure the required columns are present
        if 'correct_answer' not in df.columns or 'judge_answer' not in df.columns:
            print(f"Warning: Skipping {file} due to missing columns.")
            continue

        # Standardizing values to avoid case and space mismatches
        df['correct_answer'] = df['correct_answer'].astype(str).str.strip().str.lower()
        df['judge_answer'] = df['judge_answer'].astype(str).str.strip().str.lower()

        # Calculate correctness (True/False)
        df['correct'] = df['correct_answer'] == df['judge_answer']
        all_data.append(df)
    
    if not all_data:
        print("No valid files to process.")
        return None

    # Combine all processed data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

# Process both configurations
df_a = process_files(config_a_files)
df_b = process_files(config_b_files)

# Combine both configurations into a single dataframe
valid_dfs = [df for df in [df_a, df_b] if df is not None]

if not valid_dfs:
    print("No valid data processed from any configuration. Exiting script.")
    sys.exit(1)

combined_results = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else None

if combined_results is None or 'correct' not in combined_results.columns:
    print("No valid 'correct' column found in results. Exiting script.")
    sys.exit(1)

# Calculate overall judge accuracy
if combined_results.empty:
    print("No valid judge answers available. Cannot compute accuracy.")
    sys.exit(1)

overall_accuracy = combined_results['correct'].mean()

# Save the combined results
output_path = os.getenv("OUTPUT_CSV_PATH", "overall_judge_accuracy.csv") # Replace with your output path

if not combined_results.empty:
    output_dir = os.path.dirname(output_path)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    combined_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Print results
print(f"Overall Judge Accuracy for MMLU: {overall_accuracy:.2%}")