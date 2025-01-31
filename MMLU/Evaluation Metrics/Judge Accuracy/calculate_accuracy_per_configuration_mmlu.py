import pandas as pd
import glob
import os

# Define file paths for config A and B
config_a_files = glob.glob(os.getenv("CONFIG_A_PATH", "config_a/*.csv")) # Replace with your file paths
config_b_files = glob.glob(os.getenv("CONFIG_B_PATH", "config_b/*.csv")) # Replace with your file paths


# Function to calculate judge accuracy
def calculate_judge_accuracy(file_list):
    if not file_list:
        print("No files provided for processing.")
        return None, None
    
    all_data = []
    
    for file in file_list:
        try:
            df = pd.read_csv(file, encoding='utf-8')

            # Standardize column names by stripping spaces and converting to lowercase
            df.columns = df.columns.str.strip().str.lower()

            # Ensure required columns exist after standardization
            if 'correct_answer' not in df.columns or 'judge_answer' not in df.columns:
                print(f"Warning: Skipping {file} due to missing columns.")
                continue

            # Standardizing answers to avoid case and space mismatches
            df['correct_answer'] = df['correct_answer'].astype(str).str.strip().str.lower()
            df['judge_answer'] = df['judge_answer'].astype(str).str.strip().str.lower()

            # Calculate correctness
            df['correct'] = df['correct_answer'] == df['judge_answer']
            all_data.append(df)
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not all_data:
        print("No valid files to process.")
        return None, None
    
    # Combine all subsets into one dataframe
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Compute judge accuracy
    accuracy = combined_df['correct'].mean()
    
    return combined_df, accuracy

# Calculate accuracy for config A
df_a, judge_accuracy_a = calculate_judge_accuracy(config_a_files)
if df_a is not None:
    print(f"\nJudge Accuracy for Config A: {judge_accuracy_a:.2%}")
    output_path_a = os.getenv("OUTPUT_CSV_PATH_A", "judge_accuracy_a.csv") # Replace with your output path
    df_a.to_csv(output_path_a, index=False)
else:
    print("No valid data processed for Config A.")

# Calculate accuracy for config B
df_b, judge_accuracy_b = calculate_judge_accuracy(config_b_files)
if df_b is not None:
    print(f"\nJudge Accuracy for Config B: {judge_accuracy_b:.2%}")
    output_path_b = os.getenv("OUTPUT_CSV_PATH_B", "judge_accuracy_b.csv") # Replace with your output path
    df_b.to_csv(output_path_b, index=False)
else:
    print("No valid data processed for Config B.")

# Compare configurations if both have valid results
if df_a is not None and df_b is not None:
    if judge_accuracy_a > judge_accuracy_b:
        print("The judge was more accurate in Config A.")
    elif judge_accuracy_a < judge_accuracy_b:
        print("The judge was more accurate in Config B.")
    else:
        print("The judge performed equally in both configurations.")