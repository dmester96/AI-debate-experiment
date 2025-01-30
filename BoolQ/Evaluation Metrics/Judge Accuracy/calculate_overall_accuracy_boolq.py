import pandas as pd
import os

# Load and read result files (.csv)
config_a_path = os.getenv("CONFIG_A_PATH", "results_config_a.csv") # Replace with your file paths
config_b_path = os.getenv("CONFIG_B_PATH", "results_config_b.csv") # Replace with your file paths
output_path = os.getenv("OUTPUT_PATH", "results_combined.csv") # Replace with your output path

# Ensure files exist before reading
if not os.path.exists(config_a_path) or not os.path.exists(config_b_path):
    raise FileNotFoundError("One or both configuration result files are missing.")

df_a = pd.read_csv(config_a_path)
df_b = pd.read_csv(config_b_path)

# Concatenate both dataframes to calculate overall accuracy
df_combined = pd.concat([df_a, df_b], ignore_index=True)

# Ensure required columns exist
required_columns = {'answer', 'judge_answer'}
if not required_columns.issubset(df_combined.columns):
    raise KeyError(f"Missing required columns: {required_columns - set(df_combined.columns)}")

# Convert answers to uppercase to avoid formatting issues
df_combined['answer'] = df_combined['answer'].astype(str).str.upper()
df_combined['judge_answer'] = df_combined['judge_answer'].astype(str).str.upper()

# Calculate overall accuracy
df_combined['correct'] = df_combined['answer'] == df_combined['judge_answer']
overall_accuracy = df_combined['correct'].mean()

print(f"Overall Judge Accuracy across both configurations: {overall_accuracy:.2%}")

# Save the combined results to a new CSV file
df_combined.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")