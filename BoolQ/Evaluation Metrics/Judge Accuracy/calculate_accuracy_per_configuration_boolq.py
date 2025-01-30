import pandas as pd
import os

# Load and read result files (.csv)
config_a_path = os.getenv("CONFIG_A_PATH", "results_config_a.csv") # Replace with your file paths
config_b_path = os.getenv("CONFIG_B_PATH", "results_config_b.csv") # Replace with your file paths

# Ensure files exist before proceeding
if not os.path.exists(config_a_path) or not os.path.exists(config_b_path):
    raise FileNotFoundError("One or both configuration result files are missing.")

# Load datasets
df_a = pd.read_csv(config_a_path)
df_b = pd.read_csv(config_b_path)

# Ensure required columns exist
required_columns = {'answer', 'judge_answer'}
for df, config_name in zip([df_a, df_b], ["A", "B"]):
    if not required_columns.issubset(df.columns):
        raise KeyError(f"Missing required columns in Config {config_name}: {required_columns - set(df.columns)}")

# Standardize text format for comparison
df_a['answer'] = df_a['answer'].astype(str).str.upper()
df_a['judge_answer'] = df_a['judge_answer'].astype(str).str.upper()
df_b['answer'] = df_b['answer'].astype(str).str.upper()
df_b['judge_answer'] = df_b['judge_answer'].astype(str).str.upper()

# Compare judge's answers to the correct answers
df_a['correct'] = df_a['answer'] == df_a['judge_answer']
df_b['correct'] = df_b['answer'] == df_b['judge_answer']

# Calculate accuracy per configuration
judge_accuracy_a = df_a['correct'].mean()
judge_accuracy_b = df_b['correct'].mean()

# Print accuracy results
print(f"Judge Accuracy for Config A: {judge_accuracy_a:.2%}")
print(f"Judge Accuracy for Config B: {judge_accuracy_b:.2%}")

# Save the updated datasets back to CSV (without redundant accuracy column)
df_a.to_csv("results_config_a_updated.csv", index=False)
df_b.to_csv("results_config_b_updated.csv", index=False)

# Compare both configurations
if judge_accuracy_a > judge_accuracy_b:
    print("The judge was more accurate in Config A.")
elif judge_accuracy_a < judge_accuracy_b:
    print("The judge was more accurate in Config B.")
else:
    print("The judge performed equally in both configurations.")