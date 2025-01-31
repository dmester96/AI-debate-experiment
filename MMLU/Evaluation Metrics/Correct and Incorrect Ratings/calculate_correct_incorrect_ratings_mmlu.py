import pandas as pd
import glob
import os

# Define file paths for MMLU dataset (multiple CSV files for each configuration)
config_a_files = glob.glob(os.getenv("CONFIG_A_PATH", "config_a/*.csv")) # Replace with your file paths
config_b_files = glob.glob(os.getenv("CONFIG_B_PATH", "config_b/*.csv")) # Replace with your file paths

# Function to calculate correct and incorrect ratings for a model
def calculate_correct_incorrect_ratings(file_list, debater_name, debater_role_config_a, debater_role_config_b):
    correct_wins = 0
    incorrect_wins = 0
    correct_assignments = 0
    incorrect_assignments = 0

    if not file_list:
        print("No files provided for processing.")
        return (0, 0)

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

        # Ensure required columns are present
        if not {'correct_answer', 'winning_debater', 'answer_1', 'answer_2'}.issubset(df.columns):
            print(f"Warning: Skipping {file_path} due to missing columns.")
            continue

        # Standardizing values to avoid case mismatches
        df['correct_answer'] = df['correct_answer'].astype(str).str.strip().str.lower()
        df['answer_1'] = df['answer_1'].astype(str).str.strip().str.lower()
        df['answer_2'] = df['answer_2'].astype(str).str.strip().str.lower()
        df['winning_debater'] = df['winning_debater'].astype(str).str.strip().str.lower()

        # Determine the assigned answer for the given debater role
        debater_role = debater_role_config_a if "config_a" in file_path.lower() else debater_role_config_b
        assigned_answer = df["answer_1"] if debater_role == "Debater A" else df["answer_2"]

        # Count wins where assigned answer was correct
        correct_condition = assigned_answer == df["correct_answer"]
        incorrect_condition = assigned_answer != df["correct_answer"]

        # Count instances where the model was assigned correct/incorrect answers
        correct_assignments += correct_condition.sum()
        incorrect_assignments += incorrect_condition.sum()

        # Count wins based on judge's decision (excluding "None" cases)
        debater_role_lower = debater_role.lower()
        correct_wins += ((df['winning_debater'] == debater_role_lower) & correct_condition).sum()
        incorrect_wins += ((df['winning_debater'] == debater_role_lower) & incorrect_condition).sum()

    # Avoid division by zero
    correct_rating = correct_wins / correct_assignments if correct_assignments > 0 else 0
    incorrect_rating = incorrect_wins / incorrect_assignments if incorrect_assignments > 0 else 0

    return correct_rating, incorrect_rating

# Calculate ratings for Gemini (Debater A in Config A, Debater B in Config B)
gemini_correct_a, gemini_incorrect_a = calculate_correct_incorrect_ratings(config_a_files, "Gemini", "Debater A", "Debater B") or (0, 0)
gemini_correct_b, gemini_incorrect_b = calculate_correct_incorrect_ratings(config_b_files, "Gemini", "Debater B", "Debater A") or (0, 0)

# Calculate balanced correct and incorrect ratings for Gemini
balanced_correct_gemini = (gemini_correct_a + gemini_correct_b) / 2
balanced_incorrect_gemini = (gemini_incorrect_a + gemini_incorrect_b) / 2

# Calculate ratings for Claude (Debater B in Config A, Debater A in Config B)
claude_correct_a, claude_incorrect_a = calculate_correct_incorrect_ratings(config_a_files, "Claude", "Debater B", "Debater A") or (0, 0)
claude_correct_b, claude_incorrect_b = calculate_correct_incorrect_ratings(config_b_files, "Claude", "Debater A", "Debater B") or (0, 0)

# Calculate balanced correct and incorrect ratings for Claude
balanced_correct_claude = (claude_correct_a + claude_correct_b) / 2
balanced_incorrect_claude = (claude_incorrect_a + claude_incorrect_b) / 2

# Print results
print(f"Gemini Correct Rating: {balanced_correct_gemini:.2%}, Incorrect Rating: {balanced_incorrect_gemini:.2%}")
print(f"Claude Correct Rating: {balanced_correct_claude:.2%}, Incorrect Rating: {balanced_incorrect_claude:.2%}")

# Save results to CSV file
results = pd.DataFrame({
    "Model": ["Gemini", "Claude"],
    "Correct Rating": [balanced_correct_gemini, balanced_correct_claude],
    "Incorrect Rating": [balanced_incorrect_gemini, balanced_incorrect_claude]
})

output_path = os.getenv("OUTPUT_CSV_PATH", "correct_incorrect_ratings.csv") #Replace with your output path

# Ensure output directory exists before saving
output_dir = os.path.dirname(output_path)
if output_dir:  
    os.makedirs(output_dir, exist_ok=True)

results.to_csv(output_path, index=False)
print("Correct and Incorrect ratings saved successfully.")