import pandas as pd
import glob
import os

# Load and read result files (.csv) Note that to run descript the result files must have the columns: question, answer_1, answer_2, correct_answer, judge_answer and winning_debater
config_a_path = os.getenv("CONFIG_A_PATH", "results_config_a.csv") # Replace with your file paths
config_b_path = os.getenv("CONFIG_B_PATH", "results_config_b.csv") # Replace with your file paths

# Function to calculate correct and incorrect ratings for a given debater
def calculate_ratings(file_path, debater_name, correct_column="correct_answer", winner_column="winning_debater", answer_1_col="answer_1"):
    df = pd.read_csv(file_path, encoding='utf-8')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

    # Ensure required columns are present
    required_columns = {correct_column, winner_column, answer_1_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in {file_path}")

    # Identify when the debater was assigned the correct answer
    df['assigned_correct'] = (df[answer_1_col] == df[correct_column]) & (debater_name.lower() == "debater a".lower())
    df['assigned_incorrect'] = (df[answer_1_col] != df[correct_column]) & (debater_name.lower() == "debater a".lower())

    # Calculate wins when assigned the correct and incorrect answer
    wins_when_correct = df[(df['assigned_correct']) & (df[winner_column].str.lower() == debater_name.lower())].shape[0]
    total_when_correct = df[df['assigned_correct']].shape[0]

    wins_when_incorrect = df[(df['assigned_incorrect']) & (df[winner_column].str.lower() == debater_name.lower())].shape[0]
    total_when_incorrect = df[df['assigned_incorrect']].shape[0]

    # Calculate correct and incorrect ratings
    correct_rating = wins_when_correct / total_when_correct if total_when_correct > 0 else 0
    incorrect_rating = wins_when_incorrect / total_when_incorrect if total_when_incorrect > 0 else 0

    return correct_rating, incorrect_rating

# Calculate ratings for Gemini
correct_gemini_a, incorrect_gemini_a = calculate_ratings(config_a_path, debater_name="Debater A")
correct_gemini_b, incorrect_gemini_b = calculate_ratings(config_b_path, debater_name="Debater B")

# Calculate ratings for Claude
correct_claude_b, incorrect_claude_b = calculate_ratings(config_a_path, debater_name="Debater B")
correct_claude_a, incorrect_claude_a = calculate_ratings(config_b_path, debater_name="Debater A")

# Compute balanced ratings for Gemini and Claude
balanced_correct_gemini = (correct_gemini_a + correct_gemini_b) / 2
balanced_incorrect_gemini = (incorrect_gemini_a + incorrect_gemini_b) / 2

balanced_correct_claude = (correct_claude_a + correct_claude_b) / 2
balanced_incorrect_claude = (incorrect_claude_a + incorrect_claude_b) / 2

# Print results
print(f"Correct Rating for Gemini: {balanced_correct_gemini:.2%}")
print(f"Incorrect Rating for Gemini: {balanced_incorrect_gemini:.2%}\n")

print(f"Correct Rating for Claude: {balanced_correct_claude:.2%}")
print(f"Incorrect Rating for Claude: {balanced_incorrect_claude:.2%}")

# Save results to CSV file
results = pd.DataFrame({
    "Model": ["Gemini", "Claude"],
    "Correct Rating": [balanced_correct_gemini, balanced_correct_claude],
    "Incorrect Rating": [balanced_incorrect_gemini, balanced_incorrect_claude]
})

output_path = os.getenv("OUTPUT_CSV_PATH", "correct_incorrect_ratings_results.csv")  # replace with your output path
results.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")