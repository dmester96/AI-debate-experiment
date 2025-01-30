import pandas as pd
import os

# Load and read result files (.csv)
config_a_path = os.getenv("CONFIG_A_PATH", "results_config_a.csv") # Replace with your file paths
config_b_path = os.getenv("CONFIG_B_PATH", "results_config_b.csv") # Replace with your file paths

# Check if files exist
if not os.path.exists(config_a_path) or not os.path.exists(config_b_path):
    raise FileNotFoundError("One or both configuration CSV files are missing.")

# Function to calculate win rate for a given configuration file
def calculate_win_rate(file_path, debater_name):
    df = pd.read_csv(file_path, encoding='utf-8')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

    # Ensure required columns are present
    if 'winning_debater' not in df.columns:
        raise ValueError(f"Missing 'winning_debater' column in {file_path}")

    # Count wins for the given debater
    debater_wins = (df['winning_debater'].str.strip().str.lower() == debater_name.lower()).sum()
    total_debates = len(df)

    # Calculate win rate
    win_rate = debater_wins / total_debates if total_debates > 0 else 0
    return win_rate

# Calculate win rates for Gemini and Claude
# Gemini is Debater A in Config A and Debater B in Config B
win_rate_gemini_a = calculate_win_rate(config_a_path, debater_name="Debater A")
win_rate_gemini_b = calculate_win_rate(config_b_path, debater_name="Debater B")

# Claude is Debater B in Config A and Debater A in Config B
win_rate_claude_b = calculate_win_rate(config_a_path, debater_name="Debater B")
win_rate_claude_a = calculate_win_rate(config_b_path, debater_name="Debater A")

# Calculate the overall balanced win rate for each model
balanced_win_rate_gemini = (win_rate_gemini_a + win_rate_gemini_b) / 2
balanced_win_rate_claude = (win_rate_claude_a + win_rate_claude_b) / 2

# Print results
print(f"Win Rate for Gemini as Debater A: {win_rate_gemini_a:.2%}")
print(f"Win Rate for Gemini as Debater B: {win_rate_gemini_b:.2%}")
print(f"Overall Balanced Win Rate for Gemini: {balanced_win_rate_gemini:.2%}\n")

print(f"Win Rate for Claude as Debater A: {win_rate_claude_a:.2%}")
print(f"Win Rate for Claude as Debater B: {win_rate_claude_b:.2%}")
print(f"Overall Balanced Win Rate for Claude: {balanced_win_rate_claude:.2%}")

# Save results to CSV file
results = pd.DataFrame({
    "Model": ["Gemini", "Claude"],
    "Win Rate as Debater A": [win_rate_gemini_a, win_rate_claude_a],
    "Win Rate as Debater B": [win_rate_gemini_b, win_rate_claude_b],
    "Overall Balanced Win Rate": [balanced_win_rate_gemini, balanced_win_rate_claude]
})

output_path = os.getenv("OUTPUT_CSV_PATH", "win_rate_results.csv")  # Replace with your output path
results.to_csv(output_path, index=False)
print(f"Win rate results saved to {output_path}")