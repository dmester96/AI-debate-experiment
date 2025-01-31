import pandas as pd
import glob
import os

# Define file paths for MMLU dataset (all CSV files for each configuration)
config_a_files = glob.glob(os.getenv("CONFIG_A_PATH", "config_a/*.csv")) # Replace with your file paths
config_b_files = glob.glob(os.getenv("CONFIG_B_PATH", "config_b/*.csv")) # Replace with your file paths

# Function to calculate win rate across multiple files
def calculate_win_rate(file_list, debater_name):
    total_debates = 0
    debater_wins = 0

    if not file_list:
        print("No files provided for processing.")
        return 0

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

        # Ensure required columns are present
        if 'winning_debater' not in df.columns:
            print(f"Warning: Missing 'winning_debater' column in {file_path}")
            continue

        # Clean winning_debater column and handle missing values
        df['winning_debater'] = df['winning_debater'].astype(str).str.strip().str.lower()
        
        # Count wins for the given debater (ignoring "None" cases)
        debater_wins += (df['winning_debater'] == debater_name.lower()).sum()
        total_debates += len(df[df['winning_debater'] != "none"])

    if total_debates == 0:
        return 0  # Avoid division by zero

    # Calculate win rate
    win_rate = debater_wins / total_debates
    return win_rate

# Calculate win rates for Gemini and Claude
# Gemini is Debater A in Config A and Debater B in Config B
win_rate_gemini_a = calculate_win_rate(config_a_files, debater_name="Debater A")
win_rate_gemini_b = calculate_win_rate(config_b_files, debater_name="Debater B")

# Claude is Debater B in Config A and Debater A in Config B
win_rate_claude_b = calculate_win_rate(config_a_files, debater_name="Debater B")
win_rate_claude_a = calculate_win_rate(config_b_files, debater_name="Debater A")

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

output_path = os.getenv("OUTPUT_CSV_PATH", "win_rate.csv") # Replace with your output path

# Ensure output directory exists before saving
output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

results.to_csv(output_path, index=False)
print(f"Win rate results saved successfully to {output_path}")