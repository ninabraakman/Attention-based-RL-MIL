import pandas as pd
import numpy as np
from pathlib import Path

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Starting Analysis Script ---")

# <<< THIS IS THE FIX >>>
# The script is run from the main folder, so we must specify the path to the input file.
input_file = 'results/final_results_with_scores.csv' 
summary_file = 'results/summary_statistics.csv'
# --- END OF FIX ---

# Load the cleaned data
try:
    df = pd.read_csv(input_file)
    print(f"Successfully loaded '{input_file}' with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: '{input_file}' not found.")
    exit()

metrics_to_analyze = ['test_f1_score', 'test_auc_score']
summary_stats = df.groupby(['full_model_name', 'dataset'])[metrics_to_analyze].agg([np.mean, np.std])
summary_stats = summary_stats.round(4)

print("\n--- Summary Statistics (Mean & Std Dev over 10 Seeds) ---\n")
print(summary_stats)

# Save the summary to the 'results' folder
summary_stats.to_csv(summary_file)
print(f"\nSummary statistics saved to '{summary_file}'")
print("\n--- Analysis Script Finished ---")