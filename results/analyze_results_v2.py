# analyze_results_v2.py

import pandas as pd
import numpy as np
from pathlib import Path

# Set pandas display options for full visibility
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Starting Analysis V2: Best-of-N-Seeds Performance ---")

# --- Step 1: Load the full dataset ---
# Path to the input file, assuming it's in the 'results' folder
input_file = 'results/final_results_with_scores.csv' 
try:
    df = pd.read_csv(input_file)
    print(f"Successfully loaded '{input_file}' with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: '{input_file}' not found.")
    print("Please ensure the gather job has run successfully first.")
    exit()

# --- Step 2: Create a unified validation score column ---
# This is crucial for a fair comparison. We need one column that represents
# validation F1-score for all models. We'll use the same logic as for the test scores.
df['validation_f1_score'] = df['eval/avg-f1'].fillna(df['eval/f1_micro'])
print("Created unified 'validation_f1_score' column.")

# Drop rows where validation score is missing, as we can't rank them
df.dropna(subset=['validation_f1_score'], inplace=True)

# --- Step 3: Find the best run for each model configuration ---
# This is the key logic:
# 1. Group the DataFrame by the model name and dataset.
# 2. For each group, find the index of the row with the maximum 'validation_f1_score'.
# 3. Select these rows from the original DataFrame.
best_runs_idx = df.groupby(['full_model_name', 'dataset'])['validation_f1_score'].idxmax()
best_runs_df = df.loc[best_runs_idx]
print("Identified the best performing seed for each model based on validation F1 score.")

# --- Step 4: Format and report the results ---
# We select the columns we care about for the final report.
report_df = best_runs_df[[
    'full_model_name',
    'dataset',
    'seed', # It's useful to know which seed was the best
    'validation_f1_score', # The score used to select the model
    'test_f1_score', # The final reported score
    'test_auc_score' # The final reported score
]].sort_values(by=['full_model_name', 'dataset'])

print("\n--- Peak Performance on Test Set (from best validation seed) ---\n")
print(report_df)

# Save the report to a new CSV file
output_file = 'results/best_seed_performance.csv'
report_df.to_csv(output_file, index=False)
print(f"\nPeak performance report saved to '{output_file}'")
print("\n--- Analysis V2 Finished ---")