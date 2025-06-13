# subquestion1.py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = 'results/all_local_results.csv'
FULL_SUMMARY_OUTPUT_FILE = 'results/rq1_performance_summary_full.csv'
FINAL_F1_TABLE_OUTPUT_FILE = 'results/rq1_performance_f1_table.csv'

def analyze_main_performance(df):
    """
    Filters, cleans, and calculates performance statistics for the main thesis comparison.
    """
    # 1. --- Filtering ---
    print("Filtering data for final performance analysis...")
    
    # Keep only seeds 1 through 10
    df = df[df['seed'].between(1, 10)].copy()
    
    # Keep only the full datasets (not subsets)
    datasets_to_include = ['oulad_aggregated', 'oulad_full']
    df = df[df['dataset'].isin(datasets_to_include)]
    
    # Define the specific models we need for this analysis
    models_to_include = [
        'MIL_only',
        'EpsilonGreedy_without_replacement',
        'ILSE_without_replacement',
        'PHAM_without_replacement'
    ]
    df = df[df['model_type'].isin(models_to_include)]
    
    # A quick check to see how many rows we are working with
    if df.empty:
        print("Warning: No data found after filtering. Please check your CSV and filtering criteria.")
        return None, None
        
    print(f"Filtered down to {len(df)} rows for the analysis.")

    # 2. --- Data Cleaning (Unifying Columns) ---
    # Unify F1 scores from different model outputs into a single column
    f1_cols_to_check = ['test/avg-f1', 'test/f1', 'test/f1_micro']
    df['f1_score'] = np.nan
    for col in f1_cols_to_check:
        if col in df.columns:
            df['f1_score'] = df['f1_score'].fillna(df[col])
            
    # Do the same for AUC score
    auc_cols_to_check = ['test/auc']
    df['auc_score'] = np.nan
    for col in auc_cols_to_check:
        if col in df.columns:
            df['auc_score'] = df['auc_score'].fillna(df[col])

    # 3. --- Aggregation (For the full summary) ---
    grouping_cols = ['model_type', 'pooling_method', 'dataset']
    metrics_to_agg = ['f1_score', 'auc_score'] # Focusing on the key final metrics
    
    summary_stats = df.groupby(grouping_cols)[metrics_to_agg].agg([np.mean, np.std])
    summary_stats = summary_stats.round(4)
    
    return df, summary_stats

def create_final_f1_table(summary_df):
    """
    Takes the full summary DataFrame and creates the final, clean F1 table for the thesis.
    """
    print("\nCreating final F1-score table...")
    
    # We only care about the f1_score mean and std
    f1_data = summary_df['f1_score'][['mean', 'std']]
    
    # Unstack the 'dataset' level to turn it into columns
    f1_table = f1_data.unstack(level='dataset')
    
    # --- THIS IS THE FIX ---
    # Reorder the columns using the correct multi-level index tuples
    # This ensures we have [agg_mean, agg_std, full_mean, full_std]
    f1_table = f1_table[[
        ('mean', 'oulad_aggregated'),
        ('std',  'oulad_aggregated'),
        ('mean', 'oulad_full'),
        ('std',  'oulad_full')
    ]]
    # --- END OF FIX ---
    
    return f1_table


if __name__ == "__main__":
    print(f"Loading data from {INPUT_FILE}...")
    try:
        main_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'.")
        print("Please run gather_results.py (or the new process_all_results.py) first.")
        exit()

    # Perform the main analysis
    filtered_df, full_summary_table = analyze_main_performance(main_df)

    if full_summary_table is not None:
        # Save the full summary table
        os.makedirs('results', exist_ok=True)
        full_summary_table.to_csv(FULL_SUMMARY_OUTPUT_FILE)
        print("\n--- Full Performance Summary Table (All Metrics) ---")
        print(full_summary_table)
        print(f"\nFull summary saved to '{FULL_SUMMARY_OUTPUT_FILE}'")
        
        # --- NEW PART: Create and save the clean F1-only table ---
        final_f1_table = create_final_f1_table(full_summary_table)
        final_f1_table.to_csv(FINAL_F1_TABLE_OUTPUT_FILE)
        print("\n--- Final Thesis F1-Score Table ---")
        print(final_f1_table)
        print(f"\n✅ Final F1 table saved to '{FINAL_F1_TABLE_OUTPUT_FILE}'")
    else:
        print("Analysis finished with no data to process.")