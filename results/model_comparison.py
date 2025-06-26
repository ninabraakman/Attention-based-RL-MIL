# This file uses the csv created by gather_results.py to create a summary of all results. 
# It avarages the F1-score of all model architectures over 10 seeds and calculates the standard deviations. 
# It also performs statistical tests to prove significance based on the sub-questions of the thesis.
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations
import os

def run_statistical_tests(df):
    """
    Performs t-tests and organizes the p-values into clean summary tables.
    """
    rl_frameworks = ['RL-MIL (Epsilon-Greedy)', 'RL-MIL (Gated Attention)', 'RL-MIL (Multi-Head Attention)']
    datasets = df['Dataset'].unique()
    pooling_methods = df['Pooling'].unique()

    # Test 1: Compare RL-based models to each other (for SQ2)
    print("\n\nStatistical Test 1: Comparing RL Framework Performance (SQ2)")
    print("Hypothesis: The RL models perform comparably. A high p-value (> 0.05) supports this.")
    
    results_srq2 = []
    model_pairs = list(combinations(rl_frameworks, 2))

    for dataset in datasets:
        for pooling in pooling_methods:
            row = {'Dataset': dataset, 'Pooling': pooling}
            for model1, model2 in model_pairs:
                scores1 = df[(df['Framework'] == model1) & (df['Dataset'] == dataset) & (df['Pooling'] == pooling)]['F1_Score']
                scores2 = df[(df['Framework'] == model2) & (df['Dataset'] == dataset) & (df['Pooling'] == pooling)]['F1_Score']
                
                col_name = f"{model1.split('(')[-1].replace(')','').strip()} vs. {model2.split('(')[-1].replace(')','').strip()}"
                
                if len(scores1) > 1 and len(scores2) > 1:
                    _, p_value = ttest_ind(scores1, scores2, equal_var=False) # Welch's t-test
                    row[col_name] = f"{p_value:.4f}"
                else:
                    row[col_name] = "N/A"
            results_srq2.append(row)
    
    # Create and print the summary DataFrame for SQ2
    srq2_summary_df = pd.DataFrame(results_srq2).set_index(['Dataset', 'Pooling'])
    print("\n P-Values for RL Framework Comparisons:")
    print(srq2_summary_df)


    # Test 2: Compare Simple MIL vs. a representative RL model on OULAD Full
    print("\n\nStatistical Test 2: Comparing Simple MIL vs. RL-MIL on OULAD Full (SQ1)")
    print("Hypothesis: The RL models are significantly better. A low p-value (< 0.05) supports this.")

    results_srq1 = []
    dataset_to_test = 'oulad_full' 
    
    for pooling in pooling_methods:
        model1 = 'Simple MIL'
        model2 = 'RL-MIL (Epsilon-Greedy)' # Representative RL model, could also be RL-MIL with Gated Attention or Multi-Head Attention

        scores1 = df[(df['Framework'] == model1) & (df['Dataset'] == dataset_to_test) & (df['Pooling'] == pooling)]['F1_Score']
        scores2 = df[(df['Framework'] == model2) & (df['Dataset'] == dataset_to_test) & (df['Pooling'] == pooling)]['F1_Score']

        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            is_significant = "Yes" if p_value < 0.05 else "No"
            results_srq1.append({
                'Pooling Method': pooling,
                'Comparison': f"{model1} vs. {model2}",
                'p-value': f"{p_value:.4f}",
                'Significant (p < 0.05)': is_significant
            })
        else:
             results_srq1.append({
                'Pooling Method': pooling,
                'Comparison': f"{model1} vs. {model2}",
                'p-value': "N/A",
                'Significant (p < 0.05)': "N/A"
            })
    
    # Create and print the summary DataFrame for SRQ1
    srq1_summary_df = pd.DataFrame(results_srq1).set_index('Pooling Method')
    print("\n P-Values for Simple MIL vs. RL-MIL on OULAD Full:")
    print(srq1_summary_df)


def create_summary_table(df):
    """
    Calculates the mean and standard deviation for F1 scores, grouped by
    framework, pooling method, and dataset.
    """
    print("\n--- Generating Summary Table (Mean & Std Dev) ---")
    grouping_cols = ['Framework', 'Pooling', 'Dataset']
    summary = df.groupby(grouping_cols)['F1_Score'].agg(['mean', 'std']).round(4)
    return summary

if __name__ == "__main__":
    INPUT_FILE = 'results/final_thesis_results.csv'
    SUMMARY_TABLE_OUTPUT_FILE = 'results/performance_summary_table.csv'

    # Load Data 
    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        main_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{INPUT_FILE}'.")
        print("Please run the gather_results.py script first to generate this file.")
        exit()
    
    # Run Analysis 
    if not main_df.empty:
        summary_table = create_summary_table(main_df)
        os.makedirs('results', exist_ok=True)
        summary_table.to_csv(SUMMARY_TABLE_OUTPUT_FILE)
        print("\n Performance Summary Table:")
        print(summary_table)
        print(f"\n Summary table saved to '{SUMMARY_TABLE_OUTPUT_FILE}'")

        # Run and print the statistical tests
        run_statistical_tests(main_df)
        print("\nAnalysis complete.")
    else:
        print("Analysis finished with no data to process.")
