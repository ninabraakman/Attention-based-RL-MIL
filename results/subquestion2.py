# subquestion2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# This script reads the summary table created by subquestion1.py
INPUT_FILE = 'results/rq1_performance_summary_full.csv' 
# This is where the final plot will be saved
OUTPUT_FILE = 'results/rq2_dataset_impact_chart2.png'

def create_performance_impact_chart(summary_df, output_path):
    """
    Creates a grouped bar chart to compare the average performance of each main
    framework across the two datasets.
    """
    print("Creating aggregated framework performance chart for RQ2...")

    # The input DataFrame has a MultiIndex. We need to select the F1 score data.
    try:
        f1_data = summary_df['f1_score'].copy()
    except KeyError:
        print(f"Error: Could not find 'f1_score' in the columns of {INPUT_FILE}.")
        print("Please ensure subquestion1.py has run successfully and created the summary file.")
        return

    # --- MORE ROBUST AGGREGATION ---
    # Reset the multi-index to turn levels into columns. This is safer than grouping
    # by level names, which might not be parsed correctly from the CSV.
    f1_data_flat = f1_data.reset_index()

    # Now group by the explicit column names.
    framework_means = f1_data_flat.groupby(['model_type', 'dataset'])['mean'].mean()
    framework_stds = f1_data_flat.groupby(['model_type', 'dataset'])['mean'].std()

    # Unstack the 'dataset' level to turn it into columns for plotting.
    plot_means = framework_means.unstack(level='dataset')
    plot_stds = framework_stds.unstack(level='dataset')
    
    # Create clean names for the plot labels
    plot_means.index = plot_means.index.map(lambda x: x.replace('_without_replacement','').replace('MIL_only', 'Simple MIL'))
    plot_stds.index = plot_stds.index.map(lambda x: x.replace('_without_replacement','').replace('MIL_only', 'Simple MIL'))
    
    # --- Rename frameworks for the final plot ---
    rename_map = {
        'Simple MIL': 'MIL',
        'ILSE': 'RL-MIL (Gated Attention)',
        'PHAM': 'RL-MIL (Multi-Head Attention)',
        'EpsilonGreedy': 'RL-MIL (Epsilon-Greedy)'
    }
    plot_means.rename(index=rename_map, inplace=True)
    plot_stds.rename(index=rename_map, inplace=True)
    
    # Re-order the frameworks for a more logical presentation using the new names
    framework_order = ['MIL', 'RL-MIL (Epsilon-Greedy)', 'RL-MIL (Gated Attention)', 'RL-MIL (Multi-Head Attention)']
    plot_means = plot_means.reindex(framework_order)
    plot_stds = plot_stds.reindex(framework_order)
    
    # Get the mean and std dev values for plotting
    agg_means = plot_means.get('oulad_aggregated', pd.Series(0, index=plot_means.index))
    full_means = plot_means.get('oulad_full', pd.Series(0, index=plot_means.index))
    agg_std = plot_stds.get('oulad_aggregated', 0)
    full_std = plot_stds.get('oulad_full', 0)

    # --- Plotting Logic ---
    x = np.arange(len(plot_means.index))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, agg_means, width, label='OULAD Aggregated', yerr=agg_std, capsize=5, color='darkcyan')
    rects2 = ax.bar(x + width/2, full_means, width, label='OULAD Full', yerr=full_std, capsize=5, color='coral')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Mean F1-Score (averaged over pooling methods)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_means.index, rotation=0, ha='center', fontsize=12)
    ax.legend(fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0.6, 1.0) # Adjust ylim to better see the differences

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Aggregated framework chart saved to '{output_path}'")
    plt.show()

if __name__ == "__main__":
    print(f"Loading summary data from {INPUT_FILE}...")
    try:
        # Load the CSV. The header is on rows 0 and 1. The index is in the first 3 columns.
        # It's important to handle the specific format of the CSV.
        summary_table_df = pd.read_csv(INPUT_FILE, header=[0, 1], index_col=[0, 1, 2])
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'.")
        print("Please run subquestion1.py first to generate the summary statistics.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        print("Please ensure the CSV format is correct with a 3-level index and 2-level header.")
        exit()
    
    # Call the function to create the plot
    create_performance_impact_chart(summary_table_df, OUTPUT_FILE)



# def create_performance_impact_chart(summary_df, output_path):
#     """
#     Creates a grouped bar chart to compare model performance
#     across the two main datasets.
#     """
#     print("Creating performance comparison chart for RQ2...")

#     # The input DataFrame has a MultiIndex. We need to select the F1 score data.
#     # The columns are tuples like ('f1_score', 'mean')
#     try:
#         f1_data = summary_df['f1_score'].copy()
#     except KeyError:
#         print(f"Error: Could not find 'f1_score' in the columns of {INPUT_FILE}.")
#         print("Please ensure subquestion1.py has run successfully and created the summary file.")
#         return

#     # The dataframe is currently indexed by model_type, pooling_method, and dataset.
#     # We need to unstack the 'dataset' level to turn it into columns.
#     plot_data = f1_data.unstack(level='dataset')
    
#     # Create a single, clean name for the model architecture for the plot labels
#     plot_data.index = plot_data.index.map(lambda x: f"{x[0].replace('_without_replacement','').replace('EpsilonGreedy', 'Greedy')}-{x[1]}")
    
#     # Get the mean and std dev values for plotting
#     agg_means = plot_data.get(('mean', 'oulad_aggregated'), pd.Series(0, index=plot_data.index))
#     full_means = plot_data.get(('mean', 'oulad_full'), pd.Series(0, index=plot_data.index))
#     agg_std = plot_data.get(('std', 'oulad_aggregated'), 0)
#     full_std = plot_data.get(('std', 'oulad_full'), 0)

#     # --- Plotting Logic ---
#     x = np.arange(len(plot_data.index))  # the label locations
#     width = 0.35  # the width of the bars

#     fig, ax = plt.subplots(figsize=(20, 10))
#     rects1 = ax.bar(x - width/2, agg_means, width, label='OULAD Aggregated', yerr=agg_std, capsize=5, color='darkcyan')
#     rects2 = ax.bar(x + width/2, full_means, width, label='OULAD Full', yerr=full_std, capsize=5, color='coral')

#     # Add some text for labels, title and axes ticks
#     ax.set_ylabel('Mean F1-Score (over 10 seeds)', fontsize=14)
#     ax.set_title('Model Performance Impact of Dataset Composition', fontsize=18)
#     ax.set_xticks(x)
#     ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=12)
#     ax.legend(fontsize=14)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_ylim(0, 1.05) # F1 score is between 0 and 1

#     fig.tight_layout()
#     plt.savefig(output_path)
#     print(f"Chart saved to '{output_path}'")
#     plt.show()

# if __name__ == "__main__":
#     print(f"Loading summary data from {INPUT_FILE}...")
#     try:
#         summary_table_df = pd.read_csv(INPUT_FILE, header=[0, 1], index_col=[0, 1, 2])
#     except FileNotFoundError:
#         print(f"ERROR: Input file not found at '{INPUT_FILE}'.")
#         print("Please run subquestion1.py first to generate the summary statistics.")
#         exit()
    
#     # Call the function to create the plot
#     create_performance_impact_chart(summary_table_df, OUTPUT_FILE)