# This file uses the table created in model_comparison.py to compare performance between dataset variations. 
# It creates a chart that averages the pooling methods for each model to get a avarage model performance on the 2 dataset variations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'results/final_thesis_results.csv' 
OUTPUT_FILE = 'results/dataset_impact_chart.png'

def create_performance_impact_chart(summary_df, output_path):
    """
    Creates a grouped bar chart to compare the average performance of each main
    framework across the two datasets.
    """
    print("Creating aggregated framework performance chart for RQ2...")
    try:
        f1_data = summary_df['f1_score'].copy()
    except KeyError:
        print(f"Error: Could not find 'f1_score' in the columns of {INPUT_FILE}.")
        return

    f1_data_flat = f1_data.reset_index()
    # Group by the explicit column names
    framework_means = f1_data_flat.groupby(['model_type', 'dataset'])['mean'].mean()
    framework_stds = f1_data_flat.groupby(['model_type', 'dataset'])['mean'].std()
    plot_means = framework_means.unstack(level='dataset')
    plot_stds = framework_stds.unstack(level='dataset')
    
    plot_means.index = plot_means.index.map(lambda x: x.replace('_without_replacement','').replace('MIL_only', 'Simple MIL'))
    plot_stds.index = plot_stds.index.map(lambda x: x.replace('_without_replacement','').replace('MIL_only', 'Simple MIL'))
    
    # Rename frameworks for the final plot
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

    # Plot
    x = np.arange(len(plot_means.index)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, agg_means, width, label='OULAD Aggregated', yerr=agg_std, capsize=5, color='darkcyan')
    rects2 = ax.bar(x + width/2, full_means, width, label='OULAD Full', yerr=full_std, capsize=5, color='coral')

    ax.set_ylabel('Mean F1-Score (averaged over pooling methods)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_means.index, rotation=0, ha='center', fontsize=12)
    ax.legend(fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0.6, 1.0)

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Aggregated framework chart saved to '{output_path}'")
    plt.show()

if __name__ == "__main__":
    print(f"Loading summary data from {INPUT_FILE}...")
    try:
        summary_table_df = pd.read_csv(INPUT_FILE, header=[0, 1], index_col=[0, 1, 2])
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        exit()
    
    # Call the function to create the plot
    create_performance_impact_chart(summary_table_df, OUTPUT_FILE)