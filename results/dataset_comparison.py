# This file uses the table created in model_comparison.py to compare performance between dataset variations. 
# It creates a chart that averages the pooling methods for each model to get a avarage model performance on the 2 dataset variations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'results/performance_summary_table.csv' 
OUTPUT_FILE = 'results/dataset_impact_chart.png'

def create_performance_impact_chart(df, output_path):
    """
    Creates a grouped bar chart to compare the average performance of each main
    framework across the two datasets.
    """
    # Map the non-linear version to the generic name for plotting
    df['Framework'] = df['Framework'].replace({
        'RL-MIL (Gated Attention - non-linear)': 'RL-MIL (Gated Attention)'
    })
    framework_means = df.groupby(['Framework', 'Dataset'])['mean'].mean()
    framework_stds = df.groupby(['Framework', 'Dataset'])['mean'].std()

    plot_means = framework_means.unstack(level='Dataset')
    plot_stds = framework_stds.unstack(level='Dataset')
    
    # The framework order is updated to include both linear and non-linear models
    framework_order = [
        'Simple MIL', 
        'RL-MIL (Epsilon-Greedy)', 
        'RL-MIL (Gated Attention)', 
        'RL-MIL (Multi-Head Attention)'
    ]
    plot_means = plot_means.reindex(framework_order)
    plot_stds = plot_stds.reindex(framework_order)
    
    # Get the mean and std dev values for plotting
    agg_means = plot_means.get('oulad_aggregated', pd.Series(0, index=plot_means.index))
    full_means = plot_means.get('oulad_full', pd.Series(0, index=plot_means.index))
    agg_std = plot_stds.get('oulad_aggregated', 0).fillna(0)
    full_std = plot_stds.get('oulad_full', 0).fillna(0)

    # Plot
    x = np.arange(len(plot_means.index)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, agg_means, width, label='OULAD Aggregated', yerr=agg_std, capsize=5, color='darkcyan')
    rects2 = ax.bar(x + width/2, full_means, width, label='OULAD Full', yerr=full_std, capsize=5, color='coral')

    ax.set_ylabel('Mean F1-Score (averaged over pooling methods)', fontsize=20)
    ax.set_xticks(x)
    # The x-tick labels are updated to be concise
    ax.set_xticklabels([
        'Simple MIL', 
        'RL-MIL (Epsilon-Greedy)', 
        'RL-MIL (Gated Attention)', 
        'RL-MIL (Multi-Head Attention)'
    ], rotation=0, ha='center', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)

    ax.legend(fontsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0.6, 1.0)

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Aggregated framework chart saved to '{output_path}'")
    plt.show()

if __name__ == "__main__":
    print(f"Loading summary data from {INPUT_FILE}...")
    try:
        results_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        exit()
    
    # Call the function to create the plot
    create_performance_impact_chart(results_df, OUTPUT_FILE)
    
# # This file uses the table created in model_comparison.py to compare performance between dataset variations. 
# # It creates a chart that averages the pooling methods for each model to get a avarage model performance on the 2 dataset variations.
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# INPUT_FILE = 'results/final_thesis_results.csv' 
# OUTPUT_FILE = 'results/dataset_impact_chart.png'

# def create_performance_impact_chart(df, output_path):
#     """
#     Creates a grouped bar chart to compare the average performance of each main
#     framework across the two datasets.
#     """
    
#     mean_per_pooling = df.groupby(['Framework', 'Dataset', 'Pooling'])['F1_Score'].mean().reset_index()
    
#     framework_means = mean_per_pooling.groupby(['Framework', 'Dataset'])['F1_Score'].mean()
#     framework_stds = mean_per_pooling.groupby(['Framework', 'Dataset'])['F1_Score'].std()

#     plot_means = framework_means.unstack(level='Dataset')
#     plot_stds = framework_stds.unstack(level='Dataset')

#     framework_order = ['Simple MIL', 'RL-MIL (Epsilon-Greedy)', 'RL-MIL (Gated Attention)', 'RL-MIL (Multi-Head Attention)']
#     plot_means = plot_means.reindex(framework_order)
#     plot_stds = plot_stds.reindex(framework_order)
    
#     # Get the mean and std dev values for plotting
#     agg_means = plot_means.get('oulad_aggregated', pd.Series(0, index=plot_means.index))
#     full_means = plot_means.get('oulad_full', pd.Series(0, index=plot_means.index))
#     agg_std = plot_stds.get('oulad_aggregated', 0).fillna(0)
#     full_std = plot_stds.get('oulad_full', 0).fillna(0)

#     # Plot
#     x = np.arange(len(plot_means.index)) 
#     width = 0.35 

#     fig, ax = plt.subplots(figsize=(14, 8))
#     rects1 = ax.bar(x - width/2, agg_means, width, label='OULAD Aggregated', yerr=agg_std, capsize=5, color='darkcyan')
#     rects2 = ax.bar(x + width/2, full_means, width, label='OULAD Full', yerr=full_std, capsize=5, color='coral')

#     ax.set_ylabel('Mean F1-Score (averaged over pooling methods)', fontsize=20)
#     ax.set_xticks(x)
#     ax.set_xticklabels(plot_means.index, rotation=0, ha='center', fontsize=16)
#     ax.tick_params(axis='y', labelsize=18)

#     ax.legend(fontsize=24)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_ylim(0.6, 1.0)

#     fig.tight_layout()
#     plt.savefig(output_path)
#     print(f"Aggregated framework chart saved to '{output_path}'")
#     plt.show()

# if __name__ == "__main__":
#     print(f"Loading summary data from {INPUT_FILE}...")
#     try:
#         results_df = pd.read_csv(INPUT_FILE)
#     except FileNotFoundError:
#         print(f"ERROR: Input file not found at '{INPUT_FILE}'.")
#         exit()
#     except Exception as e:
#         print(f"An error occurred while reading the CSV: {e}")
#         exit()
    
#     # Call the function to create the plot
#     create_performance_impact_chart(results_df, OUTPUT_FILE)
