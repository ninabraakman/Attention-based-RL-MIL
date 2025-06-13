# gather_results.py
import pandas as pd
import os
import json
from pathlib import Path

def gather_all_results():
    """
    Traverses the local runs/ directory to find all results.json files for
    MIL-only, Epsilon-Greedy, and multiple Attention model configurations.
    """
    base_path = Path('./runs/classification/')
    
    if not base_path.exists():
        print(f"Error: Base path not found at '{base_path}'")
        return []

    # --- Configuration ---
    seeds = range(11) # 0 through 10
    datasets_config = {
        'oulad_aggregated_subset': '22_16_22',
        'oulad_full_subset': '20_16_20',
        'oulad_aggregated': '22_16_22',
        'oulad_full': '20_16_20',
    }
    pooling_methods = ['MeanMLP', 'MaxMLP', 'AttentionMLP', 'repset']

    # Define all the RL model subdirectories we need to search for.
    # CRUCIAL: You may need to edit the 'folder_name' to match your exact directory names.
    RL_MODEL_CONFIGS = [
        {
            "model_type": "EpsilonGreedy_without_replacement",
            "folder_name": "neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement"
        },
        {
            "model_type": "ILSE_without_replacement",
            "folder_name": "neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement"
        },
        {
            "model_type": "PHAM_without_replacement",
            "folder_name": "neg_policy_only_loss_attention_pham_reg_sum_sample_without_replacement"
        },
        {
            "model_type": "EpsilonGreedy_static",
            "folder_name": "neg_policy_only_loss_epsilon_greedy_reg_sum_sample_static"
        },
        {
            "model_type": "ILSE_static",
            "folder_name": "neg_policy_only_loss_attention_ilse_reg_sum_sample_static"
        },
        {
            "model_type": "PHAM_static",
            "folder_name": "neg_policy_only_loss_attention_pham_reg_sum_sample_static"
        }
    ]

    # --- Data Gathering ---
    results_list = []
    print("Starting data gathering from local files...")
    
    for seed in seeds:
        for dataset_name, layers in datasets_config.items():
            for pooling_base in pooling_methods:
                pooling_folder_name = f"{pooling_base}_{layers}"
                
                # Path to the specific pooling method directory
                base_model_path = (base_path / f'seed_{seed}' / dataset_name / 
                                   'instances/tabular/label/bag_size_20' / pooling_folder_name)

                if not base_model_path.is_dir():
                    continue

                # 1. Gather MIL-only baseline results
                mil_json_path = base_model_path / 'results.json'
                if mil_json_path.is_file():
                    with open(mil_json_path, 'r') as f:
                        json_content = json.load(f)
                    json_content.update({
                        'seed': seed,
                        'dataset': dataset_name,
                        'pooling_method': pooling_base,
                        'model_type': 'MIL_only'
                    })
                    results_list.append(json_content)

                # 2. Gather all RL model results
                for rl_config in RL_MODEL_CONFIGS:
                    rl_json_path = base_model_path / rl_config['folder_name'] / 'results.json'
                    if rl_json_path.is_file():
                        with open(rl_json_path, 'r') as f:
                            json_content = json.load(f)
                        json_content.update({
                            'seed': seed,
                            'dataset': dataset_name,
                            'pooling_method': pooling_base,
                            'model_type': rl_config['model_type']
                        })
                        results_list.append(json_content)
                        
    print(f"Data gathering complete. Found {len(results_list)} result files.")
    return results_list

# --- Main execution block ---
if __name__ == "__main__":
    all_results_data = gather_all_results()

    if all_results_data:
        # Convert the list of dictionaries to a pandas DataFrame
        results_df = pd.DataFrame(all_results_data)
        
        output_filename = 'results/all_local_results.csv'
        os.makedirs('results', exist_ok=True)
        
        print(f"Saving combined data to '{output_filename}'...")
        results_df.to_csv(output_filename, index=False)
        print("Successfully created the master results CSV file.")
        print("\nSample of the final data:")
        print(results_df.head())
    else:
        print("No result files were found.")