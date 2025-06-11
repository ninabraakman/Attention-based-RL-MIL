# gather_and_clean.py

import os
import json
from pathlib import Path
import pandas as pd

print("--- Starting Data Gathering and Cleaning Script ---")

def gather_results_data():
    base_path = Path('/projects/prjs1491/Attention-based-RL-MIL/runs/classification/')
    if not base_path.exists():
        print(f"Error: Base path not found at '{base_path}'")
        return []

    seeds = range(1, 11)
    datasets_config = {'oulad_full': '20_16_20', 'oulad_aggregated': '22_16_22'}
    pooling_methods = ['MeanMLP', 'MaxMLP', 'AttentionMLP', 'repset']
    rl_model_folders = {
        'RL_EpsilonGreedy': 'neg_policy_only_loss_epsilon_greedy_reg_sum_sample_static',
        'RL_Attention_ILSE': 'neg_policy_only_loss_attention_ilse_reg_sum_sample_static',
        'RL_Attention_PHAM': 'neg_policy_only_loss_attention_pham_reg_sum_sample_static'
    }
    
    results_list = []
    print("Starting file search...")
    for seed in seeds:
        for dataset_name, layers in datasets_config.items():
            for pooling_base in pooling_methods:
                pooling_folder_name = f"{pooling_base}_{layers}"
                current_path = (base_path / f'seed_{seed}' / f'{dataset_name}/instances/tabular/label/bag_size_20' / pooling_folder_name)
                if not current_path.is_dir(): continue

                mil_json_path = current_path / 'results.json'
                if mil_json_path.is_file():
                    results_list.append({'seed': seed, 'dataset': dataset_name, 'pooling_method': pooling_base, 'model_type': 'MIL_only', 'full_model_name': f'{pooling_base}_MIL', 'path': mil_json_path})
                
                for model_name, rl_folder in rl_model_folders.items():
                    rl_json_path = current_path / rl_folder / 'results.json'
                    if rl_json_path.is_file():
                        results_list.append({'seed': seed, 'dataset': dataset_name, 'pooling_method': pooling_base, 'model_type': model_name, 'full_model_name': f'{pooling_base}_{model_name}', 'path': rl_json_path})
    print(f"File search complete. Found {len(results_list)} result files.")
    return results_list

def process_and_clean_data(path_list):
    all_data_records = []
    print(f"Processing {len(path_list)} files...")
    for item in path_list:
        try:
            with open(item['path'], 'r') as f:
                json_content = json.load(f)
            record = {k: v for k, v in item.items() if k != 'path'}
            record.update(json_content)
            all_data_records.append(record)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Skipping file {item['path']} due to error: {e}")
    
    df = pd.DataFrame(all_data_records)
    print("Data loaded into DataFrame. Cleaning...")

    columns_to_drop = ['model', 'embedding_model', 'bag_size', 'label', 'test/f1', 'test/accuracy', 'test/precision', 'test/recall', 'test/ensemble-f1']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    df_cleaned['test_f1_score'] = df_cleaned['test/avg-f1'].fillna(df_cleaned['test/f1_micro'])
    df_cleaned['test_auc_score'] = df_cleaned['test/auc']
    print("Cleaning complete.")
    return df_cleaned

# --- Main execution block ---
if __name__ == "__main__":
    # <<< CHANGED >>> Create the results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    found_files = gather_results_data()
    if found_files:
        final_df = process_and_clean_data(found_files)
        
        # <<< CHANGED >>> Save the output file inside the 'results' folder
        output_filename = results_dir / 'final_results_with_scores.csv'
        final_df.to_csv(output_filename, index=False)
        print(f"Successfully saved final cleaned data to '{output_filename}'")
    else:
        print("No results were found.")

print("--- Script Finished ---")