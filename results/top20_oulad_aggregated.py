# This file does the quantitative analysis of what instance types are most often deemed important in the oulad aggregated dataset. 
# It calculates sHAP scores for the entire test set and creates a table of the instance types that most often are in the top-20 highest scores instances. 
import pandas as pd
import numpy as np
import os
import sys
import pickle
import torch
from collections import defaultdict
import time
import shap
import json

# Add project root to path to find models.py
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from models import PolicyNetwork, create_mil_model_with_dict
except (NameError, ImportError) as e:
    print("Warning: Could not import local 'models.py'. Using dummy classes.")
    class PolicyNetwork(torch.nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(1,1)
        def forward(self, x): return None, None, torch.rand(len(x), 1)
    def create_mil_model_with_dict(d): return torch.nn.Identity()


SEED_TO_ANALYZE = 8
OUTPUT_DIR = 'final_report_oulad_aggregated/'
TOP_K = 20

BASE_PATH = f'/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_{SEED_TO_ANALYZE}/oulad_aggregated/instances/tabular/label/bag_size_20/repset_22_16_22/'
RAW_DATA_PKL_PATH = '/projects/prjs1491/Attention-based-RL-MIL/data/oulad/oulad_aggregated_raw.pkl'

ILSE_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement/')
PHAM_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_pham_reg_sum_sample_without_replacement/')
GREEDY_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement/')


def load_rl_model(run_dir_path):
    """Correctly loads a trained RL policy network from specified file paths."""
    print("Attempting to load RL model...")
    device = torch.device("cpu")
    model_weights_path = os.path.join(run_dir_path, 'sweep_best_model.pt')
    rl_config_path = os.path.join(run_dir_path, 'sweep_best_model_config.json')
    mil_config_path = os.path.join(run_dir_path, '..', 'best_model_config.json')
    mil_weights_path = os.path.join(run_dir_path, '..', 'best_model.pt')
    
    try:
        with open(mil_config_path) as f: mil_config = json.load(f)
        with open(rl_config_path) as f: rl_config = json.load(f)
    except FileNotFoundError as e:
        print(f"FATAL: A config file was not found: {e}"); return None

    task_model = create_mil_model_with_dict(mil_config)
    task_model.load_state_dict(torch.load(mil_weights_path, map_location=device))
    
    policy_network = PolicyNetwork(
        task_model=task_model, state_dim=rl_config['state_dim'], hdim=rl_config['hdim'],
        learning_rate=rl_config['learning_rate'], device=device, task_type=rl_config['task_type'],
        min_clip=rl_config.get('min_clip'), max_clip=rl_config.get('max_clip'),
        sample_algorithm=rl_config.get('sample_algorithm'), no_autoencoder=rl_config.get('no_autoencoder_for_rl', False)
    )
    policy_network.load_state_dict(torch.load(model_weights_path, map_location=device))
    policy_network.eval()
    print("RL model loaded successfully.")
    return policy_network

def generate_general_labels(raw_bag_data):
    """Generates GENERALIZED labels for feature aggregation."""
    labels = []
    for instance_list in raw_bag_data:
        main_tuple = instance_list[0]; feature_type = main_tuple[0]
        if feature_type == 'assessment_type': labels.append(f"Assessment: {main_tuple[1]}")
        elif feature_type == 'activity_type': labels.append(f"VLE Clicks: {main_tuple[1]}")
        else: labels.append(feature_type.replace('_', ' ').title())
    return labels

def build_bag_to_labels_map(raw_data_path):
    """Builds a global map from bag_id to its list of general instance labels."""
    print("\n Building global map from Bag ID to Instance Labels ")
    try:
        with open(raw_data_path, 'rb') as f: data = pickle.load(f)
    except FileNotFoundError: print(f"FATAL: Raw data file not found at {raw_data_path}"); sys.exit(1)
    bag_to_labels = {str(bag_id): generate_general_labels(raw_bag) for bag_id, raw_bag in zip(data['bag_ids'], data['raw_bags'])}
    print(f"Map created with {len(bag_to_labels)} entries.")
    return bag_to_labels

def parse_feature_string(s):
    """Safely parses a string representation of a numpy array."""
    if not isinstance(s, str): return np.array([])
    return np.fromstring(s.replace('[','').replace(']','').replace('\n',' '), sep=' ')

def print_counts_table(model_name, counts_dict, bag_presence_dict, total_bags):
    """Formats and prints the final counts, including the new bag presence metric."""
    print(f"\n--- Top-{TOP_K} Feature Counts for: {model_name.upper()} ---")
    if not counts_dict:
        print("No instances were found in the top 20."); return
        
    df = pd.DataFrame(counts_dict.items(), columns=['Feature Type', 'Total Count in Top-20'])
    df['Present in X Bags'] = df['Feature Type'].map(bag_presence_dict)
    df['Bag Presence %'] = df['Present in X Bags'] / total_bags
    df['Avg Count When Present'] = df['Total Count in Top-20'] / df['Present in X Bags']
    
    df = df.sort_values(by='Total Count in Top-20', ascending=False).reset_index(drop=True)
    
    print(f"Based on {total_bags} common bags.")
    print(df.to_string())


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load data and find common bags
    ilse_file, pham_file = os.path.join(ILSE_RUN_DIR, 'attention_ilse_outputs.csv'), os.path.join(PHAM_RUN_DIR, 'attention_pham_outputs.csv')
    try:
        ilse_df, pham_df = pd.read_csv(ilse_file), pd.read_csv(pham_file)
    except FileNotFoundError as e: print(f"Could not find input file: {e}"); sys.exit(1)
        
    ilse_df, pham_df = ilse_df[ilse_df['is_padding_instance'] == False], pham_df[pham_df['is_padding_instance'] == False]
    common_bags = sorted(list(set(ilse_df['bag_id'].unique()).intersection(set(pham_df['bag_id'].unique()))))
    print(f"Found {len(common_bags)} bags common to both ILSE and PHAM files. These will be analyzed.")
    if not common_bags: print("No common bags found. Aborting."); sys.exit(1)

    ilse_df_common, pham_df_common = ilse_df[ilse_df['bag_id'].isin(common_bags)].copy(), pham_df[pham_df['bag_id'].isin(common_bags)].copy()

    # Step 2: Calculate or load SHAP values
    shap_output_file = os.path.join(OUTPUT_DIR, f'shap_scores_seed_{SEED_TO_ANALYZE}.csv')

    if os.path.exists(shap_output_file):
        print(f"Found pre-computed SHAP file. Loading from: {shap_output_file}")
        shap_df_loaded = pd.read_csv(shap_output_file)
        # Use a merge to safely add the SHAP values, aligning on the original index.
        ilse_df_common = ilse_df_common.reset_index().merge(shap_df_loaded, on='index', how='left').set_index('index')
        ilse_df_common.rename(columns={'shap_value_y': 'shap_value'}, inplace=True)
    else:
        print("No pre-computed SHAP file found. Starting new calculation...")
        greedy_model = load_rl_model(GREEDY_RUN_DIR)
        if not greedy_model: print("FATAL: Epsilon-Greedy model could not be loaded. Aborting."); sys.exit(1)
        instance_features = np.vstack(ilse_df_common['original_instance_content'].apply(parse_feature_string))
        
        def shap_wrapper_f(numpy_data):
            with torch.no_grad():
                tensor_data = torch.from_numpy(numpy_data).float().to(torch.device("cpu"))
                _, _, exp_reward = greedy_model(tensor_data)
                return exp_reward.cpu().numpy()

        print(f"Calculating SHAP values for {len(instance_features)} instances... (this will take a while)")
        explainer = shap.KernelExplainer(shap_wrapper_f, shap.sample(instance_features, min(50, len(instance_features))))
        shap_values = explainer.shap_values(instance_features)
        
        ilse_df_common['shap_value'] = np.abs(shap_values).mean(axis=1)
        print("SHAP values calculated.")
        print(f"Saving SHAP values to: {shap_output_file}")
        ilse_df_common[['shap_value']].to_csv(shap_output_file, index_label='index')

    # Step 3: Count top-20 instance types per model
    bag_to_labels_map = build_bag_to_labels_map(RAW_DATA_PKL_PATH)

    ilse_counts, pham_counts, shap_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    ilse_bag_presence, pham_bag_presence, shap_bag_presence = defaultdict(int), defaultdict(int), defaultdict(int)

    for i, bag_id in enumerate(common_bags):
        if (i + 1) % 100 == 0: print(f"  -> Processing bag {i + 1} of {len(common_bags)}...")
        if bag_id not in bag_to_labels_map: continue
        bag_labels = bag_to_labels_map[bag_id]

        bag_ilse_df = ilse_df_common[ilse_df_common['bag_id'] == bag_id]
        bag_pham_df = pham_df_common[pham_df_common['bag_id'] == bag_id]
        if len(bag_labels) != len(bag_ilse_df): continue

        # Process ilse/ Gated Attention
        top_ilse_indices = bag_ilse_df.nlargest(TOP_K, 'attention_score').index
        unique_labels_in_top20 = set()
        for idx in top_ilse_indices:
            label = bag_labels[bag_ilse_df.index.get_loc(idx)]
            ilse_counts[label] += 1
            unique_labels_in_top20.add(label)
        for label in unique_labels_in_top20: ilse_bag_presence[label] += 1

        # Process Pham/ Multi-Head Attention
        top_pham_indices = bag_pham_df.nlargest(TOP_K, 'attention_score').index
        unique_labels_in_top20 = set()
        for idx in top_pham_indices:
            label = bag_labels[bag_pham_df.index.get_loc(idx)]
            pham_counts[label] += 1
            unique_labels_in_top20.add(label)
        for label in unique_labels_in_top20: pham_bag_presence[label] += 1
            
        # Process Epsilon-greedy (SHAP)
        top_shap_indices = bag_ilse_df.nlargest(TOP_K, 'shap_value').index
        unique_labels_in_top20 = set()
        for idx in top_shap_indices:
            label = bag_labels[bag_ilse_df.index.get_loc(idx)]
            shap_counts[label] += 1
            unique_labels_in_top20.add(label)
        for label in unique_labels_in_top20: shap_bag_presence[label] += 1

    # Step 4: print final results
    print("\n\n Final results")
    print_counts_table("ILSE/Gated Attention", ilse_counts, ilse_bag_presence, len(common_bags))
    print_counts_table("PHAM/Multi-Head Attention", pham_counts, pham_bag_presence, len(common_bags))
    print_counts_table("Epsilon-Greedy (SHAP)", shap_counts, shap_bag_presence, len(common_bags))
