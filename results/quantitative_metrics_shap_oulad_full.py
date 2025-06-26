# This file prints out the top-20 most counted instance types and calculates the quantitative metrics for the oulad full dataset for Epsilon-greedy (SHAP). 
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
from scipy.stats import spearmanr

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
OUTPUT_DIR = 'final_report_oulad_full/'
TOP_K_FOR_COUNTS = 20
SAMPLE_SIZE = None
REPRODUCIBILITY_SEED = 42 

BASE_PATH = f'/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_{SEED_TO_ANALYZE}/oulad_full/instances/tabular/label/bag_size_20/repset_20_16_20' 
RAW_DATA_PKL_PATH = '/projects/prjs1491/Attention-based-RL-MIL/data/oulad/oulad_full_raw.pkl'

ILSE_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement/') 
GREEDY_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement/') 

# Helper functions
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
    labels = []
    for instance_list in raw_bag_data:
        main_tuple = instance_list[0]
        if main_tuple[0] == 'assessment_type': labels.append(f"Assessment: {main_tuple[1]}")
        elif main_tuple[0] == 'activity_type': labels.append(f"VLE Clicks: {main_tuple[1]}")
        else: labels.append(main_tuple[0].replace('_', ' ').title())
    return labels

def build_bag_to_labels_map(raw_data_path):
    try:
        with open(raw_data_path, 'rb') as f: data = pickle.load(f)
    except FileNotFoundError: print(f"FATAL: Raw data file not found at {raw_data_path}"); sys.exit(1)
    return {str(bag_id): generate_general_labels(raw_bag) for bag_id, raw_bag in zip(data['bag_ids'], data['raw_bags'])}

def parse_feature_string(s):
    if not isinstance(s, str): return np.array([])
    return np.fromstring(s.replace('[','').replace(']','').replace('\n',' '), sep=' ')

def print_counts_table(model_name, counts_dict, bag_presence_dict, total_bags):
    """Formats and prints the final counts, including the bag presence metric."""
    print(f"\n--- Top-{TOP_K_FOR_COUNTS} Feature Counts for: {model_name.upper()} ---")
    if not counts_dict:
        print("No instances were found in the top 20."); return
        
    df = pd.DataFrame(counts_dict.items(), columns=['Feature Type', 'Total Count in Top-20'])
    df['Present in X Bags'] = df['Feature Type'].map(bag_presence_dict)
    df['Bag Presence %'] = df['Present in X Bags'] / total_bags
    df['Avg Count When Present'] = df['Total Count in Top-20'] / df['Present in X Bags']
    
    df = df.sort_values(by='Total Count in Top-20', ascending=False).reset_index(drop=True)
    
    print(f"Based on {total_bags} bags.")
    print(df.to_string())

def calculate_hhi(counts_dict, total_bags):
    total_top_k_instances = total_bags * TOP_K_FOR_COUNTS
    if total_top_k_instances == 0: return 0
    return np.sum((np.array(list(counts_dict.values())) / total_top_k_instances)**2)

def calculate_gini(counts_dict):
    counts = np.array(list(counts_dict.values()), dtype=np.float64)
    if np.sum(counts) == 0: return 0
    sorted_counts, n = np.sort(counts), len(counts)
    cum_counts = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cum_counts) / cum_counts[-1]) / n

def get_counts_for_split(bags_list, df_source, score_column, bag_to_labels_map):
    counts = defaultdict(int)
    for bag_id in bags_list:
        if bag_id not in bag_to_labels_map: continue
        bag_labels, bag_df = bag_to_labels_map[bag_id], df_source[df_source['bag_id'] == bag_id]
        if len(bag_labels) != len(bag_df): continue
        for idx in bag_df.nlargest(TOP_K_FOR_COUNTS, score_column).index:
            counts[bag_labels[bag_df.index.get_loc(idx)]] += 1
    return counts

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load instance data and sample bags
    try:
        source_data_df = pd.read_csv(os.path.join(ILSE_RUN_DIR, 'attention_ilse_outputs.csv'))
    except FileNotFoundError as e:
        print(f"FATAL: Could not find the ILSE source data file.")
        sys.exit(1)
        
    source_data_df = source_data_df[source_data_df['is_padding_instance'] == False]
    all_bags = source_data_df['bag_id'].unique()
    
    if SAMPLE_SIZE is not None:
        np.random.seed(REPRODUCIBILITY_SEED)
        bags_to_process = np.random.choice(all_bags, min(SAMPLE_SIZE, len(all_bags)), replace=False)
        print(f"Randomly sampled {len(bags_to_process)} bags for SHAP analysis.")
    else:
        bags_to_process = all_bags
        print(f"Proceeding with all {len(bags_to_process)} bags for SHAP analysis.")
    
    df_for_shap = source_data_df[source_data_df['bag_id'].isin(bags_to_process)].copy()

    # Step 2: Calculate or load SHAP values
    print(f"\n--- STEP 2: CALCULATING/LOADING SHAP VALUES FOR {len(bags_to_process)} BAGS ---")
    shap_output_file = os.path.join(OUTPUT_DIR, f'shap_scores_full_sampled_seed_{SEED_TO_ANALYZE}.csv')

    if os.path.exists(shap_output_file):
        print(f"Found pre-computed SHAP file. Loading from: {shap_output_file}")
        shap_df_loaded = pd.read_csv(shap_output_file, index_col=0) # Use the original index
        # Join the loaded shap values back to the main dataframe
        df_for_shap = df_for_shap.join(shap_df_loaded)
        print("SHAP values loaded successfully.")
    else:
        print(f"No pre-computed SHAP file found at {shap_output_file}.")
        print("Starting new calculation (this may take a while)...")
        
        
        greedy_model = load_rl_model(GREEDY_RUN_DIR)
        if not greedy_model: sys.exit(1)
        
        instance_features = np.vstack(df_for_shap['original_instance_content'].apply(parse_feature_string))
        def shap_wrapper_f(x):
            with torch.no_grad(): return greedy_model(torch.from_numpy(x).float().to("cpu"))[2].cpu().numpy()
        
        explainer = shap.KernelExplainer(shap_wrapper_f, shap.sample(instance_features, 50))
        shap_values = explainer.shap_values(instance_features)
        df_for_shap['shap_value'] = np.abs(shap_values).mean(axis=1)
        
        print("SHAP values calculated.")
        print(f"Saving SHAP values to: {shap_output_file}")
        df_for_shap[['shap_value']].to_csv(shap_output_file, index_label='index')

    # Step 3: Descriptive statistics
    bag_to_labels_map = build_bag_to_labels_map(RAW_DATA_PKL_PATH)
    shap_counts = defaultdict(int)
    shap_bag_presence = defaultdict(int)

    for bag_id in bags_to_process:
        if bag_id not in bag_to_labels_map: continue
        bag_labels = bag_to_labels_map[bag_id]
        bag_df = df_for_shap[df_for_shap['bag_id'] == bag_id]
        if len(bag_labels) != len(bag_df): continue
        
        unique_labels_in_top20 = set()
        for idx in bag_df.nlargest(TOP_K_FOR_COUNTS, 'shap_value').index:
            label = bag_labels[bag_df.index.get_loc(idx)]
            shap_counts[label] += 1
            unique_labels_in_top20.add(label)
        for label in unique_labels_in_top20: shap_bag_presence[label] += 1
    
    print_counts_table("Epsilon-Greedy (SHAP)", shap_counts, shap_bag_presence, len(bags_to_process))
    
    # Step 4: Calculate and display interpretability metrics
    # 4.1: Explanation Sparsity (Gini & HHI)
    sparsity_scores = {"Model": ["Epsilon-Greedy (SHAP)"], "Gini Coefficient": [calculate_gini(shap_counts)], "HHI Score": [calculate_hhi(shap_counts, len(bags_to_process))]}
    print(pd.DataFrame(sparsity_scores).to_string(index=False))

    # 4.2: Explanation Consistency (Split-Half Method)
    shuffled_bags = np.array(bags_to_process); np.random.shuffle(shuffled_bags)
    split_point = len(shuffled_bags) // 2
    bags_a, bags_b = shuffled_bags[:split_point], shuffled_bags[split_point:]
    print(f"Splitting data into two random halves of {len(bags_a)} and {len(bags_b)} bags.")
    
    counts_a = get_counts_for_split(bags_a, df_for_shap, 'shap_value', bag_to_labels_map)
    counts_b = get_counts_for_split(bags_b, df_for_shap, 'shap_value', bag_to_labels_map)
    
    policy_df = pd.DataFrame({'split_A': counts_a, 'split_B': counts_b}).fillna(0)
    spearman_corr, _ = spearmanr(policy_df['split_A'], policy_df['split_B'])
    
    # Show top 5 most counted instance types
    top5_a = set(policy_df.nlargest(5, 'split_A').index)
    top5_b = set(policy_df.nlargest(5, 'split_B').index)
    print(f"    - Top 5 Features (Split A): {sorted(list(top5_a))}")
    print(f"    - Top 5 Features (Split B): {sorted(list(top5_b))}")
    # Also calculate for other k's
    k_values, jaccard_scores = [3, 5, 10, 15, 20], {}
    for k in k_values:
        top_k_a, top_k_b = set(policy_df.nlargest(k, 'split_A').index), set(policy_df.nlargest(k, 'split_B').index)
        jaccard_scores[f"Jaccard (K={k})"] = len(top_k_a.intersection(top_k_b)) / len(top_k_a.union(top_k_b)) if top_k_a and top_k_b else 0.0
    
    result_row = {"Model": "Epsilon-Greedy (SHAP)", "Spearman's Rank (Stability)": spearman_corr}
    result_row.update(jaccard_scores)
    print(pd.DataFrame([result_row]).to_string(index=False))