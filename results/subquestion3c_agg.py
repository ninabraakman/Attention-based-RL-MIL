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

# --- Add project root to path to find your 'models.py' file ---
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

# --- ======================================================= ---
# --- CONFIGURATION ---
# --- ======================================================= ---
SEED_TO_ANALYZE = 8
OUTPUT_DIR = 'final_report_oulad_aggregated/'
TOP_K_FOR_COUNTS = 20
BASE_PATH = f'/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_{SEED_TO_ANALYZE}/oulad_aggregated/instances/tabular/label/bag_size_20/repset_22_16_22/'
RAW_DATA_PKL_PATH = '/projects/prjs1491/Attention-based-RL-MIL/data/oulad/oulad_aggregated_raw.pkl'
ILSE_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement/')
PHAM_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_pham_reg_sum_sample_without_replacement/')
GREEDY_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement/')
# Set a seed for the random split to ensure reproducibility of consistency scores
REPRODUCIBILITY_SEED = 42

# --- ======================================================= ---
# --- HELPER FUNCTIONS ---
# --- ======================================================= ---

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


def calculate_hhi(counts_dict, total_bags):
    """Calculates the Herfindahl-Hirschman Index for explanation concentration."""
    total_top_k_instances = total_bags * TOP_K_FOR_COUNTS
    if total_top_k_instances == 0: return 0
    frequencies = np.array(list(counts_dict.values())) / total_top_k_instances
    hhi = np.sum(frequencies**2)
    return hhi

def calculate_gini(counts_dict):
    """Calculates the Gini coefficient for explanation concentration."""
    counts = np.array(list(counts_dict.values()), dtype=np.float64)
    if np.sum(counts) == 0: return 0
    sorted_counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cum_counts) / cum_counts[-1]) / n


def get_counts_for_split(bags_list, df_source, score_column, bag_to_labels_map):
    """Helper function to generate feature counts for a specific list of bags."""
    counts = defaultdict(int)
    for bag_id in bags_list:
        if bag_id not in bag_to_labels_map: continue
        bag_labels = bag_to_labels_map[bag_id]
        bag_df = df_source[df_source['bag_id'] == bag_id]
        if len(bag_labels) != len(bag_df): continue
        
        top_indices = bag_df.nlargest(TOP_K_FOR_COUNTS, score_column).index
        for idx in top_indices:
            label_index = bag_df.index.get_loc(idx)
            if label_index < len(bag_labels):
                counts[bag_labels[label_index]] += 1
    return counts


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- STEP 1: LOAD DATA AND FIND COMMON BAGS ---
    print("--- STEP 1: LOADING DATA AND FINDING COMMON BAGS ---")
    ilse_df = pd.read_csv(os.path.join(ILSE_RUN_DIR, 'attention_ilse_outputs.csv'))
    pham_df = pd.read_csv(os.path.join(PHAM_RUN_DIR, 'attention_pham_outputs.csv'))
    ilse_df, pham_df = ilse_df[ilse_df['is_padding_instance'] == False], pham_df[pham_df['is_padding_instance'] == False]
    common_bags = sorted(list(set(ilse_df['bag_id'].unique()).intersection(set(pham_df['bag_id'].unique()))))
    print(f"Found {len(common_bags)} common bags to analyze.")
    ilse_df_common = ilse_df[ilse_df['bag_id'].isin(common_bags)].copy()
    pham_df_common = pham_df[pham_df['bag_id'].isin(common_bags)].copy()

    # --- STEP 2: CALCULATE OR LOAD SHAP VALUES ---
    print("\n--- STEP 2: CALCULATING/LOADING SHAP VALUES ---")
    shap_output_file = os.path.join(OUTPUT_DIR, f'shap_scores_seed_{SEED_TO_ANALYZE}.csv')
    if os.path.exists(shap_output_file):
        print("Loading pre-computed SHAP file...")
        shap_df_loaded = pd.read_csv(shap_output_file, index_col=0)
        ilse_df_common = ilse_df_common.join(shap_df_loaded)
    else:
        print("No SHAP file found. Calculating SHAP values (this will be slow)...")
        greedy_model = load_rl_model(GREEDY_RUN_DIR)
        if not greedy_model: sys.exit(1)
        instance_features = np.vstack(ilse_df_common['original_instance_content'].apply(parse_feature_string))
        def shap_wrapper_f(x):
            with torch.no_grad(): return greedy_model(torch.from_numpy(x).float().to("cpu"))[2].cpu().numpy()
        explainer = shap.KernelExplainer(shap_wrapper_f, shap.sample(instance_features, 50))
        shap_values = explainer.shap_values(instance_features)
        ilse_df_common['shap_value'] = np.abs(shap_values).mean(axis=1)
        ilse_df_common[['shap_value']].to_csv(shap_output_file)
        print("SHAP values calculated and saved.")
    
    # --- STEP 3: GENERATE TOP-20 COUNTS FOR FULL DATASET ---
    print("\n--- STEP 3: COUNTING TOP-20 INSTANCE TYPES (FULL DATASET) ---")
    bag_to_labels_map = build_bag_to_labels_map(RAW_DATA_PKL_PATH)
    ilse_counts, pham_counts, shap_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    for bag_id in common_bags:
        if bag_id not in bag_to_labels_map: continue
        bag_labels = bag_to_labels_map[bag_id]
        bag_ilse_df = ilse_df_common[ilse_df_common['bag_id'] == bag_id]
        bag_pham_df = pham_df_common[pham_df_common['bag_id'] == bag_id]
        if len(bag_labels) != len(bag_ilse_df): continue
        for idx in bag_ilse_df.nlargest(TOP_K_FOR_COUNTS, 'attention_score').index: ilse_counts[bag_labels[bag_ilse_df.index.get_loc(idx)]] += 1
        for idx in bag_pham_df.nlargest(TOP_K_FOR_COUNTS, 'attention_score').index: pham_counts[bag_labels[bag_pham_df.index.get_loc(idx)]] += 1
        for idx in bag_ilse_df.nlargest(TOP_K_FOR_COUNTS, 'shap_value').index: shap_counts[bag_labels[bag_ilse_df.index.get_loc(idx)]] += 1
    
    # --- STEP 4: CALCULATE AND DISPLAY INTERPRETABILITY METRICS ---
    print("\n\n--- STEP 4: FINAL INTERPRETABILITY METRICS ---")
    
    # 4.1: Explanation Sparsity (Gini & HHI)
    sparsity_scores = {
        "Model": ["ILSE", "PHAM", "Epsilon-Greedy (SHAP)"],
        "Gini Coefficient": [calculate_gini(c) for c in [ilse_counts, pham_counts, shap_counts]],
        "HHI Score": [calculate_hhi(c, len(common_bags)) for c in [ilse_counts, pham_counts, shap_counts]],
    }
    sparsity_df = pd.DataFrame(sparsity_scores)
    print("\n--- METRIC 1: EXPLANATION SPARSITY (Higher is more focused) ---")
    print(sparsity_df.to_string(index=False))

    # 4.2: Explanation Consistency (Split-Half Method)
    print("\n\n--- METRIC 2: EXPLANATION CONSISTENCY (Higher is more stable) ---")
    np.random.seed(REPRODUCIBILITY_SEED)
    shuffled_bags = np.array(common_bags)
    np.random.shuffle(shuffled_bags)
    split_point = len(shuffled_bags) // 2
    bags_a, bags_b = shuffled_bags[:split_point], shuffled_bags[split_point:]
    
    print(f"Splitting data into two random halves of {len(bags_a)} and {len(bags_b)} bags.")
    
    consistency_results = []
    models_data = {
        "ILSE": (ilse_df_common, 'attention_score'),
        "PHAM": (pham_df_common, 'attention_score'),
        "Epsilon-Greedy (SHAP)": (ilse_df_common, 'shap_value'),
    }

    for model_name, (df_source, score_col) in models_data.items():
        print(f"  -> Calculating consistency for {model_name}...")
        counts_a = get_counts_for_split(bags_a, df_source, score_col, bag_to_labels_map)
        counts_b = get_counts_for_split(bags_b, df_source, score_col, bag_to_labels_map)
        
        policy_df = pd.DataFrame({'split_A': counts_a, 'split_B': counts_b}).fillna(0)
        
        spearman_corr, _ = spearmanr(policy_df['split_A'], policy_df['split_B'])
        
        # --- NEW: Calculate Jaccard for multiple K values ---
        k_values = [3, 5, 10, 15, 20]
        jaccard_scores = {}
        for k in k_values:
            top_k_a = set(policy_df.nlargest(k, 'split_A').index)
            top_k_b = set(policy_df.nlargest(k, 'split_B').index)
            # Handle case where one split might have fewer than K features with non-zero counts
            if not top_k_a or not top_k_b:
                jaccard_sim = 0.0
            else:
                jaccard_sim = len(top_k_a.intersection(top_k_b)) / len(top_k_a.union(top_k_b))
            jaccard_scores[f"Jaccard (K={k})"] = jaccard_sim
        
        result_row = {"Model": model_name, "Spearman's Rank (Stability)": spearman_corr}
        result_row.update(jaccard_scores)
        consistency_results.append(result_row)

    consistency_df = pd.DataFrame(consistency_results)
    print(consistency_df.to_string(index=False))

    print("\n✅ All analyses complete!")
