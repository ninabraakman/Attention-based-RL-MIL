import pandas as pd
import numpy as np
import os
import sys
import pickle
import torch
from collections import defaultdict
import time
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

SEED_TO_ANALYZE = 8
OUTPUT_DIR = 'final_report_oulad_full/'
TOP_K_FOR_COUNTS = 20
BASE_PATH = f'/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_{SEED_TO_ANALYZE}/oulad_full/instances/tabular/label/bag_size_20/repset_20_16_20'
RAW_DATA_PKL_PATH = '/projects/prjs1491/Attention-based-RL-MIL/data/oulad/oulad_full_raw.pkl' 

# --- Model Specific Sub-directories ---
ILSE_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement/')
PHAM_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_pham_reg_sum_sample_without_replacement/') 
# Set a seed for the random split to ensure reproducibility of consistency scores
REPRODUCIBILITY_SEED = 42


def generate_general_labels(raw_bag_data):
    """Generates GENERALIZED labels for feature aggregation."""
    labels = []
    for instance_list in raw_bag_data:
        main_tuple = instance_list[0]
        if main_tuple[0] == 'assessment_type': labels.append(f"Assessment: {main_tuple[1]}")
        elif main_tuple[0] == 'activity_type': labels.append(f"VLE Clicks: {main_tuple[1]}")
        else: labels.append(main_tuple[0].replace('_', ' ').title())
    return labels

def build_bag_to_labels_map(raw_data_path):
    """Builds a global map from bag_id to its list of general instance labels."""
    print("\n--- Building global map from Bag ID to Instance Labels ---")
    try:
        with open(raw_data_path, 'rb') as f: data = pickle.load(f)
    except FileNotFoundError: print(f"FATAL: Raw data file not found at {raw_data_path}"); sys.exit(1)
    return {str(bag_id): generate_general_labels(raw_bag) for bag_id, raw_bag in zip(data['bag_ids'], data['raw_bags'])}

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
    
    print(f"Based on {total_bags} common bags.")
    print(df.to_string())

def calculate_hhi(counts_dict, total_bags):
    """Calculates the Herfindahl-Hirschman Index for explanation concentration."""
    total_top_k_instances = total_bags * TOP_K_FOR_COUNTS
    if total_top_k_instances == 0: return 0
    frequencies = np.array(list(counts_dict.values())) / total_top_k_instances
    return np.sum(frequencies**2)

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
    try:
        ilse_df = pd.read_csv(os.path.join(ILSE_RUN_DIR, 'attention_ilse_outputs.csv'))
        pham_df = pd.read_csv(os.path.join(PHAM_RUN_DIR, 'attention_pham_outputs.csv'))
    except FileNotFoundError as e:
        print(f"FATAL: Could not find input file: {e}\nPlease update the paths in the configuration section.")
        sys.exit(1)

    ilse_df, pham_df = ilse_df[ilse_df['is_padding_instance'] == False], pham_df[pham_df['is_padding_instance'] == False]
    common_bags = sorted(list(set(ilse_df['bag_id'].unique()).intersection(set(pham_df['bag_id'].unique()))))
    print(f"Found {len(common_bags)} common bags to analyze.")
    ilse_df_common = ilse_df[ilse_df['bag_id'].isin(common_bags)].copy()
    pham_df_common = pham_df[pham_df['bag_id'].isin(common_bags)].copy()
    if not common_bags:
        print("No common bags found between ILSE and PHAM files. Aborting."); sys.exit(1)

    # --- STEP 2: GENERATE TOP-20 COUNTS FOR FULL DATASET ---
    print("\n--- STEP 2: COUNTING TOP-20 INSTANCE TYPES (FULL DATASET) ---")
    bag_to_labels_map = build_bag_to_labels_map(RAW_DATA_PKL_PATH)
    ilse_counts, pham_counts = defaultdict(int), defaultdict(int)
    ilse_bag_presence, pham_bag_presence = defaultdict(int), defaultdict(int)
    
    for bag_id in common_bags:
        if bag_id not in bag_to_labels_map: continue
        bag_labels = bag_to_labels_map[bag_id]
        
        # Process ILSE
        bag_ilse_df = ilse_df_common[ilse_df_common['bag_id'] == bag_id]
        if len(bag_labels) == len(bag_ilse_df):
            unique_labels_in_top20 = set()
            for idx in bag_ilse_df.nlargest(TOP_K_FOR_COUNTS, 'attention_score').index:
                label = bag_labels[bag_ilse_df.index.get_loc(idx)]
                ilse_counts[label] += 1
                unique_labels_in_top20.add(label)
            for label in unique_labels_in_top20: ilse_bag_presence[label] += 1

        # Process PHAM
        bag_pham_df = pham_df_common[pham_df_common['bag_id'] == bag_id]
        if len(bag_labels) == len(bag_pham_df):
            unique_labels_in_top20 = set()
            for idx in bag_pham_df.nlargest(TOP_K_FOR_COUNTS, 'attention_score').index:
                label = bag_labels[bag_pham_df.index.get_loc(idx)]
                pham_counts[label] += 1
                unique_labels_in_top20.add(label)
            for label in unique_labels_in_top20: pham_bag_presence[label] += 1

    # --- STEP 3: PRINT DESCRIPTIVE STATISTICS ---
    print("\n\n--- STEP 3: DESCRIPTIVE STATISTICS FOR OULAD_FULL ---")
    print_counts_table("ILSE", ilse_counts, ilse_bag_presence, len(common_bags))
    print_counts_table("PHAM", pham_counts, pham_bag_presence, len(common_bags))

    # --- STEP 4: CALCULATE AND DISPLAY INTERPRETABILITY METRICS ---
    print("\n\n--- STEP 4: FINAL INTERPRETABILITY METRICS FOR OULAD_FULL ---")
    
    # 4.1: Explanation Sparsity (Gini & HHI)
    sparsity_scores = {
        "Model": ["ILSE", "PHAM"],
        "Gini Coefficient": [calculate_gini(ilse_counts), calculate_gini(pham_counts)],
        "HHI Score": [calculate_hhi(ilse_counts, len(common_bags)), calculate_hhi(pham_counts, len(common_bags))],
    }
    sparsity_df = pd.DataFrame(sparsity_scores)
    print("\n--- METRIC 1: EXPLANATION SPARSITY (Higher is more focused) ---")
    print(sparsity_df.to_string(index=False))

    # 4.2: Explanation Consistency (Split-Half Method)
    print("\n\n--- METRIC 2: EXPLANATION CONSISTENCY (Higher is more stable) ---")
    np.random.seed(REPRODUCIBILITY_SEED)
    shuffled_bags = np.array(common_bags); np.random.shuffle(shuffled_bags)
    split_point = len(shuffled_bags) // 2
    bags_a, bags_b = shuffled_bags[:split_point], shuffled_bags[split_point:]
    print(f"Splitting data into two random halves of {len(bags_a)} and {len(bags_b)} bags.")
    
    consistency_results = []
    models_data = {"ILSE": (ilse_df_common, 'attention_score'), "PHAM": (pham_df_common, 'attention_score')}
    for model_name, (df_source, score_col) in models_data.items():
        print(f"  -> Calculating consistency for {model_name}...")
        counts_a = get_counts_for_split(bags_a, df_source, score_col, bag_to_labels_map)
        counts_b = get_counts_for_split(bags_b, df_source, score_col, bag_to_labels_map)
        policy_df = pd.DataFrame({'split_A': counts_a, 'split_B': counts_b}).fillna(0)
        spearman_corr, _ = spearmanr(policy_df['split_A'], policy_df['split_B'])
        
        k_values = [3, 5, 10, 15, 20]
        jaccard_scores = {}
        for k in k_values:
            top_k_a, top_k_b = set(policy_df.nlargest(k, 'split_A').index), set(policy_df.nlargest(k, 'split_B').index)
            jaccard_scores[f"Jaccard (K={k})"] = len(top_k_a.intersection(top_k_b)) / len(top_k_a.union(top_k_b)) if top_k_a and top_k_b else 0.0
        
        result_row = {"Model": model_name, "Spearman's Rank (Stability)": spearman_corr}
        result_row.update(jaccard_scores)
        consistency_results.append(result_row)

    consistency_df = pd.DataFrame(consistency_results)
    print(consistency_df.to_string(index=False))

    print("\n✅ All analyses complete!")
