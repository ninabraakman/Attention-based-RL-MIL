# This file does a case study analysis on one random chosen bag.
# It gathers the attention weights form the attention mechanisms and calculates the SHAP importance scores for the Epsilon-Greedy model.
# Then it creates an image for each model with the importance scores for each instance.
import pandas as pd
import torch
import torch.nn as nn 
import shap
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import json
from types import SimpleNamespace
import sys
import pickle
import ast
import matplotlib.patches as mpatches

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


OUTPUT_DIR = 'final_report_oulad_aggregated/'
RAW_DATA_PKL_PATH = '/projects/prjs1491/Attention-based-RL-MIL/data/oulad/oulad_aggregated_raw.pkl'

# Using 'repset' on seed 8 for all models as an example
ILSE_RUN_DIR = '/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_8/oulad_aggregated/instances/tabular/label/bag_size_20/repset_22_16_22/neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement/'
GREEDY_RUN_DIR = '/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_8/oulad_aggregated/instances/tabular/label/bag_size_20/repset_22_16_22/neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement/'
PHAM_RUN_DIR = '/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_8/oulad_aggregated/instances/tabular/label/bag_size_20/repset_22_16_22/neg_policy_only_loss_attention_pham_reg_sum_sample_without_replacement/'

CASE_STUDY_BAG_ID = "('CCC', '2014B', 623347)"

# Helper Functions
def load_rl_model(run_dir_path):
    device = torch.device("cpu")
    model_weights_path = os.path.join(run_dir_path, 'sweep_best_model.pt')
    rl_config_path = os.path.join(run_dir_path, 'sweep_best_model_config.json')
    mil_config_path = os.path.join(run_dir_path, '..', 'best_model_config.json')
    mil_weights_path = os.path.join(run_dir_path, '..', 'best_model.pt')
    with open(mil_config_path) as f: mil_config = json.load(f)
    with open(rl_config_path) as f: rl_config = json.load(f)
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
    return policy_network

def parse_feature_string(feature_str):
    if not isinstance(feature_str, str): return np.array([])
    cleaned_str = re.sub(r'[\n\[\]]', '', feature_str).strip()
    numbers = re.split(r'\s+', cleaned_str)
    return np.array([float(n) for n in numbers if n])

def generate_labels_from_raw_bag(raw_bag_data):
    """Dynamically creates clean, descriptive labels from the raw bag data structure."""
    labels = []
    assessment_count = 1
    for instance_list in raw_bag_data:
        main_tuple = instance_list[0]
        feature_type = main_tuple[0]
        if feature_type == 'assessment_type':
            labels.append(f"Assessment: {main_tuple[1]} {assessment_count}")
            assessment_count += 1
        elif feature_type == 'activity_type':
            labels.append(f"VLE Clicks: {main_tuple[1]}")
        else:
            labels.append(feature_type.replace('_', ' ').title())
    return labels

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load All Data
    print("Loading all necessary data files...")
    with open(RAW_DATA_PKL_PATH, 'rb') as f: data_from_pickle = pickle.load(f)

    ilse_df_full = pd.read_csv(os.path.join(ILSE_RUN_DIR, 'attention_ilse_outputs.csv'))
    case_study_ilse = ilse_df_full[ilse_df_full['bag_id'] == CASE_STUDY_BAG_ID].copy()
    case_study_ilse = case_study_ilse[case_study_ilse['is_padding_instance'] == False]
    ilse_attention_weights = case_study_ilse['attention_score'].values
    
    pham_df_full = pd.read_csv(os.path.join(PHAM_RUN_DIR, 'attention_pham_outputs.csv'))
    case_study_pham = pham_df_full[pham_df_full['bag_id'] == CASE_STUDY_BAG_ID].copy()
    case_study_pham = case_study_pham[case_study_pham['is_padding_instance'] == False]
    pham_attention_weights = case_study_pham['attention_score'].values

    instance_features = np.vstack(case_study_ilse['original_instance_content'].apply(parse_feature_string))
    print(f"Found {len(instance_features)} instances for Bag ID {CASE_STUDY_BAG_ID}.")

    # 2. Load Epsilon-Greedy model and calculate SHAP
    print("Loading Epsilon-Greedy model and calculating SHAP values...")
    greedy_model = load_rl_model(GREEDY_RUN_DIR)
    
    def shap_prediction_wrapper(numpy_data):
        with torch.no_grad():
            tensor_data = torch.from_numpy(numpy_data).float().to(torch.device("cpu"))
            _, _, exp_reward = greedy_model(tensor_data)
            return exp_reward.cpu().numpy()

    explainer = shap.KernelExplainer(shap_prediction_wrapper, shap.sample(instance_features, 50))
    shap_values = explainer.shap_values(instance_features)
    shap_importance = np.abs(shap_values).mean(axis=1)

    # 3. Generate Labels
    print("Generating labels for plotting...")
    temp_key = ast.literal_eval(CASE_STUDY_BAG_ID)
    search_key = (temp_key[0], temp_key[1], int(temp_key[2]))
    target_index = data_from_pickle['bag_ids'].index(search_key)
    raw_case_study_bag = data_from_pickle['raw_bags'][target_index]
    instance_labels = generate_labels_from_raw_bag(raw_case_study_bag)

    # 4. Create and save separate plots
    print("Creating and saving plots...")

    # Plot 1: Ilse/Gated Attention
    plt.figure(figsize=(8, 10))
    plt.barh(instance_labels, ilse_attention_weights, color='darkcyan')
    plt.xlabel('Attention Score', fontsize=12)
    plt.ylabel('Instance', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path_ilse = os.path.join(OUTPUT_DIR, f'case_study_agg_ilse.png')
    plt.savefig(output_path_ilse)
    print(f"ILSE plot saved to '{output_path_ilse}'")
    plt.close()

    # Plot 2: Pham/ Multi-Head Attention
    plt.figure(figsize=(8, 10))
    plt.barh(instance_labels, pham_attention_weights, color='slateblue')
    plt.xlabel('Attention Score', fontsize=12)
    plt.ylabel('Instance', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path_pham = os.path.join(OUTPUT_DIR, f'case_study_agg_pham.png')
    plt.savefig(output_path_pham)
    print(f"PHAM plot saved to '{output_path_pham}'")
    plt.close()

    # Plot 3: Epsilon-Greedy (SHAP)
    plt.figure(figsize=(8, 10))
    plt.barh(instance_labels, shap_importance, color='coral')
    plt.xlabel('SHAP Importance Value', fontsize=12)
    plt.ylabel('Instance', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path_shap = os.path.join(OUTPUT_DIR, f'case_study_agg_shap.png')
    plt.savefig(output_path_shap)
    print(f"SHAP plot saved to '{output_path_shap}'")
    plt.close()
