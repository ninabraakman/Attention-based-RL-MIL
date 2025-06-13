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

# --- Fix Python's ability to find your 'models.py' file ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from models import PolicyNetwork, create_mil_model_with_dict

# --- Configuration ---
RAW_DATA_PKL_PATH = '/projects/prjs1491/Attention-based-RL-MIL/data/oulad/oulad_aggregated_raw.pkl'
ILSE_RUN_DIR = '/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_0/oulad_aggregated_subset/instances/tabular/label/bag_size_20/MeanMLP_22_16_22/neg_policy_only_loss_attention_ilse_reg_sum_sample_without_replacement/'
GREEDY_RUN_DIR = '/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_0/oulad_aggregated_subset/instances/tabular/label/bag_size_20/MeanMLP_22_16_22/neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement/'
CASE_STUDY_BAG_ID = "('FFF', '2013J', 582235)"

# --- Helper Functions ---

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
    """
    Dynamically creates clean, descriptive labels from the raw bag data structure.
    """
    labels = []
    assessment_count = 1
    for instance_list in raw_bag_data:
        main_tuple = instance_list[0]
        feature_type = main_tuple[0]
        
        # --- NEW LABELING LOGIC ---
        if feature_type == 'assessment_type':
            labels.append(f"Assessment: {main_tuple[1]} {assessment_count}")
            assessment_count += 1
        elif feature_type == 'activity_type':
            # Changed prefix to "VLE Clicks"
            labels.append(f"VLE Clicks: {main_tuple[1]}")
        else:
            # For all others (demographics, registration, course), just use the clean title
            labels.append(feature_type.replace('_', ' ').title())
            
    return labels

# --- Main Analysis Script ---

# 1. Load Data
print("Loading all necessary data files...")
with open(RAW_DATA_PKL_PATH, 'rb') as f:
    data_from_pickle = pickle.load(f)

ilse_attention_file = os.path.join(ILSE_RUN_DIR, 'attention_ilse_outputs.csv')
attention_df = pd.read_csv(ilse_attention_file)
case_study_data = attention_df[attention_df['bag_id'] == CASE_STUDY_BAG_ID].copy()
case_study_data = case_study_data[case_study_data['is_padding_instance'] == False]
ilse_attention_weights = case_study_data['attention_score'].values
instance_features = np.vstack(case_study_data['original_instance_content'].apply(parse_feature_string))
print(f"Found {len(instance_features)} instances for Bag ID {CASE_STUDY_BAG_ID}.")

# 2. Load Models
print("Loading Epsilon-Greedy model...")
greedy_model = load_rl_model(GREEDY_RUN_DIR)

# 3. Generate SHAP Explanations
def shap_prediction_wrapper(numpy_data):
    with torch.no_grad():
        tensor_data = torch.from_numpy(numpy_data).float().to(torch.device("cpu"))
        _, _, exp_reward = greedy_model(tensor_data)
        return exp_reward.cpu().numpy()

background_data = shap.sample(instance_features, 50)
explainer = shap.KernelExplainer(shap_prediction_wrapper, background_data)
print("Calculating SHAP values for the Epsilon-Greedy model...")
shap_values = explainer.shap_values(instance_features)
shap_importance = np.abs(shap_values).mean(axis=1)

# 4. Generate Labels and Create Plot
was_selected = case_study_data['was_selected_among_top_k'].values
selected_color = 'limegreen'
not_selected_color = 'darkcyan'
ilse_colors = [selected_color if selected else not_selected_color for selected in was_selected]

print("Creating comparison plot...")
temp_key = ast.literal_eval(CASE_STUDY_BAG_ID)
search_key = (temp_key[0], temp_key[1], int(temp_key[2]))
all_bag_ids = data_from_pickle['bag_ids']
try:
    target_index = all_bag_ids.index(search_key)
    raw_case_study_bag = data_from_pickle['raw_bags'][target_index]
    print(f"Successfully found raw data for {search_key} at index {target_index}.")
    instance_labels = generate_labels_from_raw_bag(raw_case_study_bag)
except ValueError:
    raise ValueError(f"Could not find the Bag ID {search_key} in the 'bag_ids' list from the pickle file.")

# Create the plot using the descriptive labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.barh(instance_labels, ilse_attention_weights, color=ilse_colors)
ax1.set_title('ILSE Model: Intrinsic Attention Weights', fontsize=16)
ax1.set_xlabel('Attention Score', fontsize=12)
ax1.invert_yaxis()
ax1.grid(axis='x', linestyle='--', alpha=0.7)
legend_patches = [mpatches.Patch(color=selected_color, label='Selected in Top-K'),
                  mpatches.Patch(color=not_selected_color, label='Not Selected')]
ax1.legend(handles=legend_patches, loc='lower right')

ax2.barh(instance_labels, shap_importance, color='coral')
ax2.set_title('Epsilon-Greedy Model: Post-hoc SHAP Values', fontsize=16)
ax2.set_xlabel('Mean Absolute SHAP Importance', fontsize=12)
ax2.set_yticklabels([])
ax2.grid(axis='x', linestyle='--', alpha=0.7)

plt.suptitle(f'Interpretability Comparison for Student {CASE_STUDY_BAG_ID} (Aggregated Data)', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('comparison_aggregated_seleceted_wor.png')
print("\nPlot saved as 'comparison_aggregated_selected_wor.png'")



# print("Creating comparison plot...")

# # Generate the descriptive labels (this part is the same as before)
# temp_key = ast.literal_eval(CASE_STUDY_BAG_ID)
# search_key = (temp_key[0], temp_key[1], int(temp_key[2]))
# all_bag_ids = data_from_pickle['bag_ids']
# try:
#     target_index = all_bag_ids.index(search_key)
#     raw_case_study_bag = data_from_pickle['raw_bags'][target_index]
#     instance_labels = generate_labels_from_raw_bag(raw_case_study_bag)
# except ValueError:
#     raise ValueError(f"Could not find the Bag ID {search_key} in the 'bag_ids' list from the pickle file.")


# # Create the plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# # --- MODIFIED: The 'color' argument is now a single, static color ---
# # --- The legend and color list have been removed ---
# ax1.barh(instance_labels, ilse_attention_weights, color='darkcyan')
# ax1.set_title('ILSE Model: Intrinsic Attention Weights', fontsize=16)
# ax1.set_xlabel('Attention Score', fontsize=12)
# ax1.invert_yaxis()
# ax1.grid(axis='x', linestyle='--', alpha=0.7)

# # The SHAP plot on the right remains the same
# ax2.barh(instance_labels, shap_importance, color='coral')
# ax2.set_title('Epsilon-Greedy Model: Post-hoc SHAP Values', fontsize=16)
# ax2.set_xlabel('Mean Absolute SHAP Importance', fontsize=12)
# ax2.set_yticklabels([])
# ax2.grid(axis='x', linestyle='--', alpha=0.7)

# plt.suptitle(f'Interpretability Comparison for Student {CASE_STUDY_BAG_ID} (Aggregated Data)', fontsize=20)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('comparison_aggregated.png')
# print("\nPlot saved as 'comparison_aggregated.png'")
# plt.show()
