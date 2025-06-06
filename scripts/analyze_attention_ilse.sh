#!/bin/bash

#-----------------------------------------------------------------------
# SLURM JOB SUBMISSION SCRIPT FOR ATTENTION ANALYSIS (ILSE Policy) - Targeted Debug
#-----------------------------------------------------------------------
# This script is modified to target a VERY SPECIFIC trained model
# by hardcoding the prefix and limiting loops for debugging FileNotFoundError.
#-----------------------------------------------------------------------

# --- SLURM Directives (Customize to your cluster) ---
#SBATCH --job-name=debug_analyze_ilse # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00 # Shorter time for a single debug run
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/analysis/ilse_policy_sweep/debug_analyze_%A.out # Simplified log name
#SBATCH --error=../logs/analysis/ilse_policy_sweep/debug_analyze_%A.err  # Simplified log name
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# --- Environment Setup (Customize as needed) ---
echo "Setting up environment..."
module purge
module load 2023
VENV_PATH="/projects/prjs1491/Attention-based-RL-MIL/venv/bin/activate"
if [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
else
    echo "Error: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi

PROJECT_BASE_DIR="/projects/prjs1491/Attention-based-RL-MIL"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_BASE_DIR}
cd "${PROJECT_BASE_DIR}" || { echo "Error: Could not cd to ${PROJECT_BASE_DIR}"; exit 1; }

mkdir -p ../logs/analysis/ilse_policy_sweep

# --- Analysis Parameters (Hardcoded for specific model debug) ---
ANALYSIS_DATASET_NAME="oulad_full_subset"
ANALYSIS_SEED=0 # Specific seed
ANALYSIS_SPLIT="val"

TARGET_LABEL="label"
DATA_EMBEDDED_COLUMN_NAME="instances"
TASK_TYPE="classification"
MIL_AUTOENCODER_LAYER_SIZES_STR="20,16,20"
BAG_SIZE=20
EMBEDDING_MODEL="tabular"

# === DEBUGGING: Target a specific known model ===
# This is the MIL task model type under which the target RL model is saved
TARGET_MIL_BASELINE_TYPE="AttentionMLP"

# This is the EXACT prefix for the target RL model directory
# (e.g., /runs/.../AttentionMLP_20_16_20/THIS_PREFIX/)
TARGET_RL_MODEL_PREFIX="neg_policy_only_loss_attention_reg_sum_sample_static"

# Analysis script specific parameters
BATCH_SIZE_ANALYSIS=32

# --- Python Script to Execute ---
PYTHON_SCRIPT_NAME="analyze_attention_ilse.py"

echo "DEBUGGING: Targeting specific model configuration for analysis."
echo "Dataset: $ANALYSIS_DATASET_NAME, Seed: $ANALYSIS_SEED, Label: $TARGET_LABEL"
echo "MIL Task Model: $TARGET_MIL_BASELINE_TYPE, Autoencoder: $MIL_AUTOENCODER_LAYER_SIZES_STR"
echo "RL Model Prefix to use: $TARGET_RL_MODEL_PREFIX"

CMD="python ${PYTHON_SCRIPT_NAME} \
    --dataset \"${ANALYSIS_DATASET_NAME}\" \
    --random_seed ${ANALYSIS_SEED} \
    --analysis_split \"${ANALYSIS_SPLIT}\" \
    --label \"${TARGET_LABEL}\" \
    --embedding_model \"${EMBEDDING_MODEL}\" \
    --data_embedded_column_name \"${DATA_EMBEDDED_COLUMN_NAME}\" \
    --bag_size ${BAG_SIZE} \
    --baseline \"${TARGET_MIL_BASELINE_TYPE}\" \
    --autoencoder_layer_sizes ${MIL_AUTOENCODER_LAYER_SIZES_STR//_/ } \
    --prefix \"${TARGET_RL_MODEL_PREFIX}\" \
    --task_type \"${TASK_TYPE}\" \
    --gpu 0 \
    --no_wandb \
    --batch_size_analysis ${BATCH_SIZE_ANALYSIS} \
    "
    # The analyze_attention_ilse.py script will load specific RL hyperparameters
    # (like hdim, temperature, attention_size, rl_model, sample_algorithm, reg_alg) 
    # from the sweep_best_model_config.json found in the directory 
    # identified by the args above (especially the --prefix).

echo "-------------------------------------"
echo "Running Analysis Command:"
echo "${CMD}"
echo "-------------------------------------"
eval "${CMD}"
echo "-------------------------------------"
echo "Analysis command finished."

echo "Slurm debug job finished."

# #!/bin/bash
# #SBATCH --job-name=analyze_ilse_sweep # Job name
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G
# #SBATCH --time=02:00:00
# #SBATCH --partition=gpu_h100
# #SBATCH --gres=gpu:1
# #SBATCH --output=../logs/analysis/ilse_policy_sweep/analyze_%A_%a.out # Adjusted output path
# #SBATCH --error=../logs/analysis/ilse_policy_sweep/analyze_%A_%a.err  # Adjusted error path

# module purge
# module load 2023
# source ../venv/bin/activate

# # Navigate to project root
# cd /projects/prjs1491/Attention-based-RL-MIL


# # --- Analysis Parameters (Define the scope of models to analyze) ---
# ANALYSIS_DATASET_NAME="oulad_full_subset"
# ANALYSIS_SEEDS=(0) 
# ANALYSIS_SPLIT="val"

# # Parameters from your training script's loops that define the model context
# TARGET_LABELS=("label")
# DATA_EMBEDDED_COLUMN_NAME="instances"
# TASK_TYPE="classification"
# MIL_AUTOENCODER_LAYER_SIZES_STR="20,16,20"
# BAG_SIZES=(20)
# EMBEDDING_MODELS=("tabular")
# MIL_BASELINE_TYPES=("MeanMLP" "MaxMLP" "AttentionMLP" "repset") # MIL task_models used

# # --- Components for constructing the RL model's directory prefix ---
# # These should match the settings used when the RL models were trained,
# # as they form the 'prefix' directory name that get_model_save_directory expects.
# # The target prefix is like: neg_<rl_model_arch_component>_<base_prefix_component>_reg_<reg_alg_component>_sample_<sample_algorithm_component>

# # Corresponds to --rl_model argument in the original training script (e.g., attention_ilse.sh)
# RL_MODEL_ARCH_COMPONENT="policy_only"

# # Corresponds to the original --prefix argument in the training script (e.g., "loss_attention")
# RL_BASE_PREFIX_COMPONENT="loss_attention"

# # Corresponds to --reg_alg argument in the training script
# RL_REG_ALG_COMPONENT="sum"

# # Corresponds to --sample_algorithm argument in the training script
# # Use the string "None" if the sample_algorithm was not specified or was effectively None for path construction.
# # The error message showed "_sample_None", so if analyzing those, use "None".
# # For analyzing "neg_policy_only_loss_attention_reg_sum_sample_static", use "static".
# RL_SAMPLE_ALGORITHM_COMPONENT="static" 

# # Analysis script specific parameters
# BATCH_SIZE_ANALYSIS=32

# # --- Python Script to Execute ---
# PYTHON_SCRIPT_NAME="analyze_attention_ilse.py" # Your analysis script from the Canvas

# total_configs=0
# declare -a commands_to_run=()

# # Build a list of commands
# for seed_to_analyze in "${ANALYSIS_SEEDS[@]}"; do
#   for target_label in "${TARGET_LABELS[@]}"; do
#     for bag_size in "${BAG_SIZES[@]}"; do
#       for embedding_model in "${EMBEDDING_MODELS[@]}"; do
#         for mil_baseline_type in "${MIL_BASELINE_TYPES[@]}"; do # This is args.baseline for path construction

#           # Dynamically construct the full prefix for the RL model directory
#           # This must exactly match how the directories were named during training.
#           # Based on user's target: neg_policy_only_loss_attention_reg_sum_sample_static
          
#           # Handle if RL_SAMPLE_ALGORITHM_COMPONENT should result in "_sample_None"
#           sample_part_for_prefix=${RL_SAMPLE_ALGORITHM_COMPONENT}
#           if [ "${RL_SAMPLE_ALGORITHM_COMPONENT,,}" == "none" ]; then # Case-insensitive check for "none"
#               sample_part_for_prefix="None" # To match "_sample_None"
#           fi

#           CURRENT_RL_MODEL_PREFIX="neg_${RL_MODEL_ARCH_COMPONENT}_${RL_BASE_PREFIX_COMPONENT}_reg_${RL_REG_ALG_COMPONENT}_sample_${sample_part_for_prefix}"

#           echo "Preparing analysis for: dataset=$ANALYSIS_DATASET_NAME, seed=$seed_to_analyze, label=$target_label, bag=$bag_size, embed=$embedding_model, mil_task_model_type=$mil_baseline_type, RL_MODEL_DIR_PREFIX=$CURRENT_RL_MODEL_PREFIX"

#           CMD="python ${PYTHON_SCRIPT_NAME} \
#               --dataset \"${ANALYSIS_DATASET_NAME}\" \
#               --random_seed ${seed_to_analyze} \
#               --analysis_split \"${ANALYSIS_SPLIT}\" \
#               --label \"${target_label}\" \
#               --embedding_model \"${embedding_model}\" \
#               --data_embedded_column_name \"${DATA_EMBEDDED_COLUMN_NAME}\" \
#               --bag_size ${bag_size} \
#               --baseline \"${mil_baseline_type}\" \
#               --autoencoder_layer_sizes ${MIL_AUTOENCODER_LAYER_SIZES_STR//_/ } \
#               --prefix \"${CURRENT_RL_MODEL_PREFIX}\" \
#               --task_type \"${TASK_TYPE}\" \
#               --gpu 0 \
#               --no_wandb \
#               --batch_size_analysis ${BATCH_SIZE_ANALYSIS} \
#               "
#               # The analyze_attention_ilse.py script will load specific RL hyperparameters
#               # (like hdim, temperature, attention_size, rl_model, sample_algorithm, reg_alg) 
#               # from the sweep_best_model_config.json found in the directory identified by the args above.

#           commands_to_run+=("$CMD")
#           ((total_configs++))
#         done
#       done
#     done
#   done
# done

# echo "Total analysis configurations to run: $total_configs"

# # Execute all commands
# for cmd_idx in "${!commands_to_run[@]}"; do
#     echo "-------------------------------------"
#     echo "Running Analysis Config $((cmd_idx + 1)) / $total_configs"
#     echo "${commands_to_run[$cmd_idx]}"
#     echo "-------------------------------------"
#     eval "${commands_to_run[$cmd_idx]}"
#     echo "-------------------------------------"
#     echo "Finished Analysis Config $((cmd_idx + 1))"
# done

# echo "All analysis jobs submitted by this Slurm script have been processed."
