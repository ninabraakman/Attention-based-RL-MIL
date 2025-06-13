#!/bin/bash
# --- CORE JOB ARRAY DIRECTIVES ---
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --array=1-10

# --- RESOURCE REQUESTS (per job task) ---
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00

# --- DYNAMIC LOG FILES ---
#SBATCH --output=../logs/full_final/attention_ilse/Repset/%j_%a.out
#SBATCH --error=../logs/full_final/attention_ilse/Repset/%j_%a.err


module purge
module load 2023
source /projects/prjs1491/Attention-based-RL-MIL/venv/bin/activate

# Navigate to project root
cd /projects/prjs1491/Attention-based-RL-MIL

# --- CONFIGURATION ---
# Set the single baseline you want to run for this job array
baseline_type="repset"

# The random seed is now taken from the job array task ID
random_seed=$SLURM_ARRAY_TASK_ID

# Other parameters for the ILSE model
target_label="label"
wandb_entity="ninabraakman-university-of-amsterdam"
wandb_project="MasterThesis"
dataset="oulad_full"
data_embedded_column_name="instances"
task_type="classification"
autoencoder_layer_sizes="20,16,20"
bag_size=20
embedding_model="tabular"
rl_task_model="vanilla"
sample_algorithm="without_replacement"
prefix="loss_attention_ilse"
rl_model="policy_only"
reg_alg="sum"
# Attention‐sampler hyperparameters
is_linear_attention="--is_linear_attention" 

echo "Starting ILSE Run for $baseline_type, Seed: $random_seed"

# The Python call now uses the dynamic random_seed variable
CUDA_VISIBLE_DEVICES=0 python attention_ilse.py \
  --rl \
  --gpu 0 \
  --baseline "$baseline_type" \
  --autoencoder_layer_sizes "$autoencoder_layer_sizes" \
  --label "$target_label" \
  --data_embedded_column_name "$data_embedded_column_name" \
  --prefix "$prefix" \
  --dataset "$dataset" \
  --bag_size $bag_size \
  --run_sweep \
  --embedding_model "$embedding_model" \
  --train_pool_size 1 \
  --eval_pool_size 10 \
  --test_pool_size 10 \
  --balance_dataset \
  --wandb_entity "$wandb_entity" \
  --wandb_project "$wandb_project" \
  --random_seed "$random_seed" \
  --task_type "$task_type" \
  --rl_model "$rl_model" \
  --rl_task_model "$rl_task_model" \
  --sample_algorithm "$sample_algorithm" \
  --reg_alg "$reg_alg" \
  $is_linear_attention

echo "Finished ILSE Run for $baseline_type, Seed: $random_seed"

# #!/bin/bash
# #SBATCH --partition=gpu_a100
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=8G
# #SBATCH --time=4:00:00
# #SBATCH --output=../logs/subset_configs/full_subset/attention_ilse/MeanMLP/%j.out
# #SBATCH --error=../logs/subset_configs/full_subset/attention_ilse/MeanMLP/%j.err

# module purge
# module load 2023
# source ../venv/bin/activate

# # Navigate to project root
# cd /projects/prjs1491/Attention-based-RL-MIL

# # Which MIL encoders to try
# baseline_types=("MeanMLP")       # "MeanMLP" "MaxMLP" "AttentionMLP" "repset"
# target_labels=("label")
# gpus=(0)
# wandb_entity="ninabraakman-university-of-amsterdam"
# wandb_project="MasterThesis"

# dataset="oulad_full_subset"
# data_embedded_column_name="instances"
# task_type="classification"
# autoencoder_layer_sizes="20,16,20"   # "22,16,22" for oulad_aggregated and "20,16,20" for oulad_full
# bag_sizes=(20)
# embedding_models=("tabular")
# random_seed=0

# rl_task_model="vanilla"
# sample_algorithm="without_replacement"
# prefix="loss_attention_ilse"
# rl_model="policy_only"
# reg_alg="sum"


# # Attention‐sampler hyperparameters
# is_linear_attention="--is_linear_attention" 


# total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
# current_run=1

# for target_label in "${target_labels[@]}"; do
#   for bag_size in "${bag_sizes[@]}"; do
#     for embedding_model in "${embedding_models[@]}"; do
#       for baseline_type in "${baseline_types[@]}"; do

#         echo "Run $current_run/$total_runs: baseline=$baseline_type, bag_size=$bag_size, embed=$embedding_model"

#         CUDA_VISIBLE_DEVICES=0 python attention_ilse.py \
#           --rl \
#           --gpu 0 \
#           --baseline               "$baseline_type" \
#           --autoencoder_layer_sizes "$autoencoder_layer_sizes" \
#           --label                  "$target_label" \
#           --data_embedded_column_name "$data_embedded_column_name" \
#           --prefix                 "$prefix" \
#           --dataset                "$dataset" \
#           --bag_size               $bag_size \
#           --run_sweep \
#           --embedding_model        "$embedding_model" \
#           --train_pool_size        1 \
#           --eval_pool_size         10 \
#           --test_pool_size         10 \
#           --balance_dataset \
#           --wandb_entity           "$wandb_entity" \
#           --wandb_project          "$wandb_project" \
#           --random_seed            "$random_seed" \
#           --task_type              "$task_type" \
#           --rl_model               "$rl_model" \
#           --rl_task_model          "$rl_task_model" \
#           --sample_algorithm       "$sample_algorithm" \
#           --reg_alg                "$reg_alg" \
#           $is_linear_attention \

#         ((current_run++))
#       done
#     done
#   done
# done
