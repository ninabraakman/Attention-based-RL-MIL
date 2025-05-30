#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/aggregated_subset/attention/a2305_%j.out
#SBATCH --error=../logs/aggregated_subset/attention/a2305_%j.err

module purge
module load 2023
source ../venv/bin/activate

# Navigate to project root
cd /projects/prjs1491/Attention-based-RL-MIL

# Which MIL encoders to try
baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
target_labels=("label")
wandb_entity="ninabraakman-university-of-amsterdam"
wandb_project="MasterThesis"

dataset="oulad_aggregated_subset"
data_embedded_column_name="instances"
autoencoder_layer_sizes="22,16,22"   # for oulad_aggregated
bag_sizes=(20)
embedding_models=("tabular")

# RL settings
prefix="loss_attention"
rl_model="policy_only"
rl_task_model="vanilla"
sample_algorithm="static"
reg_alg="sum"
task_type="classification"

# Attention‐sampler hyperparameters
is_linear_attention="--is_linear_attention"  # omit flag for non‐linear


total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

for target_label in "${target_labels[@]}"; do
  for bag_size in "${bag_sizes[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
      for baseline_type in "${baseline_types[@]}"; do

        echo "Run $current_run/$total_runs: baseline=$baseline_type, bag_size=$bag_size, embed=$embedding_model"

        CUDA_VISIBLE_DEVICES=0 python attention2305.py \
          --rl \
          --gpu 0 \
          --baseline               "$baseline_type" \
          --autoencoder_layer_sizes "$autoencoder_layer_sizes" \
          --label                  "$target_label" \
          --data_embedded_column_name "$data_embedded_column_name" \
          --prefix                 "loss_attention" \
          --dataset                "$dataset" \
          --bag_size               $bag_size \
          --batch_size             32 \
          --run_sweep \
          --embedding_model        "$embedding_model" \
          --train_pool_size        1 \
          --eval_pool_size         10 \
          --test_pool_size         10 \
          --balance_dataset \
          --wandb_entity           "$wandb_entity" \
          --wandb_project          "$wandb_project" \
          --random_seed            0 \
          --task_type              "$task_type" \
          --rl_model               "$rl_model" \
          --rl_task_model          "$rl_task_model" \
          --sample_algorithm       "$sample_algorithm" \
          --reg_alg                "$reg_alg" \
          $is_linear_attention \

        ((current_run++))
      done
    done
  done
done

# #!/bin/bash
# #SBATCH --partition=gpu_h100
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G
# #SBATCH --time=24:00:00
# #SBATCH --output=../logs/attention2_%j.out
# #SBATCH --error=../logs/attention2_%j.err

# module purge
# module load 2023
# source ../ve/bin/activate

# # Navigate to project root
# cd /projects/prjs1491/MasterThesisNinaBraakman

# # Which MIL encoders to try
# baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
# target_labels=("label")
# wandb_entity="ninabraakman-university-of-amsterdam"
# wandb_project="MasterThesis"

# dataset="oulad_aggregated_subset"
# data_embedded_column_name="instances"
# autoencoder_layer_sizes="22,16,22"   # for oulad_aggregated
# bag_sizes=(20)
# embedding_models=("tabular")

# # RL settings
# rl_model="policy_only"
# rl_task_model="vanilla"
# sample_algorithm="static"
# reg_alg="sum"
# task_type="classification"

# # Attention‐sampler hyperparameters
# temperature=0.5
# is_linear_attention="--is_linear_attention"  # omit flag for non‐linear
# attention_size=64
# attention_dropout_p=0.3
# actor_lr=1e-4
# critic_lr=1e-4

# total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
# current_run=1

# for target_label in "${target_labels[@]}"; do
#   for bag_size in "${bag_sizes[@]}"; do
#     for embedding_model in "${embedding_models[@]}"; do
#       for baseline_type in "${baseline_types[@]}"; do

#         echo "Run $current_run/$total_runs: baseline=$baseline_type, bag_size=$bag_size, embed=$embedding_model"

#         CUDA_VISIBLE_DEVICES=0 python run_attention2.py \
#           --rl \
#           --gpu 0 \
#           --baseline               "$baseline_type" \
#           --autoencoder_layer_sizes "$autoencoder_layer_sizes" \
#           --label                  "$target_label" \
#           --data_embedded_column_name "$data_embedded_column_name" \
#           --prefix                 "loss_attention" \
#           --dataset                "$dataset" \
#           --bag_size               $bag_size \
#           --batch_size             32 \
#           --run_sweep \
#           --embedding_model        "$embedding_model" \
#           --train_pool_size        1 \
#           --eval_pool_size         10 \
#           --test_pool_size         10 \
#           --balance_dataset \
#           --wandb_entity           "$wandb_entity" \
#           --wandb_project          "$wandb_project" \
#           --random_seed            4 \
#           --task_type              "$task_type" \
#           --rl_model               "$rl_model" \
#           --rl_task_model          "$rl_task_model" \
#           --sample_algorithm       "$sample_algorithm" \
#           --reg_alg                "$reg_alg" \
#           --reg_coef               0.01 \
#           --actor_learning_rate    $actor_lr \
#           --critic_learning_rate   $critic_lr \
#           --temperature            $temperature \
#           $is_linear_attention \
#           --attention_size         $attention_size \
#           --attention_dropout_p    $attention_dropout_p

#         ((current_run++))
#       done
#     done
#   done
# done