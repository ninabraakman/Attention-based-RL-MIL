#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=../logs/full/mil/%j.out
#SBATCH --error=../logs/full/mil/%j.err

module purge
module load 2023
source ../venv/bin/activate
cd /projects/prjs1491/Attention-based-RL-MIL

baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
target_labels=("label")
gpus=(0)
wandb_entity="ninabraakman-university-of-amsterdam"
wandb_project="MasterThesis"

dataset="oulad_full"
data_embedded_column_name="instances"
task_type="classification"
autoencoder_layer_sizes="20,16,20"
bag_sizes=(20)
embedding_models=("tabular")

# Get the total number of runs
total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

# Loop over all combinations of baseline_type and target_label and embedding_model
for target_label_index in "${!target_labels[@]}"; do
  for bag_size_index in "${!bag_sizes[@]}"; do
    for embedding_model_index in "${!embedding_models[@]}"; do
      for baseline_type_index in "${!baseline_types[@]}"; do
        # Get current target label
        target_label=${target_labels[$target_label_index]}
        # Get current bag_size
        bag_size=${bag_sizes[$bag_size_index]}
        # Get current embedding_model
        embedding_model=${embedding_models[$embedding_model_index]}
        # Get current baseline_type
        baseline_type=${baseline_types[$baseline_type_index]}
        # Get current gpu
        gpu=${gpus[$target_label_index]}
        echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_$gpu ($current_run/$total_runs)"

        CUDA_VISIBLE_DEVICES=$gpu python run_mil.py \
                                      --baseline "$baseline_type" \
                                      --label "$target_label" \
                                      --bag_size "$bag_size" \
                                      --embedding_model $embedding_model \
                                      --dataset $dataset \
                                      --run_sweep \
                                      --wandb_entity $wandb_entity \
                                      --wandb_project $wandb_project \
                                      --autoencoder_layer_sizes $autoencoder_layer_sizes \
                                      --data_embedded_column_name $data_embedded_column_name \
                                      --task_type $task_type \
                                      --random_seed 10 ;
        
        ((current_run++))
      done
    done
  done
done
