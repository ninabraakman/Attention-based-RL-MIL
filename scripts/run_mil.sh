#!/bin/bash
#SBATCH --partition= #YOUR PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/oulad_full/mil/seed0_%j.out
#SBATCH --error=../logs/ouald_full/mil/seed0_%j.err

module purge
module load 2023
source ../venv/bin/activate
cd #ROOT OF YOUR PROJECT

baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
target_labels=("label")
gpus=(0)
wandb_entity="YOUR_WANDB_ENTITY"
wandb_project="YOUR_WANDB_PROJECT"

dataset="oulad_full"                              # "oulad_full" or "oulad_aggregated"
data_embedded_column_name="instances"
task_type="classification"
autoencoder_layer_sizes="20,16,20"                # "22,16,22" for oulad_aggregated and "20,16,20" for oulad_full
bag_sizes=(20)                                    # for all experiments in this project bag_size 20 is used
embedding_models=("tabular")

total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

for target_label_index in "${!target_labels[@]}"; do
  for bag_size_index in "${!bag_sizes[@]}"; do
    for embedding_model_index in "${!embedding_models[@]}"; do
      for baseline_type_index in "${!baseline_types[@]}"; do
        target_label=${target_labels[$target_label_index]}
        bag_size=${bag_sizes[$bag_size_index]}
        embedding_model=${embedding_models[$embedding_model_index]}
        baseline_type=${baseline_types[$baseline_type_index]}
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
                                      --random_seed 0 ;
        
        ((current_run++))
      done
    done
  done
done
