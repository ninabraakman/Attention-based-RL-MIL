#!/bin/bash

# module purge
# module load 2023
# source ../venv/bin/activate
if [[ -f venv/Scripts/activate ]]; then
  source venv/Scripts/activate
elif [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
else
  echo "ERROR: Cannot find venv activate script" >&2
  exit 1
fi
# Navigate to project root
# cd /projects/prjs1491/MasterThesisNinaBraakman
# cd ~/Documents/UVA/Thesis/Data\ en\ code/Attention-based-RL-MIL || exit
cd "$HOME/Documents/UVA/Thesis/Data en code/Attention-based-RL-MIL"

baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
target_labels=("label")
gpus=(0)
wandb_entity="ninabraakman-university-of-amsterdam"
wandb_project="MasterThesis"

dataset="oulad_aggregated_subset"
data_embedded_column_name="instances"
task_type="classification"
autoencoder_layer_sizes="22,16,22"
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
                                      --random_seed 0 ;
        exit
        
        ((current_run++))
      done
    done
  done
done
