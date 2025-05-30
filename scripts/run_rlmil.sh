#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=../logs/full_subset/rlmil/%j.out
#SBATCH --error=../logs/full_subset/rlmil/%j.err


module purge
module load 2023
source ../venv/bin/activate

# Navigate to project root
cd /projects/prjs1491/Attention-based-RL-MIL

baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")             # "MeanMLP" "MaxMLP" "AttentionMLP" "repset"
target_labels=("label")
gpus=(0)
wandb_entity="ninabraakman-university-of-amsterdam"
wandb_project="MasterThesis"

dataset="oulad_full_subset"
data_embedded_column_name="instances"
rl_task_model="vanilla"
sample_algorithm="static"
task_type="classification"
autoencoder_layer_sizes="20,16,20"
bag_sizes=(20)
embedding_models=("tabular")

prefix="loss"
rl_model="policy_only"
search_algorithm="epsilon_greedy"
reg_alg="sum"

total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

for target_label in "${target_labels[@]}"; do
  for bag_size in "${bag_sizes[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
      for baseline_type in "${baseline_types[@]}"; do
        gpu=${gpus[$target_label_index]}
        echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_$gpu ($current_run/$total_runs)"

        CUDA_VISIBLE_DEVICES=$gpu python run_rlmil.py --rl --baseline $baseline_type \
                                            --autoencoder_layer_sizes $autoencoder_layer_sizes \
                                            --label $target_label \
                                            --data_embedded_column_name $data_embedded_column_name \
                                            --prefix $prefix \
                                            --dataset $dataset \
                                            --bag_size $bag_size \
                                            --run_sweep \
                                            --embedding_model $embedding_model \
                                            --train_pool_size 1 --eval_pool_size 10 --test_pool_size 10 \
                                            --balance_dataset \
                                            --wandb_entity $wandb_entity \
                                            --wandb_project $wandb_project \
                                            --random_seed 0 \
                                            --task_type $task_type \
                                            --rl_model $rl_model \
                                            --search_algorithm $search_algorithm \
                                            --rl_task_model $rl_task_model \
                                            --sample_algorithm $sample_algorithm \
                                            --reg_alg $reg_alg ;
        ((current_run++))
      done
    done
  done
done


# #!/bin/bash
# #SBATCH --partition=gpu_h100
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8 
# #SBATCH --mem=16G    
# #SBATCH --time=8:00:00 
# #SBATCH --output=../logs/full/rlmil/MEAN/%j.out
# #SBATCH --error=../logs/full/rlmil/MEAN/%j.err 


# module purge
# module load 2023
# source ../venv/bin/activate

# # Navigate to project root
# cd /projects/prjs1491/Attention-based-RL-MIL

# baseline_types=("MeanMLP")         #"MeanMLP" "MaxMLP" "AttentionMLP" "repset"
# target_labels=("label")
# gpus_available=(0)
# wandb_entity="ninabraakman-university-of-amsterdam"
# wandb_project="MasterThesis"

# dataset="oulad_full"
# data_embedded_column_name="instances"
# rl_task_model="vanilla"
# sample_algorithm="static"
# task_type="classification"
# autoencoder_layer_sizes="20,16,22"
# bag_sizes=(20)
# embedding_models=("tabular")

# prefix="loss"
# rl_model="policy_only"
# search_algorithm="epsilon_greedy"
# reg_alg="sum"

# pids=()
# gpu_idx=0

# target_label=${target_labels[0]}
# bag_size=${bag_sizes[0]}
# embedding_model=${embedding_models[0]}

# for baseline_type in "${baseline_types[@]}"; do
#     if [ $gpu_idx -ge ${#gpus_available[@]} ]; then
#         echo "Warning: Not enough GPUs for all baseline_types in parallel. Some will run after others complete if you add more baseline_types than GPUs."
#     fi

#     current_gpu=${gpus_available[$gpu_idx]}

#     echo "Launching SWEEP for: $baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model on GPU $current_gpu"

#     CUDA_VISIBLE_DEVICES=$current_gpu python run_rlmil.py --rl --baseline "$baseline_type" \
#         --autoencoder_layer_sizes "$autoencoder_layer_sizes" \
#         --label "$target_label" \
#         --data_embedded_column_name "$data_embedded_column_name" \
#         --prefix "$prefix" \
#         --dataset "$dataset" \
#         --bag_size "$bag_size" \
#         --batch_size 32 \
#         --run_sweep \
#         --embedding_model "$embedding_model" \
#         --train_pool_size 1 --eval_pool_size 10 --test_pool_size 10 \
#         --balance_dataset \
#         --wandb_entity "$wandb_entity" \
#         --wandb_project "$wandb_project" \
#         --random_seed 0 \
#         --task_type "$task_type" \
#         --rl_model "$rl_model" \
#         --search_algorithm "$search_algorithm" \
#         --rl_task_model "$rl_task_model" \
#         --sample_algorithm "$sample_algorithm" \
#         --reg_alg "$reg_alg" & 

#     pids+=($!) 
#     ((gpu_idx++))
# done

# echo "Waiting for all background sweeps to complete..."
# for pid in "${pids[@]}"; do
#     wait "$pid"
# done

# echo "All sweeps completed."


# # #!/bin/bash

# # cd ..
# # source venv/bin/activate

# # baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
# # target_labels=("label")
# # gpus=(0)

# # # wandb config
# # wandb_entity="ninabraakman-university-of-amsterdam"
# # wandb_project="MasterThesis"

# # # Dataset is either: `oulad_aggregated,` `oulad_aggregated_subset,` `oulad_full,` or `oulad_full_subset`
# # dataset="oulad_aggregated_subset"
# # data_embedded_column_name="instances"

# # # Possible values: "vanilla" or "ensemble". Keep in mind before running the script with rl_task_model="ensemble" you should run the `only_ensemble` setting first.
# # rl_task_model="vanilla"

# # # These are sampling strategies for selecting $|b_i|$ instances with possible values: "static" "with_replacement" "without_replacement"
# # sample_algorithm="static"

# # # --no_autoencoder_for_rl flag can be used to run the experiments without autoencoder component

# # # ---- Constants ----
# # task_type="classification"

# # # autoencoder_layer_sizes should be a string of comma-separated integers. We used "768,256,768" in all experiments.
# # autoencoder_layer_sizes="22,16,22"

# # # The size of b_i in the paper. We used 20 in all experiments.
# # bag_sizes=(20)
# # embedding_models=("tabular")
# # prefix="loss"
# # rl_model="policy_only"
# # search_algorithm="epsilon_greedy"
# # reg_alg="sum"

# # # Get the total number of runs
# # total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
# # current_run=1

# # # Loop over all combinations of baseline_type and target_label and embedding_model
# # for target_label_index in "${!target_labels[@]}"; do
# #   for bag_size_index in "${!bag_sizes[@]}"; do
# #     for embedding_model_index in "${!embedding_models[@]}"; do
# #       for baseline_type_index in "${!baseline_types[@]}"; do
# #         # Get current target label
# #         target_label=${target_labels[$target_label_index]}
# #         # Get current bag_size
# #         bag_size=${bag_sizes[$bag_size_index]}
# #         # Get current embedding_model
# #         embedding_model=${embedding_models[$embedding_model_index]}
# #         # Get current baseline_type
# #         baseline_type=${baseline_types[$baseline_type_index]}
# #         # Get current gpu
# #         gpu=${gpus[$target_label_index]}
# #         echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_$gpu ($current_run/$total_runs)"

# #         # different RL models running
# #         SESSION_NAME="${gpu}_${sample_algorithm}_${dataset}_${target_label}_${baseline_type}"
# #         screen -dmS "$SESSION_NAME" bash -c "
# #           CUDA_VISIBLE_DEVICES=$gpu python3 run_rlmil.py --rl --baseline $baseline_type \
# #                                             --autoencoder_layer_sizes $autoencoder_layer_sizes \
# #                                             --label $target_label \
# #                                             --data_embedded_column_name $data_embedded_column_name \
# #                                             --prefix $prefix \
# #                                             --dataset $dataset \
# #                                             --bag_size $bag_size \
# #                                             --batch_size 32 \
# #                                             --run_sweep \
# #                                             --embedding_model $embedding_model \
# #                                             --train_pool_size 1 --eval_pool_size 10 --test_pool_size 10 \
# #                                             --balance_dataset \
# #                                             --wandb_entity $wandb_entity \
# #                                             --wandb_project $wandb_project \
# #                                             --random_seed 0 \
# #                                             --task_type $task_type \
# #                                             --rl_model $rl_model \
# #                                             --search_algorithm $search_algorithm \
# #                                             --rl_task_model $rl_task_model \
# #                                             --sample_algorithm $sample_algorithm \
# #                                             --reg_alg $reg_alg ;
# #           exit"

# #         # different RL models without autoencoder
# #         # SESSION_NAME="${gpu}_noauto_${sample_algorithm}_${dataset}_${target_label}_${baseline_type}"
# #         # screen -dmS "$SESSION_NAME" bash -c "
# #         #   CUDA_VISIBLE_DEVICES=$gpu python3 run_rlmil.py --rl --baseline $baseline_type \
# #         #                                     --autoencoder_layer_sizes $autoencoder_layer_sizes \
# #         #                                     --label $target_label \
# #         #                                     --data_embedded_column_name $data_embedded_column_name \
# #         #                                     --prefix $prefix \
# #         #                                     --dataset $dataset \
# #         #                                     --bag_size $bag_size \
# #         #                                     --batch_size 32 \
# #         #                                     --run_sweep \
# #         #                                     --embedding_model $embedding_model \
# #         #                                     --train_pool_size 1 --eval_pool_size 10 --test_pool_size 10 \
# #         #                                     --balance_dataset \
# #         #                                     --wandb_entity $wandb_entity \
# #         #                                     --wandb_project $wandb_project \
# #         #                                     --random_seed 0 \
# #         #                                     --task_type $task_type \
# #         #                                     --rl_model $rl_model \
# #         #                                     --search_algorithm $search_algorithm \
# #         #                                     --rl_task_model $rl_task_model \
# #         #                                     --sample_algorithm $sample_algorithm \
# #         #                                     --reg_alg $reg_alg  \
# #         #                                     --no_autoencoder_for_rl ;
# #         #   exit"

# #         # only ensemble running
# #         # SESSION_NAME="${gpu}_ens_${sample_algorithm}_${dataset}_${target_label}_${baseline_type}"
# #         # screen -dmS "$SESSION_NAME" bash -c "
# #         # CUDA_VISIBLE_DEVICES=$gpu python3 run_rlmil.py --rl --baseline $baseline_type \
# #         #                                                                 --autoencoder_layer_sizes $autoencoder_layer_sizes \
# #         #                                                                 --label $target_label \
# #         #                                                                 --data_embedded_column_name $data_embedded_column_name \
# #         #                                                                 --prefix $prefix \
# #         #                                                                 --dataset $dataset \
# #         #                                                                 --bag_size $bag_size \
# #         #                                                                 --batch_size 32 \
# #         #                                                                 --run_sweep \
# #         #                                                                 --embedding_model $embedding_model \
# #         #                                                                 --train_pool_size 1 --eval_pool_size 10 --test_pool_size 10 \
# #         #                                                                 --balance_dataset \
# #         #                                                                 --wandb_entity $wandb_entity \
# #         #                                                                 --wandb_project $wandb_project \
# #         #                                                                 --random_seed 0 \
# #         #                                                                 --task_type $task_type \
# #         #                                                                 --only_ensemble \
# #         #                                                                 --rl_model $rl_model \
# #         #                                                                 --sample_algorithm $sample_algorithm ;
# #         #                                       exit"
# #         ((current_run++))
# #       done
# #     done
# #   done
# # done
