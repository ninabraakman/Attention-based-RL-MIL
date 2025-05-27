@@ -0,0 +1,72 @@
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/aggregated_subset/attention0505_%j.out
#SBATCH --error=../logs/aggregated_subset/attention0505_%j.err

module purge
module load 2023
source ../ve/bin/activate

# Navigate to project root
cd /projects/prjs1491/MasterThesisNinaBraakman

baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
target_labels=("label")
gpus=(0)
wandb_entity="ninabraakman-university-of-amsterdam"
wandb_project="MasterThesis"

dataset="oulad_subset"
data_embedded_column_name="instances"
rl_task_model="vanilla"
sample_algorithm="without_replacement"
task_type="classification"
autoencoder_layer_sizes="23,16,23"
bag_sizes=(39)
embedding_models=("tabular")

prefix="loss_attention"
rl_model="policy_only"
reg_alg="sum"

total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

for target_label in "${target_labels[@]}"; do
  for bag_size in "${bag_sizes[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
      for baseline_type in "${baseline_types[@]}"; do

        echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_0 ($current_run/$total_runs)"

        CUDA_VISIBLE_DEVICES=0 python run_attention_rlmil.py --rl --gpu 0 \
          --baseline $baseline_type \
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
          --rl_task_model $rl_task_model \
          --sample_algorithm $sample_algorithm \
          --reg_alg $reg_alg

        ((current_run++))
      done
    done
  done
done