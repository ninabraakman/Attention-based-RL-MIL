#!/bin/bash

#SBATCH --job-name=gather_thesis_results
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=rome

# <<< CHANGED >>> Direct all logs to the results folder
#SBATCH --output=logs/gather_log-%j.out
#SBATCH --error=logs/gather_log-%j.err

# --- Job Commands ---
echo "=========================================================="
echo "Job started on $(hostname) at $(date)"
echo "=========================================================="

# 1. Navigate to the project directory
cd /projects/prjs1491/Attention-based-RL-MIL
echo "Current directory: $(pwd)"

# 2. Load modules and activate environment
module purge
module load 2023
source /projects/prjs1491/Attention-based-RL-MIL/venv/bin/activate
echo "Python environment activated."

# 3. Run the combined gathering and cleaning script
echo "Running gather_results.py.."
python results/gather_results.py

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="