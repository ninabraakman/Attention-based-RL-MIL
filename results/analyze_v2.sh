#!/bin/bash

#SBATCH --job-name=run_analysis_v2
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=rome

# Direct all logs to the results folder
#SBATCH --output=logs/analysis_v2_log-%j.out
#SBATCH --error=logs/analysis_v2_log-%j.err

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

# 2. Run the new analysis script
echo "Running analyze_results_v2.py from within results/ folder..."
python results/analyze_results_v2.py

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="