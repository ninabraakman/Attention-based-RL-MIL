#!/bin/bash

#SBATCH --job-name=run_thesis_analysis
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=rome

# <<< CHANGED >>> Direct all logs to the results folder
#SBATCH --output=logs/analysis_log-%j.out
#SBATCH --error=logs/analysis_log-%j.err

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

# 2. Run the analysis script, telling python to look inside the results folder
echo "Running analyze_results.py from within results/ folder..."
# <<< THIS IS THE FIX >>>
python results/analyze_results.py

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="