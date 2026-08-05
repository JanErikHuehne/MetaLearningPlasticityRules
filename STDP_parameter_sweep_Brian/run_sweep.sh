#!/bin/bash
#SBATCH --job-name=stdp_sweep
#SBATCH --partition=cpu-parallel-small
#SBATCH --array=0-15624%500
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --output=/shome/ge64qic/logs/sweep_%A_%a.log
#SBATCH --error=/shome/ge64qic/logs/sweep_%A_%a.err

module load miniconda/25.7.0
source /shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate brian2-env

python stdp_sweep.py
