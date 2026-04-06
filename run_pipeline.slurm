#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=intent_extraction
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

cd /proj/test_extender/Intent_Implementation_0327

source gpu_env/bin/activate

python3 main.py
