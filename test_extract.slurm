#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test_extract
#SBATCH --output=test_extract_%j.out

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate
python3 test_simple_extract.py
