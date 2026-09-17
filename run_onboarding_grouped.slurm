#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name=onboarding_grouped
#SBATCH --output=logs/onboarding_grouped_%j.out
#SBATCH --exclude=node061

cd /proj/test_extender/Intent_Implementation_0327
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source gpu_env_312/bin/activate

python test_onboarding_grouped.py
