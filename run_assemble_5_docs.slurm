#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name=assemble_onboarding
#SBATCH --output=logs/assemble_onboarding_%j.out
#SBATCH --exclude=node061

cd /proj/test_extender/Intent_Implementation_0327
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source gpu_env_312/bin/activate

python assemble_5_onboarding_docs.py
