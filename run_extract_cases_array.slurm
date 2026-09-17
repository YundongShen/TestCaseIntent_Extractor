#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=01:30:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --array=0-30%10
#SBATCH --job-name=extract_case_onboarding
#SBATCH --output=logs/extract_case_onboarding_%A_%a.out
#SBATCH --exclude=node061

cd /proj/test_extender/Intent_Implementation_0327
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source gpu_env_312/bin/activate

python extract_one_case_for_onboarding.py --task-index ${SLURM_ARRAY_TASK_ID}
