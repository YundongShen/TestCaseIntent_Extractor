#!/bin/bash
#SBATCH --job-name=onboarding_test
#SBATCH --output=onboarding_test_%j.out
#SBATCH --error=onboarding_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

# Use Flash combined result (clean 1 object, 1 goal, 29 activities)
python test_onboarding.py Result/extract_result/extract_result_20260426_185510.json
