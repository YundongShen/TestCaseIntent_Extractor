#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=01:30:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --array=0-3
#SBATCH --job-name=combined_missing
#SBATCH --output=logs/combined_missing_%A_%a.out
#SBATCH --exclude=node061

# Qwen-3.5-27B — combined extraction for the cases missing from
# dataset/intent_combined (pytest_api, pytest_unit, rtl, supertest).
# Uses gpu_env_312 + CUDA 13.0 libs, matching the working deepseek/flash runs.

cd /proj/test_extender/Intent_Implementation_0327
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source gpu_env_312/bin/activate

TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset/combined_empty_retry.txt)

echo "========================================"
echo "Array task : ${SLURM_ARRAY_TASK_ID} / 3"
echo "File       : ${TEST_FILE}"
echo "Mode       : combined"
echo "Model      : Qwen-3.5-27B"
echo "Start      : $(date)"
echo "Node       : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

MODEL_TYPE=qwen \
EXTRACT_MODE=combined \
INFERENCE_BACKEND=local \
OUTPUT_DIR=dataset/intent_combined \
TEST_FILE="${TEST_FILE}" \
python extract_for_dataset.py

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: task ${SLURM_ARRAY_TASK_ID} — ${TEST_FILE}"
else
    echo "FAILED (exit ${EXIT_CODE}): task ${SLURM_ARRAY_TASK_ID} — ${TEST_FILE}"
fi
echo "End: $(date)"
echo "========================================"
exit $EXIT_CODE
