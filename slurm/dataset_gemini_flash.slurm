#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --array=0-209%10
#SBATCH --job-name=dataset_gemini_flash
#SBATCH --output=logs/dataset_gemini_flash_%A_%a.out

cd /proj/test_extender/Intent_Implementation_0327
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source gpu_env_312/bin/activate

# Load keys from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$VERTEX_API_KEY" ]; then
    echo "[ERROR] VERTEX_API_KEY not set in .env"
    exit 1
fi

TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset/file_list.txt)

echo "========================================"
echo "Task        : Full dataset — Gemini 2.5 Flash"
echo "Array task  : ${SLURM_ARRAY_TASK_ID} / 209"
echo "File        : ${TEST_FILE}"
echo "Mode        : independent"
echo "Model       : gemini-2.5-flash"
echo "Start       : $(date)"
echo "Node        : $(hostname)"
echo "========================================"

INFERENCE_BACKEND=api \
GEMINI_MODEL=gemini-2.5-flash \
EXTRACT_MODE=independent \
OUTPUT_DIR=dataset/intent_gemini_flash \
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
