#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-209%5
#SBATCH --job-name=dataset_gemini_pro
#SBATCH --output=logs/dataset_gemini_pro_%A_%a.out

# Gemini 3.1 Pro — full dataset (210) — independent extraction.
# API-only work, but berzelius requires a GPU allocation, so we request the
# smallest possible (gpu:1). Model requires the `global` Vertex location.

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env_312/bin/activate

# Load keys from .env (VERTEX_API_KEY = Gemini Vertex/Agent Platform key)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$VERTEX_API_KEY" ]; then
    echo "[ERROR] VERTEX_API_KEY not set in .env"
    exit 1
fi

TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset/file_list.txt)

echo "========================================"
echo "Task        : Full dataset — Gemini 3.1 Pro"
echo "Array task  : ${SLURM_ARRAY_TASK_ID} / 209"
echo "File        : ${TEST_FILE}"
echo "Mode        : independent"
echo "Model       : gemini-3.1-pro-preview (location=global)"
echo "Start       : $(date)"
echo "Node        : $(hostname)"
echo "========================================"

GOOGLE_CLOUD_LOCATION=global \
INFERENCE_BACKEND=api \
GEMINI_MODEL=gemini-3.1-pro-preview \
EXTRACT_MODE=independent \
OUTPUT_DIR=dataset/intent_gemini_pro \
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
