#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=01:30:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --array=0-209%30
#SBATCH --job-name=intent_extract
#SBATCH --output=logs/intent_%A_%a.out

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

# Read the file path for this array task (file_list.txt is 1-indexed by sed)
TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset/file_list.txt)

echo "========================================"
echo "Array task : ${SLURM_ARRAY_TASK_ID} / 209"
echo "File       : ${TEST_FILE}"
echo "Mode       : independent"
echo "Model      : Qwen-3.5-27B"
echo "Start      : $(date)"
echo "Node       : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

MODEL_TYPE=qwen \
EXTRACT_MODE=independent \
INFERENCE_BACKEND=local \
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
