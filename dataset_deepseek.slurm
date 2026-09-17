#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=01:30:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --array=0-209%15
#SBATCH --job-name=dataset_deepseek
#SBATCH --output=logs/dataset_deepseek_%A_%a.out

cd /proj/test_extender/Intent_Implementation_0327
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source gpu_env_312/bin/activate

TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset/file_list.txt)

echo "========================================"
echo "Task        : Full dataset — DeepSeek"
echo "Array task  : ${SLURM_ARRAY_TASK_ID} / 209"
echo "File        : ${TEST_FILE}"
echo "Mode        : independent"
echo "Model       : DeepSeek-LLM-7B-Chat"
echo "Start       : $(date)"
echo "Node        : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

MODEL_TYPE=7b \
EXTRACT_MODE=independent \
INFERENCE_BACKEND=local \
OUTPUT_DIR=dataset/intent_deepseek \
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
