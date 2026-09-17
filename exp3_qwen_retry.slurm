#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=01:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9%10
#SBATCH --job-name=exp3_qwen_retry
#SBATCH --output=logs/exp3_qwen_retry_%A_%a.out

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" qwen_retry_filelist.txt)

# Derive the correct EXPERIMENT_RESULT_DIR from the test file path
# test_case/{level}/{framework}/ -> result/{level}_qwen/
LEVEL=$(echo "$TEST_FILE" | sed 's|.*test_case/\([^/]*\)/.*|\1|')
RESULT_DIR="Experiment/Experiment3/result/${LEVEL}_qwen"

echo "========================================"
echo "Experiment  : 3 — Qwen retry"
echo "Array task  : ${SLURM_ARRAY_TASK_ID} / 9"
echo "File        : ${TEST_FILE}"
echo "Level       : ${LEVEL}"
echo "Result dir  : ${RESULT_DIR}"
echo "Mode        : independent"
echo "Model       : Qwen-3.5-27B"
echo "Start       : $(date)"
echo "Node        : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

MODEL_TYPE=qwen \
EXTRACT_MODE=independent \
INFERENCE_BACKEND=local \
TEST_FILE="${TEST_FILE}" \
EXPERIMENT_RESULT_DIR="${RESULT_DIR}" \
python extract_for_experiment.py

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
