#!/bin/bash
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --array=0-42%10
#SBATCH --job-name=exp2_indep_gemini_flash
#SBATCH --output=logs/exp2_indep_gemini_flash_%A_%a.out

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

# Load API key from .env (never hardcode)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "[ERROR] GOOGLE_API_KEY not set. Add it to .env first."
    exit 1
fi

TEST_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" Experiment/Experiment2/file_list_exp2.txt)

echo "========================================"
echo "Experiment  : 2 — Model comparison (Gemini Flash)"
echo "Array task  : ${SLURM_ARRAY_TASK_ID} / 42"
echo "File        : ${TEST_FILE}"
echo "Mode        : independent"
echo "Model       : gemini-2.5-flash"
echo "Start       : $(date)"
echo "Node        : $(hostname)"
echo "========================================"

INFERENCE_BACKEND=api \
GEMINI_MODEL=gemini-2.5-flash \
EXTRACT_MODE=independent \
TEST_FILE="${TEST_FILE}" \
EXPERIMENT_RESULT_DIR=Experiment/Experiment2/result/independent_gemini_flash \
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
