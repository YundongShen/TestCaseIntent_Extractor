#!/bin/bash
# Submit 18 jobs sequentially via SLURM dependency chain
# Each job depends on the previous one; scheduler handles ordering after disconnect

set -e
cd /proj/test_extender/Intent_Implementation_0327

TEST_FILES=(
  "testcases/admin-index-api.test.js"
  "testcases/bug-ant-solid.test.js"
  "testcases/contact-us-form.feature"
  "testcases/ContactUs_StepDefs.java"
  "testcases/k6-api-load-test.js"
  "testcases/view-bookmarks.test.js"
)

MODES=("independent" "combined" "chain")

mkdir -p logs

PREV_JOB=""
COUNT=0
TOTAL=$((${#TEST_FILES[@]} * ${#MODES[@]}))

echo "=================================================="
echo "  Submitting ${TOTAL} jobs (Qwen 27B, 80GB GPU)"
echo "  SLURM dependency chain — continues after disconnect"
echo "=================================================="
echo ""

for test_file in "${TEST_FILES[@]}"; do
  for mode in "${MODES[@]}"; do
    COUNT=$((COUNT + 1))
    filename=$(basename "$test_file")
    safe_name=$(basename "$test_file" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-20)
    job_name="qw_${mode:0:3}_${safe_name:0:12}"
    log_file="logs/job_${COUNT}_${mode}_${safe_name}.out"

    tmp_script=$(mktemp /tmp/slurm_XXXXXX.sh)
    cat > "$tmp_script" << HEREDOC
#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=02:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_file}

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

echo "========================================"
echo "Job ${COUNT}/${TOTAL}: ${filename}"
echo "Mode: ${mode} | Model: Qwen-3.5-27B"
echo "Start: \$(date)"
echo "Node: \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

MODEL_TYPE=qwen \
EXTRACT_MODE=${mode} \
TEST_FILE=${test_file} \
INFERENCE_BACKEND=local \
python main.py

EXIT_CODE=\$?

echo ""
echo "========================================"
if [ \$EXIT_CODE -eq 0 ]; then
  echo "SUCCESS [${COUNT}/${TOTAL}]: ${filename} + ${mode}"
else
  echo "FAILED  [${COUNT}/${TOTAL}] (exit \$EXIT_CODE): ${filename} + ${mode}"
fi
echo "End: \$(date)"
echo "========================================"
HEREDOC

    if [ -z "$PREV_JOB" ]; then
      JOB_ID=$(sbatch --parsable "$tmp_script")
    else
      JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} "$tmp_script")
    fi

    PREV_JOB=$JOB_ID
    printf "  [%2d/%d] %-45s -> Job %-10s (log: %s)\n" \
      "$COUNT" "$TOTAL" "${filename} + ${mode}" "$JOB_ID" "$log_file"

    rm -f "$tmp_script"
  done
done

echo ""
echo "=================================================="
echo "  All ${TOTAL} jobs submitted"
echo "  Last Job ID: ${PREV_JOB}"
echo ""
echo "Monitor commands:"
echo "  squeue -u \$USER          # view queue"
echo "  tail -f logs/job_1_*.out  # follow first job log"
echo "=================================================="
