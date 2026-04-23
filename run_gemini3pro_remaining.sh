#!/bin/bash
# Gemini 3 Pro Preview — remaining 5 test cases x 3 modes = 15 combinations
# admin-index-api already completed, skipped here
# 60s delay between pipelines; 5s proactive delay built into each request

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

export INFERENCE_BACKEND=api
# export GOOGLE_API_KEY="your_key_here"  # do not hardcode — export manually before running

TEST_FILES=(
  "testcases/bug-ant-solid.test.js"
  "testcases/contact-us-form.feature"
  "testcases/ContactUs_StepDefs.java"
  "testcases/k6-api-load-test.js"
  "testcases/view-bookmarks.test.js"
)

MODES=("independent" "combined" "chain")

TOTAL=$((${#TEST_FILES[@]} * ${#MODES[@]}))
COUNT=0
SUCCESS=0
FAIL=0

echo "=================================================="
echo "  Gemini 3 Pro Preview — 15 combinations"
echo "  Delay between pipelines: 60s"
echo "  Start time: $(date)"
echo "=================================================="
echo ""

for test_file in "${TEST_FILES[@]}"; do
  for mode in "${MODES[@]}"; do
    COUNT=$((COUNT + 1))
    filename=$(basename "$test_file")

    echo "--------------------------------------------------"
    echo "[${COUNT}/${TOTAL}] ${filename} + ${mode}"
    echo "Time: $(date)"
    echo "--------------------------------------------------"

    EXTRACT_MODE=$mode TEST_FILE=$test_file python main.py
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
      SUCCESS=$((SUCCESS + 1))
      echo "SUCCESS [${COUNT}/${TOTAL}]: ${filename} + ${mode}"
    else
      FAIL=$((FAIL + 1))
      echo "FAILED  [${COUNT}/${TOTAL}] (exit $EXIT): ${filename} + ${mode}"
    fi

    if [ $COUNT -lt $TOTAL ]; then
      echo "Waiting 60s..."
      sleep 60
    fi
  done
done

echo ""
echo "=================================================="
echo "  Done: ${SUCCESS} succeeded / ${FAIL} failed (total ${TOTAL})"
echo "  End time: $(date)"
echo "=================================================="
