#!/bin/bash
# 用 Gemini 2.5 Flash API 顺序跑 18 个组合
# 每次请求后有 5s 主动延迟，pipeline 之间额外等 30s
# 用 nohup 跑，断网后继续

cd /proj/test_extender/Intent_Implementation_0327
source gpu_env/bin/activate

export INFERENCE_BACKEND=api
export MODEL_TYPE=qwen           # 只影响 local 模式的模型配置，API 模式忽略
export GOOGLE_API_KEY="AIzaSyDblVGh6HYbJwvO_Iko4rJce_Py_UL62Mw"

TEST_FILES=(
  "testcases/admin-index-api.test.js"
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
echo "  Gemini 2.5 Flash API — 18 个组合顺序运行"
echo "  请求间延迟: 5s (内置) + pipeline 间: 30s"
echo "  开始时间: $(date)"
echo "=================================================="
echo ""

for test_file in "${TEST_FILES[@]}"; do
  for mode in "${MODES[@]}"; do
    COUNT=$((COUNT + 1))
    filename=$(basename "$test_file")

    echo "--------------------------------------------------"
    echo "[${COUNT}/${TOTAL}] ${filename} + ${mode}"
    echo "时间: $(date)"
    echo "--------------------------------------------------"

    EXTRACT_MODE=$mode TEST_FILE=$test_file python main.py
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
      SUCCESS=$((SUCCESS + 1))
      echo "✅ [${COUNT}/${TOTAL}] 成功: ${filename} + ${mode}"
    else
      FAIL=$((FAIL + 1))
      echo "❌ [${COUNT}/${TOTAL}] 失败 (exit $EXIT): ${filename} + ${mode}"
    fi

    if [ $COUNT -lt $TOTAL ]; then
      echo "等待 30s 再跑下一个..."
      sleep 30
    fi
  done
done

echo ""
echo "=================================================="
echo "  全部完成: 成功 ${SUCCESS}/${TOTAL}，失败 ${FAIL}/${TOTAL}"
echo "  结束时间: $(date)"
echo "结果目录:"
echo "  extract:    $(ls Result/extract_result/extract_result_*.json 2>/dev/null | wc -l) 个"
echo "  onboarding: $(ls Result/onboarding_result/onboarding_*.md 2>/dev/null | wc -l) 个"
echo "=================================================="
