#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=02:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --job-name=qwen_extract_80gb
#SBATCH --output=qwen_extract_80gb_%j.out

cd /proj/test_extender/Intent_Implementation_0327

source gpu_env/bin/activate

echo "════════════════════════════════════════════════════════════════"
echo "        Qwen 3.5-27B Extraction Run (80GB GPU)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
export MODEL_TYPE=qwen
export EXTRACT_MODE=independent
echo "MODEL_TYPE=$MODEL_TYPE"
echo "EXTRACT_MODE=$EXTRACT_MODE"

echo ""
echo "[ Running Intent Extraction Pipeline ]"
python main.py

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                  Task Completed"
echo "════════════════════════════════════════════════════════════════"
echo "Finish time: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "[ Result ]"
ls -lh Result/extract_result/*.json 2>/dev/null | tail -1 | awk '{print $NF, "(" $5 ")"}'
