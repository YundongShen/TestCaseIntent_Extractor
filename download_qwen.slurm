#!/bin/bash
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=download_qwen
#SBATCH --output=download_qwen_%j.out

cd /proj/test_extender/Intent_Implementation_0327

source gpu_env/bin/activate

echo "======================================"
echo "Downloading Qwen 3.5-27B from HuggingFace"
echo "======================================"
echo "Model: Qwen/Qwen3.5-27B"
echo "Size: ~54GB (BF16)"
echo "Destination: model/models/Qwen3.5-27B"
echo "======================================"
echo

# Create destination directory
mkdir -p model/models

# Download using huggingface_hub
python << 'PYTHON_SCRIPT'
from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen3.5-27B"
cache_dir = "/proj/test_extender/Intent_Implementation_0327/model/models"

print(f"[Download] Installing Qwen 3.5-27B...")
print(f"[Download] Model: {model_id}")
print(f"[Download] Destination: {cache_dir}")
print()

try:
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"\n[Download] ✅ Successfully downloaded to: {local_dir}")
    print(f"[Download] Model ready for inference")
    
    # List downloaded files
    import subprocess
    result = subprocess.run(f"du -sh {cache_dir}/*", shell=True, capture_output=True, text=True)
    print(f"\n[Download] Downloaded files:")
    print(result.stdout)
    
except Exception as e:
    print(f"[Download] ❌ Error: {e}")
    import traceback
    traceback.print_exc()

PYTHON_SCRIPT

echo
echo "======================================"
echo "✅ Download completed"
echo "======================================"
