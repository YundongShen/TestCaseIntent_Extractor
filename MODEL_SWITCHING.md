# Model Switching Guide

## Quick Start

### Use 7B model (default)
```bash
python main.py
# or explicitly
export MODEL_TYPE=7b && python main.py
```

### Use Qwen 3.5-27B
```bash
export MODEL_TYPE=qwen && python main.py
```

### Use V3 model (multi-GPU)
```bash
export MODEL_TYPE=v3
# Single GPU
python main.py

# Multi-GPU distributed
export WORLD_SIZE=4
torchrun --nproc_per_node 4 main.py
```

## Supported Models

| Model | MODEL_TYPE | Size | Precision | Status |
|-------|-----------|------|-----------|--------|
| DeepSeek-7B-Chat | `7b` | 14GB | FP16 | Ready |
| Qwen-3.5-27B | `qwen` | 52GB | BF16 | Ready |
| DeepSeek-V3 (MoE) | `v3` | 1.3TB | FP8 | Requires multi-GPU |

## File Layout

```
model/
├── model_config_7b.py           # 7B config
├── model_config_qwen.py         # Qwen config
├── model_config_v3.py           # V3 config proxy
├── model_config_v3_official.py  # V3 official implementation
├── model_config_v3_mp.py        # V3 multi-GPU support
├── inference_service.py         # Inference service (supports all models)
└── models/
    ├── deepseek-llm-7b-chat/
    └── models--Qwen--Qwen3.5-27B/
```

## Data Flow

```
main.py
  | (reads MODEL_TYPE env var)
  |-> model_config_7b.py    (MODEL_TYPE=7b)
  |-> model_config_qwen.py  (MODEL_TYPE=qwen)
  |-> model_config_v3.py    (MODEL_TYPE=v3)
  |
inference_service.py
  | (routes to the appropriate inference method)
  |-> _infer_7b()
  |-> _infer_qwen()
  |-> _infer_v3()
  |
ObjectExtractor / ActivityExtractor / GoalExtractor
```

## Testing Different Models

### Quick Qwen test
```bash
sbatch test_qwen_load.slurm
cat test_qwen_load_*.out
```

### Full extraction comparison
```bash
# Run with 7B
python main.py

# Run with Qwen
export MODEL_TYPE=qwen && python main.py

# Compare results
diff Result/extract_result/*.json
```

## Configuration Details

### 7B model
- **File**: `model/model_config_7b.py`
- **Inference**: `load_deepseek_model()` → `_infer_7b()`
- **Characteristics**: Fast, low VRAM, FP16
- **Recommended for**: Rapid prototyping, single-GPU inference

### Qwen model
- **File**: `model/model_config_qwen.py`
- **Inference**: `load_qwen_model()` → `_infer_qwen()`
- **Characteristics**: Better quality, 27B parameters, BF16, Flash Attention
- **Recommended for**: Production-quality results

### V3 model
- **File**: `model/model_config_v3.py` + `model_config_v3_official.py`
- **Inference**: `load_deepseek_model()` → `_infer_v3()`
- **Characteristics**: Strongest capability, 671B MoE, FP8 quantization
- **Recommended for**: Highest quality, requires multi-GPU cluster

## Adding a New Model

1. Create `model/model_config_<name>.py`:
   ```python
   MODEL_CONFIG = {
       "model_id": "huggingface_model_id",
       "local_path": Path(...),
       "model_name": "Display Name",
       "quantization": "FP16",
       "parameters": "Size",
       "context_length": 2048,
   }

   def load_<name>_model(device=None):
       # return (model, tokenizer)
       pass
   ```

2. Add a branch in `main.py`:
   ```python
   elif MODEL_TYPE == "<name>":
       from model.model_config_<name> import set_seed, MODEL_CONFIG
   ```

3. Add inference method in `inference_service.py`:
   ```python
   elif model_type == "<name>":
       response = self._infer_<name>(model, tokenizer, prompt, max_tokens, device)

   def _infer_<name>(self, model, tokenizer, prompt, max_tokens, device):
       pass
   ```

## FAQ

**Q: How do I switch between 7B and Qwen quickly?**
```bash
export MODEL_TYPE=7b    # use 7B
export MODEL_TYPE=qwen  # use Qwen
```

**Q: Does Qwen support multi-GPU inference?**
A: Currently single-GPU only. For multi-GPU, consider integrating vLLM (tensor parallel), SGLang, or TensorRT-LLM.

**Q: How do I measure quality differences between models?**
```bash
MODEL_TYPE=7b python main.py
MODEL_TYPE=qwen python main.py
ls -la Result/extract_result/
diff Result/extract_result/*.json
```

**Q: Do I need to retrain after switching models?**
A: No. The system uses pretrained models for zero-shot extraction.

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `MODEL_TYPE` | `7b`, `qwen`, `v3` | Select model |
| `WORLD_SIZE` | integer | Number of GPUs for V3 |
| `CUDA_VISIBLE_DEVICES` | GPU indices | Restrict visible GPUs |

Examples:
```bash
# Run Qwen on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_TYPE=qwen python main.py

# Run V3 on 8 GPUs
export WORLD_SIZE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 main.py
```

## Workflow Comparison

### 7B (baseline)
```
Fast | Low VRAM | Lower quality
```

### Qwen (recommended balance)
```
Moderate speed | Moderate VRAM | High quality
```

### V3 (highest quality)
```
Slow | Very high VRAM | Best quality
```

---
Updated: 2026-04-20
