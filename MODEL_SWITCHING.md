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

### Use V3 model
```bash
export MODEL_TYPE=v3
python main.py
```

## Supported Models

| Model | MODEL_TYPE | Size | Precision | Status |
|-------|-----------|------|-----------|--------|
| DeepSeek-7B-Chat | `7b` | 14GB | FP16 | Ready |
| Qwen-3.5-27B | `qwen` | 52GB | BF16 | Ready |
| DeepSeek-V3 (MoE) | `v3` | 1.3TB | FP8 | Needs a large single GPU (device_map=auto) |

## File Layout

```
model/
├── model_config_7b.py           # 7B config
├── model_config_qwen.py         # Qwen config
├── model_config_v3.py           # V3 config
├── inference_service.py         # local inference (all models go through _infer_local)
├── api_inference_service.py     # Gemini API backend
└── models/
    ├── deepseek-llm-7b-chat/
    └── models--Qwen--Qwen3.5-27B/
```

## Data Flow

```
main.py
  | (reads MODEL_TYPE env var, INFERENCE_BACKEND for local vs API)
  |-> model_config_7b.py    (MODEL_TYPE=7b)
  |-> model_config_qwen.py  (MODEL_TYPE=qwen)
  |-> model_config_v3.py    (MODEL_TYPE=v3)
  |
inference_service.py
  | _infer_local() loads whichever MODEL_CONFIG was set and runs it
  |
ObjectExtractor / ActivityExtractor / GoalExtractor
```

## Testing Different Models

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
- **Characteristics**: Fast, low VRAM, FP16
- **Recommended for**: Rapid prototyping, single-GPU inference

### Qwen model
- **File**: `model/model_config_qwen.py`
- **Characteristics**: Better quality, 27B parameters, BF16, Flash Attention
- **Recommended for**: Production-quality results

### V3 model
- **File**: `model/model_config_v3.py`
- **Characteristics**: Strongest capability, 671B MoE, FP8 quantization
- **Recommended for**: Highest quality, needs a large-memory GPU

## Adding a New Model

1. Create `model/model_config_<name>.py` with a `MODEL_CONFIG` dict (`model_id`,
   `local_path`, `model_name`, `quantization`, ...) and a `set_seed()`.

2. Add a branch in `main.py`:
   ```python
   elif MODEL_TYPE == "<name>":
       from model.model_config_<name> import set_seed, MODEL_CONFIG
   ```

3. `InferenceService._load_local_model()` (in `inference_service.py`) picks the
   loader by matching `model_id` — add a branch there if the new model isn't
   Qwen- or DeepSeek-based.

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
| `INFERENCE_BACKEND` | `local`, `api` | Local weights vs. Gemini API |
| `EXTRACT_MODE` | `independent`, `combined`, `chain` | Extraction prompting mode |
| `CUDA_VISIBLE_DEVICES` | GPU indices | Restrict visible GPUs |

Example:
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE=qwen python main.py
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
Updated: 2026-09-21
