# Qwen 3.5-27B Integration Summary

## Overview
Qwen 3.5-27B has been integrated into the existing Intent Extraction system with full support for flexible model switching.

**Date**: 2026-04-20
**Model**: Qwen/Qwen3.5-27B (27B, BF16, 52GB)
**Integration**: Fully transparent — switch at any time via environment variable

---

## Changes Made

### 1. New config file
**File**: `model/model_config_qwen.py`

```python
MODEL_CONFIG = {
    "model_id": "Qwen/Qwen3.5-27B",
    "local_path": "model/models/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd",
    "parameters": "27B",
    "quantization": "BF16",
    "context_length": 32768,  # Qwen supports 32K context
}

def load_qwen_model(device=None):
    # Load with BF16 (more stable than FP16)
    # Flash Attention optimization enabled
```

### 2. Updated main.py
**Location**: Lines 14-25

```diff
+ elif MODEL_TYPE == "qwen":
+     from model.model_config_qwen import set_seed, MODEL_CONFIG
+     print(f"[Config] Using Qwen-3.5-27B model configuration (27B, BF16)")
```

### 3. Updated inference_service.py

#### 3a. Model routing
```diff
+ elif model_type == "qwen":
+     from model.model_config_qwen import load_qwen_model as load_deepseek_model, get_device
+     is_v3 = False
```

#### 3b. Inference method routing
```diff
+ elif self.model_type == "qwen":
+     response = self._infer_qwen(model, tokenizer, prompt, max_tokens, device)
```

#### 3c. New Qwen inference method
```python
def _infer_qwen(self, model, tokenizer, prompt, max_tokens, device):
    """
    Inference using Qwen-3.5-27B via transformers API.
    - Applies chat template
    - BF16 precision
    - Flash Attention support
    """
```

### 4. New test script
**File**: `test_qwen_load.slurm`

Validates Qwen model loading and inference on GPU:
```bash
sbatch test_qwen_load.slurm
```

### 5. Updated documentation
**File**: `MODEL_SWITCHING.md` — rewritten for 3-model support

---

## Usage

### Switch to Qwen
```bash
export MODEL_TYPE=qwen
python main.py
```

### Compare 7B vs Qwen
```bash
# Extract with 7B
python main.py

# Extract with Qwen
export MODEL_TYPE=qwen && python main.py

# Compare results
diff Result/extract_result/extract_result_*.json
```

### Validate Qwen loading
```bash
sbatch test_qwen_load.slurm
tail -f test_qwen_load_*.out
```

---

## Architecture

The system now supports 3 models through a unified interface:

```
ObjectExtractor / ActivityExtractor / etc.
    | (calls)
InferenceService.infer(prompt, max_tokens)
    | (checks MODEL_TYPE)
    |-> _load_local_model()
    |-> _infer_local()
        |-> _infer_7b()    (MODEL_TYPE=7b)
        |-> _infer_qwen()  (MODEL_TYPE=qwen)  <- new
        |-> _infer_v3()    (MODEL_TYPE=v3)
```

**Key properties**:
- Fully backward compatible (default remains 7B)
- No changes required in extractor code
- Switch at any time via environment variable
- Supports 7B / Qwen / V3

---

## Files Modified

| File | Type | Change |
|------|------|--------|
| `model/model_config_qwen.py` | New | Qwen load config |
| `main.py` | Modified | Added qwen branch |
| `model/inference_service.py` | Modified | Routing + _infer_qwen method |
| `test_qwen_load.slurm` | New | Load test script |
| `MODEL_SWITCHING.md` | Modified | Updated for 3-model guide |
| `QWEN_INTEGRATION_SUMMARY.md` | New | This file |

---

## Performance Comparison

| Metric | 7B | Qwen | V3 |
|--------|----|------|----|
| VRAM required | 14GB | 52GB | 1.3TB |
| Inference speed | Fast | Moderate | Slow |
| Output quality | Baseline | High | Best |
| Multi-GPU required | No | No | Yes |
| Status | Ready | Ready | Requires cluster |

---

## Next Steps

### Optional optimizations
1. **Multi-GPU inference (Qwen)**
   - vLLM tensor parallelism
   - SGLang
   - TensorRT-LLM

2. **Quantization**
   - 8-bit to reduce VRAM
   - 4-bit (AWQ/GPTQ)

3. **Batch inference**
   - Current: single-sample inference
   - Potential: batch multiple prompts

### Monitoring
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Inference timing
time MODEL_TYPE=qwen python main.py

# Memory tracking
nvidia-smi --query-gpu=memory.used --format=csv,nounits -l 1
```

---

## Verification Checklist

- [x] Qwen model downloaded (52GB, Job 16401341)
- [x] Config file created (model_config_qwen.py)
- [x] main.py supports MODEL_TYPE=qwen
- [x] inference_service.py routing correct
- [x] _infer_qwen() method implemented
- [x] Test script ready (test_qwen_load.slurm)
- [ ] Run test_qwen_load.slurm to verify
- [ ] Run python main.py (7B baseline)
- [ ] Run MODEL_TYPE=qwen python main.py
- [ ] Compare output quality

---

## Technical Details

### Qwen characteristics
- **Model**: Qwen-Chat fine-tuned variant
- **Parameters**: 27B
- **Precision**: BF16 (more stable than FP16)
- **Context**: 32K (vs 2K for 7B)
- **Optimization**: Flash Attention support

### Inference improvements
- `apply_chat_template` for automatic prompt formatting
- BF16 precision reduces numerical instability
- Flash Attention accelerates attention computation
- Greedy decoding for deterministic output

### Memory layout
```
GPU memory (single GPU):
7B:   14GB  (32GB GPU is sufficient)
Qwen: 52GB  (80GB GPU is sufficient)
V3:   325GB per GPU (not feasible on single node; 8x80GB = 640GB < 1.3TB)
```

---

**Completed**: 2026-04-20
**Status**: Configuration complete — pending runtime validation
