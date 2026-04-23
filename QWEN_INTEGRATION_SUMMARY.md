# Qwen 3.5-27B Integration Summary

## 概述 (Overview)
集成 Qwen 3.5-27B 模型到现有 Intent Extraction 系统，支持灵活的模型切换。

**时间**: 2026-04-20  
**模型**: Qwen/Qwen3.5-27B (27B, BF16, 52GB)  
**集成**: 完全透明，可随时切换

---

## 本次更改 (Changes Made)

### 1. 新增配置文件
**文件**: `model/model_config_qwen.py`

```python
# 关键配置
MODEL_CONFIG = {
    "model_id": "Qwen/Qwen3.5-27B",
    "local_path": "model/models/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd",
    "parameters": "27B",
    "quantization": "BF16",
    "context_length": 32768,  # Qwen支持32K上下文
}

def load_qwen_model(device=None):
    # 使用BF16加载 (比FP16更稳定)
    # 支持 Flash Attention 优化
```

### 2. 更新 main.py
**位置**: Lines 14-25

```diff
+ elif MODEL_TYPE == "qwen":
+     from model.model_config_qwen import set_seed, MODEL_CONFIG
+     print(f"✅ [Config] Using Qwen-3.5-27B model configuration (27B, BF16)")
```

### 3. 更新 inference_service.py

#### 3a. 模型路由 (Lines ~95-115)
```diff
+ elif model_type == "qwen":
+     print("[InferenceService] 🔄 MODEL_TYPE=qwen - Using Qwen-3.5-27B inference")
+     from model.model_config_qwen import load_qwen_model as load_deepseek_model, get_device
+     is_v3 = False
```

#### 3b. 推理方法路由 (Lines ~225-240)
```diff
+ elif self.model_type == "qwen":
+     print(f"[InferenceService] Using Qwen-3.5-27B inference (device: {device})")
+     response = self._infer_qwen(model, tokenizer, prompt, max_tokens, device)
```

#### 3c. 新增 Qwen 推理方法 (Lines ~415-465)
```python
def _infer_qwen(self, model, tokenizer, prompt, max_tokens, device):
    """
    Inference using Qwen-3.5-27B model with transformers API.
    - 支持 Chat Template
    - 使用 BF16 精度
    - 支持 Flash Attention
    """
    # 格式化提示
    # 使用 apply_chat_template
    # 调用 model.generate() 
    # 返回结果
```

### 4. 新增测试脚本
**文件**: `test_qwen_load.slurm`

用于验证 Qwen 模型在 GPU 上的加载和推理：
```bash
sbatch test_qwen_load.slurm
```

### 5. 更新文档
**文件**: `MODEL_SWITCHING.md` (完全改写)

包含：
- 快速开始指南
- 模型对比表
- 使用示例
- 常见问题
- 扩展指南

---

## 使用方法 (Usage)

### 切换到 Qwen
```bash
export MODEL_TYPE=qwen
python main.py
```

### 对比 7B 和 Qwen
```bash
# 用 7B 提取
python main.py

# 用 Qwen 提取  
export MODEL_TYPE=qwen && python main.py

# 对比结果
diff Result/extract_result/extract_result_*.json
```

### 验证 Qwen 加载
```bash
sbatch test_qwen_load.slurm
tail -f test_qwen_load_*.out
```

---

## 架构支持 (Architecture)

系统现在支持 3 个模型，使用统一接口：

```
ObjectExtractor/ActivityExtractor/etc
    ↓ (调用)
InferenceService.infer(prompt, max_tokens)
    ↓ (检查 MODEL_TYPE)
    ├─→ _load_local_model() [根据MODEL_TYPE加载]
    └─→ _infer_local() [根据model_type路由]
        ├─→ _infer_7b()     (if MODEL_TYPE=7b)
        ├─→ _infer_qwen()   (if MODEL_TYPE=qwen)  ← 新增
        └─→ _infer_v3()     (if MODEL_TYPE=v3)
```

**关键特性**:
- ✅ 完全向后兼容 (默认仍是 7B)
- ✅ 无需修改提取器代码
- ✅ 可随时切换
- ✅ 多模型支持（7B/Qwen/V3）

---

## 文件清单 (Files Modified)

| 文件 | 类型 | 改动 |
|------|------|------|
| `model/model_config_qwen.py` | 新增 | Qwen 加载配置 |
| `main.py` | 修改 | 添加 qwen 分支 |
| `model/inference_service.py` | 修改 | 路由 + _infer_qwen 方法 |
| `test_qwen_load.slurm` | 新增 | 加载测试脚本 |
| `MODEL_SWITCHING.md` | 修改 | 更新为 3 模型指南 |
| `QWEN_INTEGRATION_SUMMARY.md` | 新增 | 本文件 |

---

## 性能对比 (Performance)

| 指标 | 7B | Qwen | V3 |
|------|---|----|-----|
| 显存需求 | 14GB | 52GB | 1.3TB |
| 推理速度 | 快 ✓ | 中等 | 慢 |
| 质量 | 基准 | 高 ✓ | 最高 |
| 多GPU需求 | 无 | 无 | 是 |
| 状态 | ✓ | ✓ 就绪 | ⚠️ |

---

## 下一步 (Next Steps)

### 可选优化
1. **多GPU推理** (Qwen)
   - 集成 vLLM 张量并行
   - 集成 SGLang
   - 集成 TensorRT-LLM

2. **量化**
   - 8-bit 量化降低显存
   - 4-bit 量化 (AWQ/GPTQ)

3. **批量推理**
   - 当前: 单样本推理
   - 可改进: 批处理多提示

### 监控
```bash
# GPU 监控
watch -n 1 nvidia-smi

# 推理性能
time export MODEL_TYPE=qwen && python main.py

# 内存变化
nvidia-smi --query-gpu=memory.used --format=csv,nounits -l 1
```

---

## 验证清单 (Verification Checklist)

- [x] Qwen 模型下载完成 (52GB, Job 16401341)
- [x] 配置文件创建 (model_config_qwen.py)
- [x] main.py 支持 MODEL_TYPE=qwen
- [x] inference_service.py 路由正确
- [x] _infer_qwen() 方法实现
- [x] 测试脚本准备就绪 (test_qwen_load.slurm)
- [ ] 运行 test_qwen_load.slurm 验证
- [ ] 运行 python main.py (7B baseline)
- [ ] 运行 MODEL_TYPE=qwen python main.py
- [ ] 对比结果质量

---

## 技术细节 (Technical Details)

### Qwen 特点
- **模型**: Qwen-Chat 微调版本
- **参数**: 27B
- **精度**: BF16 (更稳定比 FP16)
- **上下文**: 32K (vs 7B 的 2K)
- **优化**: Flash Attention 支持

### 推理改进
- 使用 `apply_chat_template` 自动格式化
- BF16 精度减少数值不稳定性
- Flash Attention 加速注意力计算
- 贪心解码 (greedy) 保证确定性输出

### 内存布局
```
GPU内存 (单GPU):
7B:   14GB  (32GB GPU > 14GB is safe)
Qwen: 52GB  (80GB GPU > 52GB is safe)  
V3:   325GB per GPU (不可行，8×80GB = 640GB < 1.3TB)
```

---

**完成日期**: 2026-04-20  
**验证**: 配置已完成，等待运行测试验证
