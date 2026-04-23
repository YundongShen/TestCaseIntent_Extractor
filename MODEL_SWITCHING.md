# 模型切换说明 (Model Switching Guide)

## 快速开始 (Quick Start)

### 使用 7B 模型 (Default)
```bash
python main.py
# 或显式指定
export MODEL_TYPE=7b && python main.py
```

### 使用 Qwen 3.5-27B 模型
```bash
export MODEL_TYPE=qwen && python main.py
```

### 使用 V3 模型 (多GPU)
```bash
export MODEL_TYPE=v3
# For single GPU
python main.py

# For multi-GPU distributed
export WORLD_SIZE=4
torchrun --nproc_per_node 4 main.py
```

## 支持的模型

| 模型 | MODEL_TYPE | 大小 | 类型 | 状态 |
|------|-----------|------|------|------|
| DeepSeek-7B-Chat | `7b` | 14GB | FP16 | ✅ 就绪 |
| Qwen-3.5-27B | `qwen` | 52GB | BF16 | ✅ 就绪 (新增) |
| DeepSeek-V3 (MoE) | `v3` | 1.3TB | FP8 | ⚠️ 需要多GPU |

## 文件位置

```
model/
├── model_config_7b.py           # 7B 配置
├── model_config_qwen.py         # Qwen 配置 (新增)
├── model_config_v3.py           # V3 配置代理
├── model_config_v3_official.py  # V3 官方实现
├── model_config_v3_mp.py        # V3 多GPU支持
├── inference_service.py         # 推理服务 (已更新支持Qwen)
└── models/
    ├── deepseek-llm-7b-chat/    # 7B 模型
    └── models--Qwen--Qwen3.5-27B/ # Qwen 模型 (新增)
```

## 数据流

```
main.py 
  ↓ (读取 MODEL_TYPE 环境变量)
  ├─→ model_config_7b.py (if MODEL_TYPE=7b)
  ├─→ model_config_qwen.py (if MODEL_TYPE=qwen) 
  └─→ model_config_v3.py (if MODEL_TYPE=v3)
  ↓
inference_service.py
  ↓ (路由到相应推理方法)
  ├─→ _infer_7b()
  ├─→ _infer_qwen() (新增)
  └─→ _infer_v3()
  ↓
ObjectExtractor / ActivityExtractor / GoalExtractor (自动使用任意模型)
```

## 测试不同模型

### 快速测试 Qwen (推荐)
```bash
# 提交 SLURM 任务
sbatch test_qwen_load.slurm

# 查看输出
cat test_qwen_load_*.out
```

### 完整提取流程对比
```bash
# 用 7B 运行完整提取
python main.py

# 用 Qwen 运行完整提取  
export MODEL_TYPE=qwen && python main.py

# 对比结果
diff Result/extract_result/*.json
```

## 配置详情

### 7B 模型配置
- **文件**: `model/model_config_7b.py`
- **推理函数**: `load_deepseek_model()` → `_infer_7b()`
- **特点**: 快速、低显存、FP16精度
- **推荐**: 快速原型、单GPU推理

### Qwen 模型配置  
- **文件**: `model/model_config_qwen.py`
- **推理函数**: `load_qwen_model()` → `_infer_qwen()`
- **特点**: 更好质量、27B参数、BF16精度、Flash Attention支持
- **推荐**: 生产质量结果、需要更好的理解能力

### V3 模型配置
- **文件**: `model/model_config_v3.py` + `model_config_v3_official.py`
- **推理函数**: `load_deepseek_model()` → `_infer_v3()`
- **特点**: 最强能力、671B MoE、FP8量化
- **推荐**: 最高质量、需要多GPU集群

## 如何添加新模型

1. 创建配置文件 `model/model_config_<modelname>.py`:
   ```python
   MODEL_CONFIG = {
       "model_id": "huggingface_model_id",
       "local_path": Path(...),
       "model_name": "Display Name",
       "quantization": "FP16",
       "parameters": "Size",
       "context_length": 2048,
   }
   
   def load_<modelname>_model(device=None):
       # 返回 (model, tokenizer)
       pass
   ```

2. 在 `main.py` 添加支持:
   ```python
   elif MODEL_TYPE == "<modelname>":
       from model.model_config_<modelname> import set_seed, MODEL_CONFIG
   ```

3. 在 `inference_service.py` 添加推理方法:
   ```python
   elif model_type == "<modelname>":
       from model.model_config_<modelname> import load_<modelname>_model
       response = self._infer_<modelname>(model, tokenizer, prompt, max_tokens, device)
   
   def _infer_<modelname>(self, model, tokenizer, prompt, max_tokens, device):
       # 推理实现
       pass
   ```

## 常见问题

**Q: 如何在7B和Qwen之间快速切换?**
A: 只需设置环境变量:
```bash
export MODEL_TYPE=7b   # 使用7B
export MODEL_TYPE=qwen # 使用Qwen
```

**Q: Qwen是否支持多GPU推理?**
A: 当前使用单GPU inference。如需多GPU并行推理，可以集成：
- vLLM (带张量并行)
- SGLang
- TensorRT-LLM

**Q: 如何衡量模型质量差异?**
A: 运行完整提取并对比结果:
```bash
# 用不同模型生成结果
MODEL_TYPE=7b python main.py
MODEL_TYPE=qwen python main.py

# 查看生成结果目录
ls -la Result/extract_result/
diff Result/extract_result/*.json
```

**Q: 切换模型后需要重新训练吗?**
A: 不需要。系统使用预训练模型进行零样本提取/生成。

## 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `MODEL_TYPE` | `7b`, `qwen`, `v3` | 选择模型 |
| `WORLD_SIZE` | 整数 | V3多GPU数量 |
| `CUDA_VISIBLE_DEVICES` | GPU索引 | 指定使用的GPU |

示例:
```bash
# 使用GPU 0-3运行Qwen
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_TYPE=qwen python main.py

# 使用8张GPU运行V3  
export WORLD_SIZE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 main.py
```

## 工作流对比

### 7B (基准)
```
快速 ✓ | 低显存 ✓ | 低质量 ✗
```

### Qwen (推荐平衡)
```
较快 ✓ | 适中显存 ✓ | 高质量 ✓
```

### V3 (最高质量)
```
慢 ✗ | 高显存需求 ✗ | 最高质量 ✓
```

---
更新时间: 2026-04-20
作者: AI Assistant

