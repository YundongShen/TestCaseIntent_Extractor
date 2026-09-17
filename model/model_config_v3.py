"""DeepSeek-V3 Model Configuration
Model: DeepSeek-V3, Size: 671B MoE, Type: FP8 Quantized, Source: HuggingFace
NOTE: Requires multiple A100-80GB GPUs for inference due to MoE architecture
"""

from pathlib import Path

MODEL_CONFIG = {
    "model_id": "deepseek-ai/DeepSeek-V3",
    "local_path": Path(__file__).parent / "models" / "deepseek-v3",
    "model_name": "DeepSeek-V3",
    "quantization": "FP8",
    "parameters": "671B (MoE)",
    "context_length": 8192,
}

def get_model_path():
    """Get local model path."""
    return MODEL_CONFIG["local_path"]

def set_seed(seed=42):
    """Set random seeds for reproducibility across all runs and jobs."""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Seed] Reproducibility seed set to {seed}")

def get_device():
    """Auto-detect optimal device: CUDA > CPU"""
    import torch
    
    if torch.cuda.is_available():
        print("🚀 [Device] Using NVIDIA CUDA (V3 Model with device_map='auto')")
        return "cuda"
    else:
        print("⚙️  [Device] Using CPU (NOT RECOMMENDED for V3)")
        return "cpu"

def load_deepseek_model(device=None, quantize_8bit=True):
    """Load DeepSeek-V3 model. Downloads from HF if needed. Returns: (model, tokenizer)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from pathlib import Path
    
    if device is None:
        device = get_device()
    
    model_id = MODEL_CONFIG["model_id"]
    local_path = get_model_path()
    
    # Determine which path to load from
    if Path(local_path).exists():
        load_path = str(local_path)
        print(f"[Loading] Using local model: {load_path}")
    else:
        load_path = model_id
        print(f"[Loading] Local model not found. Will download from HuggingFace: {model_id}")
    
    print(f"[Loading] Loading V3 model to device with device_map='auto'...")
    print(f"[Loading] Model size: 671B (MoE) - May take 3-5 minutes with FP8 quantization")
    
    try:
        # Load tokenizer with fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        except (ValueError, ImportError) as e:
            print(f"[Tokenizer] First attempt failed: {e}. Trying with use_fast=False...")
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, use_fast=False)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[Tokenizer] eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"[Tokenizer] pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        
        # Load model - V3 uses FP16 with device_map='auto' for multi-GPU
        print("[Loading] Initializing V3 model with FP16 and auto device mapping...")
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16,      # Use FP16 instead of quantization
            device_map="auto",              # Automatically distribute across available GPUs
            trust_remote_code=True,
            attn_implementation="eager"     # Use eager attention instead of flash-attn
        )
        
        print("[Success] V3 Model loaded successfully (FP16, device_map='auto', eager attention)")
        return model, tokenizer
    
    except Exception as e:
        print(f"[Error] Model loading failed: {e}")
        print(f"[Help] Ensure you have:")
        print(f"  - At least 2 A100-80GB GPUs available")
        print(f"  - bitsandbytes library installed: pip install bitsandbytes")
        raise

if __name__ == "__main__":
    print("DeepSeek-V3 Model Configuration")
    print(f"Path: {get_model_path()}")
