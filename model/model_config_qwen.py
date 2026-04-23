"""Qwen-3.5-27B Model Configuration
Model: Qwen/Qwen3.5-27B, Size: 52 GB, Type: BF16, Source: HuggingFace
"""

from pathlib import Path

MODEL_CONFIG = {
    "model_id": "Qwen/Qwen3.5-27B",
    "local_path": Path(__file__).parent / "models" / "models--Qwen--Qwen3.5-27B" / "snapshots" / "b7ca741b86de18df552fd2cc952861e04621a4bd",
    "model_name": "Qwen-3.5-27B",
    "quantization": "BF16",
    "parameters": "27B",
    "context_length": 32768,
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
        print("🚀 [Device] Using NVIDIA CUDA (Qwen-27B Model)")
        return "cuda"
    else:
        print("⚙️  [Device] Using CPU")
        return "cpu"

def load_qwen_model(device=None, quantize_8bit=False):
    """Load Qwen-3.5-27B model. Downloads from HF if needed. Returns: (model, tokenizer)"""
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
    
    print(f"[Loading] Loading to device: {device}...")
    
    try:
        # Load tokenizer - Qwen uses official tokenizer
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
        
        # Load model - Qwen uses BF16 for better stability
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu" if device == "cpu" else device,
            trust_remote_code=True,
            attn_implementation="sdpa"  # Use scaled_dot_product instead of flash_attention
        )
        
        if device == "cpu":
            model = model.cpu()
        
        print("[Success] Qwen-3.5-27B Model loaded successfully (BF16)")
        return model, tokenizer
    
    except Exception as e:
        print(f"[Error] Model loading failed: {e}")
        raise

if __name__ == "__main__":
    print("Qwen-3.5-27B Model Configuration")
    print(f"Path: {get_model_path()}")
