"""DeepSeek Model Configuration
Model: DeepSeek-LLM-7B-Chat, Size: 14 GB, Source: HuggingFace
"""

from pathlib import Path

MODEL_CONFIG = {
    "model_id": "deepseek-ai/deepseek-llm-7b-chat",
    "local_path": Path(__file__).parent / "models" / "deepseek-llm-7b-chat",
    "model_name": "DeepSeek-LLM-7B-Chat",
    "quantization": "FP16",
    "parameters": "7B",
    "context_length": 2048,
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
        print("🚀 [Device] Using NVIDIA CUDA (RTX 6000)")
        return "cuda"
    else:
        print("⚙️  [Device] Using CPU")
        return "cpu"

def load_deepseek_model(device=None, quantize_8bit=True):
    """Load Llama model. Downloads from HF if needed. Returns: (model, tokenizer)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from pathlib import Path
    
    if device is None:
        device = get_device()
    
    model_id = MODEL_CONFIG["model_id"]  # NousResearch/Llama-2-13b-hf
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
        # Load tokenizer with fallback for Llama models
        # Llama models may have tokenizer.model issues, try with use_fast=False
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        except (ValueError, ImportError) as e:
            # Fallback for SentencePiece/tiktoken issues with Llama
            print(f"[Tokenizer] First attempt failed: {e}. Trying with use_fast=False...")
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, use_fast=False)
        
        # Set pad token for Llama
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[Tokenizer] eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"[Tokenizer] pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        
        # Load model - check if V3 for special handling
        is_v3 = "v3" in model_id.lower()
        
        if is_v3:
            print("[Loading] Detected V3 model - using FP16 with device_map='auto'")
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map="auto",              # Automatically distribute across GPUs
                trust_remote_code=True,
                attn_implementation="eager"    # Use eager attention instead of flash-attn
            )
            print("[Success] V3 Model loaded successfully (FP16, device_map='auto', eager attention)")
        else:
            print("[Loading] Detected 7B model - using FP16")
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map="cpu" if device == "cpu" else device,
                trust_remote_code=True
            )
            
            if device == "cpu":
                model = model.cpu()
            
            print("[Success] 7B Model loaded successfully (FP16)")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"[Error] Model loading failed: {e}")
        raise

if __name__ == "__main__":
    print("DeepSeek Model Configuration")
    print(f"Path: {get_model_path()}")
