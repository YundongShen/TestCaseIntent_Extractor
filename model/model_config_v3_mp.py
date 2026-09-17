"""
DeepSeek V3 Model - Multi-GPU Model Parallel Support.

This module provides multi-GPU distributed inference for V3 using torch.distributed.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Tuple, List
from safetensors.torch import load_model

# Add V3 inference path
V3_MODEL_PATH = Path(__file__).parent / "models" / "deepseek-v3"
V3_INFERENCE_PATH = V3_MODEL_PATH / "inference"
sys.path.insert(0, str(V3_INFERENCE_PATH))

from model import Transformer, ModelArgs


def get_device():
    """Get device based on CUDA availability."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_distributed():
    """Initialize torch.distributed for multi-GPU inference."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        print(f"[V3 MP] Initializing distributed (world_size={world_size}, rank={rank}, local_rank={local_rank})")
        dist.init_process_group("nccl")
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = "cuda"
    else:
        device = "cpu"
    
    return world_size, rank, local_rank, device


def load_deepseek_model_v3_mp(device: str = "cuda", **kwargs) -> Tuple:
    """
    Load DeepSeek V3 with multi-GPU model parallelism support.
    
    Args:
        device: 'cuda' or 'cpu'
        **kwargs: Additional args (unused)
    
    Returns:
        (model, tokenizer) tuple
    """
    
    # Initialize distributed if needed
    world_size, rank, local_rank, device = init_distributed()
    
    # Helper function for rank-0-only printing
    def rank_print(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    rank_print("\n" + "="*80)
    rank_print(f"[V3 MP] Loading V3 with {world_size}-GPU model parallelism (rank={rank})")
    rank_print("="*80)
    
    # Setup torch defaults
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    
    # Load config
    config_path = V3_MODEL_PATH / "inference" / "configs" / "config_671B.json"
    if not config_path.exists():
        config_options = [
            V3_MODEL_PATH / "inference" / "configs" / "config_671B.json",
            V3_MODEL_PATH / "inference" / "configs" / "config_236B.json",
        ]
        for opt in config_options:
            if opt.exists():
                config_path = opt
                print(f"[V3 MP] Using config: {opt.name}")
                break
    
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    
    print(f"[V3 MP] Model config: {args.n_layers} layers, {args.n_routed_experts} experts")
    
    # Create model
    print(f"[V3 MP] Creating model on {device}...")
    with torch.device(device):
        model = Transformer(args)
    model.eval()
    
    # Load weights for this rank
    weights_dir = V3_MODEL_PATH / f"deepseek-v3-mp{world_size}"
    if not weights_dir.exists():
        # Fallback to mp1 if mp{world_size} doesn't exist yet
        weights_dir = V3_MODEL_PATH / "deepseek-v3-converted"
        if not weights_dir.exists():
            weights_dir = V3_MODEL_PATH
    
    weight_file = weights_dir / f"model{rank}-mp{world_size}.safetensors"
    if weight_file.exists():
        print(f"[V3 MP] Loading weights from: {weight_file.name}")
        load_model(model, str(weight_file))
    else:
        print(f"[V3 MP] ⚠ Weight file not found: {weight_file}")
        # Fallback: try mp1 format
        weight_file_mp1 = weights_dir / "model0-mp1.safetensors"
        if weight_file_mp1.exists():
            print(f"[V3 MP] Falling back to: {weight_file_mp1.name}")
            load_model(model, str(weight_file_mp1))
    
    # Move model to device
    model = model.to(device)
    
    # Load tokenizer (only on rank 0, then broadcast)
    if rank == 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(V3_MODEL_PATH))
    else:
        tokenizer = None
    
    if world_size > 1:
        # Broadcast tokenizer to other ranks
        objects = [tokenizer]
        dist.broadcast_object_list(objects, 0)
        tokenizer = objects[0]
    
    print(f"[V3 MP] ✓ Model loaded (rank={rank}, device={device})")
    print("="*80 + "\n")
    
    return model, tokenizer


def generate_v3_mp(
    model: 'Transformer',
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 200,
    temperature: float = 0.2,
    device: str = "cuda"
) -> List[str]:
    """
    Generate text using V3 with multi-GPU support.
    
    Uses the official V3 generate function.
    """
    
    if not prompts or not tokenizer:
        return [""]
    
    try:
        import importlib.util
        
        # Load official generate function
        gen_spec = importlib.util.spec_from_file_location(
            "v3_generate",
            str(V3_INFERENCE_PATH / "generate.py")
        )
        if not gen_spec or not gen_spec.loader:
            raise ImportError("Could not load generate.py")
        
        gen_module = importlib.util.module_from_spec(gen_spec)
        gen_module.Transformer = type(model)
        gen_module.ModelArgs = ModelArgs
        gen_spec.loader.exec_module(gen_module)
        
        official_generate = getattr(gen_module, 'generate', None)
        if not official_generate:
            raise AttributeError("generate function not found")
        
        print("[Generate V3 MP] ✓ Loaded official generate")
        
        # Prepare prompts
        prompt_token_lists = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
            except:
                tokens = tokenizer.encode(prompt)
            prompt_token_lists.append(tokens)
        
        # Call generate
        print(f"[Generate V3 MP] Generating with temperature={temperature}")
        with torch.inference_mode():
            completion_tokens = official_generate(
                model,
                prompt_token_lists,
                max_new_tokens,
                tokenizer.eos_token_id,
                temperature
            )
        
        # Decode
        completions = []
        for tokens in completion_tokens:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            completions.append(text)
        
        print(f"[Generate V3 MP] ✓ Generated {len(completions)} response(s)")
        return completions
    
    except Exception as e:
        print(f"[Generate V3 MP] ✗ Generation failed: {e}")
        return [""] * len(prompts)
