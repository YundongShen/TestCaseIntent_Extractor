"""
Unified Inference Service - Supports both local models and API backends.

KEY MODIFICATION POINTS:
  1. Change local model: Edit _load_local_model() method
  2. Switch to API: Implement _infer_api() method, then set mode='api'
  3. All extractors automatically work with either backend - no other changes needed

ARCHITECTURE:
  ObjectExtractor/ActivityExtractor/etc. -> infer(prompt) -> InferenceService
  InferenceService routes to _infer_local() or _infer_api() based on mode
"""

import os
from typing import Tuple


class InferenceService:
    """Unified interface for LLM inference - supports local models and APIs.
    
    METHODS SUMMARY:
    ├─ __init__(mode)          : Initialize with 'local' or 'api' mode
    ├─ infer(prompt, max_tokens) : Route request to local or API backend (MAIN METHOD - called by extractors)
    ├─ _load_local_model()      : [Edit to change] Load/cache the local DeepSeek model
    ├─ _infer_local(...)        : Tokenize, generate, decode with local model
    ├─ _infer_api(...)          : [Edit to implement] Make API call to cloud LLM
    
    USAGE: service = get_inference_service()  # Get singleton
           response = service.infer(prompt)   # Called by ObjectExtractor, etc.
    """
    
    def __init__(self, mode: str = "local"):
        """
        Initialize inference service.
        
        Args:
            mode: "local" for local DeepSeek model, "api" for cloud API
        
        IMPORTANT: To switch to API mode, change mode="local" -> mode="api" here
        """
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.device = None
        
        print(f"[InferenceService] Initialized in '{mode}' mode")
    
    def infer(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Core method called by all extractors. Routes to local or API backend.
        Do NOT modify this method.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if self.mode == "local":
            return self._infer_local(prompt, max_tokens)
        elif self.mode == "api":
            return self._infer_api(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _load_local_model(self) -> Tuple:
        """
        EDIT THIS to change local model.
        Currently loads: model/models/deepseek-v3.2-7b-base
        
        To use different model: Modify load_deepseek_model() call below
        
        Returns: (model, tokenizer, device)
        """
        if self.model is not None:
            return self.model, self.tokenizer, self.device
        
        # Auto-check if model exists, if not download it
        from pathlib import Path
        from model.model_config import get_model_path, load_deepseek_model, get_device
        
        model_path = get_model_path()
        if not model_path.exists() or not any(model_path.iterdir()):
            print(f"[InferenceService] Model not found at {model_path}")
            print("[InferenceService] Attempting auto-download...")
            self._auto_download_model()
        
        print("[InferenceService] Loading local DeepSeek model...")
        try:
            self.device = get_device()
            self.model, self.tokenizer = load_deepseek_model(
                device=self.device, 
                quantize_8bit=True
            )
            print("[InferenceService] Model loaded successfully")
            return self.model, self.tokenizer, self.device
        except Exception as e:
            print(f"[InferenceService] Failed to load local model: {e}")
            raise
    
    def _auto_download_model(self):
        """Auto-download model if missing (using HuggingFace CLI or API)"""
        import subprocess
        from pathlib import Path
        from model.model_config import get_model_path, MODEL_CONFIG
        
        model_id = MODEL_CONFIG["model_id"]
        local_dir = get_model_path()
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("[Auto-Download] Model not found - attempting to download...")
        print(f"{'='*80}\n")
        
        methods = [
            ("huggingface-cli", self._download_with_cli),
            ("huggingface_hub snapshot_download", self._download_with_python),
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"[Download] Trying method: {method_name}")
                if method_func(model_id, str(local_dir)):
                    return
            except Exception as e:
                print(f"[Download] {method_name} failed: {e}")
        
        # All methods failed
        print(f"\n{'='*80}")
        print("[Download] ✗ Failed - Model download unsuccessful")
        print(f"{'='*80}\n")
        print("SOLUTION: Please download the model manually:")
        print(f"  Command: python download_model_direct.py")
        print(f"  Or: python model/download_model.py")
        print(f"\nAlternatively in future, set API key for cloud inference")
        raise RuntimeError("Model download failed. See solutions above.")
    
    def _download_with_cli(self, model_id, local_dir):
        """Try download using huggingface-cli"""
        import subprocess
        cmd = [
            "huggingface-cli",
            "download",
            model_id,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks=False",
            "--resume-download"
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    def _download_with_python(self, model_id, local_dir):
        """Try download using huggingface_hub Python API"""
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            return True
        except Exception as e:
            print(f"[Download] Python method error: {e}")
            return False
    
    def _infer_local(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Run local model inference.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        import torch
        
        model, tokenizer, device = self._load_local_model()
        
        # CRITICAL: Ensure model is in eval mode for inference
        model.eval()
        
        try:
            # Use chat template for better instruction following
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt
            
            # Tokenize input with attention mask
            inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                add_special_tokens=True
            )
            input_length = inputs['input_ids'].shape[1]
            inputs = inputs.to(device)
            
            # Generate with greedy decoding for stability
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode response - only the newly generated tokens
            generated_token_ids = outputs[0][input_length:]
            response = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True
            ).strip()
            
            # Clean up special characters that sometimes appear in decoded text
            # First, convert all known UTF-8 space/whitespace variants to actual spaces
            response = response.replace('Ġ', ' ')  # Common space variant
            response = response.replace('\u00A0', ' ')  # Non-breaking space
            response = response.replace('\u2000', ' ')  # En quad
            response = response.replace('\u2001', ' ')  # Em quad
            response = response.replace('\u2002', ' ')  # En space
            response = response.replace('\u2003', ' ')  # Em space
            response = response.replace('\u2004', ' ')  # Three-per-em space
            response = response.replace('\u2005', ' ')  # Four-per-em space
            response = response.replace('\u2006', ' ')  # Six-per-em space
            response = response.replace('\u2007', ' ')  # Figure space
            response = response.replace('\u2008', ' ')  # Punctuation space
            response = response.replace('\u2009', ' ')  # Thin space
            response = response.replace('\u200A', ' ')  # Hair space
            
            # Then handle other UTF-8 artifacts
            response = response.replace('Ċ', '\n')
            response = response.replace('Ĩ', 'i').replace('ē', 'e').replace('ŏ', 'o')
            response = response.replace('ĸ', 'k').replace('ł', 'l')
            
            # Finally, clean up remaining non-ASCII but keep ASCII whitespace
            response = ''.join(c if (ord(c) < 128 or c in ' \t\n') else '' for c in response)
            
            return response
        except Exception as e:
            print(f"[InferenceService] Local inference failed: {e}")
            raise
    
    def _infer_api(self, prompt: str, max_tokens: int = 200) -> str:
        """
        EDIT THIS to implement API backend inference.
        
        Example for OpenAI (replace raise NotImplementedError with this):
          from openai import OpenAI
          client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
          response = client.chat.completions.create(
              model="gpt-4",
              messages=[{"role": "user", "content": prompt}],
              max_tokens=max_tokens
          )
          return response.choices[0].message.content
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        raise NotImplementedError(
            "API mode not implemented. Edit _infer_api() method to add your API provider."
        )
        
        # TODO: Implementation template for API inference
        # Example for OpenAI:
        # from openai import OpenAI
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=max_tokens
        # )
        # return response.choices[0].message.content


# Global singleton instance
_inference_service = None


def get_inference_service(mode: str = "local") -> InferenceService:
    """
    Get global inference service (singleton pattern).
    Used by all extractors: service = get_inference_service()
    Model only loads once at first call.
    
    Args:
        mode: "local" or "api"
        
    Returns:
        InferenceService instance
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(mode=mode)
    return _inference_service


def set_inference_mode(mode: str):
    """
    Switch between local and API mode at runtime.
    Creates new InferenceService instance with specified mode.
    
    Args:
        mode: "local" or "api"
    """
    global _inference_service
    _inference_service = InferenceService(mode=mode)
    print(f"[InferenceService] Mode switched to '{mode}'")


__all__ = ['InferenceService', 'get_inference_service', 'set_inference_mode']
