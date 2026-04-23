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
    
    def __init__(self, mode: str = "local", model_config: dict = None):
        """
        Initialize inference service.
        
        Args:
            mode: "local" for local model, "api" for cloud API
            model_config: Model configuration dict from MODEL_CONFIG (should be set via set_model_config())
        
        IMPORTANT: To switch to API mode, change mode="local" -> mode="api" here
        """
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_config = model_config or _model_config
        
        print(f"[InferenceService] Initialized in '{mode}' mode")
        if self.model_config:
            print(f"[InferenceService] Using model: {self.model_config.get('model_name', 'unknown')}")
    
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
        Load local model using the configured MODEL_CONFIG.
        
        Returns: (model, tokenizer, device)
        """
        if self.model is not None:
            return self.model, self.tokenizer, self.device
        
        if not self.model_config:
            raise RuntimeError("[InferenceService] No model config set. Call set_model_config() first.")
        
        # Use the configured model path and loader
        model_name = self.model_config.get('model_name', 'unknown')
        model_id = self.model_config.get('model_id', '')
        local_path = self.model_config.get('local_path', None)
        
        print(f"[InferenceService] Loading model: {model_name}")
        
        # Auto-check if model exists
        from pathlib import Path
        if local_path and (not Path(local_path).exists() or not any(Path(local_path).iterdir())):
            print(f"[InferenceService] Model not found at {local_path}")
            print("[InferenceService] Attempting auto-download...")
            self._auto_download_model()
        
        try:
            from model.model_config import get_device
            
            self.device = get_device()
            
            # Dynamically load the appropriate loader function based on model type
            if 'Qwen' in model_id or 'qwen' in model_id.lower():
                from model.model_config_qwen import load_qwen_model
                self.model, self.tokenizer = load_qwen_model(device=self.device, quantize_8bit=True)
                print("[InferenceService] Qwen model loaded successfully")
            elif 'DeepSeek' in model_id or 'deepseek' in model_id.lower():
                from model.model_config import load_deepseek_model
                self.model, self.tokenizer = load_deepseek_model(device=self.device, quantize_8bit=True)
                print("[InferenceService] DeepSeek model loaded successfully")
            else:
                raise ValueError(f"Unknown model type: {model_id}")
            
            return self.model, self.tokenizer, self.device
        except Exception as e:
            print(f"[InferenceService] Failed to load local model: {e}")
            raise
    
    def _auto_download_model(self):
        """Auto-download model if missing (using HuggingFace CLI or API)"""
        import subprocess
        from pathlib import Path
        
        if not self.model_config:
            raise RuntimeError("[InferenceService] No model config set for download.")
        
        model_id = self.model_config.get("model_id", "")
        local_dir = self.model_config.get("local_path")
        
        if not local_dir:
            raise ValueError("[InferenceService] No local_path in model config")
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
            model_id = self.model_config.get('model_id', '')
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # For Qwen and others, use raw prompt directly
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
            
            is_qwen = 'Qwen' in model_id or 'qwen' in model_id.lower()

            # Generate response
            with torch.no_grad():
                generate_kwargs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask', None),
                    'max_new_tokens': max_tokens,
                    'pad_token_id': tokenizer.pad_token_id
                }

                if is_qwen:
                    # Qwen uses sampling mode to generate complete JSON with thinking process
                    generate_kwargs['do_sample'] = True
                    generate_kwargs['temperature'] = 0.5
                    generate_kwargs['top_p'] = 0.95
                    generate_kwargs['top_k'] = 50
                else:
                    generate_kwargs['do_sample'] = True
                    generate_kwargs['temperature'] = 0.5
                    generate_kwargs['top_p'] = 0.95
                    generate_kwargs['top_k'] = 50
                    generate_kwargs['eos_token_id'] = tokenizer.eos_token_id
                
                outputs = model.generate(**generate_kwargs)
            
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


# Global singleton instance and model configuration
_inference_service = None
_model_config = None


def set_model_config(model_config: dict):
    """
    Set the global MODEL_CONFIG before creating InferenceService.
    Should be called from main.py after selecting the appropriate model config.
    
    Args:
        model_config: Dictionary with keys: model_id, model_name, local_path, quantization, etc.
    """
    global _model_config
    _model_config = model_config
    print(f"[InferenceService] Model config set: {model_config.get('model_name', 'unknown')}")


def get_inference_service(mode: str = "local") -> InferenceService:
    """
    Get global inference service (singleton pattern).
    Uses the model config set via set_model_config().
    Model only loads once at first call.
    
    Args:
        mode: "local" or "api"
        
    Returns:
        InferenceService instance
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(mode=mode, model_config=_model_config)
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


__all__ = ['InferenceService', 'get_inference_service', 'set_inference_mode', 'set_model_config']
