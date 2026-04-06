#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download DeepSeek model - direct download of model files
"""

import os
import sys
import subprocess
from pathlib import Path

def download_with_hf_cli():
    """Download model using HuggingFace CLI"""
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("[Starting DeepSeek-V3.2-7B-base Model Download]")
    print("="*80 + "\n")
    
    model_id = "deepseek-ai/deepseek-llm-7b-base"
    local_dir = models_dir / "deepseek-v3.2-7b-base"
    
    print(f"[Info] Model ID: {model_id}")
    print(f"[Info] Save location: {local_dir}")
    print(f"[Info] Downloading... (may take 10-30 minutes depending on network)\n")
    
    try:
        # Use HuggingFace CLI
        cmd = [
            "huggingface-cli",
            "download",
            model_id,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks=False",
            "--resume-download"
        ]
        
        print("[Download] Executing command:")
        print(f"  {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0 or local_dir.exists():
            print("\n" + "="*80)
            print("[Model Download Complete!]")
            print("="*80)
            print(f"[Success] Model saved to: {local_dir}\n")
            
            # List files
            if local_dir.exists():
                print("[Info] Downloaded files:")
                total_size = 0
                for file in sorted(os.listdir(local_dir)):
                    file_path = local_dir / file
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        total_size += file_path.stat().st_size
                        print(f"  ✓ {file:<40} ({size_mb:>8.2f} MB)")
                    else:
                        print(f"  📁 {file}/")
                
                total_gb = total_size / (1024 * 1024 * 1024)
                print(f"\n[Statistics] Total size: {total_gb:.2f} GB")
            
            return str(local_dir)
        else:
            raise Exception("Download failed or directory not created")
            
    except FileNotFoundError:
        print("[Error] huggingface-cli not found")
        print("[Tip] Trying to download using Python API...\n")
        return download_with_python_api()
    except Exception as e:
        print(f"[Error] {str(e)}\n")
        print("[Tip] Trying to download using Python API...\n")
        return download_with_python_api()

def download_with_python_api():
    """Download using Python API and huggingface-hub"""
    
    from huggingface_hub import snapshot_download
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("[Using Python API to Download]")
    print("="*80 + "\n")
    
    model_id = "deepseek-ai/deepseek-llm-7b-base"
    local_dir = models_dir / "deepseek-v3.2-7b-base"
    
    print(f"[Download] Model: {model_id}")
    print(f"[Download] Path: {local_dir}\n")
    
    try:
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("\n" + "="*80)
        print("[Model Download Complete!]")
        print("="*80)
        print(f"[Success] Model saved to: {model_path}\n")
        
        # List files
        if local_dir.exists():
            print("[Info] Downloaded files:")
            total_size = 0
            for file in sorted(os.listdir(local_dir)):
                file_path = local_dir / file
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    total_size += file_path.stat().st_size
                    print(f"  ✓ {file:<40} ({size_mb:>8.2f} MB)")
                else:
                    print(f"  📁 {file}/")
            
            total_gb = total_size / (1024 * 1024 * 1024)
            print(f"\n[Statistics] Total size: {total_gb:.2f} GB")
        
        return str(model_path)
        
    except Exception as e:
        print(f"[Error] {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("\n[DeepSeek Model Download Tool]")
        print("Current environment: Python {}".format(sys.version.split()[0]))
        
        # Try HF CLI then fallback to Python API
        try:
            model_path = download_with_hf_cli()
        except:
            try:
                model_path = download_with_python_api()
            except ImportError:
                print("\n[Error] huggingface-hub not installed")
                print("[Tip] Please run: pip install huggingface-hub --upgrade")
                sys.exit(1)
            except Exception as e:
                raise e
        
        print("\n" + "="*80)
        print("[Using the Model]")
        print("="*80)
        print(f"Model location: {model_path}")
        print("\nLoad model in code:")
        print("""
from model.model_config import load_deepseek_model

model, tokenizer = load_deepseek_model(device="cpu")
""")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[Fatal Error] {str(e)}\n")
        print("[Manual Download Method]")
        print("Visit the following links to download manually:")
        print("  1. HuggingFace: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base")
        print("  2. ModelScope: https://modelscope.cn/models/deepseek-ai/deepseek-llm-7b-base")
        print("\nAfter downloading, place in: model/models/deepseek-v3.2-7b-base directory")
        sys.exit(1)
