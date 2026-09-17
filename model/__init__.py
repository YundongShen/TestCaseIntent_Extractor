"""
Model module - DeepSeek model management and usage

Provides model download, configuration, and loading functions
"""

from .model_config import load_deepseek_model, get_model_path, MODEL_CONFIG

__all__ = [
    'load_deepseek_model',
    'get_model_path',
    'MODEL_CONFIG',
]
