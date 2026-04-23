"""
Unified Inference Service Factory
统一管理本地和 API 后端的工厂函数
支持所有提取器（independent / combined / chain 模式）无缝切换
"""

import os


def get_inference_backend(backend: str = None):
    """
    统一的推理后端获取函数
    从环境变量 INFERENCE_BACKEND 读取后端类型（如果未指定）
    
    Args:
        backend: 'local' 或 'api' (如果为 None，从 INFERENCE_BACKEND 环境变量读取)
    
    Returns:
        推理服务实例 (接口完全相同：infer(prompt, max_tokens) -> str)
    
    Usage:
        # 方式 1：环境变量控制
        export INFERENCE_BACKEND="api"  # 用 Gemini
        python main.py
        
        # 方式 2：代码中指定
        service = get_inference_backend("api")
        response = service.infer(prompt, max_tokens=1000)
    """
    backend = backend or os.getenv("INFERENCE_BACKEND", "local")
    
    if backend == "api":
        print("[ServiceFactory] Using Gemini API backend")
        from model.api_inference_service import get_gemini_service
        return get_gemini_service()
    elif backend == "local":
        print("[ServiceFactory] Using local model backend")
        from model.inference_service import get_inference_service
        return get_inference_service()
    else:
        raise ValueError(f"[ServiceFactory] Unknown backend: {backend}. Use 'local' or 'api'")


__all__ = ['get_inference_backend']
