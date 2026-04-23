"""
Unified Inference Service Factory
Manages local and API backends through a single factory function.
Supports seamless backend switching for all extractors (independent / combined / chain modes).
"""

import os


def get_inference_backend(backend: str = None):
    """
    Unified inference backend accessor.
    Reads backend type from the INFERENCE_BACKEND env var if not specified explicitly.

    Args:
        backend: 'local' or 'api'. If None, reads from INFERENCE_BACKEND env var (default: 'local').

    Returns:
        Inference service instance with a unified interface: infer(prompt, max_tokens) -> str

    Usage:
        # Option 1: control via environment variable
        export INFERENCE_BACKEND="api"
        python main.py

        # Option 2: specify in code
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
