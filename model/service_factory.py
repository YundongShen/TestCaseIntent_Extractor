"""Picks the local or API inference service based on INFERENCE_BACKEND."""

import os


def get_inference_backend(backend: str = None):
    """Returns a service with infer(prompt, max_tokens). backend defaults to
    the INFERENCE_BACKEND env var, or "local" if that's not set."""
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
