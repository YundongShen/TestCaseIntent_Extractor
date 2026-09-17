"""
Google Gemini API Inference Service
Standalone API inference layer dedicated to Gemini API.
Fully isolated from local model inference (inference_service.py).
"""

import os
import time
from typing import Optional


class GeminiInferenceService:
    """Gemini API inference service — standalone implementation."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini inference service.

        If VERTEX_API_KEY is set, uses Vertex AI client (vertexai=True).
        Otherwise falls back to GOOGLE_API_KEY with standard Gemini API.
        """
        self.vertex_api_key = os.getenv("VERTEX_API_KEY")
        self.use_vertex = bool(self.vertex_api_key)

        if self.use_vertex:
            self.api_key = self.vertex_api_key
        else:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "[GeminiInference] No API key found. Set VERTEX_API_KEY or GOOGLE_API_KEY."
            )

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        self.client = None
        self._initialize_client()
        mode = "Vertex AI" if self.use_vertex else "Gemini API"
        print(f"[GeminiInference] Service initialized with {mode}")

    def _initialize_client(self):
        """Initialize the Gemini API client (Vertex AI or standard)."""
        try:
            from google import genai
            if self.use_vertex:
                self.client = genai.Client(vertexai=True, api_key=self.api_key)
                print(f"[GeminiInference] Vertex AI client configured")
            else:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[GeminiInference] Gemini API client configured")
        except ImportError:
            raise ImportError(
                "[GeminiInference] google-genai not installed. "
                "Run: pip install google-genai"
            )
        except Exception as e:
            raise RuntimeError(f"[GeminiInference] Failed to configure client: {e}")

    def infer(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call the Gemini API, retrying with backoff on 429s."""
        if not self.client:
            raise RuntimeError("[GeminiInference] Client not initialized")

        from google.genai import types as genai_types

        wait_times = [15, 30, 60]
        last_exc = None
        for attempt, wait in enumerate([0] + wait_times):
            if wait:
                print(f"[GeminiInference] 429 rate limit, waiting {wait}s (attempt {attempt+1}/{len(wait_times)+1})...")
                time.sleep(wait)
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.5,
                        top_p=0.95,
                    )
                )
                if not response.text:
                    print("[GeminiInference] Warning: Empty response from Gemini")
                    return ""
                result = response.text.strip()
                time.sleep(5)  # proactive delay: 5s between calls keeps well under 15 RPM
                return result
            except Exception as e:
                last_exc = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    continue
                print(f"[GeminiInference] API call failed: {e}")
                raise
        print(f"[GeminiInference] All retries exhausted: {last_exc}")
        raise last_exc

    def infer_json(self, prompt: str, max_tokens: int = 1000) -> str:
        """Same as infer(); kept as a separate name for call sites expecting JSON back."""
        return self.infer(prompt, max_tokens)

    def infer_no_thinking(self, prompt: str, max_tokens: int = 1000) -> str:
        """Gemini API has no extended-thinking toggle here; same as infer()."""
        return self.infer(prompt, max_tokens)


# Global singleton instance
_gemini_service: Optional[GeminiInferenceService] = None


def get_gemini_service(api_key: Optional[str] = None) -> GeminiInferenceService:
    """Get the shared GeminiInferenceService, creating it on first call."""
    global _gemini_service

    if _gemini_service is None:
        _gemini_service = GeminiInferenceService(api_key=api_key)

    return _gemini_service


def reset_gemini_service():
    """Reset the Gemini service (useful for testing or switching API keys)."""
    global _gemini_service
    _gemini_service = None


__all__ = [
    'GeminiInferenceService',
    'get_gemini_service',
    'reset_gemini_service'
]
