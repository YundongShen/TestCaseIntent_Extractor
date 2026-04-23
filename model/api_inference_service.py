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

        Args:
            api_key: Google API key. If None, read from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "[GeminiInference] GOOGLE_API_KEY not provided and not in environment"
            )

        self.model_name = "gemini-3-pro-preview"
        self.client = None
        self._initialize_client()
        print(f"[GeminiInference] Service initialized with Gemini API")

    def _initialize_client(self):
        """Initialize the Gemini API client."""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            print(f"[GeminiInference] Gemini API client configured")
        except ImportError:
            raise ImportError(
                "[GeminiInference] google-genai not installed. "
                "Run: pip install google-genai"
            )
        except Exception as e:
            raise RuntimeError(f"[GeminiInference] Failed to configure Gemini: {e}")

    def infer(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Run inference using the Gemini API.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum number of output tokens.

        Returns:
            Generated text string.
        """
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
        """
        Run JSON inference using the Gemini API.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum number of output tokens.

        Returns:
            Generated JSON text string.
        """
        return self.infer(prompt, max_tokens)


# Global singleton instance
_gemini_service: Optional[GeminiInferenceService] = None


def get_gemini_service(api_key: Optional[str] = None) -> GeminiInferenceService:
    """
    Get the Gemini inference service (singleton).

    Args:
        api_key: Optional API key for initialization or reconfiguration.

    Returns:
        GeminiInferenceService instance.
    """
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
