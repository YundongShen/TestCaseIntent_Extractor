"""
Google Gemini API Configuration
Standalone API configuration, fully isolated from local model setup.
"""

import os


class GeminiConfig:
    """Gemini API configuration."""

    API_KEY = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME = "gemini-pro"
    TEMPERATURE = 0.5
    TOP_P = 0.95

    @classmethod
    def validate(cls):
        """Validate that the API key is set."""
        if not cls.API_KEY:
            raise ValueError(
                "[Gemini] GOOGLE_API_KEY not set. "
                "Set environment variable: export GOOGLE_API_KEY='...'"
            )
        print(f"[Gemini] API key configured")
        return True


def set_gemini_api_key(api_key: str):
    """Set the Gemini API key at runtime."""
    GeminiConfig.API_KEY = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    print(f"[Gemini] API key updated")


__all__ = ['GeminiConfig', 'set_gemini_api_key']
