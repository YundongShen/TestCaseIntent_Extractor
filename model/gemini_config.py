"""
Google Gemini API Configuration
独立的 API 推理服务，与本地模型完全隔离
"""

import os


class GeminiConfig:
    """Gemini API 配置"""
    
    API_KEY = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME = "gemini-pro"
    TEMPERATURE = 0.5
    TOP_P = 0.95
    
    @classmethod
    def validate(cls):
        """验证 API key"""
        if not cls.API_KEY:
            raise ValueError(
                "[Gemini] GOOGLE_API_KEY not set. "
                "Set environment variable: export GOOGLE_API_KEY='...'"
            )
        print(f"✅ [Gemini] API key configured")
        return True


# 全局 API key 管理
def set_gemini_api_key(api_key: str):
    """设置 Gemini API key"""
    GeminiConfig.API_KEY = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    print(f"[Gemini] API key set")


__all__ = ['GeminiConfig', 'set_gemini_api_key']
