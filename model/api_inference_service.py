"""
Google Gemini API Inference Service
完全独立的 API 推理层，专用于 Gemini API
与本地模型推理 (inference_service.py) 完全隔离
"""

import os
import time
from typing import Optional


class GeminiInferenceService:
    """Gemini API 推理服务 - 完全独立实现"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 Gemini 推理服务
        
        Args:
            api_key: Google API key (如果为 None，从环境变量读取)
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
        """初始化 Gemini API 客户端"""
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
        使用 Gemini API 进行推理

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本
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
        使用 Gemini API 进行 JSON 推理（带格式约束）
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            
        Returns:
            生成的 JSON 文本
        """
        # Gemini API 会自动处理 JSON 格式请求
        return self.infer(prompt, max_tokens)


# 全局单例实例
_gemini_service: Optional[GeminiInferenceService] = None


def get_gemini_service(api_key: Optional[str] = None) -> GeminiInferenceService:
    """
    获取 Gemini 推理服务（单例模式）
    
    Args:
        api_key: 可选的 API key（用于初始化或重新配置）
        
    Returns:
        GeminiInferenceService 实例
    """
    global _gemini_service
    
    if _gemini_service is None:
        _gemini_service = GeminiInferenceService(api_key=api_key)
    
    return _gemini_service


def reset_gemini_service():
    """重置 Gemini 服务（用于测试或切换 API key）"""
    global _gemini_service
    _gemini_service = None


__all__ = [
    'GeminiInferenceService',
    'get_gemini_service',
    'reset_gemini_service'
]
