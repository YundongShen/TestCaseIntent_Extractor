"""
Business layer - Provide onboarding document prompt template.
"""

from layers.business.onboarding_template import PROMPT


class BusinessLayer:
    """Business layer for prompt template provision."""
    
    def __init__(self):
        print("[BusinessLayer] Initializing")
        print(f"[BusinessLayer] Prompt loaded ({len(PROMPT)} chars)")
    
    def process(self, intent_data):
        """Combine intent data with prompt template.
        Returns: data with prompt added."""
        print("[BusinessLayer] Processing")
        
        return {
            **intent_data,
            "prompt": PROMPT
        }


__all__ = ['BusinessLayer', 'PROMPT']
