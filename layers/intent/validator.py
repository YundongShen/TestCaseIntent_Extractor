"""
Validator - Validate if extracted objects, goals, and activities are related.
"""

import re
import json

class Validator:
    """Validate extracted data relevance using inference service."""
    
    def __init__(self):
        print("[Validator] Initializing")
    
    def validate(self, objects, goals, activities):
        """Validate if three elements are mutually related using LLM only. 
        Returns: {"matched": bool, "reasoning": str, "details": {...}}"""
        return self._validate_with_model(objects, goals, activities)
    
    def _validate_with_model(self, objects, goals, activities):
        """Validate using LLM inference service"""
        from model.service_factory import get_inference_backend

        prompt = f"""You are a test intent analysis expert. Check if the following three extracted test intent elements are mutually related and matched.

[Test Objects] (Main objects being tested)
{json.dumps(objects, ensure_ascii=False, indent=2)}

[Test Goals] (Expected content to verify)
{json.dumps(goals, ensure_ascii=False, indent=2)}

[Test Activities] (Sequence of operations executed)
{json.dumps(activities, ensure_ascii=False, indent=2)}

[Validation Task]
Determine if these three elements are mutually related, i.e.:
- Do activities operate on these objects?
- Do goals verify the state or behavior of these objects?
- Are activities executed to achieve these goals?

Return JSON format result:
{{
  "matched": true/false,
  "reasoning": "Brief explanation of match degree (if not matched, explain missing associations)"
}}

Output ONLY JSON, no other text."""
        
        # Use inference service instead of direct model calls
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=200)
        
        try:
            match = re.search(r'\{.*"matched".*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return {
                    "matched": result.get("matched", False),
                    "reasoning": result.get("reasoning", ""),
                    "details": {
                        "objects_count": len(objects),
                        "goals_count": len(goals),
                        "activities_count": len(activities)
                    }
                }
        except:
            pass
        
        return {
            "matched": False,
            "error": "Validation failed",
            "reasoning": ""
        }
