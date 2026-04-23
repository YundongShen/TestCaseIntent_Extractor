"""
Chain Goal Extractor - Extract test goals (second step in chain mode, uses objects context).
"""

import re
import json

class ChainGoalExtractor:
    """Extract test goals from code using inference service, with objects context."""
    
    def __init__(self):
        print("[ChainGoalExtractor] Initializing (chain mode)")
    
    @staticmethod
    def _extract_json_object(text):
        """
        Extract a valid JSON object from text using bracket/brace matching.
        Handles nested structures.
        """
        for i, char in enumerate(text):
            if char == '{':
                brace_count = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[i:j+1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                pass
        return None
    
    def extract(self, code_text, objects=None):
        """
        Extract test goals from code using LLM, with objects context.
        
        Args:
            code_text: The test code
            objects: Previously extracted objects (to be used as context)
        
        Returns: {"goals": [list]}
        """
        return self._extract_with_model(code_text, objects=objects)
    
    def _extract_with_model(self, code_text, objects=None):
        """Extract using LLM inference service"""
        from model.service_factory import get_inference_backend
        
        # Build context from objects if provided
        context_info = ""
        if objects:
            objects_str = ", ".join(objects) if isinstance(objects, list) else str(objects)
            context_info = f"\n[Previously Extracted Objects]\n{objects_str}\n"
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"goals": ["goal1", "goal2", ...]}}
Output ONLY JSON, no explanations or other text.

[Definition]
Test Goal is the expected verification direction and quality judgment criteria formulated for the test object, supported by test activities.

[Extraction Steps]
1. Use the previously extracted test objects (listed below) as the primary entities. These are the target objects for which test goals need to be defined.
2. Determine the verification target of the primary test object based on explicit information in the code (e.g., assertions, descriptions). The derived goal must faithfully summarize the actual verification purpose.
3. (Optional) Adjust granularity as needed—this step may be omitted; output the goal at a reasonable level of detail.

Extract the PRIMARY/DOMINANT goal(s) for this test case - typically just 1-2 core verification purposes, not all individual assertions.
{context_info}
[Test Code]
```
{code_text}
```

Extract the PRIMARY/DOMINANT goal(s) for this test case - typically just 1-2 core verification purposes, not all individual assertions.

Final answer (ONLY JSON):"""
        
        # Use inference service
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=3000)
        
        print("\n[ChainGoalExtractor] LLM Response:")
        print(response)
        print()
        
        # Try Method 1: Balanced bracket matching (handles nested structures)
        data = self._extract_json_object(response)
        if data and isinstance(data, dict):
            goals = data.get("goals", [])
            if isinstance(goals, list) and len(goals) > 0:
                return {"goals": goals}
        
        # Try Method 2: Original regex patterns (fallback for compatibility)
        try:
            patterns = [
                r'\{[^}]*"goals"[^}]*\}',      # Simple non-nested
                r'\{.*?"goals".*?\}',           # Non-greedy
                r'\{[\s\S]*?"goals"[\s\S]*?\}', # With special chars and newlines
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        data = json.loads(json_str)
                        goals = data.get("goals", [])
                        if isinstance(goals, list) and len(goals) > 0:
                            return {"goals": goals}
                    except json.JSONDecodeError:
                        pass
                        
        except Exception:
            pass
        
        # Try Method 3: Clean and retry
        try:
            cleaned = ''.join(c if ord(c) < 128 else ' ' for c in response)
            data = self._extract_json_object(cleaned)
            if data and isinstance(data, dict):
                goals = data.get("goals", [])
                if isinstance(goals, list) and len(goals) > 0:
                    return {"goals": goals}
        except Exception:
            pass
        
        return {"goals": []}
