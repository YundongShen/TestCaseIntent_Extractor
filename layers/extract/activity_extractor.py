"""
Activity Extractor - Extract test activities using LLM inference service.
"""

import re
import json

class ActivityExtractor:
    """Extract test activities from code using inference service."""
    
    def __init__(self):
        print("[ActivityExtractor] Initializing")
    
    @staticmethod
    def _extract_json_object(text):
        """
        Extract a valid JSON object from text using bracket/brace matching.
        Handles nested structures like: {"activities": ["item with {nested} object"]}
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
    
    def extract(self, code_text):
        """Extract test activities from code using LLM only. Returns: {"activities": [list]}"""
        if len(code_text) > 2000:
            code_text = code_text[:2000]
        return self._extract_with_model(code_text)
    
    def _extract_with_model(self, code_text):
        """Extract using LLM inference service"""
        from model.inference_service import get_inference_service
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"activities": ["activity1", "activity2", ...]}}
Output ONLY JSON, no explanations or other text.

[Definition of Test Activities]
Test Activities are a purposeful set of operations directed toward test objects to trigger, observe, record, and inspect their behaviors, attributes, and states. They form the execution flow that realizes test intent.

[Extraction Steps]

Step 1: Extract every action executed during the test execution from the code. Collect all candidate operations without omission.

Step 2: Simplify overly detailed implementation statements into concise natural language, and arrange all extracted actions strictly in their original execution order. Write clear, readable descriptions for each activity.

Step 3 (Optional): Aggregate adjacent actions of the same type or operations serving the same purpose when necessary. For this extraction, output the sequence without aggregation unless it significantly improves clarity(Be careful not to over-aggregate).

[Test Code]
```
{code_text}
```

Extract all test activities following the three-step process. Maintain execution order and use 
clear, concise language for each activity step.

Final answer (ONLY JSON):"""
        
        # Use inference service
        service = get_inference_service()
        response = service.infer(prompt, max_tokens=2000)
        
        print("\n[ActivityExtractor] LLM Response:")
        print(response)
        print()
        
        # Try Method 1: Balanced bracket matching (handles nested structures)
        data = self._extract_json_object(response)
        if data and isinstance(data, dict):
            activities = data.get("activities", [])
            if isinstance(activities, list) and len(activities) > 0:
                return {"activities": activities}
        
        # Try Method 2: Original regex patterns (fallback for compatibility)
        try:
            patterns = [
                r'\{[^}]*"activities"[^}]*\}',      # Simple non-nested
                r'\{.*?"activities".*?\}',           # Non-greedy  
                r'\{[\s\S]*?"activities"[\s\S]*?\}', # With special chars and newlines
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        data = json.loads(json_str)
                        activities = data.get("activities", [])
                        if isinstance(activities, list) and len(activities) > 0:
                            return {"activities": activities}
                    except json.JSONDecodeError:
                        pass
                        
        except Exception:
            pass
        
        # Try Method 3: Clean and retry
        try:
            cleaned = ''.join(c if ord(c) < 128 else ' ' for c in response)
            data = self._extract_json_object(cleaned)
            if data and isinstance(data, dict):
                activities = data.get("activities", [])
                if isinstance(activities, list) and len(activities) > 0:
                    return {"activities": activities}
        except Exception:
            pass
        
        return {"activities": []}
