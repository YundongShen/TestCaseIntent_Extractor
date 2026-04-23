"""
Chain Activity Extractor - Extract test activities (third step in chain mode, uses objects and goals context).
"""

import re
import json

class ChainActivityExtractor:
    """Extract test activities from code using inference service, with objects and goals context."""
    
    def __init__(self):
        print("[ChainActivityExtractor] Initializing (chain mode)")
    
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
    
    def extract(self, code_text, objects=None, goals=None):
        """
        Extract test activities from code using LLM, with objects and goals context.
        
        Args:
            code_text: The test code
            objects: Previously extracted objects (to be used as context)
            goals: Previously extracted goals (to be used as context)
        
        Returns: {"activities": [list]}
        """
        return self._extract_with_model(code_text, objects=objects, goals=goals)
    
    def _extract_with_model(self, code_text, objects=None, goals=None):
        """Extract using LLM inference service"""
        from model.service_factory import get_inference_backend
        
        # Build context from objects and goals if provided
        context_info = ""
        if objects:
            objects_str = ", ".join(objects) if isinstance(objects, list) else str(objects)
            context_info += f"[Previously Extracted Objects]\n{objects_str}\n\n"
        if goals:
            goals_str = "\n".join(goals) if isinstance(goals, list) else str(goals)
            context_info += f"[Previously Extracted Goals]\n{goals_str}\n\n"
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"activities": ["activity1", "activity2", ...]}}
Output ONLY JSON, no explanations or other text.

[Definition of Test Activities]
Test Activities are a purposeful set of operations directed toward test objects to trigger, observe, record, and inspect their behaviors, attributes, and states. They form the execution flow that realizes test intent.

[Extraction Steps]

Step 1: Extract every action executed during the test execution from the code. Collect all candidate operations without omission.

Step 2: Simplify overly detailed implementation statements into concise natural language, and arrange all extracted actions strictly in their original execution order. Write clear, readable descriptions for each activity.

Step 3: Based on the previously extracted test objects and goals, filter and retain only those activities that are relevant (directly or indirectly) to achieving the test goals on the target objects. Aggregate adjacent actions of the same type or operations serving the same purpose to create a concise, focused activity sequence that supports the test objectives.
{context_info}
[Test Code]
```
{code_text}
```

Extract test activities following the three-step process. Focus on activities that drive the test objectives. Maintain execution order and use clear, concise language.

Final answer (ONLY JSON):"""
        
        # Use inference service
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=3000)
        
        print("\n[ChainActivityExtractor] LLM Response:")
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
