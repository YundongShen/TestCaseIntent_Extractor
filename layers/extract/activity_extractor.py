"""
Activity Extractor - Extract test activities using DeepSeek LLM with COT reasoning.
"""

import re
import json

class ActivityExtractor:
    """Extract test activities from code using inference service."""
    
    def __init__(self):
        print("[ActivityExtractor] Initializing (uses InferenceService)")

    @staticmethod
    def _extract_json_candidates(text):
        """
        Extract all possible JSON object candidates by balanced brace matching.
        """
        candidates = []
        for i, char in enumerate(text):
            if char != "{":
                continue
            brace_count = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    brace_count += 1
                elif text[j] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[i:j+1]
                        try:
                            candidates.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            pass
                        break
        return candidates
    
    def extract(self, code_text):
        """Extract test activities from code. Returns: {"activities": [list]}"""
        if len(code_text) > 2000:
            code_text = code_text[:2000]
        
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
        
        from model.service_factory import get_inference_backend
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=9000)

        print("\n[ActivityExtractor] LLM Response:")
        print(response)
        print()

        # Prefer the last valid JSON candidate containing the target key.
        for data in reversed(self._extract_json_candidates(response)):
            if isinstance(data, dict) and "activities" in data:
                activities = data.get("activities", [])
                if isinstance(activities, list):
                    return {"activities": activities}

        # Regex fallback for malformed wrappers.
        try:
            matches = re.findall(r"\{[\s\S]*?\"activities\"[\s\S]*?\}", response, re.DOTALL)
            for match in reversed(matches):
                data = json.loads(match)
                activities = data.get("activities", [])
                if isinstance(activities, list):
                    return {"activities": activities}
        except Exception:
            pass
        
        return {"activities": []}
