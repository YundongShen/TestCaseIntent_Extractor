"""
Object Extractor - Extract test objects using DeepSeek LLM with COT reasoning.
"""

import re
import json

class ObjectExtractor:
    """Extract test objects from code using inference service."""
    
    def __init__(self):
        print("[ObjectExtractor] Initializing (uses InferenceService)")
    
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
        """Extract test objects from code. Returns: {"objects": [list]}"""
        
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"objects": ["object1", "object2", ...]}}
Output ONLY JSON, no explanations or other text.

[Definition]
Test Objects - these refer to the target entities in software testing, which is directly subjected to test activities and whose behaviors, attributes or states are observed and verified.

[Extraction Steps]
1. Identify all entities whose behaviors, attributes, or states are observed (directly or indirectly) during test activities. This forms the complete candidate set.
2. From the candidate set, exclude non-core auxiliary objects AND test information. Retain only entities that are the direct, primary target of test activities (the entity test actions are intentionally applied to).

[Exclusion Rules: Non-Core Auxiliary Objects + Test Information]
Remove any entity that falls into one of the following categories. This list is illustrative, not exhaustive:
- Test framework syntax and APIs: constructs used to define test suites, test cases, hooks, and mocks
- Assertion constructs: matchers, expect/assert statements, and verification predicates
- Test utility calls: helper functions that interact with testing libraries (used to perform testing, not the target)
- Auxiliary and temporary variables: objects for test configuration, setup, or environment
- Test lifecycle management: functions that prepare or clean up test state
- Infrastructure wrappers: containers of test results and generic built-ins
- Test information: entities whose behaviors/attributes/states are observed only for indirect support of verification; they are not the direct, primary target of test activities

[Final Constraint]

Output only entities that are the direct, primary target of test activities. Do not include entities observed only for indirect support.

[Test Code]
```
{code_text}
```

Extract all test objects following the two-step process. Output only the core target entities.

Final answer (ONLY JSON):"""
        
        # Use inference service (consistent with GoalExtractor and ActivityExtractor)
        from model.service_factory import get_inference_backend
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=9000)
        
        print("\n[ObjectExtractor] LLM Response:")
        print(response)
        print()
        
        # Prefer the last valid JSON candidate containing the target key.
        for data in reversed(self._extract_json_candidates(response)):
            if isinstance(data, dict) and "objects" in data:
                objects = data.get("objects", [])
                if isinstance(objects, list):
                    return {"objects": objects}

        # Try Method 1 (legacy): first valid JSON candidate
        data = self._extract_json_object(response)
        if data and isinstance(data, dict):
            objects = data.get("objects", [])
            if isinstance(objects, list):
                return {"objects": objects}
        
        # Try Method 2: Original regex patterns (fallback for compatibility)
        try:
            patterns = [
                r'\{[^}]*"objects"[^}]*\}',      # Simple non-nested
                r'\{.*?"objects".*?\}',           # Non-greedy
                r'\{[\s\S]*?"objects"[\s\S]*?\}', # With special chars and newlines
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        data = json.loads(json_str)
                        objects = data.get("objects", [])
                        if isinstance(objects, list):
                            return {"objects": objects}
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        
        return {"objects": []}

