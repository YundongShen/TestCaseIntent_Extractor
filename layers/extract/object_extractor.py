"""
Object Extractor - Extract test objects using LLM inference service.
"""

import re
import json

class ObjectExtractor:
    """Extract test objects from code using inference service."""
    
    def __init__(self):
        print("[ObjectExtractor] Initializing")
    
    def extract(self, code_text):
        """Extract test objects from code using LLM only. Returns: {"objects": [list]}"""
        if len(code_text) > 2000:
            code_text = code_text[:2000]
        return self._extract_with_model(code_text)
    
    def _extract_with_model(self, code_text):
        """Extract using LLM inference service"""
        from model.inference_service import get_inference_service
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"objects": ["obj1", "obj2", ...]}}
Output ONLY JSON, no explanations or other text.

[Definition]
Test Objects are the target entities in software testing that are directly subjected to test activities. Their behaviors, attributes, or states are observed and verified.

[Extraction Steps]

Step 1: Identify all entities whose behaviors, attributes, or states are observed (directly or indirectly) during test activities. This forms the complete candidate set.

Step 2: From the candidate set, exclude non-core auxiliary objects AND test information. Retain only entities that are the direct, primary target of test activities (the entity test actions are intentionally applied to).

[Exclusion Rules (Non-Core Auxiliary Objects + Test Information)]

Remove any entity that falls into one of the following categories. This list is illustrative, not exhaustive:
- Test framework syntax and APIs - constructs used to define test suites, test cases, hooks, and mocks.
- Assertion constructs - matchers, expect/assert statements, and verification predicates.
- Test utility calls - helper functions that interact with testing libraries (used to perform testing, not the target).
- Auxiliary and temporary variables - objects for test configuration, setup, or environment.
- Test lifecycle management - functions that prepare or clean up test state.
- Infrastructure wrappers - containers of test results and generic built-ins.
- Test information - entities whose behaviors/attributes/states are observed only for indirect support of verification; they are not the direct, primary target of test activities.

[Final Constraint]

Output only entities that are the direct, primary target of test activities. Do not include entities observed only for indirect support.

[Test Code]
```
{code_text}
```

Extract Test Objects following the extraction steps and exclusion rules.

Final answer (ONLY JSON):"""
        
        # Use inference service
        service = get_inference_service()
        response = service.infer(prompt, max_tokens=3000)
        
        print("\n[ObjectExtractor] LLM Response:")
        print(response)
        print()
        
        try:
            # Remove markdown code blocks if present
            cleaned_response = response
            if "```json" in cleaned_response:
                # Extract content between ```json and ```
                match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_response)
                if match:
                    cleaned_response = match.group(1)
            elif "```" in cleaned_response:
                # Extract content between ``` and ```
                match = re.search(r'```\s*([\s\S]*?)\s*```', cleaned_response)
                if match:
                    cleaned_response = match.group(1)
            
            # First, try to parse the entire response as JSON directly
            data = json.loads(cleaned_response)
            
            # Handle both array format and object format
            if isinstance(data, list):
                # Direct array format: ["obj1", "obj2", ...]
                if len(data) > 0 and isinstance(data[0], str):
                    return {"objects": data}
            elif isinstance(data, dict):
                # Object format: {"test_objects": [...], ...}
                # Try multiple key names for objects (flexible for different prompt formats)
                for key in ["objects", "test_objects", "testObjects", "extracted_objects"]:
                    if key in data:
                        value = data[key]
                        if isinstance(value, list) and len(value) > 0:
                            # If it's a list of strings, return directly
                            if isinstance(value[0], str):
                                return {"objects": value}
                            # If it's a list of objects/dicts, extract 'name' field
                            elif isinstance(value[0], dict):
                                names = []
                                for item in value:
                                    if isinstance(item, dict) and "name" in item:
                                        names.append(item["name"])
                                if names:
                                    return {"objects": names}
            
            # If direct key lookup fails, try regex patterns as fallback
            patterns = [
                r'\{[^}]*"objects"[^}]*\}',      # Simple non-nested
                r'\{.*?"objects".*?\}',           # Non-greedy
                r'\{[\s\S]*?"objects"[\s\S]*?\}', # With special chars and newlines
            ]
            
            for pattern in patterns:
                match = re.search(pattern, cleaned_response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        data = json.loads(json_str)
                        objects = data.get("objects", [])
                        if isinstance(objects, list) and len(objects) > 0:
                            return {"objects": objects}
                    except json.JSONDecodeError:
                        pass
                        
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try cleaning and regex
            try:
                # Remove any remaining non-ASCII characters
                cleaned = ''.join(c if ord(c) < 128 else ' ' for c in cleaned_response)
                data = json.loads(cleaned)
                
                # Try flexible key lookup on cleaned data
                for key in ["objects", "test_objects", "testObjects"]:
                    if key in data:
                        value = data[key]
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], str):
                                return {"objects": value}
                            elif isinstance(value[0], dict) and "name" in value[0]:
                                names = [item.get("name", "") for item in value if "name" in item]
                                if names:
                                    return {"objects": names}
            except:
                pass
                    
        except Exception:
            pass
        
        return {"objects": []}
