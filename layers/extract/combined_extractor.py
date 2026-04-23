"""
Combined Extractor - Extract objects, goals, and activities in a single LLM call.
"""

import re
import json

class CombinedExtractor:
    """Extract objects, activities, and goals from code in a single LLM inference."""
    
    def __init__(self):
        print("[CombinedExtractor] Initializing")
    
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
    
    def extract(self, code_text):
        """
        Extract objects, activities, and goals from code using a single LLM call.
        Returns: {
            "objects": [...],
            "activities": [...],
            "goals": [...]
        }
        """
        if len(code_text) > 2000:
            code_text = code_text[:2000]
        return self._extract_with_model(code_text)
    
    def _extract_with_model(self, code_text):
        """Extract using a single LLM inference call"""
        from model.service_factory import get_inference_backend
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"objects": "primary_object_name", "goals": "verification_purpose", "activities": ["action1", "action2", ...]}}
Output ONLY JSON, no explanations or other text.

⚠️ CRITICAL: Use EXACTLY these field names - NO OTHER NAMES ALLOWED:
- "objects" (NOT testObjects, NOT test_objects)
- "goals" (NOT test_goals, NOT goal)
- "activities" (NOT activity, NOT test_activities)

## Test Object
Definition: The direct, primary target entity that test actions are intentionally applied to.

Steps:
1. Identify all observed entities (directly or indirectly) to form a candidate set.
2. Exclude all auxiliary and supporting information. Keep only the direct, primary target.

Exclusion categories (remove any entity belonging to these):
1. Basic auxiliary objects: test drivers, test tools, test frameworks, test environments.
2. Test framework core syntax/API: test suite/case definitions (describe, it, @Test, etc.).
3. Assertion matchers/expressions: generic assertions (expect, assert, assertEquals, etc.).
4. Test library tool calls: rendering/locating libraries (render, screen.getByTestId, findElement, etc.).
5. Test auxiliary/temporary variables: language-level basic libraries (React, java.util), common temporary variables.
6. Test lifecycle management elements: environment cleanup (cleanup, teardown, etc.).
7. All entities providing only supporting information.

Output: a single string representing the direct, primary test object. Never output null or empty. If no clear primary object exists, output the most significant entity being tested.

## Test Goal
Definition: Expected verification direction and quality judgment for the test object.

Steps:
1. Identify the primary test object.
2. Based on assertions and descriptions, determine what the test verifies about that object.
3. Summarize into 1-2 core verification purposes (not every assertion).

Output: a short string describing the primary verification goal. Never output null or empty.

## Test Activities
Definition: Purposeful operations directed at test objects to trigger, observe, record, and inspect behaviors and states.

Steps:
1. Extract every action from the code without omission.
2. Simplify to concise natural language, preserve execution order.
3. Do not aggregate unless clarity greatly improves (avoid over-aggregation).
4. Write activity names in natural language (no underscores or camelCase).

Output: an array of strings representing each activity in sequence. Never output empty.

[Test Code]
```
{code_text}
```

Extract test objects, goals, and activities systematically according to the definitions above.

REMINDER: FIELD NAMES MUST BE "objects", "goals", "activities" - NOTHING ELSE!

Final answer (ONLY JSON):"""
        
        # Use inference service
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=3000)
        
        print("\n[CombinedExtractor] LLM Response:")
        print(response)
        print()
        
        # Try Method 1: Balanced bracket matching (handles nested structures)
        data = self._extract_json_object(response)
        if data and isinstance(data, dict):
            # Try primary field names first, then fallback to alternative names
            objects = data.get("objects", "") or data.get("testObjects", "")
            activities = data.get("activities", []) or data.get("activity", [])
            goals = data.get("goals", "") or data.get("goal", "") or data.get("test_goals", "")
            
            # Convert single strings to single-item lists for consistency with pipeline
            objects_list = [objects] if isinstance(objects, str) and objects.strip() else (objects if isinstance(objects, list) else [])
            goals_list = [goals] if isinstance(goals, str) and goals.strip() else (goals if isinstance(goals, list) else [])
            activities_list = activities if isinstance(activities, list) else []
            
            if objects_list or activities_list or goals_list:
                return {
                    "objects": objects_list,
                    "activities": activities_list if activities_list else [],
                    "goals": goals_list
                }
        
        # Try Method 2: Original regex patterns (fallback for compatibility)
        try:
            patterns = [
                r'\{[\s\S]*?"objects"[\s\S]*?"activities"[\s\S]*?"goals"[\s\S]*?\}',
                r'\{.*?"objects".*?"activities".*?"goals".*?\}',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        data = json.loads(json_str)
                        # Try primary field names first, then fallback to alternative names
                        objects = data.get("objects", "") or data.get("testObjects", "")
                        activities = data.get("activities", []) or data.get("activity", [])
                        goals = data.get("goals", "") or data.get("goal", "") or data.get("test_goals", "")
                        
                        # Convert single strings to single-item lists for consistency
                        objects_list = [objects] if isinstance(objects, str) and objects.strip() else (objects if isinstance(objects, list) else [])
                        goals_list = [goals] if isinstance(goals, str) and goals.strip() else (goals if isinstance(goals, list) else [])
                        activities_list = activities if isinstance(activities, list) else []
                        
                        if objects_list or activities_list or goals_list:
                            return {
                                "objects": objects_list,
                                "activities": activities_list if activities_list else [],
                                "goals": goals_list
                            }
                    except json.JSONDecodeError:
                        pass
                        
        except Exception:
            pass
        
        # Try Method 3: Clean and retry
        try:
            cleaned = ''.join(c if ord(c) < 128 else ' ' for c in response)
            data = self._extract_json_object(cleaned)
            if data and isinstance(data, dict):
                objects = data.get("objects", "")
                activities = data.get("activities", [])
                goals = data.get("goals", "")
                
                # Convert single strings to single-item lists for consistency
                objects_list = [objects] if isinstance(objects, str) and objects.strip() else (objects if isinstance(objects, list) else [])
                goals_list = [goals] if isinstance(goals, str) and goals.strip() else (goals if isinstance(goals, list) else [])
                activities_list = activities if isinstance(activities, list) else []
                
                if objects_list or activities_list or goals_list:
                    return {
                        "objects": objects_list,
                        "activities": activities_list if activities_list else [],
                        "goals": goals_list
                    }
        except Exception:
            pass
        
        return {
            "objects": [],
            "activities": [],
            "goals": []
        }
