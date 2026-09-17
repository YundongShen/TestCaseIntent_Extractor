"""
Goal Extractor - Extract test goals using DeepSeek LLM with COT reasoning.
"""

import re
import json

class GoalExtractor:
    """Extract test goals from code using inference service."""
    
    def __init__(self):
        print("[GoalExtractor] Initializing (uses InferenceService)")

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
        """Extract test goals from code. Returns: {"goals": [list]}"""
        
        prompt = f"""IMPORTANT: Return ONLY valid JSON in this exact format: {{"goals": ["goal1", "goal2", ...]}}
Output ONLY JSON, no explanations or other text.

[Definition]
Test Goal is the expected verification direction and quality judgment criteria formulated for the test object, supported by test activities.

[Extraction Steps]
1. Identify the primary test object from the test code (for test case, there is only one main object). Although multiple objects may appear, a typical test case has one dominant verification purpose; the goal is defined for the most important object.
2. Determine the verification target of the primary test object based on explicit information in the code (e.g., assertions, descriptions). The derived goal must faithfully summarize the actual verification purpose.
3. (Optional) Adjust granularity as needed—this step may be omitted; output the goal at a reasonable level of detail.

Extract the PRIMARY/DOMINANT goal(s) for this test case - typically just 1-2 core verification purposes, not all individual assertions.

[Test Code]
```
{code_text}
```

Extract the PRIMARY/DOMINANT goal(s) for this test case - typically just 1-2 core verification purposes, not all individual assertions.

Final answer (ONLY JSON):"""
        
        from model.service_factory import get_inference_backend
        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=9000)

        print("\n[GoalExtractor] LLM Response:")
        print(response)
        print()

        # Prefer the last valid JSON candidate containing the target key.
        for data in reversed(self._extract_json_candidates(response)):
            if isinstance(data, dict) and "goals" in data:
                goals = data.get("goals", [])
                if isinstance(goals, list):
                    return {"goals": goals}

        # Regex fallback for malformed wrappers.
        try:
            matches = re.findall(r"\{[\s\S]*?\"goals\"[\s\S]*?\}", response, re.DOTALL)
            for match in reversed(matches):
                data = json.loads(match)
                goals = data.get("goals", [])
                if isinstance(goals, list):
                    return {"goals": goals}
        except Exception:
            pass
        
        return {"goals": []}
