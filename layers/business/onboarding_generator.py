"""Turns test intents into onboarding documents via the LLM."""

import json
import re
from datetime import datetime
import os

from layers.business.goal_grouping_template import build_prompt as build_grouping_prompt


class OnboardingGenerator:
    """Uses LLM to generate formatted Onboarding documents"""

    def __init__(self):
        print("[ONBOARDING GENERATOR] Initializing generator")

    def group_cases(self, suite_id, cases):
        """Cluster case-level records (each {"id", "object", "goal"}) under a
        shared testing question. Returns {"status", "groups", "error"} where
        groups is [{"shared_question", "test_case_ids"}]."""
        from model.service_factory import get_inference_backend

        prompt = build_grouping_prompt(suite_id, cases)
        print("[ONBOARDING GENERATOR] Calling LLM to group test cases by shared testing question")
        service = get_inference_backend()
        raw = service.infer_no_thinking(prompt, max_tokens=1000)

        parsed = self._parse_json_object(raw)
        if parsed is None or "groups" not in parsed:
            return {"status": "error", "groups": [], "error": f"could not parse grouping JSON:\n{raw}"}

        all_ids = {c["id"] for c in cases}
        seen = set()
        for g in parsed["groups"]:
            seen.update(g.get("test_case_ids", []))
        missing = all_ids - seen
        if missing:
            print(f"[ONBOARDING GENERATOR] Warning: grouping omitted case ids {missing}; appending as their own group")
            parsed["groups"].append({
                "shared_question": "What does this test case verify on its own?",
                "test_case_ids": sorted(missing),
            })

        return {"status": "success", "groups": parsed["groups"], "error": None}

    @staticmethod
    def _parse_json_object(text):
        """Pull out the groups JSON, in case the model left reasoning text
        around it. Scans top-level {...} spans by brace counting and takes
        the last one that parses and has a "groups" key."""
        if not text:
            return None
        candidates = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start:i + 1])
        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "groups" in obj:
                return obj
        return None

    def generate_suite_document(self, prompt_template, suite_info, cases, groups):
        """Render the suite-level onboarding doc from suite_info, the
        case-level records, and the groups from group_cases()."""
        prompt = self._build_suite_prompt(prompt_template, suite_info, cases, groups)
        print("[ONBOARDING GENERATOR] Calling LLM to generate suite-level document")
        return self._generate_with_model(prompt, marker="## Test Suite Overview", use_thinking=False)

    @staticmethod
    def _build_suite_prompt(format_guide, suite_info, cases, groups):
        by_id = {c["id"]: c for c in cases}

        cases_desc = []
        for c in cases:
            deps = ", ".join(f"`{d}`" for d in c.get("dependencies", []))
            cases_desc.append(f"- {c['id']} (dependencies: {deps})")
        cases_block = "\n".join(cases_desc)

        groups_desc = []
        for gi, g in enumerate(groups, 1):
            groups_desc.append(f"Group {gi}: {', '.join(g['test_case_ids'])}")
            groups_desc.append(f"Shared testing question: {g['shared_question']}")
            for cid in g["test_case_ids"]:
                c = by_id.get(cid)
                if not c:
                    continue
                acts = "\n".join(f"  - {a}" for a in c.get("activities", []))
                groups_desc.append(
                    f"  [{cid}]\n"
                    f"  Object: {c['object']}\n"
                    f"  Goal: {c['goal']}\n"
                    f"  Activities:\n{acts}"
                )
            groups_desc.append("")
        groups_block = "\n".join(groups_desc)

        deps_str = ", ".join(f"`{d}`" for d in suite_info.get("dependencies", []))

        return f"""{format_guide}

【Suite Info】
Suite ID: {suite_info['suite_id']}
Test cases: {len(cases)}
Suite description: {suite_info.get('description', '')}
Unique Dependencies: {deps_str}

【Test Cases (in order)】
{cases_block}

【Groups (already decided, do not re-group)】
{groups_block}

Based on the above content and format requirements, generate the Onboarding document:"""

    def generate(self, prompt_template, objects, goals, activities):
        """Single-case onboarding doc: format guidance plus the object/goal/activity lists."""
        prompt = self._build_prompt(prompt_template, objects, goals, activities)
        
        print("[ONBOARDING GENERATOR] Calling LLM to generate")
        return self._generate_with_model(prompt)
    
    def _generate_with_model(self, prompt, marker="# Test Intent-Driven Onboarding Document", use_thinking=True):
        """Generate document using LLM inference service"""
        from model.service_factory import get_inference_backend

        service = get_inference_backend()
        if use_thinking:
            document = service.infer(prompt, max_tokens=8000)
        else:
            document = service.infer_no_thinking(prompt, max_tokens=8000)

        # Strip any thinking preamble — find the last occurrence of the document title
        # (Qwen sometimes writes the marker in its thinking too, so we take the last match)
        if marker in document:
            document = document[document.rindex(marker):]

        print("[ONBOARDING GENERATOR] Generation completed")
        return {"status": "success", "document": document, "error": None}
    
    def _build_prompt(self, format_guide, objects, goals, activities):
        """
        Build generation prompt: format guidance + test intent content.
        """
        objects_str = "\n".join([f"- {obj}" for obj in objects])
        goals_str = "\n".join([f"- {goal}" for goal in goals])
        activities_str = "\n".join([f"- {activity}" for activity in activities])
        
        return f"""{format_guide}

【Test Intent Content to Generate】
Objects:
{objects_str}

Goals:
{goals_str}

Activities:
{activities_str}

Based on the above content and format requirements, generate Onboarding document:"""
    
    def save_document(self, document):
        """Save to onboarding_result/ with an auto-generated timestamped filename."""
        try:
            # Create onboarding_result folder (if not exists)
            result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "onboarding_result")
            os.makedirs(result_dir, exist_ok=True)
            
            # Generate timestamped filename: onboarding_20260326_141530.md
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"onboarding_{timestamp}.md"
            output_path = os.path.join(result_dir, filename)
            
            # Save document
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(document)
            
            print(f"[ONBOARDING GENERATOR] Document saved to: {output_path}")
            
            return {
                "status": "success",
                "path": output_path,
                "error": None
            }
        except Exception as e:
            print(f"[ONBOARDING GENERATOR] Save failed: {e}")
            return {
                "status": "error",
                "path": "",
                "error": str(e)
            }
