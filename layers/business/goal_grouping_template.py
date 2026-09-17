"""
Prompt for the goal-grouping step: turns each case's Test Goal into a higher
level "shared testing question" and clusters cases that answer the same one.
Works on already-extracted case-level records (object + goal) only — it
doesn't re-extract or merge anything.
"""

PROMPT = """You are analyzing a set of test cases that belong to one test suite. Each test case already has its own independently extracted Test Object and Test Goal.

Your task, for the test cases given below:
1. For each test case, abstract its specific Test Goal into a more GENERAL testing question: a higher-level question about the capability, property, or behavior class being verified. Do not restate the specific scenario (input values, error strings, HTTP codes); describe the general dimension it tests.
2. Group test cases whose specific goals are different facets of the SAME general testing question into one group. Give a fundamentally different dimension its own group.
3. Prefer a small number of meaningful groups over many single-case groups, but never force unrelated test cases into the same group just to reduce the count.
4. Keep the original given order of test cases; a group's test_case_ids must list them in that same relative order, and groups must be ordered by the position of their first test case.

【Test Suite: {suite_id}】
{cases_block}

Return ONLY valid JSON, no explanations, no markdown fences, in exactly this shape:
{{"groups": [{{"shared_question": "<general testing question, phrased as a question>", "test_case_ids": ["<id>", "..."]}}]}}

IMPORTANT: Do not show your reasoning, analysis, or alternative options. Output only the final JSON object, starting your response immediately with "{{"."""


def build_prompt(suite_id, cases):
    """Fill in the prompt with each case's id/object/goal."""
    lines = []
    for c in cases:
        lines.append(f"- {c['id']} | Object: {c['object']} | Goal: {c['goal']}")
    cases_block = "\n".join(lines)
    return PROMPT.format(suite_id=suite_id, cases_block=cases_block)
