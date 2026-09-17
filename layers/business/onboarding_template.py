"""
Prompt template for the suite-level onboarding document. Takes case-level
Test Intent records that already belong to one suite and are already
grouped, and just formats them — extraction still happens per test case.
"""

PROMPT = """You are a test onboarding document expert. Generate a structured onboarding document strictly following the format below. All content must be derived only from the suite information, groups, and per-test-case Test Intent records provided to you. Do not invent test cases, objects, goals, or activities that are not given.

【Document Format Structure】

## Test Suite Overview

**Suite ID:** `(suite id)`
**Test cases:** (number of test cases)

(One paragraph summarizing, in plain language, what this suite as a whole verifies.)

- **(short descriptive name for the suite)**
  - **Unique Dependencies:** `(dep)`, `(dep)`, ...
  - **(test case id):** (short 2-4 word label for the case) (dependencies: `(dep)`, `(dep)`)
  - (repeat one bullet per test case, in the given order)

---

(Repeat the following block once per group, in the given group order.)

## Group (n): (comma-separated test case ids in this group)

**Shared testing question:** (the general testing question given for this group, phrased as a single question)

### `(test case id)` - (short 2-5 word title for the case)

**Test Objects**

(the object(s) for this test case)

**Test Goals**

(the goal for this test case)

**Test Activities**

- Step 1: (activity)
- Step 2: (activity)
- ... (one bullet per activity, in order)

(repeat the `###` block for every test case in this group, then "---" before the next group; no "---" after the last group)

【Format Rules】
- Use plain markdown, no emoji, no extra commentary.
- Keep the exact section markers shown above: "## Test Suite Overview", "## Group (n): ...", "### `TC-ID` - Title".
- The suite's short name, per-case labels, and per-case titles must be concise and derived from the actual objects/goals given, not generic placeholders.
- Do not renumber, merge, or drop any test case. Every test case given to you must appear exactly once, in exactly the group it was assigned to.
- Do not re-decide the grouping: use the groups and shared testing questions exactly as given.

【Generation Requirements】
Fill in every section with concrete content derived only from the given suite info, groups, and per-test-case Test Intent data. Do not leave placeholder text.

IMPORTANT: Output the final document directly. Do not include any thinking process, reasoning steps, analysis, or preamble. Start your response immediately with "## Test Suite Overview" and output every section in order."""
