#!/usr/bin/env python3
"""
Same grouped-suite template, tried on the 5 cases in SilderMenu.test.js
(getMeunMatcheys path matching). Cases come from the reviewed Groundtruth.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["INFERENCE_BACKEND"] = "local"
os.environ["MODEL_TYPE"] = "qwen"

from model.model_config_qwen import MODEL_CONFIG
from model.inference_service import set_model_config
set_model_config(MODEL_CONFIG)

from layers.business.onboarding_generator import OnboardingGenerator
from layers.business.onboarding_template import PROMPT


SUITE_INFO = {
    "suite_id": "MENU-MATCH-SUITE-01",
    "description": (
        "This suite verifies how the getMeunMatcheys function matches a "
        "requested URL path against a fixed list of menu routes, covering "
        "exact matches, non-matches, and parameterized route patterns."
    ),
    "dependencies": ["./SiderMenu"],
}

CASES = [
    {
        "id": "TC-MENU-01",
        "object": "The `getMeunMatcheys` function",
        "goal": "Verify that an exact simple path is correctly matched.",
        "activities": [
            "Call `getMeunMatcheys(meun, '/dashboard')`.",
            "Assert the return value as `['/dashboard']`.",
        ],
        "dependencies": ["./SiderMenu"],
    },
    {
        "id": "TC-MENU-02",
        "object": "The `getMeunMatcheys` function",
        "goal": "Verify that a non-matching path returns an empty result.",
        "activities": [
            "Call `getMeunMatcheys(meun, '/dashboardname')`.",
            "Assert the return value as `[]`.",
        ],
        "dependencies": ["./SiderMenu"],
    },
    {
        "id": "TC-MENU-03",
        "object": "The `getMeunMatcheys` function",
        "goal": "Verify that a second-level path is correctly matched.",
        "activities": [
            "Call `getMeunMatcheys(meun, '/dashboard/name')`.",
            "Assert the return value as `['/dashboard/name']`.",
        ],
        "dependencies": ["./SiderMenu"],
    },
    {
        "id": "TC-MENU-04",
        "object": "The `getMeunMatcheys` function",
        "goal": "Verify that a parameterized path is correctly matched against the route pattern.",
        "activities": [
            "Call `getMeunMatcheys(meun, '/userinfo/2144')`.",
            "Assert the return value as `['/userinfo/:id']`.",
        ],
        "dependencies": ["./SiderMenu"],
    },
    {
        "id": "TC-MENU-05",
        "object": "The `getMeunMatcheys` function",
        "goal": "Verify that a nested parameterized path is correctly matched against the route pattern.",
        "activities": [
            "Call `getMeunMatcheys(meun, '/userinfo/2144/info')`.",
            "Assert the return value as `['/userinfo/:id/info']`.",
        ],
        "dependencies": ["./SiderMenu"],
    },
]


def main():
    gen = OnboardingGenerator()

    print("=" * 60)
    print("STEP 1: goal grouping")
    print("=" * 60)
    group_input = [{"id": c["id"], "object": c["object"], "goal": c["goal"]} for c in CASES]
    group_result = gen.group_cases(SUITE_INFO["suite_id"], group_input)
    if group_result["status"] != "success":
        print(f"[FAILED] grouping: {group_result['error']}")
        return
    for g in group_result["groups"]:
        print(f"  Group {g['test_case_ids']} -> {g['shared_question']}")

    print()
    print("=" * 60)
    print("STEP 2: document generation")
    print("=" * 60)
    doc_result = gen.generate_suite_document(PROMPT, SUITE_INFO, CASES, group_result["groups"])
    if doc_result["status"] != "success":
        print(f"[FAILED] document generation: {doc_result['error']}")
        return

    out_path = "Result/onboarding_result/onboarding_silermenu_qwen.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc_result["document"])

    print(f"\n[SUCCESS] Document saved: {out_path}\n")
    print("=" * 60)
    print(doc_result["document"])


if __name__ == "__main__":
    main()
