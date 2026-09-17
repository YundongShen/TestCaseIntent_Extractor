#!/usr/bin/env python3
"""
Load the 31 per-case extraction results written by
extract_one_case_for_onboarding.py and generate the 5 onboarding documents.
"""
import os
import json

os.environ["INFERENCE_BACKEND"] = "local"
os.environ["MODEL_TYPE"] = "qwen"

from model.model_config_qwen import MODEL_CONFIG
from model.inference_service import set_model_config
set_model_config(MODEL_CONFIG)

from suites_data import SUITES
from layers.business.onboarding_generator import OnboardingGenerator
from layers.business.onboarding_template import PROMPT

CASE_DIR = "Result/onboarding_result/case_extractions"


def load_cases(suite):
    cases = []
    for i in range(1, len(suite["cases"]) + 1):
        case_id = f"TC-{suite['suite_id'].split('-')[0]}-{i:02d}"
        path = os.path.join(CASE_DIR, f"{suite['suite_id']}_{case_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing extraction for {suite['suite_id']} / {case_id}: {path}")
        with open(path, encoding="utf-8") as f:
            cases.append(json.load(f))
    return cases


def main():
    gen = OnboardingGenerator()
    all_results = []

    for suite in SUITES:
        print("\n" + "#" * 70)
        print(f"# SUITE: {suite['suite_id']}  ({len(suite['cases'])} test cases)")
        print("#" * 70)

        cases = load_cases(suite)
        for c in cases:
            print(f"  {c['case_id']}: obj={c['object']!r} goal={c['goal']!r}")

        suite_info = {
            "suite_id": suite["suite_id"],
            "description": suite["description"],
            "dependencies": suite["dependencies"],
        }

        group_input = [{"id": c["case_id"], "object": c["object"], "goal": c["goal"]} for c in cases]
        group_result = gen.group_cases(suite_info["suite_id"], group_input)
        if group_result["status"] != "success":
            print(f"[FAILED] grouping: {group_result['error']}")
            continue
        for g in group_result["groups"]:
            print(f"  Group {g['test_case_ids']} -> {g['shared_question']}")

        cases_for_doc = [
            {"id": c["case_id"], "object": c["object"], "goal": c["goal"],
             "activities": c["activities"], "dependencies": c["dependencies"]}
            for c in cases
        ]
        doc_result = gen.generate_suite_document(PROMPT, suite_info, cases_for_doc, group_result["groups"])
        if doc_result["status"] != "success":
            print(f"[FAILED] document generation: {doc_result['error']}")
            continue

        safe_name = suite["suite_id"].lower().replace("-", "_")
        out_path = f"Result/onboarding_result/onboarding_{safe_name}_qwen_independent.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc_result["document"])
        print(f"[SUCCESS] Document saved: {out_path}")
        all_results.append(out_path)

    print("\n" + "=" * 70)
    print(f"Done. Generated {len(all_results)}/{len(SUITES)} documents:")
    for p in all_results:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
