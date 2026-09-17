#!/usr/bin/env python3
"""
Quick test: load an existing extract result and run Layer 4 + 5 only.
Usage: python test_onboarding.py <extract_result.json>
"""
import sys
import os
import json
from datetime import datetime

# Set local Qwen backend
os.environ["INFERENCE_BACKEND"] = "local"
os.environ["MODEL_TYPE"] = "qwen"

from model.model_config_qwen import MODEL_CONFIG
from model.inference_service import set_model_config
set_model_config(MODEL_CONFIG)

from layers import BusinessLayer, OutputLayer


def main():
    extract_file = sys.argv[1] if len(sys.argv) > 1 else \
        "Result/extract_result/extract_result_20260426_185510.json"

    print(f"[test_onboarding] Loading extract result: {extract_file}")
    with open(extract_file, "r", encoding="utf-8") as f:
        extract = json.load(f)

    # Simulate Layer 3 passthrough data
    data3 = {
        "objects":    extract.get("objects", []),
        "goals":      extract.get("goals", []),
        "activities": extract.get("activities", []),
        "test_case_id": extract.get("test_case_id", "unknown"),
        "user_id":    extract.get("user_id", "unknown"),
        "specificity": 0.7,
        "business_purpose": "Document generation",
    }

    print(f"  objects:    {data3['objects']}")
    print(f"  goals:      {len(data3['goals'])} items")
    print(f"  activities: {len(data3['activities'])} items\n")

    # Layer 4: Business (loads PROMPT template, no LLM)
    print("=" * 60)
    print("Layer 4: BUSINESS LAYER")
    print("=" * 60)
    data4 = BusinessLayer().process(data3)
    print(f"Prompt loaded: {len(data4.get('prompt', ''))} chars\n")

    # Layer 5: Output (calls Qwen to generate document)
    print("=" * 60)
    print("Layer 5: OUTPUT LAYER")
    print("=" * 60)
    result = OutputLayer().process(data4)

    if result.get("success"):
        print(f"\n[SUCCESS] Document saved: {result.get('filepath')}")
        # Print document content
        with open(result["filepath"], "r", encoding="utf-8") as f:
            print("\n" + "=" * 60)
            print(f.read())
    else:
        print(f"\n[FAILED] {result.get('error')}")


if __name__ == "__main__":
    main()
