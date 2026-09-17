#!/usr/bin/env python3
"""
Extract one isolated test case (independent mode, Qwen3.5-27B), one task of
a SLURM array over suites_data.SUITES so the 31 cases run in parallel.

Usage: python extract_one_case_for_onboarding.py --task-index <0..30>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["INFERENCE_BACKEND"] = "local"
os.environ["MODEL_TYPE"] = "qwen"

from model.model_config_qwen import MODEL_CONFIG
from model.inference_service import set_model_config
set_model_config(MODEL_CONFIG)

from layers import InputLayer, ExtractLayer
from suites_data import SUITES, isolate_case_source


def flatten_tasks():
    tasks = []
    for suite in SUITES:
        for i, (title, body) in enumerate(suite["cases"], 1):
            case_id = f"TC-{suite['suite_id'].split('-')[0]}-{i:02d}"
            tasks.append((suite, case_id, title, body))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, required=True)
    args = ap.parse_args()

    tasks = flatten_tasks()
    suite, case_id, title, body = tasks[args.task_index]

    out_dir = "Result/onboarding_result/case_extractions"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{suite['suite_id']}_{case_id}.json")

    print(f"[task {args.task_index}] suite={suite['suite_id']} case={case_id} title={title!r}")

    source = isolate_case_source(suite, title, body)
    input_layer = InputLayer()
    extract_layer = ExtractLayer(extract_mode="independent")

    pre = input_layer.process({"content": source})
    result = extract_layer.process(pre)
    objects = result.get("objects", [])
    goals = result.get("goals", [])
    activities = result.get("activities", [])

    print(f"  objects={objects}")
    print(f"  goals={goals}")
    print(f"  activities={activities}")

    record = {
        "suite_id": suite["suite_id"],
        "case_id": case_id,
        "title": title,
        "object": "; ".join(objects) if objects else "(none extracted)",
        "goal": "; ".join(goals) if goals else "(none extracted)",
        "activities": activities if activities else [],
        "dependencies": suite["dependencies"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[SUCCESS] wrote {out_path}")


if __name__ == "__main__":
    main()
