#!/usr/bin/env python3
"""
Dataset Ground Truth Extraction Script
Runs only Layer 1 (Input) + Layer 2 (Extract) and saves results to dataset/intent/.
Skips Intent / Business / Output layers to reduce compute time.

Usage:
    TEST_FILE=dataset/raw/jest_test/admin-index-api.test.js \
    MODEL_TYPE=qwen EXTRACT_MODE=independent INFERENCE_BACKEND=local \
    python extract_for_dataset.py
"""

import sys
import os
import json
from datetime import datetime

# ── Model configuration (mirrors main.py) ────────────────────────────────────
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "local")
MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen")
os.environ["MODEL_TYPE"] = MODEL_TYPE

if INFERENCE_BACKEND == "api":
    from model.api_inference_service import get_gemini_service
    _svc = get_gemini_service()
    MODEL_CONFIG = {
        "model_name": _svc.model_name,
        "model_id": _svc.model_name,
        "quantization": "API",
    }
    print(f"[Config] Using API backend: {_svc.model_name}")
else:
    if MODEL_TYPE == "qwen":
        from model.model_config_qwen import set_seed, MODEL_CONFIG
        print("[Config] Using Qwen-3.5-27B (27B, BF16)")
    elif MODEL_TYPE == "v3":
        from model.model_config_v3 import set_seed, MODEL_CONFIG
        print("[Config] Using V3 model (671B MoE, FP8)")
    else:
        from model.model_config_7b import set_seed, MODEL_CONFIG
        print("[Config] Using 7B model (FP16)")

    from model.inference_service import set_model_config
    set_model_config(MODEL_CONFIG)
# ─────────────────────────────────────────────────────────────────────────────

from layers import InputLayer, ExtractLayer


def derive_output_path(source_file: str) -> str:
    """
    Map dataset/raw/{framework}/filename.ext → dataset/intent/{framework}/filenameIntent.json
    Works whether source_file is absolute or relative.
    """
    # Normalise to forward slashes and strip leading ./
    parts = source_file.replace("\\", "/").lstrip("./").split("/")

    # Find 'raw' segment to locate the framework name
    try:
        raw_idx = parts.index("raw")
    except ValueError:
        raise ValueError(f"Cannot derive output path: 'raw' not found in '{source_file}'")

    framework = parts[raw_idx + 1]
    basename = os.path.splitext(parts[-1])[0]  # drop original extension
    output_filename = f"{basename}Intent.json"

    # Build output dir relative to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    intent_root = os.getenv("OUTPUT_DIR", "dataset/intent")
    output_dir = os.path.join(project_root, intent_root, framework)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, output_filename), framework


def load_test_file(path: str) -> dict:
    """Load a raw test file and return the input dict for the pipeline."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    basename = os.path.splitext(os.path.basename(path))[0]
    test_case_id = "tc_" + basename.replace("-", "_").replace(".", "_")

    return {
        "content": content,
        "user_id": "dataset_builder",
        "test_case_id": test_case_id,
    }


def run_extraction(source_file: str):
    """Run Layer 1 + Layer 2 and save result to dataset/intent/."""
    extract_mode = os.getenv("EXTRACT_MODE", "independent")

    print("=" * 70)
    print("[Dataset Ground Truth Extraction]")
    print(f"  File   : {source_file}")
    print(f"  Mode   : {extract_mode}")
    print(f"  Model  : {MODEL_CONFIG.get('model_name', 'unknown')}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Layer 1: Input ────────────────────────────────────────────────────────
    raw_data = load_test_file(source_file)
    data1 = InputLayer().process(raw_data)
    print(f"[Layer 1] Done — {len(data1.get('content', ''))} chars")

    # ── Layer 2: Extract ──────────────────────────────────────────────────────
    data2 = ExtractLayer(extract_mode=extract_mode).process(data1)
    objects = data2.get("objects", [])
    goals = data2.get("goals", [])
    activities = data2.get("activities", [])
    print(f"[Layer 2] Done — objects={len(objects)} goals={len(goals)} activities={len(activities)}")

    # ── Save result ───────────────────────────────────────────────────────────
    output_path, framework = derive_output_path(source_file)

    result = {
        "test_case_id": data2.get("test_case_id", raw_data["test_case_id"]),
        "source_file": source_file,
        "framework": framework,
        "extraction_timestamp": datetime.now().isoformat(),
        "extract_mode": extract_mode,
        "model": MODEL_CONFIG.get("model_name", "unknown"),
        "objects": objects,
        "goals": goals,
        "activities": activities,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Save]   Result written → {output_path}")
    print(f"[Done]   {datetime.now().isoformat()}")
    return output_path


def main():
    source_file = os.getenv("TEST_FILE")
    if not source_file:
        print("ERROR: TEST_FILE environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(source_file):
        print(f"ERROR: File not found: {source_file}", file=sys.stderr)
        sys.exit(1)

    try:
        run_extraction(source_file)
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
