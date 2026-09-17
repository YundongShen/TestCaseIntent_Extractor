#!/usr/bin/env python3
"""
correct_intent.py — Quality-correction pass for extracted test intents.

For each intent JSON in dataset/intent/{framework}/, loads the original raw
source file and calls the Gemini API to produce a corrected TO/TG/TA that
follows the ground truth quality criteria:
  - Objects: English, high-level domain entities only
  - Goals:   1-2 concise verification statements
  - Activities: ordered, aggregated (no repetition across scenarios)
  - Language: always English regardless of source file language

Backs up originals as {name}Intent.orig.json before overwriting.

Usage:
    GOOGLE_API_KEY=<key> python correct_intent.py --framework cucumber
    GOOGLE_API_KEY=<key> python correct_intent.py --framework all
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ground truth reference example (tc_admin_index_api_test)
# ---------------------------------------------------------------------------
GT_EXAMPLE = {
    "objects": [
        "Admin Index API",
        "Admin User",
        "Environment Configuration",
        "Index Metadata",
    ],
    "goals": [
        "Verify the correctness of the Admin Index API details endpoint and metadata structure",
        "Verify synchronization trigger functionality across valid and invalid strategies",
    ],
    "activities": [
        "Set enable index module environment variable",
        "Create an admin user",
        "Request admin index details and verify response status is 200",
        "Verify metadata length is 7 and structure for Product and ProductVariant",
        "Trigger index sync with default, full, and reset strategies and verify status 200",
        "Attempt to trigger sync with invalid strategy and verify status 400",
        "Verify metadata entities have pending or processing status after sync",
        "Delete enable index module environment variable",
    ],
}

CORRECTION_PROMPT_TEMPLATE = """You are producing ground truth test intent for a research dataset.
Re-extract TO/TG/TA directly from the original test source below.
Use the current extraction only as a rough reference — re-derive everything from the source.

[Definitions]

Test Objects (TO):
  The direct, primary entities whose behaviors/attributes/states are tested.
  - English only. Noun phrases. High-level domain names (e.g. "Login Page", "Admin Index API").
  - Exclude: test-framework constructs, assertions, temporary variables, setup helpers.
  - Typically 1-4 objects total.

Test Goals (TG):
  The core verification direction and quality criteria for the test objects.
  - 1-2 sentences. Pattern: "Verify <what> <expected behavior/criterion>".
  - Derived from the scenario descriptions and assertions, NOT individual step details.
  - Include specific correctness criteria where stated (e.g. status codes, counts, messages).

Test Activities (TA):
  The ordered execution steps realizing the test.
  - Follow the ORIGINAL execution order in the source file strictly.
  - Write as clear imperative sentences in English.
  - Preserve concrete values: specific input data, expected messages, URLs, status codes, counts.
  - Each scenario's steps appear separately in order — do NOT merge steps from different scenarios.
  - Simplify implementation syntax into natural language but keep specificity.
  - Exclude pure test-framework boilerplate (imports, describe/it wrappers, assertion library calls).
  - Language: always English even if source is in another language.
  - Fix any encoding corruption (e.g. "Notificacin" -> "Notification").

[Ground Truth Quality Example]
Objects:    {gt_objects}
Goals:      {gt_goals}
Activities: {gt_activities}

[Current (rough) Extraction — for reference only]
{current_json}

[Original Test Source — extract from this]
{raw_source}

Return ONLY valid JSON:
{{"objects": [...], "goals": [...], "activities": [...]}}
No explanations, no markdown fences, no extra keys."""


def build_prompt(current: dict, raw_source: str) -> str:
    return CORRECTION_PROMPT_TEMPLATE.format(
        gt_objects=json.dumps(GT_EXAMPLE["objects"]),
        gt_goals=json.dumps(GT_EXAMPLE["goals"]),
        gt_activities=json.dumps(GT_EXAMPLE["activities"][:4]),
        current_json=json.dumps(
            {k: current[k] for k in ("objects", "goals", "activities")},
            ensure_ascii=False,
            indent=2,
        ),
        raw_source=raw_source[:3000],  # cap to avoid token overflow
    )


def extract_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping any markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end])


def find_raw_file(framework: str, intent_stem: str):
    """
    Given intent filename stem (e.g. 'Login' from 'LoginIntent.json'),
    find the corresponding raw source file in dataset/raw/{framework}/.
    """
    raw_dir = Path("dataset/raw") / framework
    if not raw_dir.exists():
        return None
    # The intent stem is the original filename without extension, e.g. "Login"
    # Raw file could be Login.feature, Login.js, etc.
    for candidate in raw_dir.iterdir():
        if candidate.stem.lower() == intent_stem.lower():
            return candidate
    # Fallback: case-insensitive prefix match
    for candidate in raw_dir.iterdir():
        if candidate.stem.lower() == intent_stem.lower():
            return candidate
    return None


def correct_framework(framework: str, service) -> None:
    intent_dir = Path("dataset/intent") / framework
    if not intent_dir.exists():
        print(f"[SKIP] No intent directory for framework: {framework}")
        return

    intent_files = sorted(
        f for f in intent_dir.glob("*.json") if not f.name.endswith(".orig.json")
    )
    if not intent_files:
        print(f"[SKIP] No intent files in {intent_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Framework: {framework} ({len(intent_files)} files)")
    print(f"{'='*60}")

    for intent_path in intent_files:
        # Derive raw filename stem: "LoginIntent.json" → "Login"
        stem = intent_path.stem  # e.g. "LoginIntent"
        if stem.endswith("Intent"):
            raw_stem = stem[: -len("Intent")]
        else:
            raw_stem = stem

        raw_path = find_raw_file(framework, raw_stem)
        if raw_path is None:
            print(f"  [WARN] Raw file not found for {intent_path.name}, skipping")
            continue

        # Load
        with open(intent_path, encoding="utf-8") as f:
            current = json.load(f)
        with open(raw_path, encoding="utf-8", errors="replace") as f:
            raw_source = f.read()

        print(f"  Correcting: {intent_path.name}  ←  {raw_path.name}")

        # Build prompt and call API
        prompt = build_prompt(current, raw_source)
        try:
            response = service.infer(prompt, max_tokens=1500)
        except Exception as e:
            print(f"    [ERROR] API call failed: {e}")
            continue

        # Parse response
        try:
            corrected = extract_json(response)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"    [ERROR] JSON parse failed: {e}")
            print(f"    Raw response: {response[:300]}")
            continue

        # Validate keys
        for key in ("objects", "goals", "activities"):
            if key not in corrected or not isinstance(corrected[key], list):
                print(f"    [ERROR] Missing or invalid key '{key}' in response")
                corrected = None
                break

        if corrected is None:
            continue

        # Backup original
        backup_path = intent_path.with_suffix(".orig.json")
        if not backup_path.exists():
            shutil.copy2(intent_path, backup_path)

        # Merge: keep metadata fields, update TO/TG/TA
        current["objects"] = corrected["objects"]
        current["goals"] = corrected["goals"]
        current["activities"] = corrected["activities"]
        current["correction_model"] = service.model_name

        with open(intent_path, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)

        print(f"    ✓ objects={len(corrected['objects'])}  "
              f"goals={len(corrected['goals'])}  "
              f"activities={len(corrected['activities'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct extracted test intents via Gemini API")
    parser.add_argument(
        "--framework",
        default="cucumber",
        help="Framework subdirectory to correct, or 'all' for every framework",
    )
    args = parser.parse_args()

    # Initialise Gemini service
    from model.api_inference_service import get_gemini_service
    try:
        service = get_gemini_service()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Set GOOGLE_API_KEY environment variable before running.")
        sys.exit(1)

    if args.framework == "all":
        intent_root = Path("dataset/intent")
        frameworks = sorted(d.name for d in intent_root.iterdir() if d.is_dir())
    else:
        frameworks = [args.framework]

    for fw in frameworks:
        correct_framework(fw, service)

    print("\nDone.")


if __name__ == "__main__":
    main()
