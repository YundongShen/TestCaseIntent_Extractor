#!/usr/bin/env python3
"""
LLM-judge alignment scorer for Test Intent extraction (Section 6.2).

For each (case x extraction-set) it compares the extracted <objects, goals,
activities> against the human ground-truth intent and assigns one holistic,
lenient label: Fully Aligned / Partially Aligned / Misaligned.

Empty extractions (all three lists empty) are auto-labelled Misaligned without
an API call. Results are appended to eval/scores/scores.jsonl and the run is
resumable (already-scored case+set pairs are skipped).

Env: same Vertex interface as the Gemini 3.1 Pro extraction —
    VERTEX_API_KEY (from .env), GOOGLE_CLOUD_LOCATION=global.

Usage:
    python eval/judge.py                      # all 98 cases x 6 sets
    python eval/judge.py --framework k6       # trial: one framework
    python eval/judge.py --sets gemini_pro    # limit to some sets
"""
import os, sys, json, time, argparse, glob

SETS = ["qwen_independent", "qwen_combined", "qwen_chain",
        "deepseek", "gemini_flash", "gemini_pro"]

RUBRIC = """You are evaluating whether an AUTOMATICALLY EXTRACTED Test Intent matches a \
human-written GROUND-TRUTH Test Intent for the same test case.

Judge the OVERALL semantic alignment holistically and LENIENTLY:
- Objects align if they refer to the same entity/entities, even with different
  wording, a different angle, or minor extra/missing secondary objects.
- Goal aligns if the verification purpose is similar, even if worded differently.
- Activities align if they describe the same operations, even at a different level
  of detail or with non-critical steps missing.

Assign exactly ONE label:
- "Fully Aligned": overall the extracted intent means the same as the ground truth
  (allowing all the lenient differences above).
- "Partially Aligned": it captures the general intent but misses or gets wrong a
  NON-TRIVIAL part (e.g. wrong or missing core object, only part of the goal, or
  major activities missing/wrong).
- "Misaligned": the overall meaning does not match, the verification target is
  wrong, or the extraction is empty.

Respond with ONLY a JSON object: {"label": "...", "rationale": "<=1 sentence"}"""


def build_prompt(gt_text, ext):
    return f"""{RUBRIC}

=== GROUND TRUTH ===
{gt_text.strip()}

=== EXTRACTED TEST INTENT ===
Test Objects: {ext.get('objects')}
Test Goal: {ext.get('goals')}
Test Activities: {ext.get('activities')}

JSON:"""


def parse_label(text):
    import re
    t = text.strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            lab = d.get("label", "").strip()
            for canon in ("Fully Aligned", "Partially Aligned", "Misaligned"):
                if canon.lower() in lab.lower():
                    return canon, d.get("rationale", "")
        except Exception:
            pass
    low = t.lower()
    if "fully" in low:   return "Fully Aligned", t[:120]
    if "partial" in low: return "Partially Aligned", t[:120]
    if "misalign" in low: return "Misaligned", t[:120]
    return "PARSE_ERROR", t[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", default=None)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--out", default="eval/scores/scores.jsonl")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    rows = json.load(open("eval/mapping_98.json"))
    if args.framework:
        rows = [r for r in rows if r["framework"] == args.framework]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                d = json.loads(line); done.add((d["case_key"], d["set"]))
            except Exception:
                pass

    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, api_key=os.environ["VERTEX_API_KEY"])
    MODEL = "gemini-3.1-pro-preview"

    out = open(args.out, "a")
    n_api = n_empty = n_skip = 0
    for r in rows:
        case_key = f"{r['framework']}/{r['case']}"
        gt_text = open(r["gt"]).read()
        for s in args.sets:
            if (case_key, s) in done:
                n_skip += 1; continue
            ext = json.load(open(r[s]))
            empty = not (ext.get("objects") or ext.get("goals") or ext.get("activities"))
            if empty:
                rec = {"case_key": case_key, "framework": r["framework"],
                       "category": r["category"], "set": s,
                       "label": "Misaligned", "rationale": "empty extraction",
                       "auto": True}
                n_empty += 1
            else:
                prompt = build_prompt(gt_text, ext)
                for attempt in range(5):
                    try:
                        resp = client.models.generate_content(
                            model=MODEL, contents=prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=2000, temperature=0.0))
                        label, rationale = parse_label(resp.text or "")
                        break
                    except Exception as e:
                        if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                            time.sleep([10, 20, 40, 60][attempt]); continue
                        label, rationale = "API_ERROR", str(e)[:150]
                        break
                rec = {"case_key": case_key, "framework": r["framework"],
                       "category": r["category"], "set": s,
                       "label": label, "rationale": rationale, "auto": False}
                n_api += 1
                time.sleep(3)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            print(f"[{case_key:22s} {s:16s}] {rec['label']}")
    out.close()
    print(f"\nDone. api={n_api} empty_auto={n_empty} skipped={n_skip}")


if __name__ == "__main__":
    main()
