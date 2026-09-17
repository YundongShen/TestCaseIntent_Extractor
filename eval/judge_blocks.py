#!/usr/bin/env python3
"""
Block-level (test-case-level) LLM-judge scorer. ~211 test cases.

Each ground-truth BLOCK is one test case. The extraction under comparison is the
file-level Intent.json (may cover several test cases from the same file). The
judge decides whether THIS test case's intent is captured/aligned in the
extraction: Fully / Partially / Misaligned.

Resumable; appends to eval/scores/scores_blocks.jsonl.
Env: VERTEX_API_KEY (.env), GOOGLE_CLOUD_LOCATION=global.

Usage:
    python eval/judge_blocks.py --sets deepseek qwen_independent gemini_flash gemini_pro
    python eval/judge_blocks.py --sets qwen_combined qwen_chain
"""
import os, sys, json, time, argparse

SETS = ["qwen_independent", "qwen_combined", "qwen_chain",
        "deepseek", "gemini_flash", "gemini_pro"]

RUBRIC = """You are evaluating whether an AUTOMATICALLY EXTRACTED Test Intent captures a \
specific human-annotated test case.

The GROUND TRUTH below is the intent of ONE test case (its Test Object, Test Goal,
Test Activities). The EXTRACTION was produced for the whole test file, which may
contain SEVERAL test cases, so it can include content for other test cases too.

Decide whether THIS test case's intent is captured in the extraction, judging the
overlap holistically and LENIENTLY:
- Objects align if they refer to the same entity, even with different wording, a
  different angle, or minor extra/missing secondary objects.
- Goal aligns if the verification purpose is similar, even if worded differently.
- Activities align if they describe the same operations, even at a different level
  of detail or with non-critical steps missing.

Assign exactly ONE label:
- "Fully Aligned": this test case's intent is clearly present in the extraction and
  overall means the same thing (allowing the lenient differences above).
- "Partially Aligned": this test case is partly captured but a NON-TRIVIAL part is
  missing or wrong (e.g. its goal only partly covered, or its main activities absent).
- "Misaligned": this test case's intent is not represented in the extraction, or the
  extraction is empty.

Respond with ONLY a JSON object: {"label": "...", "rationale": "<=1 sentence"}"""


def build_prompt(gt_block, ext):
    return f"""{RUBRIC}

=== GROUND TRUTH (one test case) ===
{gt_block.strip()}

=== EXTRACTED TEST INTENT (file-level; may cover multiple test cases) ===
Test Objects: {ext.get('objects')}
Test Goal: {ext.get('goals')}
Test Activities: {ext.get('activities')}

JSON:"""


def parse_label(text):
    import re
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            lab = d.get("label", "")
            for canon in ("Fully Aligned", "Partially Aligned", "Misaligned"):
                if canon.lower() in lab.lower():
                    return canon, d.get("rationale", "")
        except Exception:
            pass
    low = t.lower()
    if "fully" in low: return "Fully Aligned", t[:120]
    if "partial" in low: return "Partially Aligned", t[:120]
    if "misalign" in low: return "Misaligned", t[:120]
    return "PARSE_ERROR", t[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", default=None)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--out", default="eval/scores/scores_blocks.jsonl")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    rows = json.load(open("eval/mapping_blocks.json"))
    if args.framework:
        rows = [r for r in rows if r["framework"] == args.framework]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                d = json.loads(line); done.add((d["block_id"], d["set"]))
            except Exception:
                pass

    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, api_key=os.environ["VERTEX_API_KEY"])
    MODEL = "gemini-3.1-pro-preview"

    out = open(args.out, "a")
    n_api = n_empty = n_skip = 0
    for r in rows:
        for s in args.sets:
            if (r["block_id"], s) in done:
                n_skip += 1; continue
            ext = json.load(open(r[s]))
            empty = not (ext.get("objects") or ext.get("goals") or ext.get("activities"))
            if empty:
                rec = {"block_id": r["block_id"], "framework": r["framework"],
                       "category": r["category"], "set": s,
                       "label": "Misaligned", "rationale": "empty extraction", "auto": True}
                n_empty += 1
            else:
                prompt = build_prompt(r["gt_block_text"], ext)
                label = rationale = None
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
                        label, rationale = "API_ERROR", str(e)[:150]; break
                rec = {"block_id": r["block_id"], "framework": r["framework"],
                       "category": r["category"], "set": s,
                       "label": label, "rationale": rationale, "auto": False}
                n_api += 1
                time.sleep(3)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            print(f"[{r['block_id']:26s} {s:16s}] {rec['label']}")
    out.close()
    print(f"\nDone. api={n_api} empty_auto={n_empty} skipped={n_skip}")


if __name__ == "__main__":
    main()
