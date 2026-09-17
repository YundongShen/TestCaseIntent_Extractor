#!/usr/bin/env python3
"""
Parse each ground-truth file into test-case-level intent blocks and build the
block-level evaluation mapping (~211 test cases).

Each block = one (Test Object, Test Goal, Test Activities) triple inside a GT
file. The extraction under comparison is the file-level Intent.json of the same
raw file (it may cover several test cases from that file).

Output: eval/mapping_blocks.json
"""
import glob, re, os, json, csv

FWS = ["cucumber","cypress","jest_spec","jest_test","junit_spring","junit_unit",
       "k6","locust","playwright","pytest_api","pytest_unit","rtl","supertest"]
CAT = {"cucumber":"BDD","cypress":"Browser Automation","playwright":"Browser Automation",
       "k6":"Performance","locust":"Performance","pytest_api":"API","supertest":"API",
       "junit_spring":"Integration","jest_spec":"Unit","jest_test":"Unit",
       "junit_unit":"Unit","pytest_unit":"Unit","rtl":"Unit"}
SETS = {"qwen_independent":"intent","qwen_combined":"intent_combined","qwen_chain":"intent_chain",
        "deepseek":"intent_deepseek","gemini_flash":"intent_gemini_flash","gemini_pro":"intent_gemini_pro"}


def split_blocks(text):
    idxs = [m.start() for m in re.finditer(r'Test\s*Object', text, re.I)]
    if not idxs:
        return [text.strip()] if text.strip() else []
    out = []
    for i, st in enumerate(idxs):
        en = idxs[i+1] if i+1 < len(idxs) else len(text)
        blk = text[st:en].strip()
        if blk:
            out.append(blk)
    return out


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    raw_index = {fw: {stem(p): p for p in glob.glob(f"dataset/raw/{fw}/*")} for fw in FWS}

    rows = []
    for fw in FWS:
        for gt in sorted(glob.glob(f"TI_GT_DS/Groundtruth/{fw}/*")):
            s = stem(gt)
            blocks = split_blocks(open(gt).read())
            n = len(blocks)
            for bi, btext in enumerate(blocks, 1):
                block_id = f"{fw}/{s}#{bi}" if n > 1 else f"{fw}/{s}"
                row = {"block_id": block_id, "framework": fw, "category": CAT[fw],
                       "gt_file": gt, "block_index": bi, "n_blocks": n,
                       "gt_block_text": btext,
                       "raw": raw_index[fw].get(s, "")}
                for name, d in SETS.items():
                    p = f"dataset/{d}/{fw}/{s}Intent.json"
                    row[name] = p if os.path.exists(p) else ""
                rows.append(row)

    os.makedirs("eval", exist_ok=True)
    json.dump(rows, open("eval/mapping_blocks.json", "w"), indent=2, ensure_ascii=False)
    print(f"blocks: {len(rows)} -> eval/mapping_blocks.json")
    from collections import Counter
    print("per framework:", dict(Counter(r["framework"] for r in rows)))
    print("per category :", dict(Counter(r["category"] for r in rows)))
    miss = [r["block_id"] for r in rows if not r["raw"]]
    print("blocks with no matching raw file:", len(miss), miss[:10])


if __name__ == "__main__":
    main()
