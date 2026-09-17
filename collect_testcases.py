"""
Collect test case files from GitHub for ground truth dataset construction.

Usage:
    python collect_testcases.py
    GITHUB_TOKEN="ghp_xxx" python collect_testcases.py

Output:
    dataset/raw/<framework>/<filename>   - downloaded test files
    dataset/index.json                   - metadata for each file

Framework quotas (target 210, trim to 200):
    jest_test      30   (.test.js  - unit/integration)
    jest_spec      20   (.spec.js  - unit/integration)
    cypress        15   (E2E)
    playwright     15   (E2E)
    rtl            10   (React Testing Library - component)
    supertest      10   (API integration)
    pytest_unit    25   (Python unit)
    pytest_api     15   (Python API integration)
    locust         10   (Python performance)
    junit_unit     20   (Java unit)
    junit_spring   15   (Java Spring integration)
    cucumber       15   (BDD/acceptance)
    k6             10   (JS performance)
"""

import os
import json
import time
import re
import requests
from pathlib import Path
from collections import Counter

# ─── Configuration ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
TARGET_DIR   = Path("dataset/raw")
INDEX_FILE   = Path("dataset/index.json")
MIN_LINES    = 20
MAX_LINES    = 600
SLEEP_SEARCH = 5    # seconds between search requests
SLEEP_FILE   = 1    # seconds between file downloads

# Each entry: (framework_label, quota, search_query, quality_patterns, min_pattern_matches, test_count_pattern, max_tests)
# test_count_pattern: regex to count individual test methods; max_tests enforces test-case (not suite) granularity
FRAMEWORK_SPECS = [
    (
        "jest_test", 30,
        'filename:.test.js "describe" "it(" "expect("',
        [r"describe\s*\(", r"\bit\s*\(", r"expect\s*\(", r"beforeEach\s*\(|afterEach\s*\(|beforeAll\s*\("],
        2, r"\bit\s*\(", 8,
    ),
    (
        "jest_spec", 20,
        'filename:.spec.js "describe" "it(" "expect("',
        [r"describe\s*\(", r"\bit\s*\(", r"expect\s*\("],
        2, r"\bit\s*\(", 8,
    ),
    (
        "cypress", 15,
        'filename:.cy.js "describe" "cy." "it("',
        [r"\bcy\.", r"describe\s*\(", r"\bit\s*\("],
        2, r"\bit\s*\(", 8,
    ),
    (
        "playwright", 15,
        'filename:.spec.ts "test(" "expect(" "page."',
        [r"\bpage\.", r"\btest\s*\(", r"expect\s*\("],
        2, r"\btest\s*\(", 8,
    ),
    (
        "rtl", 10,
        'filename:.test.tsx "render" "screen" "userEvent"',
        [r"\brender\s*\(", r"\bscreen\.", r"expect\s*\("],
        2, r"\bit\s*\(", 8,
    ),
    (
        "supertest", 10,
        'filename:.test.js "supertest" "request(" "expect("',
        [r"\brequest\s*\(", r"\.get\s*\(|\.post\s*\(|\.put\s*\(|\.delete\s*\(", r"expect\s*\("],
        2, r"\bit\s*\(", 8,
    ),
    (
        "pytest_unit", 25,
        'filename:test_.py "def test_" "assert" "mock"',
        [r"def test_\w+", r"\bassert\b", r"mock|patch|MagicMock"],
        2, r"def test_\w+", 5,
    ),
    (
        "pytest_api", 15,
        'filename:test_.py "def test_" "requests" "assert"',
        [r"def test_\w+", r"\bassert\b", r"requests\.|httpx\.|client\."],
        2, r"def test_\w+", 5,
    ),
    (
        "locust", 10,
        'filename:locustfile.py "HttpUser" "task" "self.client"',
        [r"HttpUser|FastHttpUser", r"@task", r"self\.client\."],
        2, r"@task", 6,
    ),
    (
        "junit_unit", 20,
        'filename:Test.java "@Test" "@Mock" "assert"',
        [r"@Test", r"@Mock|@InjectMocks|Mockito", r"assert\w*\(|Assert\."],
        2, r"@Test", 5,
    ),
    (
        "junit_spring", 15,
        'filename:Test.java "@SpringBootTest" "@Test" "assert"',
        [r"@SpringBootTest|@WebMvcTest|@DataJpaTest", r"@Test", r"assert\w*\(|Assert\."],
        2, r"@Test", 5,
    ),
    (
        "cucumber", 15,
        'filename:.feature "Scenario" "Given" "When" "Then"',
        [r"Scenario:|Scenario Outline:", r"Given\s+", r"When\s+", r"Then\s+"],
        3, r"Scenario:|Scenario Outline:", 5,
    ),
    (
        "k6", 10,
        'filename:.js "import http from" "check(" "sleep("',
        [r"import http from ['\"]k6/http", r"\bcheck\s*\(", r"\bsleep\s*\("],
        2, r"\bgroup\s*\(", 8,
    ),
]

# ─── GitHub API helpers ─────────────────────────────────────────────────────────

def make_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def search_files(query, per_page=30, page=1):
    url = "https://api.github.com/search/code"
    params = {"q": query, "per_page": per_page, "page": page}
    resp = requests.get(url, headers=make_headers(), params=params, timeout=15)
    if resp.status_code == 403:
        print("  [rate limit] waiting 60s...")
        time.sleep(60)
        resp = requests.get(url, headers=make_headers(), params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  [warn] search failed ({resp.status_code}): {query[:60]}")
        return []
    return resp.json().get("items", [])


def fetch_raw_content(item):
    repo = item["repository"]["full_name"]
    path = item["path"]
    for branch in ["HEAD", "main", "master"]:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text
        time.sleep(0.3)
    return None


# ─── Quality filter ─────────────────────────────────────────────────────────────

def passes_quality(content, patterns, min_matches, test_count_pattern=None, max_tests=None):
    lines = content.splitlines()
    n = len(lines)
    if n < MIN_LINES or n > MAX_LINES:
        return False
    matched = sum(1 for p in patterns if re.search(p, content))
    if matched < min_matches:
        return False
    if test_count_pattern and max_tests:
        count = len(re.findall(test_count_pattern, content))
        if count > max_tests:
            return False
    return True


# ─── Main collection loop ───────────────────────────────────────────────────────

def collect():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    index = []
    seen  = set()   # "repo/path" dedup
    total = 0
    counts = {}

    for fw, quota, query, patterns, min_matches, test_count_pat, max_tests in FRAMEWORK_SPECS:
        counts[fw] = 0
        needed = quota
        page   = 1

        print(f"\n[{fw}] quota={quota}  query: {query[:65]}")

        while counts[fw] < needed and page <= 4:
            items = search_files(query, per_page=30, page=page)
            print(f"  page {page}: {len(items)} candidates")
            time.sleep(SLEEP_SEARCH)

            if not items:
                break

            for item in items:
                if counts[fw] >= needed:
                    break

                repo     = item["repository"]["full_name"]
                path     = item["path"]
                filename = Path(path).name
                uid      = f"{repo}/{path}"

                if uid in seen:
                    continue
                seen.add(uid)

                content = fetch_raw_content(item)
                time.sleep(SLEEP_FILE)

                if content is None:
                    print(f"  [skip] fetch failed: {uid[:60]}")
                    continue

                if not passes_quality(content, patterns, min_matches, test_count_pat, max_tests):
                    print(f"  [skip] quality ({len(content.splitlines())} lines): {filename}")
                    continue

                fw_dir = TARGET_DIR / fw
                fw_dir.mkdir(parents=True, exist_ok=True)

                safe_name = re.sub(r"[^\w.\-]", "_", filename)
                dest = fw_dir / safe_name
                counter = 1
                while dest.exists():
                    stem = Path(safe_name).stem
                    suf  = Path(safe_name).suffix
                    dest = fw_dir / f"{stem}_{counter}{suf}"
                    counter += 1

                dest.write_text(content, encoding="utf-8")

                total   += 1
                counts[fw] += 1
                entry = {
                    "id":          total,
                    "framework":   fw,
                    "filename":    dest.name,
                    "path":        str(dest),
                    "source_repo": repo,
                    "source_path": path,
                    "source_url":  f"https://github.com/{repo}/blob/HEAD/{path}",
                    "lines":       len(content.splitlines()),
                }
                index.append(entry)
                print(f"  [{total:03d}] {fw} ({counts[fw]}/{needed})  {dest.name}  ({entry['lines']} lines)")

            page += 1

        if counts[fw] < needed:
            print(f"  [warn] only collected {counts[fw]}/{needed} for {fw}")

    INDEX_FILE.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Done. {total} files total.")
    print(f"Index: {INDEX_FILE}")
    print(f"\nBreakdown:")
    for fw, cnt in counts.items():
        quota = next(q for f, q, *_ in FRAMEWORK_SPECS if f == fw)
        status = "OK" if cnt >= quota else f"SHORT ({cnt}/{quota})"
        print(f"  {fw:<20} {cnt:>3}  {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("[warn] No GITHUB_TOKEN — rate limited to 10 req/min (unauthenticated)")
        print("       Set: export GITHUB_TOKEN='ghp_...'")
        print()
    collect()
