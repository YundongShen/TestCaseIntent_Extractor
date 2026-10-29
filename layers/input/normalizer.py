import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestCase:
    case_id: str
    suite: str
    title: str
    source: str
    start_line: int
    end_line: int
    format: str
    context: Dict = field(default_factory=dict)


def _empty_body(lines, a, b):
    text = "\n".join(lines[a:b + 1])
    if "{" not in text:
        return False
    return not text[text.index("{") + 1:text.rindex("}")].strip()


def _within(inner, outer):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


class Normalizer:
    def __init__(self, clean=True, link_context=True, keep_empty=False, cleaner=None):
        self.do_clean = clean
        self.do_link = link_context
        self.keep_empty = keep_empty
        self.cleaner = cleaner

    def run(self, doc):
        cases = []
        for case in self.split(doc):
            if self.do_clean:
                case.source = self.clean(case.source)
            if self.do_link:
                case.context = self.link_context(case, doc)
            cases.append(case)
        return cases

    def _kept(self, doc):
        return [n for n, (a, b) in enumerate(doc.cases)
                if self.keep_empty or not _empty_body(doc.lines, a, b)]

    def _case_lines(self, doc, keep):
        drop = set()
        for n, (a, b) in enumerate(doc.cases):
            if n != keep:
                drop.update(range(a, b + 1))
        kept = doc.cases[keep]
        for group in doc.groups:
            others = any(_within(c, group) for n, c in enumerate(doc.cases) if n != keep)
            if others and not _within(kept, group):
                drop.update(range(group[0], group[1] + 1))
        out, last = [], None
        for i, line in enumerate(doc.lines):
            if i in drop:
                continue
            seam = last is not None and i != last + 1
            if seam and not line.strip() and out and not out[-1].strip():
                last = i
                continue
            out.append(line)
            last = i
        return out, drop

    def split(self, doc):
        stem = os.path.basename(doc.path).rsplit(".", 1)[0]
        kept = self._kept(doc)
        if not kept or (len(kept) == 1 and len(doc.cases) == 1):
            with open(doc.path, encoding="utf-8") as f:
                whole = f.read()
        if not kept:
            return [TestCase(case_id=stem, suite=doc.path, title=stem, source=whole,
                             start_line=1, end_line=len(doc.lines), format=doc.format)]
        if len(kept) == 1 and len(doc.cases) == 1:
            a, b = doc.cases[0]
            return [TestCase(case_id=stem, suite=doc.path, title=doc.titles[0], source=whole,
                             start_line=a + 1, end_line=b + 1, format=doc.format)]
        out = []
        for k, n in enumerate(kept, 1):
            lines, _ = self._case_lines(doc, n)
            a, b = doc.cases[n]
            out.append(TestCase(case_id=f"{stem}_{k}", suite=doc.path, title=doc.titles[n],
                                source="\n".join(lines).rstrip("\n") + "\n",
                                start_line=a + 1, end_line=b + 1, format=doc.format))
        return out

    def clean(self, source):
        if self.cleaner is not None:
            return self.cleaner(source)
        lines = [l.rstrip() for l in source.splitlines()]
        merged = []
        for line in lines:
            if not line and merged and not merged[-1]:
                continue
            merged.append(line)
        return "\n".join(merged).strip("\n") + "\n"

    def link_context(self, case, doc):
        n = next((i for i, c in enumerate(doc.cases) if c[0] + 1 == case.start_line and c[1] + 1 == case.end_line), None)
        span = doc.cases[n] if n is not None else (case.start_line - 1, case.end_line - 1)
        enclosing = [g for g in doc.groups if _within(span, g)]
        setup = [s for s in doc.setup
                 if not any(_within(s, c) for c in doc.cases)
                 and (not any(_within(s, g) for g in doc.groups) or any(_within(s, g) for g in enclosing))]
        imports = [l.strip() for l in doc.lines
                   if re.match(r"\s*(import\s|from\s+\S+\s+import\s|const\s+.*=\s*require\(|package\s)", l)]
        return {
            "suite_file": os.path.basename(doc.path),
            "format": doc.format,
            "line_range": [span[0] + 1, span[1] + 1],
            "imports": imports,
            "enclosing_blocks": [doc.lines[g[0]].strip() for g in enclosing],
            "shared_setup": ["\n".join(doc.lines[s[0]:s[1] + 1]) for s in setup],
            "sibling_cases": [t for i, t in enumerate(doc.titles) if i != n],
        }


def normalize(doc, **options) -> List[TestCase]:
    return Normalizer(**options).run(doc)


def find_case(cases: List[TestCase], title: str) -> Optional[TestCase]:
    return next((c for c in cases if c.title == title), None)
