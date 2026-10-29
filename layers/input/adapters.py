import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

Range = Tuple[int, int]


@dataclass
class SuiteDocument:
    path: str
    format: str
    lines: List[str]
    cases: List[Range] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    groups: List[Range] = field(default_factory=list)
    setup: List[Range] = field(default_factory=list)


REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^")


def _skip_regex(line, j):
    k, in_class = j + 1, False
    while k < len(line):
        c = line[k]
        if c == "\\":
            k += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            return k
        k += 1
    return j


def close_line(lines, start, open_ch="(", close_ch=")"):
    depth, opened, quote = 0, False, None
    for i in range(start, len(lines)):
        line, j = lines[i], 0
        while j < len(line):
            ch = line[j]
            if quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif line.startswith("//", j):
                break
            elif ch == "/" and not line.startswith("/*", j):
                prev = line[:j].rstrip()
                if not prev or prev[-1] in REGEX_BEFORE or re.search(r"\b(return|typeof)$", prev):
                    j = _skip_regex(line, j)
            elif ch == open_ch:
                depth += 1
                opened = True
            elif ch == close_ch:
                depth -= 1
                if opened and depth == 0:
                    return i
            j += 1
        if quote in ("'", '"'):
            quote = None
    return len(lines) - 1


def _attach_above(lines, start, prefixes):
    while start > 0 and lines[start - 1].strip().startswith(prefixes):
        start -= 1
    return start


def _offsets(lines):
    out = [0]
    for line in lines:
        out.append(out[-1] + len(line) + 1)
    return out


class FormatAdapter:
    name = "base"
    extensions: Tuple[str, ...] = ()

    def matches(self, path):
        return path.lower().endswith(self.extensions)

    def parse(self, path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        doc = SuiteDocument(path=path, format=self.name, lines=lines)
        self.locate(doc)
        return doc

    def locate(self, doc):
        raise NotImplementedError


class JavaScriptAdapter(FormatAdapter):
    name = "code/javascript"
    extensions = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
    case_pattern = re.compile(r"\b(?:it|test|electronTest)(?:\.only|\.skip)?\s*\(\s*(['\"`])(.*?)\1")
    group_pattern = re.compile(r"\b(?:test\.)?(?:describe|context)(?:\.only|\.skip)?\s*\(")
    setup_pattern = re.compile(r"\b(?:test\.)?(?:beforeAll|beforeEach|afterAll|afterEach|before|after)\s*\(")

    def locate(self, doc):
        lines, offs = doc.lines, _offsets(doc.lines)
        text = "\n".join(lines)

        def line_of(pos):
            return next(i for i in range(len(lines)) if offs[i + 1] > pos)

        def in_comment(i, pos):
            return "//" in lines[i][:pos - offs[i]]

        for m in self.case_pattern.finditer(text):
            i = line_of(m.start())
            if in_comment(i, m.start()):
                continue
            doc.cases.append((_attach_above(lines, i, ("//",)), close_line(lines, i)))
            doc.titles.append(m.group(2))
        for pattern, target in ((self.group_pattern, doc.groups), (self.setup_pattern, doc.setup)):
            for m in pattern.finditer(text):
                i = line_of(m.start())
                if not in_comment(i, m.start()):
                    target.append((i, close_line(lines, i)))


class JavaAdapter(FormatAdapter):
    name = "code/java"
    extensions = (".java",)
    case_pattern = re.compile(r"@Test\b[\s\S]*?void\s+(\w+)\s*\(")
    setup_pattern = re.compile(r"@(?:Before|BeforeEach|BeforeAll|BeforeClass|After|AfterEach|AfterAll|AfterClass)\b[\s\S]*?void\s+\w+\s*\(")

    def locate(self, doc):
        lines = doc.lines
        text = "\n".join(lines)
        for pattern, cases in ((self.case_pattern, True), (self.setup_pattern, False)):
            for m in pattern.finditer(text):
                first = text.count("\n", 0, m.start())
                method = text.count("\n", 0, m.end())
                span = (_attach_above(lines, first, ("//", "/*", "*")), close_line(lines, method, "{", "}"))
                if cases:
                    doc.cases.append(span)
                    doc.titles.append(m.group(1))
                else:
                    doc.setup.append(span)


class PythonAdapter(FormatAdapter):
    name = "code/python"
    extensions = (".py",)
    case_pattern = re.compile(r"^(\s*)(?:async\s+)?def\s+(test\w*)\s*\(")
    setup_pattern = re.compile(r"^(\s*)(?:async\s+)?def\s+(setUp|tearDown|setUpClass|tearDownClass|setup_method|teardown_method|setup_module|teardown_module)\s*\(")

    def _span(self, lines, i, indent):
        start = i
        j = i - 1
        while j >= 0 and lines[j].strip():
            j -= 1
        if j + 1 < i and lines[j + 1].lstrip().startswith("@"):
            start = j + 1
        sig_end, depth = i, 0
        for k in range(i, len(lines)):
            depth += lines[k].count("(") - lines[k].count(")")
            if depth <= 0 and lines[k].rstrip().endswith(":"):
                sig_end = k
                break
        end = sig_end
        for k in range(sig_end + 1, len(lines)):
            if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= indent:
                break
            if lines[k].strip():
                end = k
        return _attach_above(lines, start, ("#",)), end

    def locate(self, doc):
        lines = doc.lines
        for i, line in enumerate(lines):
            m = self.case_pattern.match(line)
            if m:
                doc.cases.append(self._span(lines, i, len(m.group(1))))
                doc.titles.append(m.group(2))
                continue
            m = self.setup_pattern.match(line)
            if m:
                doc.setup.append(self._span(lines, i, len(m.group(1))))
            if line.lstrip().startswith("@pytest.fixture"):
                k = i + 1
                while k < len(lines) and not lines[k].lstrip().startswith("def "):
                    k += 1
                if k < len(lines):
                    indent = len(lines[k]) - len(lines[k].lstrip())
                    doc.setup.append(self._span(lines, k, indent))


class GherkinAdapter(FormatAdapter):
    name = "gherkin"
    extensions = (".feature",)
    case_pattern = re.compile(r"^\s*(Scenario|Scenario Outline|Scenario Template|Example)\s*:\s*(.*)")
    feature_pattern = re.compile(r"^\s*Feature\s*:")
    rule_pattern = re.compile(r"^\s*Rule\s*:")
    background_pattern = re.compile(r"^\s*Background\s*:")

    def locate(self, doc):
        lines = doc.lines
        starts = [i for i, l in enumerate(lines) if self.case_pattern.match(l)]
        heads = [i for i, l in enumerate(lines) if self.feature_pattern.match(l) or self.rule_pattern.match(l)]
        backgrounds = [i for i, l in enumerate(lines) if self.background_pattern.match(l)]
        bounds = sorted(set(starts + heads + backgrounds))

        def block_end(i):
            nxt = [b for b in bounds if b > i]
            end = (nxt[0] if nxt else len(lines)) - 1
            while end > i and (not lines[end].strip() or lines[end].strip().startswith(("@", "#"))):
                end -= 1
            return end

        for i in starts:
            doc.cases.append((_attach_above(lines, i, ("@", "#")), block_end(i)))
            doc.titles.append(self.case_pattern.match(lines[i]).group(2).strip())
        for pattern, stop in ((self.feature_pattern, (self.feature_pattern,)), (self.rule_pattern, (self.rule_pattern, self.feature_pattern))):
            for i in [k for k, l in enumerate(lines) if pattern.match(l)]:
                nxt = [b for b in heads if b > i and any(p.match(lines[b]) for p in stop)]
                doc.groups.append((_attach_above(lines, i, ("@", "#")), (nxt[0] if nxt else len(lines)) - 1))
        for i in backgrounds:
            doc.setup.append((i, block_end(i)))


class ManualAdapter(FormatAdapter):
    name = "manual"
    extensions = (".md", ".txt", ".rst")
    case_pattern = re.compile(r"^\s*(?:#{1,6}\s*)?(?:(?:Test\s*Case|TC)[\s\-_#:]*[\w.\-]*\s*[:\-]?\s*)(.*)$", re.I)

    def locate(self, doc):
        lines = doc.lines
        starts = [i for i, l in enumerate(lines) if self.case_pattern.match(l) and l.strip()]
        for n, i in enumerate(starts):
            end = (starts[n + 1] if n + 1 < len(starts) else len(lines)) - 1
            while end > i and not lines[end].strip():
                end -= 1
            doc.cases.append((i, end))
            doc.titles.append(self.case_pattern.match(lines[i]).group(1).strip() or lines[i].strip())


ADAPTERS: List[FormatAdapter] = [JavaScriptAdapter(), JavaAdapter(), PythonAdapter(), GherkinAdapter(), ManualAdapter()]


def register_adapter(adapter, first=True):
    if first:
        ADAPTERS.insert(0, adapter)
    else:
        ADAPTERS.append(adapter)


def adapter_for(path):
    for adapter in ADAPTERS:
        if adapter.matches(path):
            return adapter
    raise ValueError(f"no format adapter for {os.path.basename(path)}")


def parse_suite(path):
    return adapter_for(path).parse(path)
