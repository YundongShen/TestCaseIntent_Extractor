import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

FIELDS = ("objects", "goals", "activities")


@dataclass
class ValidationReport:
    complete: bool
    supported: bool
    consistent: bool
    issues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    @property
    def passed(self):
        return self.complete and self.supported and self.consistent

    def feedback(self):
        return "\n".join(f"- {i}" for i in self.issues)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [v for v in value if str(v).strip()]


def parse_json(text):
    if not text:
        return None
    spans, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    for span in reversed(spans):
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue
    return None


def ask(service, prompt, max_tokens):
    call = getattr(service, "infer_no_thinking", None) or service.infer
    return call(prompt, max_tokens=max_tokens)


SUPPORTED_PROMPT = """Check whether every element of the test intent below is supported by the test code.
An element is supported only if the code actually shows it; anything the code does not show is unsupported.

[Test code]
{source}

[Test intent]
Objects: {objects}
Goals: {goals}
Activities: {activities}

Return ONLY JSON: {{"supported": true/false, "unsupported": ["<element> - <reason>", ...]}}"""

CONSISTENT_PROMPT = """Check whether the three parts of the test intent below are consistent with each other:
the activities operate on the objects, the goals verify the objects, and the activities serve the goals.

Objects: {objects}
Goals: {goals}
Activities: {activities}

Return ONLY JSON: {{"consistent": true/false, "issues": ["<issue>", ...]}}"""


class SemanticValidator:
    def __init__(self, service=None, check_supported=True, check_consistent=True, max_tokens=600):
        self.service = service
        self.check_supported = check_supported
        self.check_consistent = check_consistent
        self.max_tokens = max_tokens

    def _service(self):
        if self.service is None:
            from model.service_factory import get_inference_backend
            self.service = get_inference_backend()
        return self.service

    def complete(self, intent):
        missing = [f for f in FIELDS if not _as_list(intent.get(f))]
        return not missing, [f"{f} is missing or empty" for f in missing]

    def supported(self, intent, source):
        prompt = SUPPORTED_PROMPT.format(source=source, **{f: json.dumps(_as_list(intent.get(f)), ensure_ascii=False) for f in FIELDS})
        result = parse_json(ask(self._service(), prompt, self.max_tokens)) or {}
        ok = bool(result.get("supported", False))
        return ok, [f"not supported by the code: {u}" for u in result.get("unsupported", [])] or ([] if ok else ["could not confirm support from the code"])

    def consistent(self, intent):
        prompt = CONSISTENT_PROMPT.format(**{f: json.dumps(_as_list(intent.get(f)), ensure_ascii=False) for f in FIELDS})
        result = parse_json(ask(self._service(), prompt, self.max_tokens)) or {}
        ok = bool(result.get("consistent", False))
        return ok, [f"inconsistent: {i}" for i in result.get("issues", [])] or ([] if ok else ["could not confirm consistency"])

    def validate(self, intent, source=""):
        complete, issues = self.complete(intent)
        details = {"complete": complete}
        supported = consistent = complete
        if complete and self.check_supported:
            supported, more = self.supported(intent, source)
            issues += more
        if complete and self.check_consistent:
            consistent, more = self.consistent(intent)
            issues += more
        details.update(supported=supported, consistent=consistent)
        return ValidationReport(complete=complete, supported=supported, consistent=consistent, issues=issues, details=details)
