import json
from typing import Dict, Optional

from layers.intent.semantic_validator import FIELDS, ask, parse_json, _as_list

LEVEL_WORDS = [(0.25, "very specific: keep exact identifiers, values, paths and calls"),
               (0.5, "specific: keep key identifiers and values, drop incidental detail"),
               (0.75, "general: describe behavior in plain terms, few identifiers"),
               (1.01, "very general: one short plain-language statement, no identifiers")]

ADJUST_PROMPT = """Rewrite the test intent below so each part matches the requested level of detail.
Do not add anything that is not already in the intent. Keep the same number of activities unless the level asks to merge them.

[Task requirements]
{requirements}

[Requested level per part]
Objects: {objects_level}
Goals: {goals_level}
Activities: {activities_level}

[Test intent]
Objects: {objects}
Goals: {goals}
Activities: {activities}

Return ONLY JSON: {{"objects": [...], "goals": [...], "activities": [...]}}"""


def describe_level(value):
    value = min(max(float(value), 0.0), 1.0)
    return next(text for bound, text in LEVEL_WORDS if value < bound)


class GranularityAdjuster:
    def __init__(self, service=None, default_levels: Optional[Dict[str, float]] = None, max_tokens=1500):
        self.service = service
        self.default_levels = default_levels or {"objects": 0.2, "goals": 0.6, "activities": 0.4}
        self.max_tokens = max_tokens

    def _service(self):
        if self.service is None:
            from model.service_factory import get_inference_backend
            self.service = get_inference_backend()
        return self.service

    def adjust(self, intent, levels: Optional[Dict[str, float]] = None, requirements=""):
        levels = {**self.default_levels, **(levels or {})}
        prompt = ADJUST_PROMPT.format(
            requirements=requirements or "general documentation",
            **{f"{f}_level": describe_level(levels[f]) for f in FIELDS},
            **{f: json.dumps(_as_list(intent.get(f)), ensure_ascii=False) for f in FIELDS})
        result = parse_json(ask(self._service(), prompt, self.max_tokens)) or {}
        adjusted = dict(intent)
        for f in FIELDS:
            value = _as_list(result.get(f))
            if value:
                adjusted[f] = value
        adjusted["granularity"] = {f: float(levels[f]) for f in FIELDS}
        return adjusted
