from typing import Callable, Dict, Optional

from layers.intent.granularity_adjuster import GranularityAdjuster
from layers.intent.semantic_validator import SemanticValidator

Extractor = Callable[[str, str], Dict]


class SemanticLayer:
    def __init__(self, extractor: Optional[Extractor] = None, validator=None, adjuster=None,
                 max_reextract=2, adjust=True):
        self.extractor = extractor
        self.validator = validator or SemanticValidator()
        self.adjuster = adjuster or GranularityAdjuster()
        self.max_reextract = max_reextract
        self.do_adjust = adjust

    def process(self, intent: Dict, source: str, requirements="", levels=None):
        attempts = [{"intent": intent}]
        report = self.validator.validate(intent, source)
        attempts[-1]["report"] = report
        while not report.passed and self.extractor is not None and len(attempts) <= self.max_reextract:
            intent = self.extractor(source, report.feedback())
            report = self.validator.validate(intent, source)
            attempts.append({"intent": intent, "report": report})
        final = self.adjuster.adjust(intent, levels, requirements) if self.do_adjust else dict(intent)
        return {
            **final,
            "validation": {"passed": report.passed, "complete": report.complete, "supported": report.supported,
                           "consistent": report.consistent, "issues": report.issues},
            "reextract_count": len(attempts) - 1,
        }


def extractor_from_layer(extract_layer) -> Extractor:
    def run(source, feedback=""):
        result = extract_layer.process({"content": source})
        return {k: result.get(k, []) for k in ("objects", "goals", "activities")}
    return run
