from typing import Iterable, List

from layers.input.adapters import parse_suite
from layers.input.normalizer import Normalizer, TestCase


class InputPipeline:
    def __init__(self, normalizer=None):
        self.normalizer = normalizer or Normalizer()

    def run(self, path) -> List[TestCase]:
        return self.normalizer.run(parse_suite(path))

    def run_many(self, paths: Iterable[str]) -> List[TestCase]:
        cases = []
        for path in paths:
            cases.extend(self.run(path))
        return cases

    @staticmethod
    def as_records(cases: List[TestCase]):
        return [{"content": c.source, "test_case_id": c.case_id, "title": c.title,
                 "suite": c.suite, "context": c.context} for c in cases]
