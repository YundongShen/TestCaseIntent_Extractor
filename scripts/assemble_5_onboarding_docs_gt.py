#!/usr/bin/env python3
"""
Same 5 documents as assemble_5_onboarding_docs.py, but built from the
reviewed Groundtruth instead of fresh extraction, for comparison.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["INFERENCE_BACKEND"] = "local"
os.environ["MODEL_TYPE"] = "qwen"

from model.model_config_qwen import MODEL_CONFIG
from model.inference_service import set_model_config
set_model_config(MODEL_CONFIG)

from suites_data import SUITES
from layers.business.onboarding_generator import OnboardingGenerator
from layers.business.onboarding_template import PROMPT

# suite_id -> list of (object, goal, [activities...]), taken verbatim from
# TI_GT_DS/Groundtruth/, in the same order as suites_data.SUITES[*]["cases"].
GT_CASES = {
    "MENU-MATCH-SUITE-01": [
        ("The `getMeunMatcheys` function", "Verify that an exact simple path is correctly matched.",
         ["Call `getMeunMatcheys(meun, '/dashboard')`.", "Assert the return value as `['/dashboard']`."]),
        ("The `getMeunMatcheys` function", "Verify that a non-matching path returns an empty result.",
         ["Call `getMeunMatcheys(meun, '/dashboardname')`.", "Assert the return value as `[]`."]),
        ("The `getMeunMatcheys` function", "Verify that a second-level path is correctly matched.",
         ["Call `getMeunMatcheys(meun, '/dashboard/name')`.", "Assert the return value as `['/dashboard/name']`."]),
        ("The `getMeunMatcheys` function", "Verify that a parameterized path is correctly matched against the route pattern.",
         ["Call `getMeunMatcheys(meun, '/userinfo/2144')`.", "Assert the return value as `['/userinfo/:id']`."]),
        ("The `getMeunMatcheys` function", "Verify that a nested parameterized path is correctly matched against the route pattern.",
         ["Call `getMeunMatcheys(meun, '/userinfo/2144/info')`.", "Assert the return value as `['/userinfo/:id/info']`."]),
    ],
    "LEVENSTEIN-SUITE-01": [
        ("The `levenstein` function", "Verify that the edit distance between two same strings is calculated correctly.",
         ["Call `levenstein('aaa', 'aaa')`.", "Assert the return value to 0."]),
        ("The `levenstein` function", "Verify that the edit distance between two different strings is calculated correctly.",
         ["Call `levenstein('abc', 'def')`.", "Assert the return value to 3."]),
        ("The `levenstein` function", "Verify that the edit distance between two same email strings is calculated correctly.",
         ["Call `levenstein('gmail.com', 'gnail.com')`.", "Assert the return value to be 1."]),
        ("`levenstein` function", "Verify that the edit distance between two different email strings is calculated correctly.",
         ["Call `levenstein('gmail.com', 'gnaul.com')`.", "Assert the return value to be 2."]),
        ("`levenstein` function", "Verify that the edit distance between two strings is calculated correctly.",
         ["Call `levenstein('mail.com', 'gnaul.com')`.", "Assert the return value to be 3."]),
    ],
    "BOOTSTRAP4-SUITE-01": [
        ("The page's `body` element", "Confirm that the `body` element is hidden during initial page load.",
         ["Visit the homepage `/`.", "Check that the `body` element is hidden."]),
        ("The page's `body` element", "Confirm that the `body` element becomes visible after the page loads.",
         ["Visit the homepage `/`.", "Check that the `body` element eventually becomes visible."]),
        ("The style of the `.alert-primary` element", "Verify that the alert style is applied after Bootstrap CSS loads.",
         ["Visit the homepage `/`.", "Get the initial background color of the first `.alert-primary` element.",
          "Wait for the Bootstrap CSS request to complete.", "Get the background color of the same element again.",
          "Assert that the two colors are different."]),
        ('Modal element `[role="dialog"]`', "Confirm that there is only one modal on the page, and its initial state is hidden.",
         ["Visit the homepage `/`.", 'Locate the `[role="dialog"]` element.', "Assert that its quantity is 1.",
          "Get the `aria-hidden` attribute of this element.", 'Assert that the attribute value is `"true"`.']),
        ('"Launch demo modal" button', "Confirm that a button for opening a modal exists on the page.",
         ["Visit the homepage `/`.", 'Find the button element with the text `"Launch demo modal"`.',
          "Assert that the element is defined (exists)."]),
        ('Modal `[role="dialog"]` and its launch button', "Verify that the modal opens after clicking the button.",
         ["Access the homepage `/`.", "Locate the modal and obtain its `aria-hidden` attribute.",
          'Find the `"Launch demo modal"` button and click it.',
          'Check if the modal\'s `aria-hidden` attribute value is `"false"`.']),
    ],
    "TRIANGLIFY-OPTS-SUITE-01": [
        ("The `getSizes` and `getVariances` functions", "Confirm that both functions return reasonable default values when no arguments are passed.",
         ["Call `getSizes()` without arguments and use snapshot assertion to check the return value.",
          "Call `getVariances()` without arguments and use snapshot assertion to check the return value."]),
        ("The `getSizes` function", "Verify that a single resolution string is correctly parsed into an array of sizes.",
         ["Call `getSizes('200x200')`.", "Assert the result as `[{ h: 200, w: 200 }]`."]),
        ("The `getSizes` function", "Verify that an array of multiple resolution strings is correctly parsed into an array of objects of corresponding sizes.",
         ["Call `getSizes(['100x200', '300x400'])`.", "Assert the result as `[{ w: 100, h: 200 }, { w: 300, h: 400 }]`."]),
        ("The `getSizes` function", "An exception is thrown if an incorrectly formatted resolution string is passed in.",
         ["Call `getSizes('100xFoo')`.", "The assertion function throws an exception."]),
        ("The `getVariances` function", "Verify that a single variance string can be correctly parsed as an array.",
         ["Call `getVariances('0.61')`.", "Assert the result as `[0.61]`."]),
        ("`getVariances` function", "Verifies that multiple variance string arrays can be correctly parsed into corresponding numeric arrays.",
         ["Call `getVariances(['0.333', '0.1', '1'])`.", "Assert the result as `[0.333, 0.1, 1]`."]),
        ("`getVariances` function", "Throws an exception if an out-of-range or invalid variance value is passed in.",
         ["Call `getVariances('1.1')`, `getVariances('-1')`, and `getVariances('foo')` respectively.",
          "Assert that an exception is thrown on each call."]),
    ],
    "BOOKS-API-SUITE-01": [
        ("`GET /api/books` Interface", "Verify that calling this interface returns the expected status code and the response body is an array.",
         ["Send a GET request to `/api/books`.", "Check that the status code is 200.", "Check that the response body is an array."]),
        ("`GET /api/books/:id` Interface", "Verify that requesting an existing resource returns a success status code with the correct data.",
         ["Send a GET request to `/api/books/1`.", "Check that the status code is 200.", "Check that the `id` field in the returned body is 1."]),
        ("`GET /api/books/:id` API", "Verify that requesting a non-existent resource returns an error status code.",
         ["Send a GET request to `/api/books/999`.", "Check that the status code is 404."]),
        ("`POST /api/books` API", "Verify that creating a new book with valid data returns a success response with the correct book information.",
         ['Construct new book data (title: "Dune", author: "Frank Herbert", genre: "Sci-Fi", copiesAvailable: 4).',
          "Send a POST request to `/api/books`, carrying the above JSON data.", "Check that the status code is 201.",
          'Check that the `title` in the returned body is "Dune".']),
        ("`PUT /api/books/:id` API", "Verify that updating an existing book returns a success status code and the book data is updated.",
         ['Send a PUT request to `/api/books/1`, with the body containing `{ title: "Updated Title" }`.',
          "Check that the status code is 200.", 'Check that the `title` in the returned body is "Updated Title".']),
        ("`PUT /api/books/:id` API", "Verify that updating a non-existent book returns an error.",
         ['Send a PUT request to `/api/books/999`, with the body containing `{ title: "No Book" }`.',
          "Check that the status code is 404."]),
        ("`DELETE /api/books/:id` Interface", "Verify that deleting an existing book returns a success status code.",
         ["Send a DELETE request to `/api/books/1`.", "Check that the status code is 200."]),
        ("`DELETE /api/books/:id` Interface", "Verify that deleting a non-existent book returns an error.",
         ["Send a DELETE request to `/api/books/999`.", "The status code is 404."]),
    ],
}


def main():
    gen = OnboardingGenerator()
    all_results = []

    for suite in SUITES:
        sid = suite["suite_id"]
        print("\n" + "#" * 70)
        print(f"# SUITE (Groundtruth): {sid}  ({len(suite['cases'])} test cases)")
        print("#" * 70)

        gt = GT_CASES[sid]
        assert len(gt) == len(suite["cases"]), f"{sid}: GT has {len(gt)} cases, suite defines {len(suite['cases'])}"

        cases = []
        for i, (obj, goal, acts) in enumerate(gt, 1):
            case_id = f"TC-{sid.split('-')[0]}-{i:02d}"
            cases.append({"id": case_id, "object": obj, "goal": goal,
                          "activities": acts, "dependencies": suite["dependencies"]})
            print(f"  {case_id}: obj={obj!r} goal={goal!r}")

        suite_info = {"suite_id": sid, "description": suite["description"], "dependencies": suite["dependencies"]}

        group_input = [{"id": c["id"], "object": c["object"], "goal": c["goal"]} for c in cases]
        group_result = gen.group_cases(sid, group_input)
        if group_result["status"] != "success":
            print(f"[FAILED] grouping: {group_result['error']}")
            continue
        for g in group_result["groups"]:
            print(f"  Group {g['test_case_ids']} -> {g['shared_question']}")

        doc_result = gen.generate_suite_document(PROMPT, suite_info, cases, group_result["groups"])
        if doc_result["status"] != "success":
            print(f"[FAILED] document generation: {doc_result['error']}")
            continue

        safe_name = sid.lower().replace("-", "_")
        out_path = f"Result/onboarding_result/onboarding_{safe_name}_groundtruth.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc_result["document"])
        print(f"[SUCCESS] Document saved: {out_path}")
        all_results.append(out_path)

    print("\n" + "=" * 70)
    print(f"Done. Generated {len(all_results)}/{len(SUITES)} documents:")
    for p in all_results:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
