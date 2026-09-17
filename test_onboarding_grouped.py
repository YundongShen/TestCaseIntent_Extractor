#!/usr/bin/env python3
"""
Grouped-suite onboarding template, tried on the 4 cases in dog.test.js.
Cases are taken straight from the reviewed Groundtruth, no re-extraction —
just checks the grouping + document generation step.
"""
import os

os.environ["INFERENCE_BACKEND"] = "local"
os.environ["MODEL_TYPE"] = "qwen"

from model.model_config_qwen import MODEL_CONFIG
from model.inference_service import set_model_config
set_model_config(MODEL_CONFIG)

from layers.business.onboarding_generator import OnboardingGenerator
from layers.business.onboarding_template import PROMPT


SUITE_INFO = {
    "suite_id": "LOGIN-SUITE-01",
    "description": (
        "This suite verifies how the POST /users/connection login endpoint "
        "handles a successful sign-in and the different ways a sign-in "
        "attempt can be rejected."
    ),
    "dependencies": ["supertest", "mongoose", "mongodb-memory-server", "bcrypt"],
}

CASES = [
    {
        "id": "TC-LOGIN-01",
        "object": "`POST /users/connection` Login Interface",
        "goal": "Verify that login with correct credentials returns a successful result with a valid token.",
        "activities": [
            "Create a user before each test.",
            "Send a POST request to `/users/connection` with the correct email address and password.",
            "Check that the status code is 200, `result` in the body is true, and `token` and `username` are consistent with those created.",
        ],
        "dependencies": ["supertest", "bcrypt"],  # beforeEach (bcrypt) + request(app) (supertest) run for every case
    },
    {
        "id": "TC-LOGIN-02",
        "object": "`POST /users/connection` Login Interface",
        "goal": "Verify that login without required fields returns a failure result with an appropriate error message.",
        "activities": [
            "Send a POST request to `/users/connection`, sending only the password and not the email address.",
            "Check that the status code is 200, `result` is false, and `error` is \"Missing or empty fields\".",
        ],
        "dependencies": ["supertest", "bcrypt"],  # beforeEach (bcrypt) + request(app) (supertest) run for every case
    },
    {
        "id": "TC-LOGIN-03",
        "object": "`POST /users/connection` Login Interface",
        "goal": "Verify that login with an incorrect password fails and returns the appropriate error message.",
        "activities": [
            "Send a POST request with a correct email address but an incorrect password \"wrongpassword\".",
            "Check for a 200 status code, `result` of false, and `error` of \"User not found or wrong password\".",
        ],
        "dependencies": ["supertest", "bcrypt"],  # beforeEach (bcrypt) + request(app) (supertest) run for every case
    },
    {
        "id": "TC-LOGIN-04",
        "object": "`POST /users/connection` Login Interface",
        "goal": "Verify that login with a non-existent email address fails and returns the appropriate error message.",
        "activities": [
            "Send a POST request with a non-existent email address \"nonexistent@example.com\" and a correct password.",
            "Check for a 200 status code, `result` of false, and `error` of \"User not found or wrong password\".",
        ],
        "dependencies": ["supertest", "bcrypt"],  # beforeEach (bcrypt) + request(app) (supertest) run for every case
    },
]


def main():
    gen = OnboardingGenerator()

    print("=" * 60)
    print("STEP 1: goal grouping")
    print("=" * 60)
    group_input = [{"id": c["id"], "object": c["object"], "goal": c["goal"]} for c in CASES]
    group_result = gen.group_cases(SUITE_INFO["suite_id"], group_input)
    if group_result["status"] != "success":
        print(f"[FAILED] grouping: {group_result['error']}")
        return
    for g in group_result["groups"]:
        print(f"  Group {g['test_case_ids']} -> {g['shared_question']}")

    print()
    print("=" * 60)
    print("STEP 2: document generation")
    print("=" * 60)
    doc_result = gen.generate_suite_document(PROMPT, SUITE_INFO, CASES, group_result["groups"])
    if doc_result["status"] != "success":
        print(f"[FAILED] document generation: {doc_result['error']}")
        return

    out_path = "Result/onboarding_result/onboarding_grouped_qwen.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc_result["document"])

    print(f"\n[SUCCESS] Document saved: {out_path}\n")
    print("=" * 60)
    print(doc_result["document"])


if __name__ == "__main__":
    main()
