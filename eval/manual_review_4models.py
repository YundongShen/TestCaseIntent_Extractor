#!/usr/bin/env python3
"""
Manual stricter re-review of the 211 test-case x 4-model alignment.
Order per cell: [DeepSeek, Qwen, Flash, Pro]. F=Fully, P=Partially, M=Misaligned.

Stricter principle vs the Gemini pre-labels: a file-level extraction that only
covers a test case via a merged/generic umbrella goal (without distinctly
verifying THAT case) is Partial; a case absent / wrong / from an empty or
unusable extraction is Misaligned.
"""
import json, os

L = {
 # ===== cucumber =====
 "cucumber/Login#1": "FFFF", "cucumber/Login#2": "FFFF", "cucumber/Login#3": "FFFF",
 "cucumber/US06#1": "PFPF", "cucumber/US06#2": "MFPF",
 "cucumber/US10#1": "PFFF", "cucumber/US10#2": "PFFF",
 "cucumber/US13#1": "MPPP", "cucumber/US13#2": "PFFF",
 "cucumber/boot": "PFFF",
 "cucumber/curl#1": "MFPF", "cucumber/curl#2": "MFPF", "cucumber/curl#3": "MFPF",
 "cucumber/demo#1": "FFFF", "cucumber/demo#2": "FFFF",
 "cucumber/main#1": "PFPP", "cucumber/main#2": "PFPP", "cucumber/main#3": "PFFP", "cucumber/main#4": "PPFF",
 # ===== cypress =====
 "cypress/Amazon.cy#1": "MFFF", "cypress/Amazon.cy#2": "FFFF",
 "cypress/Frames.cy#1": "FFFF", "cypress/Frames.cy#2": "PPPF", "cypress/Frames.cy#3": "MPPF",
 "cypress/Game.cy": "PFFF",
 "cypress/kalkulator.cy#1": "FFMF", "cypress/kalkulator.cy#2": "FFMF", "cypress/kalkulator.cy#3": "FFFF",
 "cypress/nav.cy": "MFFF",
 "cypress/spec.cy#1": "FFFF", "cypress/spec.cy#2": "FFFF", "cypress/spec.cy#3": "MFFF",
 "cypress/teste2e.cy#1": "PFFF", "cypress/teste2e.cy#2": "MMMM", "cypress/teste2e.cy#3": "MFFF", "cypress/teste2e.cy#4": "PFFF",
 "cypress/xhr.cy#1": "MFFF", "cypress/xhr.cy#2": "MFFF", "cypress/xhr.cy#3": "MFFF",
 # ===== jest_spec =====
 "jest_spec/App.spec#1": "FFFF", "jest_spec/App.spec#2": "FFFF",
 "jest_spec/cli.spec#1": "PFFF", "jest_spec/cli.spec#2": "PFFF", "jest_spec/cli.spec#3": "MFFF", "jest_spec/cli.spec#4": "MFFF",
 "jest_spec/cssCode.directive.spec": "PFFF",
 "jest_spec/grid.spec#1": "FFFF", "jest_spec/grid.spec#2": "FFFF",
 "jest_spec/opts.spec#1": "PPPF", "jest_spec/opts.spec#2": "MPPF", "jest_spec/opts.spec#3": "MPPF",
 "jest_spec/opts.spec#4": "MPPF", "jest_spec/opts.spec#5": "MPPF", "jest_spec/opts.spec#6": "MPPF", "jest_spec/opts.spec#7": "MPPF",
 "jest_spec/player.spec": "PFFF",
 "jest_spec/server.spec_1#1": "FFFF", "jest_spec/server.spec_1#2": "PFFF", "jest_spec/server.spec_1#3": "FFFF",
 "jest_spec/utils.spec#1": "FFFF", "jest_spec/utils.spec#2": "FFFF", "jest_spec/utils.spec#3": "FFFF",
 "jest_spec/utils.spec#4": "FFFF", "jest_spec/utils.spec#5": "FFFF",
 # ===== jest_test =====
 "jest_test/SilderMenu.test#1": "PFFF", "jest_test/SilderMenu.test#2": "MFMF", "jest_test/SilderMenu.test#3": "PFFF",
 "jest_test/SilderMenu.test#4": "PFFF", "jest_test/SilderMenu.test#5": "PFFF",
 "jest_test/configurationeditor.test#1": "PFFF", "jest_test/configurationeditor.test#2": "MFFF",
 "jest_test/get_output.test#1": "MFFF", "jest_test/get_output.test#2": "MFFF",
 "jest_test/jsx.test#1": "MFFF", "jest_test/jsx.test#2": "MFFF",
 "jest_test/lyraexport.test": "PFFF",
 "jest_test/paths.test#1": "PPFF", "jest_test/paths.test#2": "PPFF", "jest_test/paths.test#3": "PPFF", "jest_test/paths.test#4": "PPFF",
 "jest_test/sample.test_3#1": "MMMM", "jest_test/sample.test_3#2": "PFFF", "jest_test/sample.test_3#3": "PFFF", "jest_test/sample.test_3#4": "PFFF",
 "jest_test/vgSpecEditor.test": "PFFF",
 "jest_test/vlSpecEditor.test": "PFFP",
 # ===== junit_spring =====
 "junit_spring/ContinentRepositoryTest": "PFFF",
 "junit_spring/CountryResourceTest": "FFFF",
 "junit_spring/DemoApplicationIntegrationTest": "PFFF",
 "junit_spring/ESClientTest#1": "MFFF", "junit_spring/ESClientTest#2": "MFFF", "junit_spring/ESClientTest#3": "MFPF", "junit_spring/ESClientTest#4": "MFPP",
 "junit_spring/MapperTest#1": "PFFF", "junit_spring/MapperTest#2": "MFFF",
 "junit_spring/SampleSmokeTest": "MFFF",
 "junit_spring/TestMapperTest": "FFFF",
 # ===== junit_unit =====
 "junit_unit/AssertTest": "FFFF",
 "junit_unit/ConcurrencyTest#1": "FFFF", "junit_unit/ConcurrencyTest#2": "MFFF",
 "junit_unit/EstadistiquesTest_1": "PFFF",
 "junit_unit/ExampleJMockTest": "PFFF",
 "junit_unit/FirstTest#1": "FFFF", "junit_unit/FirstTest#2": "FFFF",
 "junit_unit/GraphTest": "FFFF",
 "junit_unit/StubHandlerTest": "MFFF",
 "junit_unit/UserServiceTest#1": "MFFF", "junit_unit/UserServiceTest#2": "MFFF",
 # ===== k6 =====
 "k6/Auth": "MFFF", "k6/delete": "PFPF", "k6/nextload": "PFFF",
 "k6/script": "PFFF", "k6/simian": "MFFF", "k6/soak": "MFPF",
 # ===== locust =====
 "locust/locustfile": "PFFF", "locust/locustfile_1": "PFFF", "locust/locustfile_2": "PFFF",
 "locust/locustfile_3": "PFFF", "locust/locustfile_6": "PFFF", "locust/locustfile_8": "PFFF",
 # ===== playwright =====
 "playwright/api.spec#1": "MFFF", "playwright/api.spec#2": "MFFF",
 "playwright/csp.spec#1": "FFFF", "playwright/csp.spec#2": "FFFF", "playwright/csp.spec#3": "FFFF", "playwright/csp.spec#4": "FFFF",
 "playwright/e2e.spec#1": "PFFF", "playwright/e2e.spec#2": "MFFF",
 "playwright/module.spec#1": "MPPP", "playwright/module.spec#2": "MPPP", "playwright/module.spec#3": "PPPF",
 "playwright/module.spec#4": "MPPP", "playwright/module.spec#5": "MPPP", "playwright/module.spec#6": "MPPF",
 "playwright/qa.spec#1": "PFFF", "playwright/qa.spec#2": "FFFF", "playwright/qa.spec#3": "FFFF", "playwright/qa.spec#4": "FFFF",
 "playwright/rss.spec#1": "MFFF", "playwright/rss.spec#2": "MFFF", "playwright/rss.spec#3": "MFFF",
 "playwright/top.spec#1": "MFFF", "playwright/top.spec#2": "MFFF", "playwright/top.spec#3": "MFFF",
 "playwright/umd.spec#1": "MFFF", "playwright/umd.spec#2": "PFFF",
 # ===== pytest_api =====
 "pytest_api/REMOTE_FILE_test_#1": "PFFF", "pytest_api/REMOTE_FILE_test_#2": "MFFF", "pytest_api/REMOTE_FILE_test_#3": "MFFF",
 "pytest_api/test__11": "PFFF", "pytest_api/test__2": "PFFF", "pytest_api/test__3": "FFFF",
 "pytest_api/test__6": "PPFF", "pytest_api/test__9": "PFFF", "pytest_api/unit_test_": "MFFF",
 # ===== pytest_unit =====
 "pytest_unit/ABS_test_#1": "FFFF", "pytest_unit/ABS_test_#2": "FFFF", "pytest_unit/ABS_test_#3": "FFFF",
 "pytest_unit/ADD_test_#1": "PFFF", "pytest_unit/ADD_test_#2": "PFFF", "pytest_unit/ADD_test_#3": "MFFF",
 "pytest_unit/BUTTER_test_": "PFFF", "pytest_unit/DET_test_": "PFFF",
 "pytest_unit/LOG_test_#1": "FFFF", "pytest_unit/LOG_test_#2": "FFFF", "pytest_unit/LOG_test_#3": "FFFF",
 "pytest_unit/POPULATE_test_#1": "PFFF", "pytest_unit/POPULATE_test_#2": "MFFF", "pytest_unit/POPULATE_test_#3": "MFFF",
 "pytest_unit/test_#1": "MFFF", "pytest_unit/test_#2": "PFFF", "pytest_unit/test_#3": "PFFF",
 "pytest_unit/test__1#1": "FFFF", "pytest_unit/test__1#2": "FFFF", "pytest_unit/test__1#3": "FFFF",
 "pytest_unit/test__3#1": "FFFF", "pytest_unit/test__3#2": "FFFF",
 "pytest_unit/test__5": "PFFF",
 # ===== rtl =====
 "rtl/A.test#1": "FFFF", "rtl/A.test#2": "MFFF", "rtl/A.test#3": "FFFF",
 "rtl/Drawer.test": "FFFF",
 "rtl/Row.test#1": "PFFF", "rtl/Row.test#2": "PFFF", "rtl/Row.test#3": "MFFF", "rtl/Row.test#4": "MFFF",
 "rtl/basic.test": "PFFF", "rtl/index.test": "FFFF",
 # ===== supertest =====
 "supertest/App.test#1": "PFFF", "supertest/App.test#2": "MFFF",
 "supertest/api.test_1#1": "PPPP", "supertest/api.test_1#2": "PPPP", "supertest/api.test_1#3": "MPPP", "supertest/api.test_1#4": "MPPP",
 "supertest/api.test_1#5": "MPPP", "supertest/api.test_1#6": "MPPP", "supertest/api.test_1#7": "MPPP", "supertest/api.test_1#8": "MPPP",
 "supertest/app.test#1": "MFFF", "supertest/app.test#2": "MFFF", "supertest/app.test#3": "MPPP",
 "supertest/auth.test": "PFFF",
 "supertest/dog.test#1": "FFFF", "supertest/dog.test#2": "MFFF", "supertest/dog.test#3": "MFFF", "supertest/dog.test#4": "MFFF",
 "supertest/index.test#1": "MFFF", "supertest/index.test#2": "MFFF",
 "supertest/posts.test": "MFFF",
 "supertest/put.test#1": "PFFF", "supertest/put.test#2": "MFFF",
}

FULL = {"F": "Fully Aligned", "P": "Partially Aligned", "M": "Misaligned"}
MODELS = ["deepseek", "qwen_independent", "gemini_flash", "gemini_pro"]

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    mapping = json.load(open("eval/mapping_blocks.json"))
    ids = [m["block_id"] for m in mapping]
    missing = [i for i in ids if i not in L]
    extra = [i for i in L if i not in set(ids)]
    print("mapping blocks:", len(ids), "| labeled:", len(L))
    print("MISSING (in mapping, not labeled):", len(missing))
    for i in missing: print("   ", i)
    print("EXTRA (labeled, not in mapping):", len(extra))
    for i in extra: print("   ", i)
    bad = {k: v for k, v in L.items() if len(v) != 4 or any(c not in "FPM" for c in v)}
    print("malformed:", bad)
    if missing or extra or bad:
        print("\n*** FIX BEFORE WRITING ***"); return

    catmap = {m["block_id"]: (m["framework"], m["category"]) for m in mapping}
    out = open("eval/scores/manual_review_4models.jsonl", "w")
    for m in mapping:
        codes = L[m["block_id"]]
        for mi, s in enumerate(MODELS):
            out.write(json.dumps({"block_id": m["block_id"], "framework": m["framework"],
                                  "category": m["category"], "set": s,
                                  "label": FULL[codes[mi]]}, ensure_ascii=False) + "\n")
    out.close()
    print("\nwrote eval/scores/manual_review_4models.jsonl")

if __name__ == "__main__":
    main()
