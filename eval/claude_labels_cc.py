#!/usr/bin/env python3
"""
Claude's stricter review for Qwen combined & chain (for Table 1).
Each value = 2 chars: [combined, chain]. F/P/M.
Then merge with the 4-model review (claude_labels.L) and aggregate Tables 1/2/3.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claude_labels import L as L4, FULL  # 4-model codes: [DS,Qw,Fl,Pr]

CC = {
 # cucumber
 "cucumber/Login#1":"MF","cucumber/Login#2":"MF","cucumber/Login#3":"MF",
 "cucumber/US06#1":"PF","cucumber/US06#2":"PF","cucumber/US10#1":"PF","cucumber/US10#2":"PF",
 "cucumber/US13#1":"PM","cucumber/US13#2":"PM","cucumber/boot":"MF",
 "cucumber/curl#1":"PP","cucumber/curl#2":"PP","cucumber/curl#3":"PP",
 "cucumber/demo#1":"FF","cucumber/demo#2":"FF",
 "cucumber/main#1":"PF","cucumber/main#2":"PF","cucumber/main#3":"PF","cucumber/main#4":"PF",
 # cypress
 "cypress/Amazon.cy#1":"FF","cypress/Amazon.cy#2":"FF",
 "cypress/Frames.cy#1":"FF","cypress/Frames.cy#2":"PF","cypress/Frames.cy#3":"PF",
 "cypress/Game.cy":"FF",
 "cypress/kalkulator.cy#1":"PF","cypress/kalkulator.cy#2":"PF","cypress/kalkulator.cy#3":"FF",
 "cypress/nav.cy":"FF",
 "cypress/spec.cy#1":"FF","cypress/spec.cy#2":"FF","cypress/spec.cy#3":"FF",
 "cypress/teste2e.cy#1":"PM","cypress/teste2e.cy#2":"MM","cypress/teste2e.cy#3":"MM","cypress/teste2e.cy#4":"PM",
 "cypress/xhr.cy#1":"MM","cypress/xhr.cy#2":"MM","cypress/xhr.cy#3":"MM",
 # jest_spec
 "jest_spec/App.spec#1":"FF","jest_spec/App.spec#2":"FF",
 "jest_spec/cli.spec#1":"PF","jest_spec/cli.spec#2":"PF","jest_spec/cli.spec#3":"PF","jest_spec/cli.spec#4":"PF",
 "jest_spec/cssCode.directive.spec":"FM",
 "jest_spec/grid.spec#1":"FF","jest_spec/grid.spec#2":"FF",
 "jest_spec/opts.spec#1":"MP","jest_spec/opts.spec#2":"MP","jest_spec/opts.spec#3":"MP","jest_spec/opts.spec#4":"MP",
 "jest_spec/opts.spec#5":"MP","jest_spec/opts.spec#6":"MP","jest_spec/opts.spec#7":"MP",
 "jest_spec/player.spec":"FM",
 "jest_spec/server.spec_1#1":"FM","jest_spec/server.spec_1#2":"FM","jest_spec/server.spec_1#3":"FM",
 "jest_spec/utils.spec#1":"FF","jest_spec/utils.spec#2":"FF","jest_spec/utils.spec#3":"FF",
 "jest_spec/utils.spec#4":"FF","jest_spec/utils.spec#5":"FF",
 # jest_test
 "jest_test/SilderMenu.test#1":"PF","jest_test/SilderMenu.test#2":"MF","jest_test/SilderMenu.test#3":"PF",
 "jest_test/SilderMenu.test#4":"PF","jest_test/SilderMenu.test#5":"PF",
 "jest_test/configurationeditor.test#1":"FF","jest_test/configurationeditor.test#2":"FF",
 "jest_test/get_output.test#1":"FM","jest_test/get_output.test#2":"FM",
 "jest_test/jsx.test#1":"FF","jest_test/jsx.test#2":"FF",
 "jest_test/lyraexport.test":"MF",
 "jest_test/paths.test#1":"PM","jest_test/paths.test#2":"PM","jest_test/paths.test#3":"PM","jest_test/paths.test#4":"PM",
 "jest_test/sample.test_3#1":"MM","jest_test/sample.test_3#2":"MM","jest_test/sample.test_3#3":"MM","jest_test/sample.test_3#4":"MM",
 "jest_test/vgSpecEditor.test":"MM","jest_test/vlSpecEditor.test":"MM",
 # junit_spring
 "junit_spring/ContinentRepositoryTest":"FF","junit_spring/CountryResourceTest":"FM",
 "junit_spring/DemoApplicationIntegrationTest":"FF",
 "junit_spring/ESClientTest#1":"FF","junit_spring/ESClientTest#2":"FF","junit_spring/ESClientTest#3":"FF","junit_spring/ESClientTest#4":"PP",
 "junit_spring/MapperTest#1":"FF","junit_spring/MapperTest#2":"FF",
 "junit_spring/SampleSmokeTest":"FF","junit_spring/TestMapperTest":"FF",
 # junit_unit
 "junit_unit/AssertTest":"FF",
 "junit_unit/ConcurrencyTest#1":"FM","junit_unit/ConcurrencyTest#2":"FM",
 "junit_unit/EstadistiquesTest_1":"FF","junit_unit/ExampleJMockTest":"FF",
 "junit_unit/FirstTest#1":"FF","junit_unit/FirstTest#2":"FF","junit_unit/GraphTest":"FF",
 "junit_unit/StubHandlerTest":"MF","junit_unit/UserServiceTest#1":"FF","junit_unit/UserServiceTest#2":"FF",
 # k6
 "k6/Auth":"FF","k6/delete":"PF","k6/nextload":"FF","k6/script":"FM","k6/simian":"PF","k6/soak":"PP",
 # locust
 "locust/locustfile":"FF","locust/locustfile_1":"FM","locust/locustfile_2":"FF",
 "locust/locustfile_3":"FF","locust/locustfile_6":"FF","locust/locustfile_8":"FF",
 # playwright
 "playwright/api.spec#1":"FF","playwright/api.spec#2":"FF",
 "playwright/csp.spec#1":"PF","playwright/csp.spec#2":"PF","playwright/csp.spec#3":"PF","playwright/csp.spec#4":"PF",
 "playwright/e2e.spec#1":"FF","playwright/e2e.spec#2":"FF",
 "playwright/module.spec#1":"PP","playwright/module.spec#2":"PP","playwright/module.spec#3":"PP",
 "playwright/module.spec#4":"PP","playwright/module.spec#5":"PP","playwright/module.spec#6":"PP",
 "playwright/qa.spec#1":"FM","playwright/qa.spec#2":"PM","playwright/qa.spec#3":"PM","playwright/qa.spec#4":"PM",
 "playwright/rss.spec#1":"FF","playwright/rss.spec#2":"FF","playwright/rss.spec#3":"FF",
 "playwright/top.spec#1":"FF","playwright/top.spec#2":"FF","playwright/top.spec#3":"FF",
 "playwright/umd.spec#1":"FF","playwright/umd.spec#2":"FF",
 # pytest_api
 "pytest_api/REMOTE_FILE_test_#1":"FF","pytest_api/REMOTE_FILE_test_#2":"FF","pytest_api/REMOTE_FILE_test_#3":"FF",
 "pytest_api/test__11":"FF","pytest_api/test__2":"FF","pytest_api/test__3":"FM",
 "pytest_api/test__6":"FF","pytest_api/test__9":"FM","pytest_api/unit_test_":"FF",
 # pytest_unit
 "pytest_unit/ABS_test_#1":"FF","pytest_unit/ABS_test_#2":"FF","pytest_unit/ABS_test_#3":"FF",
 "pytest_unit/ADD_test_#1":"FF","pytest_unit/ADD_test_#2":"FF","pytest_unit/ADD_test_#3":"FF",
 "pytest_unit/BUTTER_test_":"FF","pytest_unit/DET_test_":"FF",
 "pytest_unit/LOG_test_#1":"MM","pytest_unit/LOG_test_#2":"MM","pytest_unit/LOG_test_#3":"MM",
 "pytest_unit/POPULATE_test_#1":"PP","pytest_unit/POPULATE_test_#2":"PP","pytest_unit/POPULATE_test_#3":"PP",
 "pytest_unit/test_#1":"PP","pytest_unit/test_#2":"PF","pytest_unit/test_#3":"PF",
 "pytest_unit/test__1#1":"PF","pytest_unit/test__1#2":"PF","pytest_unit/test__1#3":"PF",
 "pytest_unit/test__3#1":"FF","pytest_unit/test__3#2":"FF","pytest_unit/test__5":"FM",
 # rtl
 "rtl/A.test#1":"FF","rtl/A.test#2":"FF","rtl/A.test#3":"FF","rtl/Drawer.test":"FF",
 "rtl/Row.test#1":"FF","rtl/Row.test#2":"FF","rtl/Row.test#3":"FF","rtl/Row.test#4":"FF",
 "rtl/basic.test":"FF","rtl/index.test":"FF",
 # supertest
 "supertest/App.test#1":"FM","supertest/App.test#2":"FM",
 "supertest/api.test_1#1":"PP","supertest/api.test_1#2":"PP","supertest/api.test_1#3":"PP","supertest/api.test_1#4":"PP",
 "supertest/api.test_1#5":"PP","supertest/api.test_1#6":"PP","supertest/api.test_1#7":"PP","supertest/api.test_1#8":"PP",
 "supertest/app.test#1":"PM","supertest/app.test#2":"PM","supertest/app.test#3":"PM",
 "supertest/auth.test":"FF",
 "supertest/dog.test#1":"FM","supertest/dog.test#2":"PM","supertest/dog.test#3":"PM","supertest/dog.test#4":"PM",
 "supertest/index.test#1":"FF","supertest/index.test#2":"PF",
 "supertest/posts.test":"FF","supertest/put.test#1":"FF","supertest/put.test#2":"FF",
}

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    mapping = json.load(open("eval/mapping_blocks.json"))
    ids = [m["block_id"] for m in mapping]
    miss = [i for i in ids if i not in CC]; extra = [i for i in CC if i not in set(ids)]
    bad = {k:v for k,v in CC.items() if len(v)!=2 or any(c not in "FPM" for c in v)}
    print(f"blocks={len(ids)} CC-labeled={len(CC)} missing={len(miss)} extra={len(extra)} bad={len(bad)}")
    for i in miss[:20]: print("  missing", i)
    for i in extra[:20]: print("  extra", i)
    if miss or extra or bad: print("*** FIX ***"); return

    # write full 6-set review
    out = open("eval/scores/claude_review_all.jsonl","w")
    for m in mapping:
        bid=m["block_id"]; c4=L4[bid]; cc=CC[bid]
        allsets=[("deepseek",c4[0]),("qwen_independent",c4[1]),("gemini_flash",c4[2]),
                 ("gemini_pro",c4[3]),("qwen_combined",cc[0]),("qwen_chain",cc[1])]
        for s,code in allsets:
            out.write(json.dumps({"block_id":bid,"framework":m["framework"],
                "category":m["category"],"set":s,"label":FULL[code]},ensure_ascii=False)+"\n")
    out.close()
    print("wrote eval/scores/claude_review_all.jsonl")

if __name__=="__main__":
    main()
