"""Shared suite/test-case definitions for the 5-suite independent-extraction onboarding run."""

SUITES = [
    {
        "suite_id": "MENU-MATCH-SUITE-01",
        "framework": "jest_test",
        "source_file": "dataset/raw/jest_test/SilderMenu.test.js",
        "description": (
            "This suite verifies how the getMeunMatcheys function matches a "
            "requested URL path against a fixed list of menu routes."
        ),
        "dependencies": ["./SiderMenu"],
        "wrapper": (
            "import {{ getMeunMatcheys }} from './SiderMenu';\n\n"
            "const meun = ['/dashboard', '/userinfo', '/dashboard/name', '/userinfo/:id', '/userinfo/:id/info'];\n\n"
            "describe('test meun match', () => {{\n"
            "  it('{title}', () => {{\n"
            "    {body}\n"
            "  }});\n"
            "}});\n"
        ),
        "cases": [
            ("simple path", "expect(getMeunMatcheys(meun, '/dashboard')).toEqual(['/dashboard']);"),
            ("error path", "expect(getMeunMatcheys(meun, '/dashboardname')).toEqual([]);"),
            ("Secondary path", "expect(getMeunMatcheys(meun, '/dashboard/name')).toEqual(['/dashboard/name']);"),
            ("Parameter path", "expect(getMeunMatcheys(meun, '/userinfo/2144')).toEqual(['/userinfo/:id']);"),
            ("three parameter path", "expect(getMeunMatcheys(meun, '/userinfo/2144/info')).toEqual(['/userinfo/:id/info']);"),
        ],
    },
    {
        "suite_id": "LEVENSTEIN-SUITE-01",
        "framework": "jest_spec",
        "source_file": "dataset/raw/jest_spec/utils.spec.js",
        "description": (
            "This suite verifies that the levenstein function correctly "
            "computes the edit distance between pairs of strings."
        ),
        "dependencies": ["chai", "./utils"],
        "wrapper": (
            "'use strict'\n\n"
            "const {{ expect }} = require('chai')\n"
            "const levenstein = require('./utils')\n\n"
            "describe('Utils', () => {{\n"
            "  describe('Levenstein', () => {{\n"
            "    it('{title}', () => {{\n"
            "      {body}\n"
            "    }})\n"
            "  }})\n"
            "}})\n"
        ),
        "cases": [
            ("should calculate the distance between two identical strings", "expect(levenstein('aaa', 'aaa')).eq(0)"),
            ("should calculate the distance between two different strings", "expect(levenstein('abc', 'def')).eq(3)"),
            ("should calculate the distance between two similar strings", "expect(levenstein('gmail.com', 'gnail.com')).eq(1)"),
            ("should calculate the distance between gmail.com & gnaul.com", "expect(levenstein('gmail.com', 'gnaul.com')).eq(2)"),
            ("should calculate the distance between mail.com & gnaul.com", "expect(levenstein('mail.com', 'gnaul.com')).eq(3)"),
        ],
    },
    {
        "suite_id": "BOOTSTRAP4-SUITE-01",
        "framework": "playwright",
        "source_file": "dataset/raw/playwright/module.spec.ts",
        "description": (
            "This suite verifies Bootstrap 4 page content visibility, alert "
            "styling, and modal dialog behavior."
        ),
        "dependencies": ["@playwright/test"],
        # Cases 1-3 sit directly under 'Bootstrap 4'; cases 4-6 sit under the
        # nested 'Modal' describe, which also inherits the outer beforeEach.
        "wrapper": (
            "import {{ test, expect }} from '@playwright/test';\n\n"
            "test.beforeEach(async ({{ page }}) => {{\n"
            "\tawait page.goto('/');\n"
            "}});\n\n"
            "test.describe('Bootstrap 4', () => {{\n"
            "{body}"
            "}});\n"
        ),
        "cases": [
            ("should initially hide page content",
             "\ttest('should initially hide page content', async ({ page }) => {\n"
             "\t\tawait expect(page.locator('body')).toBeHidden();\n\t});\n"),
            ("should eventually display page content",
             "\ttest('should eventually display page content', async ({ page }) => {\n"
             "\t\tawait expect(page.locator('body')).toBeVisible();\n\t});\n"),
            ("should style alert messages",
             "\ttest('should style alert messages', async ({ page }) => {\n"
             "\t\tconst alert = page.locator('.alert-primary').first();\n"
             "\t\tconst initialColor = await alert.evaluate((el) => {\n"
             "\t\t\treturn getComputedStyle(el).backgroundColor;\n\t\t});\n"
             "\t\tawait page.waitForResponse(\n"
             "\t\t\t'https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css'\n\t\t);\n"
             "\t\tconst styledColor = await alert.evaluate((el) => getComputedStyle(el).backgroundColor);\n"
             "\t\texpect(styledColor !== initialColor);\n\t});\n"),
            ("should have one hidden modal",
             "\ttest.describe('Modal', () => {\n"
             "\t\ttest('should have one hidden modal', async ({ page }) => {\n"
             "\t\t\tconst modal = page.locator('[role=\"dialog\"]');\n"
             "\t\t\tconst isHidden = await modal.getAttribute('aria-hidden');\n"
             "\t\t\texpect((await modal.count()).valueOf() === 1);\n"
             "\t\t\texpect(isHidden.valueOf() === 'true');\n\t\t});\n\t});\n"),
            ("should have button to launch modal",
             "\ttest.describe('Modal', () => {\n"
             "\t\ttest('should have button to launch modal', async ({ page }) => {\n"
             "\t\t\tconst button = page.locator('text=Launch demo modal');\n"
             "\t\t\texpect(button).toBeDefined();\n\t\t});\n\t});\n"),
            ("should launch modal on button click",
             "\ttest.describe('Modal', () => {\n"
             "\t\ttest('should launch modal on button click', async ({ page }) => {\n"
             "\t\t\tconst modal = page.locator('[role=\"dialog\"]');\n"
             "\t\t\tconst isHidden = await modal.getAttribute('aria-hidden');\n"
             "\t\t\tconst button = page.locator('text=Launch demo modal');\n"
             "\t\t\tawait button.click();\n"
             "\t\t\texpect(isHidden.valueOf() === 'false');\n\t\t});\n\t});\n"),
        ],
        "no_format": True,  # wrapper already fully formatted per-case; skip .format()
    },
    {
        "suite_id": "TRIANGLIFY-OPTS-SUITE-01",
        "framework": "jest_spec",
        "source_file": "dataset/raw/jest_spec/opts.spec.js",
        "description": (
            "This suite verifies that getSizes and getVariances correctly "
            "parse, default, and validate resolution and variance options."
        ),
        "dependencies": ["./opts"],
        "wrapper": (
            "const {{ getSizes, getVariances, getSeed }} = require('./opts');\n\n"
            "describe('themer-wallpaper-trianglify options', () => {{\n"
            "  it('{title}', () => {{\n"
            "    {body}\n"
            "  }});\n"
            "}});\n"
        ),
        "cases": [
            ("should return proper defaults if none provided",
             "expect(getSizes()).toMatchSnapshot();\n    expect(getVariances()).toMatchSnapshot();"),
            ("should parse a single resolution option",
             "expect(getSizes('200x200')).toEqual([{ h: 200, w: 200 }]);"),
            ("should parse multiple resolution options",
             "expect(getSizes(['100x200', '300x400'])).toEqual([{ w: 100, h: 200}, { w: 300, h: 400 }]);"),
            ("should throw when a malformed resolution option is given",
             "expect(() => getSizes('100xFoo')).toThrow();"),
            ("should parse a single variance option",
             "expect(getVariances('0.61')).toEqual([0.61]);"),
            ("should parse multiple variance options",
             "expect(getVariances(['0.333', '0.1', '1'])).toEqual([0.333, 0.1, 1]);"),
            ("should throw when an invalid variance option is given",
             "expect(() => getVariances('1.1')).toThrow();\n    expect(() => getVariances('-1')).toThrow();\n    expect(() => getVariances('foo')).toThrow();"),
        ],
    },
    {
        "suite_id": "BOOKS-API-SUITE-01",
        "framework": "supertest",
        "source_file": "dataset/raw/supertest/api.test_1.js",
        "description": (
            "This suite verifies the CRUD behavior of the Books API, "
            "including handling of valid and invalid book IDs."
        ),
        "dependencies": ["supertest"],
        "wrapper": (
            "const request = require('supertest');\n"
            "const app = require('../server');\n\n"
            "describe('Books API', () => {{\n\n"
            "    test('{title}', async () => {{\n"
            "        {body}\n"
            "    }});\n\n"
            "}});\n"
        ),
        "cases": [
            ("GET /api/books should return all books",
             "const res = await request(app).get('/api/books');\n"
             "        expect(res.statusCode).toBe(200);\n"
             "        expect(Array.isArray(res.body)).toBe(true);"),
            ("GET /api/books/:id should return a book",
             "const res = await request(app).get('/api/books/1');\n"
             "        expect(res.statusCode).toBe(200);\n"
             "        expect(res.body.id).toBe(1);"),
            ("GET /api/books/:id should return 404 for invalid ID",
             "const res = await request(app).get('/api/books/999');\n"
             "        expect(res.statusCode).toBe(404);"),
            ("POST /api/books should create a new book",
             "const newBook = {\n"
             "            title: \"Dune\",\n"
             "            author: \"Frank Herbert\",\n"
             "            genre: \"Sci-Fi\",\n"
             "            copiesAvailable: 4\n"
             "        };\n\n"
             "        const res = await request(app)\n"
             "            .post('/api/books')\n"
             "            .send(newBook);\n\n"
             "        expect(res.statusCode).toBe(201);\n"
             "        expect(res.body.title).toBe(\"Dune\");"),
            ("PUT /api/books/:id should update a book",
             "const res = await request(app)\n"
             "            .put('/api/books/1')\n"
             "            .send({ title: \"Updated Title\" });\n\n"
             "        expect(res.statusCode).toBe(200);\n"
             "        expect(res.body.title).toBe(\"Updated Title\");"),
            ("PUT /api/books/:id should return 404 if not found",
             "const res = await request(app)\n"
             "            .put('/api/books/999')\n"
             "            .send({ title: \"No Book\" });\n\n"
             "        expect(res.statusCode).toBe(404);"),
            ("DELETE /api/books/:id should delete a book",
             "const res = await request(app).delete('/api/books/1');\n"
             "        expect(res.statusCode).toBe(200);"),
            ("DELETE /api/books/:id should return 404 if not found",
             "const res = await request(app).delete('/api/books/999');\n"
             "        expect(res.statusCode).toBe(404);"),
        ],
    },
]



def isolate_case_source(suite, title, body):
    if suite.get("no_format"):
        return suite["wrapper"].format(body=body)
    return suite["wrapper"].format(title=title, body=body)

