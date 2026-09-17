# Section 6.2 Quantitative Results — Claude stricter review, test-case level (n=211)

Judge: Claude manual holistic review. Empty/placeholder extractions -> Misaligned.

## Table 1 — Prompting Techniques (model fixed = Qwen 3.5-27B)
| Technique | Fully | Partially | Misaligned |
|---|---|---|---|
| Independent Extraction | 178 (84.4%) | 31 (14.7%) | 2 (0.9%) |
| Combined Extraction | 111 (52.6%) | 72 (34.1%) | 28 (13.3%) |
| Chained Extraction | 131 (62.1%) | 30 (14.2%) | 50 (23.7%) |

## Table 2 — Language Models (prompting fixed = Independent)
| Model | Fully | Partially | Misaligned |
|---|---|---|---|
| DeepSeek 7B | 54 (25.6%) | 73 (34.6%) | 84 (39.8%) |
| Qwen 3.5-27B | 178 (84.4%) | 31 (14.7%) | 2 (0.9%) |
| Gemini 2.5 Flash | 170 (80.6%) | 36 (17.1%) | 5 (2.4%) |
| Gemini 3.1 Pro | 190 (90.0%) | 19 (9.0%) | 2 (0.9%) |

## Table 3 — Testing Categories (fixed = Qwen 3.5-27B + Independent)
| Category | Fully | Partially | Misaligned | n |
|---|---|---|---|---|
| Unit | 79 (86.8%) | 11 (12.1%) | 1 (1.1%) | 91 |
| Integration | 11 (100.0%) | 0 (0.0%) | 0 (0.0%) | 11 |
| API | 22 (68.8%) | 10 (31.2%) | 0 (0.0%) | 32 |
| Browser Automation | 37 (80.4%) | 8 (17.4%) | 1 (2.2%) | 46 |
| Performance | 12 (100.0%) | 0 (0.0%) | 0 (0.0%) | 12 |
| BDD | 17 (89.5%) | 2 (10.5%) | 0 (0.0%) | 19 |
