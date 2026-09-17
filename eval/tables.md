# Section 6.2 Quantitative Results (LLM-judge pre-label + human review)

- Evaluable cases: **98** (raw files with ground-truth annotations; 1:1 basename match).
- Judge: `gemini-3.1-pro-preview` (Vertex, global), holistic lenient rubric.
- Empty extractions auto-labelled Misaligned.
- Source: `eval/scores/scores.jsonl` · Mapping: `eval/mapping_98.json`

## Table 1 — Prompting Techniques (model fixed = Qwen 3.5-27B), n=98
| Technique | Fully | Partially | Misaligned |
|---|---|---|---|
| Independent Extraction | 94 (95.9%) | 4 (4.1%) | 0 (0.0%) |
| Combined Extraction | 83 (84.7%) | 5 (5.1%) | 10 (10.2%) |
| Chained Extraction | 48 (49.0%) | 41 (41.8%) | 9 (9.2%) |

## Table 2 — Language Models (prompting fixed = Independent), n=98
| Model | Fully | Partially | Misaligned |
|---|---|---|---|
| DeepSeek 7B | 27 (27.6%) | 63 (64.3%) | 8 (8.2%) |
| Qwen 3.5-27B | 94 (95.9%) | 4 (4.1%) | 0 (0.0%) |
| Gemini 2.5 Flash | 92 (93.9%) | 6 (6.1%) | 0 (0.0%) |
| Gemini 3.1 Pro | 98 (100.0%) | 0 (0.0%) | 0 (0.0%) |

## Table 3 — Testing Categories (fixed = Qwen 3.5-27B + Independent)
| Category | Fully | Partially | Misaligned | n |
|---|---|---|---|---|
| Unit | 40 (100.0%) | 0 | 0 | 40 |
| Integration | 7 (100.0%) | 0 | 0 | 7 |
| API | 15 (100.0%) | 0 | 0 | 15 |
| Browser Automation | 15 (93.8%) | 1 (6.2%) | 0 | 16 |
| Performance | 9 (75.0%) | 3 (25.0%) | 0 | 12 |
| BDD | 8 (100.0%) | 0 | 0 | 8 |

## Notes / caveats
- Misaligned composition: Combined 10 = 6 empty-auto + 4 judged; Chained 9 = 3 empty-auto + 6 judged; DeepSeek 8 = all judged.
- **Table 1 conflicts with the current paper prose**, which claims Independent and Chained are "broadly comparable" and both better than Combined. The data shows a clear ordering: **Independent > Combined > Chained** on Fully-Aligned. Chained produces far more Partially-Aligned (41/98), consistent with its activity-filtering/compression behaviour, but this makes it the weakest, not comparable to Independent.
- Self-evaluation caveat: the judge (Gemini Pro) also scores its own extraction set (perfect 100%). Recommend a Gemini-Flash cross-judge on the gemini_pro column.
- Small n for Integration (7), BDD (8) — category percentages are coarse.
