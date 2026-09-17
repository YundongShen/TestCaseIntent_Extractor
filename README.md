# Test Intent Extraction

Extracts a structured Test Intent (Object, Goal, Activities) from test code
using an LLM, and can turn a set of extracted intents into an onboarding
document. Built around a five-layer pipeline: Input -> Extract -> Intent ->
Business -> Output.

## Layout

```
layers/
  input/        preprocessing (trim, dedupe blank lines, strip comments)
  extract/      object/goal/activity extraction — independent, combined, chain modes
  extract/chain/  the chain-mode extractors (object -> goal -> activity, passing context along)
  intent/       validates and adjusts the extracted triplet
  business/     onboarding document templates + generator (grouping, formatting)
  output/       writes the final document to disk

model/          model configs (Qwen, DeepSeek, DeepSeek-V3) and the local/API inference services
eval/           scripts and data for scoring extraction against groundtruth
testcases/      a handful of sample test files for quick manual runs
scripts/        supporting one-off scripts (dataset collection, onboarding-doc assembly, md->docx, ...)
slurm/          job scripts for running the pipeline on a GPU cluster

main.py                     runs one file through all five layers
extract_for_dataset.py      runs layers 1-2 only, for building a dataset in bulk
```

Raw test corpus, groundtruth annotations, and per-model extraction results
live in a separate dataset repo, not here.

## Running it

```bash
pip install -r requirements.txt
export MODEL_TYPE=qwen        # or 7b, deepseek, v3
python main.py
```

`MODEL_TYPE` and `EXTRACT_MODE` (independent/combined/chain) are read from
the environment; see `MODEL_SWITCHING.md` for the full list of options.
Local models are downloaded on first use; set `INFERENCE_BACKEND=api` plus
the relevant key in `.env` to use the Gemini API instead.

Generated documents land in `Result/onboarding_result/`.
