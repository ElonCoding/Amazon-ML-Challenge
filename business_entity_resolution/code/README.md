# Business Entity Resolution — Reproduction Guide

This source package contains the preprocessing, candidate generation, pairwise features in bounded batches, LightGBM training, evaluation, and inference code for the Amazon ML Challenge 2026 project. The archive does not include the multi-gigabyte challenge data; provide the challenge train/test directories when running it.

## Run from the repository checkout

From `D:\ML challenge` in PowerShell:

```powershell
python -m pip install -r business_entity_resolution\code\requirements.txt
python business_entity_resolution\code\src\train.py `
  --train-dir dataset\student_resource\dataset\train `
  --model-out models\lgb_model.joblib
python business_entity_resolution\code\src\inference.py `
  --test-dir dataset\student_resource\dataset\test `
  --model-path models\lgb_model.joblib `
  --output-dir output
```

After extracting the submitted archive elsewhere, pass the locations of the supplied `train/` and `test/` directories with `--train-dir` and `--test-dir`; pass writable paths with `--model-out`, `--model-path`, and `--output-dir` as needed. The default paths are convenient for the repository checkout and may not match the evaluator's data placement.

Training also accepts `--max-train-entities N`. This makes a deterministic sample of Source 1 entities but still loads all Source 2 and Source 3 records, so it is not a substitute for measuring target-index memory use.

## Outputs

- `matching_results.tsv`: one row per test Source 1 entity; empty IDs mean no predicted match.
- `candidate_pairs.tsv`: one row per test Source 1 entity; contains the final candidates that were scored.

Both are tab-separated with the challenge's exact headers. Matched IDs must be included in the corresponding candidate list. Candidate lists are capped at 20 by the current project configuration; the challenge asks for small candidate sets but does not specify this numeric cap.

## Validation and packaging

The package utility runs the supplied validator before archiving. Add `--check-ids` to the packaging command when sufficient memory is available; without it, ID-existence checks are skipped. The validator checks formatting and consistency but does not compute F₀.₅.
## Scale and compliance

The inputs contain millions of rows per source. The current implementation keeps tables, candidate maps, and portions of its inverted indexes in memory; benchmark runtime and peak memory before a full run. Blocking recall, candidate count, threshold, and F₀.₅ are experimental results and must be measured before they are reported. Do not use external business identity lookup, registry, geocoding, or augmentation services.

