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

Training accepts `--max-train-entities N` for a bounded diagnostic run. It samples Source 1 by stable seeded hash while scanning chunks, filters ground truth to sampled Source 1 IDs, then scans each target TSV while retaining sampled gold targets and up to `2*N` seeded distractors. Each run writes a `.metrics.json` file beside the model. This evaluates a sampled target corpus and is not a full-data score.

To override the candidate cap, pass `--max-candidates-per-entity N`. A 5,000-anchor comparison measured 0.9980 candidate recall and 0.9847 macro F₀.₅ with cap 40, compared with 0.9288 and 0.9590 at cap 20. The configured default is 40 based on that sample; verify on a separate holdout before final model selection.

## Outputs

- `matching_results.tsv`: one row per test Source 1 entity; empty IDs mean no predicted match.
- `candidate_pairs.tsv`: one row per test Source 1 entity; contains the final candidates that were scored.

Both are tab-separated with the challenge's exact headers. Matched IDs must be included in the corresponding candidate list. Candidate lists are capped at 40 by the current project configuration; the challenge asks for small candidate sets but does not specify this numeric cap.

## Validation and packaging

The package utility runs the supplied validator before archiving. Add `--check-ids` to the packaging command when sufficient memory is available; without it, ID-existence checks are skipped. The validator checks formatting and consistency but does not compute F₀.₅.
## Scale and compliance

The inputs contain millions of rows per source. Full training and inference have not been completed. A 10,000-anchor confirmation run was stopped at about 96% system memory use. Benchmark runtime and peak memory on a larger-memory machine before full runs. Sampled blocking recall, candidate count, threshold, and F₀.₅ are recorded in `output/TRAINING_RUN_ANALYSIS.md` and the model-side metrics files; do not present them as full-data scores. Do not use external business identity lookup, registry, geocoding, or augmentation services.

