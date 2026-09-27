# Amazon ML Challenge 2026: Business Entity Resolution

This repository implements a local pipeline for matching Source 1 business records to records in Source 2 and Source 3. It uses only the challenge data. It does not call external identity, registry, geocoding, or enrichment services.

## Project status

The repository contains the training, blocking, scoring, inference, validator, and packaging components. The challenge TSVs are in `dataset/student_resource/dataset/`. Local fitted models and sampled-validation metrics are in `models/`; no final test TSV outputs have been generated. The sampled results are documented in `output/TRAINING_RUN_ANALYSIS.md` and must not be presented as full-corpus or leaderboard results.

## Data and outputs

Each input record has `entity_id`, `business_name`, `business_address`, and `country`; ground truth has `source1_entity_id` and comma-separated `matched_entity_ids`. All files are TSV and must be read with `sep="\t"`.

The pipeline writes:

- `output/matching_results.tsv`: one row for every test Source 1 entity, with a comma-separated list of Source 2/3 matches or an empty list.
- `output/candidate_pairs.tsv`: one row for every test Source 1 entity, with the final candidate IDs actually scored or an empty list.

Every output match must exist in the corresponding candidate list. The challenge data contains France in the test set; country labels must remain open-set.

## Current architecture

1. `src/preprocess.py` reads Arrow-backed strings, normalizes names and addresses, retains country labels as strings, and drops duplicate raw text columns after creating the features used downstream.
2. `src/blocking.py` uses name character n-grams, address tokens, and address digits to retrieve a capped candidate set.
3. `src/feature_extraction.py` computes 26 lexical, numeric, field-agreement, and source-indicator features.
4. `src/train.py` trains a LightGBM pair classifier and tunes a threshold on a Source 1 holdout for macro F₀.₅.
5. `src/inference.py` generates and scores test candidates, then writes the two TSVs.

The current cap is 40 candidates per Source 1. A 5,000-anchor validation comparison measured candidate recall of 0.9980 at cap 40 versus 0.9288 at cap 20, while average candidates rose from 20.00 to 33.18. Macro F₀.₅ on that sampled validation split was 0.9847 at cap 40 and 0.9590 at cap 20. These are sampled results only; cap 40 is a project setting, not a numeric challenge limit. The blocker partitions by exact country labels; this processes unseen labels such as France but assumes matching records share the same label. Measure or revise that assumption before finalizing results.

## Environment

Use the pinned dependencies in this directory's `requirements.txt`. The project environment inspected for this version used Python 3.14.3. The full dataset has millions of rows per source, so monitor memory and runtime. The current pipeline loads files and indexes into memory; benchmark on representative data before launching a full run on a constrained machine.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python business_entity_resolution\code\src\train.py
python business_entity_resolution\code\src\inference.py
```

The default input paths now resolve to `dataset/student_resource/dataset/{train,test}`. Model and output paths default to `models/lgb_model.joblib` and `output/`.

For a deterministic bounded training/validation run, pass `--max-train-entities N`; Source 1 is sampled while streaming, and each target source is scanned while retaining the sampled anchors' gold targets plus up to `2*N` seeded distractors. The run writes a `.metrics.json` file beside the model with pair counts, candidate recall, calibrated metrics, and provenance. This is a sampled target corpus, not a full-corpus performance result.

Example sampled run:

```powershell
python business_entity_resolution\code\src\train.py `
  --train-dir dataset\student_resource\dataset\train `
  --model-out models\validation_model.joblib `
  --max-train-entities 5000 `
  --max-candidates-per-entity 40
```

Full training and inference have not been completed on this machine. A 10,000-anchor confirmation run reached 96% system memory use and was stopped; benchmark memory on a larger-memory system before attempting full runs.

## Validation and package

The package utility runs the challenge-provided validator before creating the ZIP. It checks the output schema, Source 1 coverage, duplicates, and candidate/match subset relationship. Add `--check-ids` to validate target IDs against the test source files when sufficient memory is available; without it, ID-existence checks are skipped. The validator does not compute F₀.₅.

Build the ZIP after both output files exist and every measured-results placeholder in the methodology document has been completed:

```powershell
python utils\package_submission.py --team-name team_innovators
# Add --check-ids when sufficient memory is available.
```

The archive contains `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, and the filled `Documentation_template.md` at the ZIP root. Do not include raw datasets, `.venv`, caches, or Git metadata.
## Evaluation and compliance

Optimize the exact macro-average F₀.₅ over Source 1 entities. Correctly predicting an empty list for a true singleton scores 1.0; predicting any match for it scores 0.0. The final model must meet the challenge's MIT/Apache 2.0 license and up-to-8-billion-parameter constraints. Do not use external lookups or external data augmentation.

Do not publish sampled candidate-recall, reduction-ratio, threshold, or F₀.₅ as full-data or leaderboard results. Record the validation split and run configuration for any measured claim.


