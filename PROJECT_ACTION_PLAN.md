# Amazon ML Challenge 2026 — Project Action Plan

**Purpose:** Turn the challenge brief and the current `D:\ML challenge` repository into a reliable, reproducible entity-resolution submission. This document is based on the existing project files and a streaming read of the supplied TSVs. It does not claim that the current model has been run successfully or that the README's performance figures have been reproduced.

**Implementation status (2026-09-26):** Initial project corrections are in place. Default paths now point to the supplied nested data directory. Training samples Source 1 randomly rather than taking the first rows and builds retrieval indexes once for train and validation. The blocker uses Arrow-backed input, drops duplicate raw text after normalization, skips very high-frequency postings, caps numeric postings it only queries partially, and retrieves Source 2/Source 3 separately before a balanced merge. Pair features are generated in bounded Source 1 batches and inference scores them batch by batch. README/methodology claims were replaced with code-grounded descriptions, dependencies were pinned to the installed environment, the stale notebook was rewritten as a streaming data audit, and a packaging utility now stages the required ZIP layout. A full run and measured candidate recall are still outstanding; full-data memory/runtime risk is not cleared. At the latest resource check this machine had 15.8 GB physical RAM, about 1.7 GB available, and about 24.7 GB free on D:, so a full run was not launched.

## 1. Project objective

For every Source 1 test entity, identify **all** matching records in Source 2 and Source 3. A Source 1 entity may have no matches, one match, or multiple matches. Produce:

- `output/matching_results.tsv` — final links, with exactly one row per Source 1 test entity.
- `output/candidate_pairs.tsv` — the final candidate list actually scored by the matching model, with exactly one row per Source 1 test entity.

The final matches for each Source 1 ID must be a subset of its candidate IDs. The challenge uses macro-average F₀.₅ by Source 1 entity, includes singletons, and reviews candidate generation as part of final ranking. External business lookups and enrichment are prohibited.

## 2. What is actually in this project

The repository already contains a LightGBM pair-scoring pipeline, text preprocessing, a multi-index blocker, feature extraction, an evaluator, inference serialization, two copies of a submission validator, a notebook, and a methodology template. The input data and submission outputs are present in the project tree, but the outputs and models directories currently contain only `.gitkeep` files; there are no generated predictions or trained model to inspect. The repository reports no commits yet and its project files are untracked.

The challenge brief is present as `amazon_ml_challenge_2026_brief.md`. The original PDF was not present in the project tree I inspected.

### Data inventory observed

These counts come from streaming the TSV files, not from the README:

| File | Rows | Key observations |
|---|---:|---|
| Train Source 1 | 2,206,821 | US: 1,323,633; India: 883,188. No blank names or addresses observed. |
| Train Source 2 | 5,034,616 | US: 3,016,817; India: 2,017,799. 168,967 blank addresses. |
| Train Source 3 | 5,285,603 | US: 3,170,056; India: 2,115,547. 175,916 blank addresses. |
| Train ground truth | 2,206,821 | Match-list lengths range from 0 to 11. 123,247 rows have no matches (about 5.6%). |
| Test Source 1 | 1,732,544 | India: 809,986; US: 663,106; France: 259,452. France is about 15% of Source 1 test records. |
| Test Source 2 | 4,887,273 | US, India, and France. 129,408 blank addresses. |
| Test Source 3 | 5,082,316 | US, India, and France. 136,098 blank addresses. |

The combined test target set has nearly 10 million records. The training target set also has over 10 million records. This is large enough that memory use and candidate-generation runtime must be treated as core design requirements.

The ground-truth match-count distribution observed is:

| Matches for a Source 1 entity | Count |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

This makes an empty prediction important, but the training labels are not singleton-dominated: about 94.4% of Source 1 rows have at least one match. Do not assume the hidden test has the same distribution without evidence.

## 3. Highest-priority issues to resolve

### P0 — Fix the data paths before relying on the documented commands

The actual challenge data is under:

```text
D:\ML challenge\dataset\student_resource\dataset\train
D:\ML challenge\dataset\student_resource\dataset\test
```

The pipeline defaults in `business_entity_resolution\code\src\config.py` originally pointed to the empty root-level placeholder folders. They now point to the supplied `dataset\student_resource\dataset\{train,test}` directories. Keep this path relationship in the reproduction commands and package README.

Until the defaults are corrected, pass explicit paths to both scripts. For example, from PowerShell:

```powershell
Set-Location 'D:\ML challenge'
python business_entity_resolution\code\src\train.py `
  --train-dir 'D:\ML challenge\dataset\student_resource\dataset\train' `
  --model-out 'D:\ML challenge\models\lgb_model.joblib'

python business_entity_resolution\code\src\inference.py `
  --test-dir 'D:\ML challenge\dataset\student_resource\dataset\test' `
  --model-path 'D:\ML challenge\models\lgb_model.joblib' `
  --output-dir 'D:\ML challenge\output'
```

These commands show the corrected paths; they are **not a recommendation to launch the full run yet**. First address the scale risks below.

### P0 — Make blocking and feature generation fit the data size

The blocker still retains Python posting lists and the preprocessed source tables in memory for millions of target rows. The implementation now omits terms with document frequency above 50,000, caps numeric postings at the ten IDs queried, and builds target indexes once for the sampled Source 1 population. Pair extraction now uses a pandas index and bounded 5,000-Source-1 batches; inference no longer materializes the entire pair-feature matrix at once. These changes reduce transient memory but have not been measured on the full corpus and do not make the run memory-safe by proof.

Recommended implementation direction:

1. Continue replacing high-overhead postings with compact storage if measurements show memory pressure; the code no longer retains a `Counter` per document, but still retains postings and source tables.
2. Candidate maps and serialized candidate lists still accumulate across the full Source 1 set. If they dominate memory, stream candidate generation/output in batches while preserving one final row per Source 1 entity.
3. Measure peak memory, indexing time, query throughput, feature throughput, and candidate counts on representative subsets before a full run.
4. Preserve every Source 1 ID through batching, including records with empty fields or no candidates.
5. Track candidate recall and mean/maximum candidate count on the held-out training entities while changing the blocker.

A `--max-train-entities` argument currently limits Source 1 rows only; it still loads all Source 2 and Source 3 rows and builds indexes over them. It is not by itself a safe small-data benchmark. Use a deliberately small, internally consistent data sample for initial runtime checks, then test a larger sample before deciding on the full-data architecture.

### P0 — Correct the package layout and build process

The challenge package should have this shape:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The current `build_submission.sh` archives `business_entity_resolution/` at the ZIP root, which leaves `code/` out of the required position and places the methodology template inside that directory. It also assumes the validator's `dataset/test` default path, which does not contain the real files here. Fix the build process to stage the required tree explicitly and validate against the actual test directory before creating the ZIP. Keep raw TSVs, `.venv`, `.git`, caches, and temporary model artifacts out of the submission archive.

### P1 — Make the README and methodology match the code and measured results

Several current claims are either unsupported by the code or not demonstrated by saved run outputs:

- The project initially said a 20-candidate ceiling was required. The README and methodology now describe it as a project setting; the challenge asks for a **small** candidate set and rewards smaller sets but does not prescribe a maximum of 20.
- The README/template report blocking recall above 98%, reduction ratios above 99.8–99.98%, macro F₀.₅ of 0.912, a threshold near 0.70, and a 2.4 MB model. The model and output folders are empty, and no run report supports these figures. Remove them or replace them with reproducibly measured results, including the split and exact metric implementation.
- `train.py` makes one random 20% Source 1 holdout; it is not stratified or out-of-fold. The methodology now states that accurately.
- The methodology initially listed phonetic, URL/domain, and phone features that are not in `feature_extraction.py`. It now documents the 26 implemented features from the available name, address, country, and ID fields.
- README quickstart paths and model filenames do not consistently match the script defaults and current data placement. `code/README.md` refers to a `src/pipeline.py` that is absent.
- The notebook imported missing functions/classes and used wrong column names and paths. It has been replaced with a streaming TSV audit; no model results are generated by the notebook.
- The dependency files initially used lower bounds (`>=`). They now pin the versions installed in the project `.venv`.

Keep the methodology document factual. Report only experiments that were actually run and can be repeated from the submitted code and documented commands.

## 4. Recommended solution workflow

### Step 1 — Freeze an auditable input and environment

Keep the raw challenge data unchanged and out of Git. Record file names, byte sizes, row counts, headers, and hashes in a local run log or methodology appendix. Pin the runtime and package versions used for the final run. Preserve the source `entity_id` values exactly; use normalized text only for retrieval and features.

All TSV readers must explicitly use a tab separator and string IDs. Retain empty fields as empty strings. The observed auxiliary address fields include hundreds of thousands of blanks, so missing values must not crash tokenization or be treated as evidence of a match.

### Step 2 — Create a held-out validation design

Split by complete Source 1 entities, not individual candidate pairs. Keep every validation Source 1 entity in the score denominator, including those with no generated candidates. Do not use the test set to tune thresholds or features.

For each validation Source 1 entity, run the exact production blocking and scoring path. Measure:

- Candidate pair recall: fraction of true IDs present in the candidate list.
- Candidate count distribution per Source 1 entity and the total candidate volume.
- Final macro F₀.₅ per Source 1 entity, including the source-specified singleton behavior.
- False matches on training singletons and errors by country and target source.

`evaluate.py` contains the expected singleton rules and per-entity macro F₀.₅ logic. Preserve the explicit empty-truth cases: empty prediction scores 1.0; any predicted ID scores 0.0. Confirm the rest of the metric against the challenge brief before using it to select a threshold.

### Step 3 — Build high-recall, bounded candidate generation

The current `MultiIndexBlocker` uses character n-gram name retrieval, address-token retrieval, numeric address tokens, and a top-20 union cap. That is a reasonable prototype, but evaluate it rather than assuming its claims.

Improve and measure it in this order:

1. Keep the candidate sources separate for Source 2 and Source 3 so one source cannot crowd out the other in a shared top-K list.
2. Combine complementary retrieval signals: exact normalized name, informative name tokens, character n-grams, address tokens, and numeric address components. Keep missing-address entities eligible through name-based routes.
3. Avoid allowing very common character n-grams or address tokens to create huge posting traversals. Use frequency-aware indexing and bounded retrieval while measuring recall.
4. Keep country as an open-set signal. The current dynamic exact-country partition can process France, but it excludes cross-country-label candidates entirely. The challenge says not to hard-code/filter countries; it does not state that differently labeled records can never match. Test country agreement as a feature or add a controlled fallback retrieval path so country partitioning is not the sole route.
5. Tune any candidate cap against held-out candidate recall and candidate counts. A small candidate set is a challenge requirement; exactly 20 is not.
6. Save `candidate_pairs.tsv` from the final candidate list the model actually scored, after all retrieval and filtering stages.

### Step 4 — Train and calibrate the pair scorer

The current 26 implemented features cover country/name/address equality and similarities, digit overlap, lengths/token counts, and target-source indicators. They are a baseline for the fields available. Train a binary pair scorer on positive links from ground truth and nonmatching candidate pairs, with care that negatives resemble the hard candidates encountered during inference.

`train.py` currently adds any missing gold links to the training candidate map after it measures validation blocking. That can provide positive training pairs, but it means the training feature distribution includes pairs the blocker failed to retrieve. Keep that distinction explicit. The validation score must continue to reflect the real pipeline: a gold pair missing from validation candidates is unrecoverable and must count as missed.

Tune the probability threshold on held-out Source 1 entities for macro F₀.₅, not pair-level accuracy. Preserve multi-match behavior: emit each candidate that meets the calibrated decision rule; do not impose one-to-one matching. If tuning a source-specific or country-specific threshold, require enough validation evidence and document it to avoid overfitting.

The final model must meet the challenge's MIT/Apache 2.0 license and up-to-8-billion-parameter limits. LightGBM is an MIT-licensed, small model family, but document the actual model choice and license rather than relying only on a README badge.

### Step 5 — Run full inference and validate both files

For every test Source 1 row, serialize a row even when there are no candidates or no predicted matches. Output exactly these headers:

```text
source1_entity_id<TAB>matched_entity_ids
source1_entity_id<TAB>candidate_entity_ids
```

Use commas only between IDs in each list; use an empty second field for none. Verify unique Source 1 rows, valid Source 2/Source 3 IDs, no repeated IDs per list, and final matches as a subset of candidates. Include France rows; do not filter them because they were unseen in training.

The project validator exposes ID-existence checking as optional (`--check-ids`). For the final run, pass both output paths, the actual test directory, and `--check-ids` when memory permits. Do not mistake a default PASS that skipped ID existence checks for complete verification. The challenge validator checks formatting/integrity, not the model's score.

### Step 6 — Reproduce, document, and package

Before packaging, start from a clean output directory, run the documented training and inference commands, run the validator, and build the exact required ZIP tree. The reproduction guide should give commands that work from the stated directory and use the actual data paths. Include pinned dependencies and filled methodology documentation. Check the ZIP contents after building.

## 5. Specific risks in the current implementation

| Risk | Why it matters | Action |
|---|---|---|
| Data paths were pointed at empty placeholders | Training/inference would not find the provided TSVs. | Fixed `config.py` defaults; retain explicit path args for reproducibility. |
| Python indexes over ~10M target records | May exhaust memory or take impractical time. | Reduced high-frequency/numeric postings and added feature batching; still needs full-scale resource measurement and possibly compact or streamed blocking. |
| Strict country partition as the only retrieval path | It may miss links if country labels differ; open-set support alone does not prove recall. | Validate and add a controlled non-country fallback. |
| Shared S2/S3 top-K retrieval | One source may crowd candidates out for the other. | Retrieve per source, then merge and rank. |
| Candidate cap fixed at 20 | Not a challenge rule; may reduce recall. | Select cap from held-out candidate recall and candidate-volume evidence. |
| One random holdout presented as OOF/stratified | Documentation would misstate the experiment. | Correct docs or implement and record the stated validation design. |
| Unverified performance numbers | Cannot be defended in the final methodology. | Removed from current README/methodology; re-run, save logs/metrics, and report only measured values. |
| Stale notebook/API references | Notebook previously did not match current modules/schema. | Replaced with a streaming TSV audit; do not use it as model evidence. |
| Packaging script archives wrong tree and wrong test path | Final ZIP can fail reproduction/review despite valid predictions. | Added `utils/package_submission.py` and a PowerShell build wrapper for the exact required archive layout; still needs a completed output package to exercise. |

## 6. Do not do these things

- Do not use commercial ER APIs, business registries, geocoding services, internet lookups, or external data augmentation.
- Do not hard-code only US and India or omit the 259,452 France Source 1 test records.
- Do not brute-force every Source 1 against all Source 2/3 records.
- Do not assume a 20-candidate maximum is required by the challenge.
- Do not force one-to-one matches or limit each Source 1 entity to one output ID.
- Do not treat a missing candidate as a model false negative that threshold tuning can fix; improve blocking because the scorer cannot recover an omitted record.
- Do not use random pair-level validation, score only non-singletons, or tune against test labels.
- Do not leave unsupported scores, thresholds, or feature claims in README/methodology materials.
- Do not commit raw TSV data, `.venv`, caches, `.git`, or large generated model/output artifacts unless explicitly required for the final ZIP.
- Do not package until both output files pass the validator and the archive has the exact required directory structure.

## 7. Completion checklist

- [ ] Data paths resolve to the `dataset/student_resource/dataset/{train,test}` files.
- [ ] Full-scale runtime and peak memory are measured; no all-pairs comparison is performed.
- [ ] Validation is split by Source 1 entities and reports candidate recall, macro F₀.₅, singleton behavior, and candidate counts.
- [ ] Blocking preserves strong recall while keeping candidate sets small; no fixed 20 cap is presented as an official rule.
- [ ] Candidate generation includes both Source 2 and Source 3 and supports France/open country labels.
- [ ] `candidate_pairs.tsv` is exactly the final inference candidate set.
- [ ] Every test Source 1 ID appears exactly once in both output files; lists are valid and deduplicated.
- [ ] Every final match appears in that entity's candidate list.
- [ ] Validator runs with the correct test directory and ID checks enabled where feasible.
- [ ] Model/license and parameter-limit claims are documented accurately.
- [ ] README, notebook, methodology, and commands match the code that is actually shipped.
- [ ] Dependencies are pinned; the run is reproducible from documented commands.
- [ ] ZIP layout matches the challenge specification and contains both outputs, runnable code, and methodology.

**Review boundary:** I inspected the project inventory, source code, configuration, README files, methodology template, notebook cells, tests, validators, packaging script, and streamed all seven challenge TSVs for row counts, country distribution, blank names/addresses, and ground-truth list lengths. I did not run the model, submission tests, or validator; the project currently has no generated output/model artifacts to verify.
