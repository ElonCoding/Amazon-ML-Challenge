# Amazon ML Challenge 2026 — Business Entity Resolution Brief

**Source:** `_amazon_ml_challenge_2026.pdf` (8 pages). This brief summarizes the source requirements and labels source-stated suggestions as tips.

## 1. Objective

Build an ML solution that determines which records across three independent data sources refer to the same real-world business entity. **Source 1 is the deduplicated reference source.** For every Source 1 entity, find all matching records in Source 2 and Source 3. A Source 1 entity may match zero, one, or many records from either source.

## 2. Input files and data schema

All challenge data and submissions use tab-separated `.tsv` files. Read and write with a tab separator; commas occur inside addresses and comma-separated ID lists.

Each source record file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) has:

| Column | Meaning |
|---|---|
| `entity_id` | Unique record identifier. Prefix indicates source: `S1-`, `S2-`, or `S3-`. |
| `business_name` | Business name; may include abbreviations, legal-suffix variation, typos, or transliterations. |
| `business_address` | Business address; may be partial, variably formatted, missing components, or landmark-based. |
| `country` | Country label. Training covers US and India; test additionally contains France. Treat as an open set of string labels: do not hard-code, filter, or one-hot only `{US, India}`. Every test entity, including France, must appear in the submission. |

There is no separate source column: source is identified by the `entity_id` prefix and the file containing the record.

**Training files**

- `dataset/train/train_source1.tsv` — Source 1 training records (deduplicated reference source).
- `dataset/train/train_source2.tsv` — Source 2 training records.
- `dataset/train/train_source3.tsv` — Source 3 training records.
- `dataset/train/train_ground_truth.tsv` — labels with columns `source1_entity_id` and `matched_entity_ids`. The latter is a comma-separated list of matching Source 2 and/or Source 3 IDs; empty when there are no matches.

**Test files**

- `dataset/test/test_source1.tsv` — every Source 1 test entity requires an output row.
- `dataset/test/test_source2.tsv` — Source 2 test records.
- `dataset/test/test_source3.tsv` — Source 3 test records.
- No test ground truth is provided. The source recommends holding out training data for validation and scoring it with the stated F₀.₅ formula.

## 3. Expected noise patterns

| Field | Source-stated variations |
|---|---|
| Names | Abbreviations (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`); inconsistent legal suffixes; DBA/trade names; punctuation (`&`/`and`); word-order transpositions; typos. |
| Addresses | Abbreviations (`Rd`/`Road`, `St`/`Street`); transliteration variants; missing components (for example, PIN code or state); landmark-based references (for example, “Near SBI ATM”); municipal numbering formats; component reordering. |

## 4. Blocking and candidate generation

**Mandatory requirements**

- Blocking/candidate generation must scale: comparing every record with every other record is not an option. Cut the search space to a **small candidate set per Source 1 entity**.
- Candidate generation counts toward final ranking. Organizers review `candidate_pairs.tsv` and the code that produces it alongside the `matching_results.tsv` score. A smaller candidate set per Source 1 entity is ranked higher in the final evaluation beyond the public/private leaderboard.
- `candidate_pairs.tsv` must represent the **last candidate list actually fed to the matching model for inference**, after all blocking/filtering stages—not an earlier blocking pass.
- Every ID in `matching_results.tsv` must also be present as a candidate for that Source 1 entity. A match absent from candidates signals a pipeline bug and the validator warns about it.

### `candidate_pairs.tsv` semantics

Two tab-separated columns:

| Column | Meaning |
|---|---|
| `source1_entity_id` | Source 1 record ID. |
| `candidate_entity_ids` | Comma-separated candidate IDs from Source 2 and/or Source 3. |

Include exactly one row per test Source 1 entity. Leave `candidate_entity_ids` empty if blocking found no candidates. Lists may contain only existing test-set `S2-`/`S3-` IDs and must not contain duplicates. Final matches must be a subset of these candidates. This file is included in the final package for blocking analysis (including recall ceiling and reduction ratio) and pipeline verification; it is **not** scored on the leaderboard.

## 5. `matching_results.tsv` semantics

Two tab-separated columns:

| Column | Meaning |
|---|---|
| `source1_entity_id` | Source 1 record ID. |
| `matched_entity_ids` | Comma-separated matching IDs from Source 2 and/or Source 3. |

Include exactly one row for every Source 1 test entity. Leave the list empty for entities with no matches (singletons). Each list must contain no duplicate IDs and only existing test-set Source 2 or Source 3 IDs. Do not include Source 1 self-matches. This is the file uploaded to the Portal and the only file scored on the leaderboard.

## 6. Validation rules

Before submission, verify both TSVs:

- Exact required column names and tab-separated format.
- Every test Source 1 ID occurs in exactly one row in each file; no missing or duplicate `source1_entity_id` rows.
- `matching_results.tsv` contains only existing test-set Source 2/Source 3 IDs; no Source 1 IDs.
- `candidate_pairs.tsv` contains only existing test-set Source 2/Source 3 IDs.
- No duplicate IDs within any comma-separated ID list.
- Empty ID list is permitted and required when predicting no matches / finding no candidates.
- Every final match is included in the corresponding candidate list.

The source provides `utils/validate_submission.py` (standard library only, no dependencies). Run from the `student_resource/` directory:

```text
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

It prints `PASS` and exits 0 when safe to submit, or prints numbered issues and exits 1. It reads output and test source files; it does not compute score. Submissions that fail validation will not be evaluated.

## 7. Evaluation: F₀.₅ and singletons

Evaluation uses **Fβ with β = 0.5**, a precision-heavy metric that penalizes false merges more than missed matches:

\[
F_{0.5} = \frac{1.25 \times Precision \times Recall}{0.25 \times Precision + Recall}
\]

The score is a **macro-average per Source 1 entity**, then averaged across all Source 1 entities in the evaluation set. Singletons are included. For a Source 1 entity with no true matches, predicting an empty list scores **1.0**; predicting any match scores **0.0**. Correct singleton detection earns credit, and false matches on a singleton are penalized. The source states F₀.₅ weights precision 2× over recall.

## 8. Leaderboard and submission requirements

- **During the challenge:** upload full-test-set `matching_results.tsv` through the Portal, tab-separated and with the exact column names. Public leaderboard scoring uses a subset of test; the private leaderboard uses the remaining portion after the challenge ends. Submit predictions for the full test set in both cases; the split is applied during scoring.
- **Final rankings:** based on the private leaderboard.
- **Final package:** every team must submit one ZIP containing code, both output files, and the methodology document. Top teams’ packages are reviewed before final rankings are confirmed. Correctly formatted submissions should show `SCORED` status with an F₀.₅ score.

## 9. Final submission package structure

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                         # all source code
│       ├── README.md                    # exact end-to-end run instructions
│       └── requirements.txt             # pinned dependencies/environment
└── Documentation_template.md            # filled-in methodology write-up
```

The pipeline must be self-contained and runnable. Anyone should be able to regenerate both output files from the training/test data using only what is in `code/business_entity_resolution/`. A filled-in `.md` methodology document is acceptable; PDF export also works. The source says there is no need to rename the template.

## 10. Model and license constraints

The final model must be licensed under **MIT or Apache 2.0** and have **up to 8 billion parameters**. The source says submissions and pipelines will be reviewed for model-license and fair-play rules.

## 11. Methodology documentation requirements

Fill in the provided `Documentation_template.md` and include it in the ZIP. Describe:

- Methodology used.
- Candidate-generation/blocking strategy.
- Model architecture and feature engineering.
- Any other relevant information about the approach.

The source sets no page limit and says to prioritize clarity and technical depth over brevity.

## 12. Prohibited external data lookup

Participants are **strictly not allowed** to use external databases, APIs, or services to look up business identities or resolve entities. The source explicitly includes:

- Commercial entity-resolution APIs or services.
- Government business-registration database lookups.
- Geocoding APIs to normalize addresses.
- External data augmentation from internet sources.

The source says approaches, methodologies, and code pipelines will be reviewed and verified, and evidence of external data lookup results in immediate disqualification. It frames the challenge as testing ML/data-science skills using only the provided training data.

## 13. Source-stated tips for success (suggestions, not requirements)

- Invest in strong blocking/candidate generation; it determines the upper bound of recall.
- Explore string-similarity features such as Jaccard, Levenshtein, and TF-IDF cosine for name and address matching.
- Pay attention to country-specific address patterns.
- Consider the precision–recall trade-off carefully because F₀.₅ rewards precision more than recall.
- Do not neglect singletons; correctly predicting “no match” is worth 1.0 for that entity.
- Validate output format against the stated rules before submitting.
