# Methodology — Business Entity Resolution

**Team:** team innovators  
**Team Members:** Parikshit Sharma, Aditi Sharma  
**Challenge:** Amazon ML Challenge 2026

> Complete the measured-results fields after running the final pipeline. Do not submit unverified metrics or claims.

## 1. Objective and inputs

The pipeline links each deduplicated Source 1 business record to zero, one, or multiple matching records from Source 2 and Source 3. Inputs are the challenge-provided tab-separated files. Record fields used by the current implementation are `entity_id`, `business_name`, `business_address`, and `country`; labels come from `train_ground_truth.tsv`.

## 2. Preprocessing

`src/preprocess.py` reads tab-separated files with Arrow-backed strings, normalizes names and addresses in 100,000-row chunks, lowercases/cleans punctuation and whitespace, applies a limited set of business suffix and address abbreviation rewrites, and extracts numeric address tokens. Country labels are normalized as open-set categories. After canonicalization, duplicate raw text fields are dropped from the working table.

## 3. Candidate generation

`src/blocking.py` generates candidates using separate per-country, per-source inverted indexes over business-name character n-grams and address word tokens, plus a bounded numeric-token lookup. Terms with document frequency above 50,000 are omitted from the text postings. The Source 2 and Source 3 candidate rankings are merged in alternating rank order and capped at 20 per Source 1 entity. Country partitioning uses exact normalized labels; unseen country labels are handled dynamically, while cross-label matches are not retrieved by the current blocker. The cap and frequency cutoff are project settings, not numeric requirements in the challenge brief.

Final report fields to fill after validation:

- Candidate recall on held-out Source 1 entities: **[measure and enter]**
- Mean / median / maximum candidates per Source 1 entity: **[measure and enter]**
- Candidate reduction ratio: **[measure and enter]**
- Country-partition failure analysis / fallback behavior: **[describe the implemented and measured behavior]**

## 4. Pairwise model and features

`src/feature_extraction.py` computes 26 features from the available fields: exact country/name/address agreements; name edit, token, and substring similarities; address token similarities; numeric overlap; string/token lengths; and Source 2/Source 3 indicators. No URL, phone, external registry, or geocoding features are used by the current source code.

`src/train.py` trains a binary LightGBM classifier on candidate pairs. Ground-truth links omitted by training blocking are added to the training candidate map so the pair scorer can see positive examples. Validation scoring uses the actual validation candidate map; a true link missing from that map remains unrecoverable and lowers validation score.

## 5. Validation and threshold

The current training code selects a deterministic random holdout of Source 1 entities (20%) and searches a threshold grid using macro-average F₀.₅. It is a single holdout, not out-of-fold or stratified validation. The metric averages per-entity F₀.₅ and explicitly scores a true singleton as 1.0 for an empty prediction and 0.0 for a non-empty prediction.

- Validation split and seed: **[confirm from run log]**
- Selected threshold: **[measure and enter]**
- Macro F₀.₅: **[measure and enter]**
- Singleton false-positive rate / score: **[measure and enter]**
- Important error patterns: **[inspect validation errors and enter]**

## 6. Test inference and outputs

`src/inference.py` applies the blocker and classifier to test records and writes `matching_results.tsv` and `candidate_pairs.tsv`. Both files must contain exactly one row for every test Source 1 ID. ID lists are comma-separated; empty lists represent no matches or no candidates. Every final match must be among the corresponding candidates.

## 7. Runtime and reproducibility

- Python version: **[record exact version]**
- Dependency lock/version file: `code/requirements.txt`
- Training command and arguments: **[record exact command]**
- Inference command and arguments: **[record exact command]**
- Runtime / peak memory: **[measure and enter]**
- Validator result: **[record result and whether ID checks were enabled]**

The current implementation holds input tables and indexes in memory. Record resource use on the final environment. Include the commands and dependencies necessary to regenerate both output files from the provided training/test data.

## 8. Model license and fair play

The current pair model is LightGBM, which is MIT-licensed. Verify and document the actual final model and its license; ensure it is within the challenge's 8-billion-parameter ceiling. The solution uses only the supplied challenge data and makes no external business-identity, registry, geocoding, or internet data lookups.

