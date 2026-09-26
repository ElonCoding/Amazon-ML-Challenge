# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** team innovators  
**Team Members:** Parikshit Sharma, Aditi Sharma  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We present a high-precision, scalable entity resolution pipeline designed for the Amazon ML Challenge 2026. The solution links deduplicated Source 1 business records across noisy, multi-million record candidate corpora (Source 2 and Source 3) under zero-match (singletons) and multi-match conditions. Combining an open-set, country-partitioned TF-IDF n-gram inverted index blocker with a 26-feature LightGBM gradient boosted classifier calibrated for Macro $F_{0.5}$, the system achieves a validation Macro $F_{0.5}$ of **0.9615** (pair recall 0.9368, reduction ratio 0.999465) with strictly zero external APIs, internet calls, or data enrichment.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale and Memory Boundaries:** Training data spans over 12.5 million cumulative records across Source 1 (2,206,821), Source 2 (5,034,616), and Source 3 (5,285,603), while test data includes ~11.7 million records. An all-pairs comparison ($O(N \cdot M)$) requires $>10^{13}$ pair evaluations, necessitating sub-linear inverted indexing and memory-bounded streaming.
- **Label Distribution & Singletons:** Ground-truth match counts per Source 1 entity range from 0 to 11 (mean ~3.4), with 123,247 true singletons (5.6% zero-match rate). Macro $F_{0.5}$ heavily penalizes false merges (precision-biased, $\beta=0.5$), requiring strict threshold calibration and exact handling of empty string predictions.
- **Open-Set Geography & Noise:** While training consists primarily of US and India records, the test set features unseen geographic partitions (France comprising ~15% of test Source 1, 259,452 records). Robust preprocessing must support arbitrary international characters and unseen country strings without static whitelists.
- **Text & Address Variations:** Significant abbreviations (e.g., "Pvt Ltd", "Corp", "Blvd", "St", "Rd"), missing address fields (344,883 empty addresses in S2/S3), and PIN/ZIP code formats require robust multi-token and digit-level feature extraction.

### 2.2 Solution Strategy
- **Approach Type:** Hybrid Multi-Index Blocking + Calibrated Pairwise Gradient Boosted Tree Classifier (LightGBM).
- **Core Innovation:** 
  1. *Dynamic Open-Set Country Partitioning:* Indexes and filters candidates within normalized country partitions dynamically, eliminating cross-country false merges and bounding index size while naturally handling France and open-world entities.
  2. *Dual-Signal Term-Capped Inverted Index:* Character n-grams ($3 \le n \le 5$) for fuzzy typo tolerance and word tokens for address matching, pruned with a document-frequency ceiling ($\text{DF} \le 50,000$) to drop uninformative stop-terms and retain rare discriminatory tokens.
  3. *Balanced Rank-Interleaved Retrieval:* Evaluates Source 2 and Source 3 independently before merging candidates in alternating rank order, preventing Source 2 dominance and guaranteeing balanced recall across target sources.
  4. *Macro $F_{0.5}$ Threshold Calibration:* Explicitly optimizes candidate acceptance threshold $\tau$ on held-out Source 1 entities accounting for singleton metrics ($1.0$ if both sets empty, $0.0$ if asymmetric).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Country Key:** Cleaned, lowercased open-set country category (exact match partition).
  - **Business Name Index:** Character $n$-grams (lengths 3 to 5) padded with whitespace boundaries, weighted by TF-IDF with cosine similarity scoring ($\text{top\_k}=12, \text{min\_sim}=0.12$).
  - **Address Index:** Word tokens ($len > 1$) with TF-IDF cosine similarity ($\text{top\_k}=6, \text{min\_sim}=0.15$).
  - **Numeric Token Index:** Exact overlap lookup for numerical sequences (PIN/ZIP codes, plot/suite numbers).
- **Candidate pairs generated:** Capped at a maximum of 20 candidates per Source 1 entity. Average candidates per entity: **19.99**.
- **How true matches were not lost:**
  - Independent retrieval from Source 2 and Source 3 guarantees representation for both catalogs.
  - Frequency cutoff (50k) strips ultra-frequent generic words ("enterprises", "solutions", "limited") while preserving rare distinctive tokens.
  - Achieved **0.9368 pair recall** and **0.999465 reduction ratio** on validation holdout.

---

## 4. Matching Model

**Features used (26 total engineered features):**
- **Exact Agreement Flags (3):** Exact normalized country match, exact clean name match, exact clean address match.
- **Name Lexical Similarity (6):** Levenshtein distance ratio, Jaro-Winkler similarity, token sort ratio, token set ratio, partial token ratio, longest common substring ratio.
- **Address Lexical Similarity (4):** Address token sort ratio, token set ratio, partial ratio, token Jaccard similarity.
- **Numeric & Postal Features (3):** Exact digit overlap count, binary digit overlap flag, digit set Jaccard similarity.
- **Structural Ratios & Lengths (8):** Name absolute character length difference, name length ratio, name token count difference, name token ratio, address length difference, address length ratio, address token count difference, address token ratio.
- **Source Indicators (2):** Binary flags for Source 2 (`feat_target_is_s2`) and Source 3 (`feat_target_is_s3`).

**Model type:** LightGBM Binary Classifier (`LGBMClassifier` / `Booster`)
- `objective`: binary
- `boosting_type`: gbdt
- `learning_rate`: 0.05
- `num_leaves`: 31
- `max_depth`: 6
- `feature_fraction`: 0.8
- `n_estimators`: 150

**Threshold selection method:**
Grid search over decision probability $\tau \in [0.10, 0.95]$ evaluating exact entity-level Macro $F_{0.5}$ on held-out validation entities. Optimal threshold calibrated at **$\tau = 0.860$**.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.9615** (Validation holdout).
- **Blocking Metrics:** Pair Recall = 0.9368, Reduction Ratio = 0.999465.
- **Common false positives (wrong merges):**
  - High-franchise businesses sharing identical company names across different cities/suites when postal digits are absent.
  - Multi-tenant commercial complexes where different businesses share identical street addresses.
  - Addressed by high calibrated threshold ($\tau = 0.860$) favoring precision over recall.
- **Common false negatives (missed matches):**
  - Severe abbreviations omitting the core brand name (e.g., acronyms without phonetic overlap).
  - Target records with completely blank address fields (168k in S2, 175k in S3) lacking numeric validation.

---

## 6. Conclusion

The developed architecture achieves state-of-the-art entity resolution performance (Macro $F_{0.5} = 0.9615$) on the Amazon ML Challenge 2026 dataset. By leveraging Arrow-backed streaming preprocessing, dual-index candidate blocking, and precision-calibrated LightGBM classification, the pipeline delivers sub-linear computational complexity, robust open-world generalization, and strict compliance with challenge integrity rules.

---

## Appendix

### A. Code Artefacts
All code resides under `code/business_entity_resolution/`:
- `src/config.py`: Central hyperparameters, schema constants, and file path resolvers.
- `src/preprocess.py`: Arrow-backed chunked text canonicalization, legal suffix mapping, address digit parsing.
- `src/blocking.py`: TF-IDF char n-gram and word token inverted index blocker.
- `src/feature_extraction.py`: 26-dimensional pairwise similarity and structural feature extraction.
- `src/train.py`: Model fitting and validation threshold grid search for Macro $F_{0.5}$.
- `src/inference.py`: End-to-end test set inference and strict TSV serialization.
- `src/evaluate.py`: Macro $F_{0.5}$ entity scoring including singleton penalization.
- `tests/test_pipeline.py`: Comprehensive test suite verifying schema, France open-set, text cleaning, and metrics.
- `utils/validate_submission.py`: Challenge format and integrity verification.
- `utils/package_submission.py`: Submission zip packager enforcing archive structure.

### B. Additional Results
- **Model Parameter Count:** LightGBM booster with 150 trees, max depth 6 (~4,650 total split nodes, well within the 8-billion parameter ceiling).
- **License Compliance:** LightGBM is open-source under the MIT License; all supporting libraries (scikit-learn, rapidfuzz, pandas, pyarrow) are BSD/Apache/MIT compliant.
- **Fair Play Guarantee:** Zero external APIs, commercial entity registries, or internet lookups were utilized.
