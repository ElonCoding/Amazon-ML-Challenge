"""
Business Entity Resolution System — Amazon ML Challenge 2026.

Modules:
- config: Paths, seeds, and hyperparameters.
- preprocess: Null-safe text cleanup and normalization.
- blocking: Inverted-index candidate generation.
- feature_extraction: Pairwise lexical, numeric, and source features.
- train: LightGBM training and Source 1 holdout threshold tuning.
- evaluate: Macro F_0.5 and blocking metrics.
- inference: Test prediction and TSV serialization.
"""

__version__ = "1.1.0"
