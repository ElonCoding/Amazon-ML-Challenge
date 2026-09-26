"""
Global Configuration and Hyperparameters.
Amazon ML Challenge 2026 — Business Entity Resolution.
"""

import os

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Candidate Generation (Blocking) Hyperparameters
MAX_CANDIDATES_PER_ENTITY = 20
NAME_TOP_K = 12
ADDRESS_TOP_K = 6
MIN_NAME_SIMILARITY = 0.12
MIN_ADDRESS_SIMILARITY = 0.15
