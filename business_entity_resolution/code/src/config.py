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

# Evaluation & Metric Hyperparameters
BETA = 0.5
BETA_SQ = BETA ** 2  # 0.25
DEFAULT_THRESHOLD = 0.70
THRESHOLD_GRID_START = 0.40
THRESHOLD_GRID_STOP = 0.90
THRESHOLD_GRID_STEP = 0.02
