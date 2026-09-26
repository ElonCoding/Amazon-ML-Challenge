"""
Feature Engineering Engine for Candidate Pair Resolution.
"""
import math
from difflib import SequenceMatcher
from typing import Dict, Iterator, List, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def longest_common_substring_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    return (2.0 * match.size) / (len(s1) + len(s2))

FEATURE_NAMES = [
    "feat_country_exact", "feat_name_clean_exact", "feat_addr_exact",
    "feat_name_levenshtein_ratio", "feat_name_jaro_winkler", "feat_name_token_sort_ratio",
    "feat_name_token_set_ratio", "feat_name_partial_ratio", "feat_name_longest_common_sub_ratio",
    "feat_addr_token_sort_ratio", "feat_addr_token_set_ratio", "feat_addr_partial_ratio",
    "feat_addr_token_jaccard", "feat_digits_exact_overlap_count", "feat_digits_has_overlap",
    "feat_digits_jaccard", "feat_name_len_diff", "feat_name_len_ratio", "feat_name_token_diff",
    "feat_name_token_ratio", "feat_addr_len_diff", "feat_addr_len_ratio", "feat_addr_token_diff",
    "feat_addr_token_ratio", "feat_target_is_s2", "feat_target_is_s3",
]
