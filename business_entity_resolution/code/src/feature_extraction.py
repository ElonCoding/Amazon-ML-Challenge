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

def extract_pair_features(s1_row: pd.Series, target_row: pd.Series) -> Dict[str, float]:
    s1_country = s1_row["clean_country"]
    t_country = target_row["clean_country"]
    s1_name_clean = s1_row["clean_name"]
    t_name_clean = target_row["clean_name"]
    s1_addr_clean = s1_row["clean_address"]
    t_addr_clean = target_row["clean_address"]
    s1_digits = s1_row["digits"]
    t_digits = target_row["digits"]
    t_id = target_row["entity_id"]

    feats = {}
    feats["feat_country_exact"] = 1.0 if s1_country == t_country else 0.0
    feats["feat_name_clean_exact"] = 1.0 if s1_name_clean == t_name_clean else 0.0
    feats["feat_addr_exact"] = 1.0 if s1_addr_clean == t_addr_clean else 0.0
    feats["feat_name_levenshtein_ratio"] = fuzz.ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_jaro_winkler"] = float(distance.JaroWinkler.similarity(s1_name_clean, t_name_clean))
    feats["feat_name_token_sort_ratio"] = fuzz.token_sort_ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_token_set_ratio"] = fuzz.token_set_ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_partial_ratio"] = fuzz.partial_ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_longest_common_sub_ratio"] = longest_common_substring_ratio(s1_name_clean, t_name_clean)

    s1_addr_tokens = s1_addr_clean.split()
    t_addr_tokens = t_addr_clean.split()
    feats["feat_addr_token_sort_ratio"] = fuzz.token_sort_ratio(s1_addr_clean, t_addr_clean) / 100.0
    feats["feat_addr_token_set_ratio"] = fuzz.token_set_ratio(s1_addr_clean, t_addr_clean) / 100.0
    feats["feat_addr_partial_ratio"] = fuzz.partial_ratio(s1_addr_clean, t_addr_clean) / 100.0
    feats["feat_addr_token_jaccard"] = token_jaccard(s1_addr_tokens, t_addr_tokens)

    overlap_digits = set(s1_digits) & set(t_digits)
    feats["feat_digits_exact_overlap_count"] = float(len(overlap_digits))
    feats["feat_digits_has_overlap"] = 1.0 if len(overlap_digits) > 0 else 0.0
    feats["feat_digits_jaccard"] = token_jaccard(s1_digits, t_digits)

    s1_name_len, t_name_len = len(s1_name_clean), len(t_name_clean)
    feats["feat_name_len_diff"] = float(abs(s1_name_len - t_name_len))
    max_nl = max(s1_name_len, t_name_len, 1)
    feats["feat_name_len_ratio"] = min(s1_name_len, t_name_len) / max_nl

    s1_ntok, t_ntok = len(s1_name_clean.split()), len(t_name_clean.split())
    feats["feat_name_token_diff"] = float(abs(s1_ntok - t_ntok))
    max_nt = max(s1_ntok, t_ntok, 1)
    feats["feat_name_token_ratio"] = min(s1_ntok, t_ntok) / max_nt

    s1_addr_len, t_addr_len = len(s1_addr_clean), len(t_addr_clean)
    feats["feat_addr_len_diff"] = float(abs(s1_addr_len - t_addr_len))
    max_al = max(s1_addr_len, t_addr_len, 1)
    feats["feat_addr_len_ratio"] = min(s1_addr_len, t_addr_len) / max_al

    feats["feat_addr_token_diff"] = float(abs(len(s1_addr_tokens) - len(t_addr_tokens)))
    max_at = max(len(s1_addr_tokens), len(t_addr_tokens), 1)
    feats["feat_addr_token_ratio"] = min(len(s1_addr_tokens), len(t_addr_tokens)) / max_at

    feats["feat_target_is_s2"] = 1.0 if t_id.startswith("S2-") else 0.0
    feats["feat_target_is_s3"] = 1.0 if t_id.startswith("S3-") else 0.0
    return feats
