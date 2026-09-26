"""
Feature Engineering Engine for Candidate Pair Resolution.
Extracts 26 lexical, token, numeric, and structural features
for candidate pairs (e_1, e_target).
"""

import math
from difflib import SequenceMatcher
from typing import Dict, Iterator, List, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


FEATURE_NAMES = [
    # 1. Exact match flags
    "feat_country_exact",
    "feat_name_clean_exact",
    "feat_addr_exact",
    # 2. Name lexical similarity
    "feat_name_levenshtein_ratio",
    "feat_name_jaro_winkler",
    "feat_name_token_sort_ratio",
    "feat_name_token_set_ratio",
    "feat_name_partial_ratio",
    "feat_name_longest_common_sub_ratio",
    # 3. Address lexical similarity
    "feat_addr_token_sort_ratio",
    "feat_addr_token_set_ratio",
    "feat_addr_partial_ratio",
    "feat_addr_token_jaccard",
    # 4. Numeric and PIN/ZIP overlap
    "feat_digits_exact_overlap_count",
    "feat_digits_has_overlap",
    "feat_digits_jaccard",
    # 5. Structural and length ratios
    "feat_name_len_diff",
    "feat_name_len_ratio",
    "feat_name_token_diff",
    "feat_name_token_ratio",
    "feat_addr_len_diff",
    "feat_addr_len_ratio",
    "feat_addr_token_diff",
    "feat_addr_token_ratio",
    # 6. Source indicators
    "feat_target_is_s2",
    "feat_target_is_s3",
]


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


def extract_pair_features(
    s1_row: pd.Series, target_row: pd.Series
) -> Dict[str, float]:
    """
    Computes pairwise feature dictionary between s1_row and target_row.
    """
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

    # 1. Exact match
    feats["feat_country_exact"] = 1.0 if s1_country == t_country else 0.0
    feats["feat_name_clean_exact"] = 1.0 if s1_name_clean == t_name_clean else 0.0
    feats["feat_addr_exact"] = 1.0 if s1_addr_clean == t_addr_clean else 0.0

    # 2. Name lexical
    feats["feat_name_levenshtein_ratio"] = fuzz.ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_jaro_winkler"] = float(distance.JaroWinkler.similarity(s1_name_clean, t_name_clean))
    feats["feat_name_token_sort_ratio"] = fuzz.token_sort_ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_token_set_ratio"] = fuzz.token_set_ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_partial_ratio"] = fuzz.partial_ratio(s1_name_clean, t_name_clean) / 100.0
    feats["feat_name_longest_common_sub_ratio"] = longest_common_substring_ratio(s1_name_clean, t_name_clean)

    # 3. Address lexical
    s1_addr_tokens = s1_addr_clean.split()
    t_addr_tokens = t_addr_clean.split()
    feats["feat_addr_token_sort_ratio"] = fuzz.token_sort_ratio(s1_addr_clean, t_addr_clean) / 100.0
    feats["feat_addr_token_set_ratio"] = fuzz.token_set_ratio(s1_addr_clean, t_addr_clean) / 100.0
    feats["feat_addr_partial_ratio"] = fuzz.partial_ratio(s1_addr_clean, t_addr_clean) / 100.0
    feats["feat_addr_token_jaccard"] = token_jaccard(s1_addr_tokens, t_addr_tokens)

    # 4. Numeric & PIN/ZIP overlap
    overlap_digits = set(s1_digits) & set(t_digits)
    feats["feat_digits_exact_overlap_count"] = float(len(overlap_digits))
    feats["feat_digits_has_overlap"] = 1.0 if len(overlap_digits) > 0 else 0.0
    feats["feat_digits_jaccard"] = token_jaccard(s1_digits, t_digits)

    # 5. Structural & length ratios
    s1_name_len = len(s1_name_clean)
    t_name_len = len(t_name_clean)
    feats["feat_name_len_diff"] = float(abs(s1_name_len - t_name_len))
    max_nl = max(s1_name_len, t_name_len, 1)
    feats["feat_name_len_ratio"] = min(s1_name_len, t_name_len) / max_nl

    s1_ntok = len(s1_name_clean.split())
    t_ntok = len(t_name_clean.split())
    feats["feat_name_token_diff"] = float(abs(s1_ntok - t_ntok))
    max_nt = max(s1_ntok, t_ntok, 1)
    feats["feat_name_token_ratio"] = min(s1_ntok, t_ntok) / max_nt

    s1_addr_len = len(s1_addr_clean)
    t_addr_len = len(t_addr_clean)
    feats["feat_addr_len_diff"] = float(abs(s1_addr_len - t_addr_len))
    max_al = max(s1_addr_len, t_addr_len, 1)
    feats["feat_addr_len_ratio"] = min(s1_addr_len, t_addr_len) / max_al

    feats["feat_addr_token_diff"] = float(abs(len(s1_addr_tokens) - len(t_addr_tokens)))
    max_at = max(len(s1_addr_tokens), len(t_addr_tokens), 1)
    feats["feat_addr_token_ratio"] = min(len(s1_addr_tokens), len(t_addr_tokens)) / max_at

    # 6. Source indicators
    feats["feat_target_is_s2"] = 1.0 if t_id.startswith("S2-") else 0.0
    feats["feat_target_is_s3"] = 1.0 if t_id.startswith("S3-") else 0.0

    return feats


def build_pair_feature_matrix(
    df_s1: pd.DataFrame,
    df_target: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    candidate_map: Dict[str, List[str]],
    target_index=None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Extracts features for all candidate pairs in candidate_map in bounded
    Source 1 batches.
    Returns:
        features_df: DataFrame with feature columns
        pair_identifiers: List of (s1_id, target_id)
    """
    feature_batches = []
    pair_identifiers: List[Tuple[str, str]] = []
    for features, pairs in iter_pair_feature_matrices(
        df_s1, df_target, candidate_map, batch_size=5_000,
        target_index=target_index,
    ):
        if not features.empty:
            feature_batches.append(features.astype(np.float32, copy=False))
            pair_identifiers.extend(pairs)
    if not feature_batches:
        return pd.DataFrame(columns=FEATURE_NAMES), pair_identifiers
    return pd.concat(feature_batches, ignore_index=True), pair_identifiers


def iter_pair_feature_matrices(
    df_s1: pd.DataFrame,
    df_target: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    candidate_map: Dict[str, List[str]],
    batch_size: int = 5_000,
    target_index=None,
) -> Iterator[Tuple[pd.DataFrame, List[Tuple[str, str]]]]:
    """Yield pair features in bounded Source 1 batches.

    The prior implementation converted the entire multi-million-row target
    table into a nested Python dictionary and materialized every candidate
    feature row at once. Indexing the frames once and looking up only IDs used
    by each batch keeps that transient memory bounded by the batch candidate
    count.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    s1_index = df_s1.set_index("entity_id", drop=False)
    split_targets = isinstance(df_target, tuple)
    if target_index is None:
        if split_targets:
            target_index = tuple(
                frame.set_index("entity_id", drop=False) for frame in df_target
            )
        else:
            target_index = df_target.set_index("entity_id", drop=False)
    s1_ids = list(candidate_map)

    for start in range(0, len(s1_ids), batch_size):
        batch_ids = s1_ids[start : start + batch_size]
        batch_map = {s1_id: candidate_map[s1_id] for s1_id in batch_ids}
        valid_s1_ids = [s1_id for s1_id in batch_ids if s1_id in s1_index.index]
        if not valid_s1_ids:
            continue

        used_target_ids = list(
            dict.fromkeys(
                target_id
                for s1_id in valid_s1_ids
                for target_id in batch_map[s1_id]
            )
        )
        if not used_target_ids:
            continue

        s1_rows = s1_index.loc[valid_s1_ids].to_dict(orient="index")
        if split_targets:
            target_rows = {}
            for prefix, source_index in zip(("S2-", "S3-"), target_index):
                source_ids = [item for item in used_target_ids if item.startswith(prefix)]
                if source_ids:
                    target_rows.update(
                        source_index.reindex(source_ids)
                        .dropna(subset=["entity_id"])
                        .to_dict(orient="index")
                    )
        else:
            target_rows = target_index.reindex(used_target_ids).dropna(
                subset=["entity_id"]
            ).to_dict(orient="index")
        rows = []
        pairs: List[Tuple[str, str]] = []
        for s1_id in valid_s1_ids:
            s1_row = s1_rows[s1_id]
            for target_id in batch_map[s1_id]:
                target_row = target_rows.get(target_id)
                if target_row is None:
                    continue
                rows.append(extract_pair_features(s1_row, target_row))
                pairs.append((s1_id, target_id))

        if rows:
            yield pd.DataFrame(rows, columns=FEATURE_NAMES), pairs

