"""
Evaluation Metrics Engine.
Amazon ML Challenge 2026 — Business Entity Resolution.
Implements:
- Entity-level Macro F_0.5 with exact singleton accounting.
- Candidate blocking Recall and Reduction Ratio.
"""

from typing import Dict, List, Set, Tuple


def compute_entity_f_beta(
    true_matches: Set[str], pred_matches: Set[str], beta: float = 0.5
) -> float:
    """
    Computes F_beta score for a single anchor entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    - If true is empty (singleton):
        - score is 1.0 if pred is empty.
        - score is 0.0 if pred is non-empty (false merge penalty).
    - If true is non-empty:
        - score is 0.0 if pred is empty.
        - otherwise standard F_beta on matched ID sets.
    """
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    if len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    beta_sq = beta ** 2  # 0.25 for beta=0.5
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0

    f_beta = ((1 + beta_sq) * precision * recall) / denominator
    return f_beta


def compute_macro_f05(
    ground_truth_map: Dict[str, List[str]],
    prediction_map: Dict[str, List[str]],
) -> Tuple[float, Dict[str, float]]:
    """
    Computes Macro F_0.5 across all entities in ground_truth_map.
    Returns:
        overall_macro_f05: float
        per_entity_scores: dict mapping s1_id -> f05 score
    """
    scores = {}
    for s1_id, true_list in ground_truth_map.items():
        true_set = set(true_list)
        pred_set = set(prediction_map.get(s1_id, []))
        score = compute_entity_f_beta(true_set, pred_set, beta=0.5)
        scores[s1_id] = score

    if not scores:
        return 0.0, {}

    macro_f05 = sum(scores.values()) / len(scores)
    return macro_f05, scores


def compute_blocking_metrics(
    ground_truth_map: Dict[str, List[str]],
    candidate_map: Dict[str, List[str]],
    total_s2_count: int,
    total_s3_count: int,
) -> Dict[str, float]:
    """
    Computes candidate generation / blocking metrics:
    - Pair Recall (upper bound for classification model)
    - Reduction Ratio
    - Average candidate count per entity
    - Max candidate count per entity
    """
    total_true_pairs = 0
    covered_true_pairs = 0
    total_candidates = 0
    max_candidates = 0

    for s1_id, true_list in ground_truth_map.items():
        true_set = set(true_list)
        cand_set = set(candidate_map.get(s1_id, []))
        total_true_pairs += len(true_set)
        covered_true_pairs += len(true_set & cand_set)
        c_len = len(cand_set)
        total_candidates += c_len
        if c_len > max_candidates:
            max_candidates = c_len

    pair_recall = (
        (covered_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    )
    n1 = len(ground_truth_map)
    avg_candidates = total_candidates / n1 if n1 > 0 else 0.0

    search_space = n1 * (total_s2_count + total_s3_count)
    reduction_ratio = (
        1.0 - (total_candidates / search_space) if search_space > 0 else 1.0
    )

    return {
        "pair_recall": pair_recall,
        "covered_pairs": covered_true_pairs,
        "total_true_pairs": total_true_pairs,
        "total_candidates": total_candidates,
        "avg_candidates_per_entity": avg_candidates,
        "max_candidates_per_entity": max_candidates,
        "reduction_ratio": reduction_ratio,
    }
