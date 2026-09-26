"""
Evaluation Metrics Engine.
Amazon ML Challenge 2026 — Business Entity Resolution.
"""
from typing import Dict, List, Set, Tuple

def compute_entity_f_beta(
    true_matches: Set[str], pred_matches: Set[str], beta: float = 0.5
) -> float:
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0
    if len(pred_matches) == 0:
        return 0.0
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)
    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denominator

def compute_macro_f05(
    ground_truth_map: Dict[str, List[str]],
    prediction_map: Dict[str, List[str]],
) -> Tuple[float, Dict[str, float]]:
    scores = {}
    for s1_id, true_list in ground_truth_map.items():
        true_set = set(true_list)
        pred_set = set(prediction_map.get(s1_id, []))
        scores[s1_id] = compute_entity_f_beta(true_set, pred_set, beta=0.5)
    if not scores:
        return 0.0, {}
    return sum(scores.values()) / len(scores), scores
