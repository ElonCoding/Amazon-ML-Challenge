"""
Inference & Prediction Scoring CLI.
"""
import os, sys, argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from preprocess import load_and_preprocess_tsv
from blocking import MultiIndexBlocker
from feature_extraction import iter_pair_feature_matrices
from train import EntityResolutionClassifier

def serialize_candidate_pairs(candidate_map: Dict[str, List[str]], s1_ids_order: List[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_order:
            cands = candidate_map.get(s1_id, [])
            seen = set()
            clean_cands = []
            for c in cands:
                if c not in seen and not c.startswith("S1-"):
                    clean_cands.append(c)
                    seen.add(c)
            f.write(f"{s1_id}\t{','.join(clean_cands)}\n")

def serialize_matching_results(
    prediction_map: Dict[str, List[str]], s1_ids_order: List[str],
    output_path: str, candidate_map: Dict[str, List[str]] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids_order:
            matches = prediction_map.get(s1_id, [])
            valid_cands = set(candidate_map.get(s1_id, [])) if candidate_map is not None else None
            seen = set()
            clean_matches = []
            for m in matches:
                if m not in seen and not m.startswith("S1-"):
                    if valid_cands is not None and m not in valid_cands:
                        continue
                    clean_matches.append(m)
                    seen.add(m)
            f.write(f"{s1_id}\t{','.join(clean_matches)}\n")
