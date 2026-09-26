"""
Model Training & Threshold Calibration CLI.
"""
import os, sys, pickle, argparse, csv
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from preprocess import load_and_preprocess_tsv
from blocking import MultiIndexBlocker
from feature_extraction import build_pair_feature_matrix, FEATURE_NAMES
from evaluate import compute_macro_f05, compute_blocking_metrics

def parse_ground_truth(gt_path: str) -> Dict[str, List[str]]:
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Ground-truth file must contain columns {sorted(required)}")
        for row in reader:
            s1_id = (row.get("source1_entity_id") or "").strip()
            matched = (row.get("matched_entity_ids") or "").strip()
            if not s1_id:
                raise ValueError("Ground-truth file contains an empty source1_entity_id")
            if s1_id in gt_map:
                raise ValueError(f"Duplicate ground-truth row for {s1_id}")
            gt_map[s1_id] = (
                [match.strip() for match in matched.split(",") if match.strip()]
                if matched else []
            )
    return gt_map
