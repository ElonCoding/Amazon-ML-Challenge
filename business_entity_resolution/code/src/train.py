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

class EntityResolutionClassifier:
    def __init__(self, params: Dict = None, n_estimators: int = N_ESTIMATORS):
        self.params = params or LIGHTGBM_PARAMS
        self.n_estimators = n_estimators
        self.booster: Optional[lgb.Booster] = None
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES
        self.optimal_threshold = DEFAULT_THRESHOLD

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        X_tr = X_train[self.feature_names].to_numpy(dtype=np.float32, copy=False)
        y_tr = y_train.astype(np.float32)
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = X_val[self.feature_names].to_numpy(dtype=np.float32, copy=False)
            y_v = y_val.astype(np.float32)
            dval = lgb.Dataset(X_v, label=y_v, reference=dtrain, feature_name=self.feature_names)
            valid_sets.append(dval)
            valid_names.append("valid")
        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=self.n_estimators,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.booster is None:
            raise RuntimeError("Model is not fitted.")
        if X.empty:
            return np.array([], dtype=np.float32)
        return self.booster.predict(X[self.feature_names].values.astype(np.float32))
