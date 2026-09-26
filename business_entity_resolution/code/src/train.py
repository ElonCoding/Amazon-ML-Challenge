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

    def optimize_threshold(
        self, X_val: pd.DataFrame, val_pairs: List[Tuple[str, str]], val_ground_truth: Dict[str, List[str]]
    ) -> Tuple[float, float]:
        probs = self.predict_proba(X_val)
        pair_prob_map = {pair: float(prob) for pair, prob in zip(val_pairs, probs)}
        s1_to_targets = {}
        for (s1_id, t_id), p in pair_prob_map.items():
            s1_to_targets.setdefault(s1_id, []).append((t_id, p))
        best_tau = DEFAULT_THRESHOLD
        best_f05 = -1.0
        test_taus = np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_STOP + 1e-5, THRESHOLD_GRID_STEP)
        for tau in test_taus:
            pred_map = {s1_id: [] for s1_id in val_ground_truth}
            for s1_id, targets in s1_to_targets.items():
                if s1_id in pred_map:
                    pred_map[s1_id] = [t_id for t_id, prob in targets if prob >= tau]
            score, _ = compute_macro_f05(val_ground_truth, pred_map)
            if score > best_f05:
                best_f05 = score
                best_tau = float(tau)
        self.optimal_threshold = best_tau
        return best_tau, best_f05

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "booster": self.booster, "is_fitted": self.is_fitted,
                "optimal_threshold": self.optimal_threshold,
                "feature_names": self.feature_names, "params": self.params,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "EntityResolutionClassifier":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.booster = data["booster"]
        instance.is_fitted = data["is_fitted"]
        instance.optimal_threshold = data.get("optimal_threshold", DEFAULT_THRESHOLD)
        instance.feature_names = data.get("feature_names", FEATURE_NAMES)
        instance.params = data.get("params", instance.params)
        return instance
