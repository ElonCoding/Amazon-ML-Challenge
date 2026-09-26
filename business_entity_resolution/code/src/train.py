"""
Model Training & Threshold Calibration CLI.
Amazon ML Challenge 2026 — Business Entity Resolution.
"""

import os
import sys
import pickle
import argparse
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RANDOM_SEED,
    VAL_SPLIT_RATIO,
    LIGHTGBM_PARAMS,
    N_ESTIMATORS,
    THRESHOLD_GRID_START,
    THRESHOLD_GRID_STOP,
    THRESHOLD_GRID_STEP,
    DEFAULT_THRESHOLD,
    DEFAULT_TRAIN_DIR,
    DEFAULT_MODEL_PATH,
    MAX_CANDIDATES_PER_ENTITY,
)
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
                if matched
                else []
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
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
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
        self,
        X_val: pd.DataFrame,
        val_pairs: List[Tuple[str, str]],
        val_ground_truth: Dict[str, List[str]],
    ) -> Tuple[float, float]:
        probs = self.predict_proba(X_val)
        pair_prob_map = {pair: float(prob) for pair, prob in zip(val_pairs, probs)}

        s1_to_targets: Dict[str, List[Tuple[str, float]]] = {}
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
            pickle.dump(
                {
                    "booster": self.booster,
                    "is_fitted": self.is_fitted,
                    "optimal_threshold": self.optimal_threshold,
                    "feature_names": self.feature_names,
                    "params": self.params,
                },
                f,
            )

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


def train_model(train_dir: str, model_out: str, max_train_entities: Optional[int] = None):
    print("=" * 70)
    print("Starting Model Training Pipeline")
    print("=" * 70)

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("Loading ground truth...")
    gt_map = parse_ground_truth(gt_path)

    print("Loading Source 1...")
    df_s1 = load_and_preprocess_tsv(s1_path)

    if max_train_entities and max_train_entities < len(df_s1):
        print(f"Sampling {max_train_entities} Source 1 entities with seed {RANDOM_SEED}.")
        df_s1 = df_s1.sample(n=max_train_entities, random_state=RANDOM_SEED).reset_index(drop=True)
        sampled_s1_ids = set(df_s1["entity_id"])
        target_ids = {t for s in sampled_s1_ids for t in gt_map.get(s, [])}
        distractor_count = min(50_000, max_train_entities * 2)
        print(f"Loading Source 2 with {len(target_ids)} gold targets and {distractor_count} distractors...")
        df_s2 = load_and_preprocess_tsv(s2_path, filter_ids=target_ids, max_extra_rows=distractor_count)
        print(f"Loading Source 3 with {len(target_ids)} gold targets and {distractor_count} distractors...")
        df_s3 = load_and_preprocess_tsv(s3_path, filter_ids=target_ids, max_extra_rows=distractor_count)
    else:
        print("Loading Source 2...")
        df_s2 = load_and_preprocess_tsv(s2_path)
        print("Loading Source 3...")
        df_s3 = load_and_preprocess_tsv(s3_path)

    np.random.seed(RANDOM_SEED)
    all_s1_ids = df_s1["entity_id"].values
    n_val = int(len(all_s1_ids) * VAL_SPLIT_RATIO)
    val_indices = np.random.choice(len(all_s1_ids), size=n_val, replace=False)
    val_mask = np.zeros(len(all_s1_ids), dtype=bool)
    val_mask[val_indices] = True

    df_s1_val = df_s1[val_mask].copy().reset_index(drop=True)
    df_s1_train = df_s1[~val_mask].copy().reset_index(drop=True)

    train_ids = set(df_s1_train["entity_id"])
    val_ids = set(df_s1_val["entity_id"])
    gt_train = {k: v for k, v in gt_map.items() if k in train_ids}
    gt_val = {k: v for k, v in gt_map.items() if k in val_ids}

    blocker = MultiIndexBlocker(max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY)
    # Build target indexes once for the sampled Source 1 population. Rebuilding
    # the same very large indexes separately for train and validation doubles
    # blocking work without changing the retrieval corpus.
    cands_all = blocker.generate_candidates(df_s1, df_s2, df_s3)
    cands_train = {s1_id: cands_all.get(s1_id, []) for s1_id in gt_train}
    cands_val = {s1_id: cands_all.get(s1_id, []) for s1_id in gt_val}
    del cands_all

    source2_count, source3_count = len(df_s2), len(df_s3)
    b_metrics = compute_blocking_metrics(gt_val, cands_val, source2_count, source3_count)
    print(
        "Blocking validation: "
        f"pair_recall={b_metrics['pair_recall']:.4f}, "
        f"reduction_ratio={b_metrics['reduction_ratio']:.6f}, "
        f"avg_candidates={b_metrics['avg_candidates_per_entity']:.2f}, "
        f"max_candidates={b_metrics['max_candidates_per_entity']}"
    )

    for s1_id, true_targets in gt_train.items():
        existing = set(cands_train.get(s1_id, []))
        for t_id in true_targets:
            if t_id not in existing:
                cands_train[s1_id].append(t_id)

    target_frames = (df_s2, df_s3)
    target_index = tuple(
        frame.set_index("entity_id", drop=False) for frame in target_frames
    )
    X_train_df, train_pairs = build_pair_feature_matrix(
        df_s1_train, target_frames, cands_train, target_index=target_index
    )
    y_train = np.array([1 if t_id in gt_train.get(s1_id, []) else 0 for s1_id, t_id in train_pairs])

    X_val_df, val_pairs = build_pair_feature_matrix(
        df_s1_val, target_frames, cands_val, target_index=target_index
    )
    y_val = np.array([1 if t_id in gt_val.get(s1_id, []) else 0 for s1_id, t_id in val_pairs])
    del target_index, target_frames, df_s2, df_s3

    clf = EntityResolutionClassifier()
    clf.fit(X_train_df, y_train, X_val_df, y_val)
    del X_train_df, y_train, train_pairs

    best_tau, best_f05 = clf.optimize_threshold(X_val_df, val_pairs, gt_val)
    print(f"Calibration Result: Optimal Threshold={best_tau:.3f} | Macro F_0.5={best_f05:.4f}")

    clf.save(model_out)
    print(f"Saved model to: {model_out}")
    return clf, best_tau, best_f05


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM entity resolution model.")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-train-entities", type=int, default=None)
    args = parser.parse_args()

    train_model(args.train_dir, args.model_out, args.max_train_entities)
