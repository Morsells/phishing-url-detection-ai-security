from pathlib import Path
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features():
    """Load engineered feature tables and return train/test matrices."""
    train_path = PROCESSED_DIR / "urls_train_features.csv"
    test_path = PROCESSED_DIR / "urls_test_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}. Run `python src/features.py` first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}. Run `python src/features.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_cols = {"url", "label"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError(f"Train file missing columns {required_cols}: {train_path}")
    if not required_cols.issubset(test_df.columns):
        raise ValueError(f"Test file missing columns {required_cols}: {test_path}")

    # URL entfernen → nicht als Feature
    X_train = train_df.drop(columns=["url", "label"])
    y_train = train_df["label"].astype(int)

    X_test = test_df.drop(columns=["url", "label"])
    y_test = test_df["label"].astype(int)

    return X_train, y_train, X_test, y_test


def _safe_predict_proba(model, X) -> Optional[np.ndarray]:
    """
    Return probability for class 1 if available, else None.
    Many tree models provide predict_proba; some models do not.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # proba shape: (n_samples, 2)
        return proba[:, 1]
    return None


def evaluate(model, X_test, y_test, name: str) -> Dict[str, Any]:
    """Compute standard classification metrics and print a short summary."""
    pred = model.predict(X_test)

    proba_1 = _safe_predict_proba(model, X_test)

    metrics: Dict[str, Any] = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[tn, fp],[fn, tp]]
    }

    # AUC metrics require scores/probabilities
    if proba_1 is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba_1))
        metrics["pr_auc"] = float(average_precision_score(y_test, proba_1))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    print(f"\n===== {name} =====")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"mcc:       {metrics['mcc']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"roc_auc:   {metrics['roc_auc']:.4f}")
        print(f"pr_auc:    {metrics['pr_auc']:.4f}")
    print(f"confusion_matrix [[tn, fp],[fn,tp]]: {metrics['confusion_matrix']}")

    return metrics


def save_results(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    out_path = RESULTS_DIR / "ml_results.json"
    payload = {
        "meta": meta,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Ergebnisse gespeichert: {out_path}")


def main():
    X_train, y_train, X_test, y_test = load_features()

    meta = {
        "seed": SEED,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
        "features": list(X_train.columns),
        "note": "Models trained on engineered URL features from src/features.py",
    }

    results = []

    # 1) Random Forest
    rf_params = dict(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
    )
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, "RandomForest")
    rf_metrics["params"] = rf_params
    results.append(rf_metrics)

    # 2) XGBoost
    xgb_params = dict(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train)
    xgb_metrics = evaluate(xgb, X_test, y_test, "XGBoost")
    xgb_metrics["params"] = xgb_params
    results.append(xgb_metrics)

    # 3) LightGBM
    lgb_params = dict(
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=64,
        random_state=SEED,
    )
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_train, y_train)
    lgb_metrics = evaluate(lgb, X_test, y_test, "LightGBM")
    lgb_metrics["params"] = lgb_params
    results.append(lgb_metrics)

    save_results(results, meta)


if __name__ == "__main__":
    main()
