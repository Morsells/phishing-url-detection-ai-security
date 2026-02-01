from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Lädt die erzeugten Feature-Tabellen und gibt Train/Test als (X_train, y_train, X_test, y_test) zurück.

    Erwartete Dateien (aus src/features.py):
      - data/processed/urls_train_features.csv
      - data/processed/urls_test_features.csv
    """
    train_path = PROCESSED_DIR / "urls_train_features.csv"
    test_path = PROCESSED_DIR / "urls_test_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Datei fehlt: {train_path}\n"
            "Bitte zuerst Features erzeugen: python src/features.py"
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Datei fehlt: {test_path}\n"
            "Bitte zuerst Features erzeugen: python src/features.py"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_cols = {"url", "label"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError(f"Train-Datei hat nicht alle Spalten {required_cols}: {train_path}")
    if not required_cols.issubset(test_df.columns):
        raise ValueError(f"Test-Datei hat nicht alle Spalten {required_cols}: {test_path}")

    # URL und Label sind keine Features → url/label entfernen
    X_train = train_df.drop(columns=["url", "label"])
    y_train = train_df["label"].astype(int)

    X_test = test_df.drop(columns=["url", "label"])
    y_test = test_df["label"].astype(int)

    # Mini-Check: Feature-Spalten sollten identisch sein
    if list(X_train.columns) != list(X_test.columns):
        missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
        missing_in_train = [c for c in X_test.columns if c not in X_train.columns]
        raise ValueError(
            "Train/Test Feature-Spalten passen nicht zusammen.\n"
            f"Fehlt im Test: {missing_in_test}\n"
            f"Fehlt im Train: {missing_in_train}"
        )

    return X_train, y_train, X_test, y_test


def _safe_predict_proba(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Gibt p(y=1) zurück, falls das Modell predict_proba unterstützt, sonst None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Erwartete Form: (n_samples, 2) → Index 1 entspricht Klasse "1"
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    return None


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> Dict[str, Any]:
    """
    Berechnet Standard-Metriken und gibt ein Dict zurück (inkl. Confusion Matrix).
    """
    pred = model.predict(X_test)
    proba_1 = _safe_predict_proba(model, X_test)

    metrics: Dict[str, Any] = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[tn, fp], [fn, tp]]
        "roc_auc": None,
        "pr_auc": None,
    }

    # AUC-Metriken benötigen Scores/Wahrscheinlichkeiten
    if proba_1 is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba_1))
        metrics["pr_auc"] = float(average_precision_score(y_test, proba_1))

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
    """
    Speichert die Ergebnisse als JSON: results/metrics/ml_results.json
    """
    out_path = RESULTS_DIR / "ml_results.json"
    payload = {"meta": meta, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Ergebnisse gespeichert: {out_path}")


def main() -> None:
    X_train, y_train, X_test, y_test = load_features()

    meta: Dict[str, Any] = {
        "seed": SEED,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
        "features": list(X_train.columns),
        "note": "Modelle auf engineered URL-Features (src/features.py)",
    }

    results: List[Dict[str, Any]] = []

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
        n_jobs=-1,
    )
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_train, y_train)
    lgb_metrics = evaluate(lgb, X_test, y_test, "LightGBM")
    lgb_metrics["params"] = lgb_params
    results.append(lgb_metrics)

    save_results(results, meta)


if __name__ == "__main__":
    main()
