from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline


# ============================================================
# Konfiguration
# ============================================================
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "urls_train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "urls_test.csv"
OUT_METRICS_PATH = BASE_DIR / "results" / "metrics" / "ngram_results.json"

# Char-n-gram Baseline
NGRAM_RANGE: Tuple[int, int] = (2, 3)
MIN_DF = 2

# Logistic Regression
MAX_ITER = 2000
SOLVER = "liblinear"
CLASS_WEIGHT = "balanced"


# ============================================================
# Hilfsfunktionen
# ============================================================
def set_global_seed(seed: int) -> None:
    """Setzt Seeds für reproduzierbare Läufe."""
    random.seed(seed)
    np.random.seed(seed)


def load_split_csv(path: Path) -> pd.DataFrame:
    """
    Lädt einen Split (train/test) aus CSV und validiert Spalten.
    Erwartete Spalten: url, label (0/1).
    """
    df = pd.read_csv(path)

    required_cols = {"url", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in {path}: {missing}. Erwartet: {required_cols}")

    df["url"] = df["url"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def ensure_parent_dir(file_path: Path) -> None:
    """Erstellt den Parent-Ordner für eine Datei, falls nötig."""
    file_path.parent.mkdir(parents=True, exist_ok=True)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    ROC-AUC schlägt fehl, wenn im Test nur eine Klasse vorkommt.
    In dem Fall geben wir None zurück.
    """
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    PR-AUC ist i.d.R. stabiler, aber auch hier: falls nur eine Klasse vorkommt,
    ist die Kennzahl nicht sinnvoll.
    """
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


# ============================================================
# Hauptlogik
# ============================================================
def main() -> None:
    set_global_seed(SEED)

    # Daten laden
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train-Datei nicht gefunden: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test-Datei nicht gefunden: {TEST_PATH}")

    train_df = load_split_csv(TRAIN_PATH)
    test_df = load_split_csv(TEST_PATH)

    X_train = train_df["url"].values
    y_train = train_df["label"].values
    X_test = test_df["url"].values
    y_test = test_df["label"].values

    # Modell: TF-IDF (Char-n-grams) + Logistic Regression
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        lowercase=True,
    )

    clf = LogisticRegression(
        max_iter=MAX_ITER,
        solver=SOLVER,
        class_weight=CLASS_WEIGHT,
        random_state=SEED,
    )

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    # Vorhersagen
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Metriken
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "roc_auc": safe_roc_auc(y_test, y_proba),
        "pr_auc": safe_pr_auc(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[tn, fp],[fn, tp]]
    }

    payload: Dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "positive_rate_train": float(np.mean(y_train)),
            "positive_rate_test": float(np.mean(y_test)),
            "vectorizer": {
                "type": "TfidfVectorizer",
                "analyzer": "char",
                "ngram_range": list(NGRAM_RANGE),
                "min_df": MIN_DF,
                "lowercase": True,
            },
            "classifier": {
                "type": "LogisticRegression",
                "solver": SOLVER,
                "class_weight": CLASS_WEIGHT,
                "max_iter": MAX_ITER,
                "random_state": SEED,
            },
            "beschreibung": "Char-TF-IDF (2-3) + Logistic Regression Baseline für URL-Phishing-Erkennung.",
        },
        "metrics": metrics,
    }

    ensure_parent_dir(OUT_METRICS_PATH)
    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Konsolen-Output
    print("===== Char n-gram Baseline (TF-IDF 2-3 grams + LogReg) =====")
    print(f"Saved metrics to: {OUT_METRICS_PATH}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"mcc:       {metrics['mcc']:.4f}")
    print(f"roc_auc:   {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'None'}")
    print(f"pr_auc:    {metrics['pr_auc'] if metrics['pr_auc'] is not None else 'None'}")
    print(f"confusion_matrix [[tn, fp],[fn,tp]]: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
