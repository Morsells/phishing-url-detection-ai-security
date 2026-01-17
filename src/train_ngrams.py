from pathlib import Path
import json
import random
from typing import Dict, Any, Optional

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


# -----------------------------
# Config
# -----------------------------
SEED = 42

TRAIN_PATH = "data/processed/urls_train.csv"
TEST_PATH = "data/processed/urls_test.csv"
OUT_METRICS_PATH = "results/metrics/ngram_results.json"

# Char n-gram baseline settings
NGRAM_RANGE = (2, 3)
MIN_DF = 2

# Logistic Regression settings
MAX_ITER = 2000
SOLVER = "liblinear"
CLASS_WEIGHT = "balanced"


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_split_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"url", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Expected at least {required_cols}")
    df["url"] = df["url"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    set_global_seed(SEED)

    # Load data
    train_df = load_split_csv(TRAIN_PATH)
    test_df = load_split_csv(TEST_PATH)

    X_train = train_df["url"].values
    y_train = train_df["label"].values
    X_test = test_df["url"].values
    y_test = test_df["label"].values

    # Model: Char n-gram TF-IDF + Logistic Regression
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

    # Predictions
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[tn, fp],[fn, tp]]
    }

    payload = {
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
            "note": "Character TF-IDF n-gram baseline for URL phishing detection.",
        },
        "metrics": metrics,
    }

    ensure_parent_dir(OUT_METRICS_PATH)
    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("===== Char n-gram Baseline (TF-IDF 2-3 grams + LogReg) =====")
    print(f"Saved metrics to: {OUT_METRICS_PATH}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"mcc:       {metrics['mcc']:.4f}")
    print(f"roc_auc:   {metrics['roc_auc']:.4f}")
    print(f"pr_auc:    {metrics['pr_auc']:.4f}")
    print(f"confusion_matrix [[tn, fp],[fn,tp]]: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
