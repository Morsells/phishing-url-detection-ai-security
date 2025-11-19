from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features():
    train_path = PROCESSED_DIR / "urls_train_features.csv"
    test_path = PROCESSED_DIR / "urls_test_features.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # URL entfernen → nicht als Feature
    X_train = train_df.drop(columns=["url", "label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["url", "label"])
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test


def evaluate(model, X_test, y_test, name: str):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }

    print(f"\n===== {name} =====")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

    return metrics


def save_results(results: list):
    out_path = RESULTS_DIR / "ml_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[+] Ergebnisse gespeichert: {out_path}")


def main():
    X_train, y_train, X_test, y_test = load_features()

    # Optional: Scaling (vor allem für Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1) Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate(rf, X_test, y_test, "RandomForest"))

    # 2) XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate(xgb, X_test, y_test, "XGBoost"))

    # 3) LightGBM
    lgb = LGBMClassifier(
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42
    )
    lgb.fit(X_train, y_train)
    results.append(evaluate(lgb, X_test, y_test, "LightGBM"))

    save_results(results)


if __name__ == "__main__":
    main()
