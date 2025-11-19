from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
BERT_DIR = BASE_DIR / "models" / "bert"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------- BERT Dataset nur für Inferenz --------

class URLInferDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.urls = df["url"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        enc = self.tokenizer(
            url,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }


def get_bert_features(df: pd.DataFrame, tokenizer, model, device, batch_size=64, max_len=64):
    dataset = URLInferDataset(df, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_logits = []
    all_probas = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            out = model(input_ids, attention_mask=mask)
            logits = out.logits  # (B, 2)
            prob = torch.softmax(logits, dim=1)[:, 1]

            all_logits.extend(logits[:, 1].cpu().tolist())
            all_probas.extend(prob.cpu().tolist())

    return pd.DataFrame(
        {
            "bert_logit": all_logits,
            "bert_proba": all_probas,
        }
    )


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # -------- 1) Features laden --------
    train_feat_path = PROCESSED_DIR / "urls_train_features.csv"
    test_feat_path = PROCESSED_DIR / "urls_test_features.csv"

    train_feat = pd.read_csv(train_feat_path)
    test_feat = pd.read_csv(test_feat_path)

    # für Hybrid-Experiment: Größe begrenzen (CPU-freundlich)
    N_TRAIN = 20000
    N_TEST = 50000

    if len(train_feat) > N_TRAIN:
        train_feat = train_feat.sample(n=N_TRAIN, random_state=42).reset_index(drop=True)
    if len(test_feat) > N_TEST:
        test_feat = test_feat.sample(n=N_TEST, random_state=42).reset_index(drop=True)

    print(f"[+] Hybrid Train-Size: {len(train_feat)}")
    print(f"[+] Hybrid Test-Size : {len(test_feat)}")

    # -------- 2) BERT + Tokenizer laden --------
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR)
    model.to(device)

    # -------- 3) BERT-Features berechnen --------
    print("[+] Berechne BERT-Features für Train...")
    bert_train = get_bert_features(train_feat, tokenizer, model, device)
    print("[+] Berechne BERT-Features für Test...")
    bert_test = get_bert_features(test_feat, tokenizer, model, device)

    # an DataFrames anhängen (Index passt, weil gleiche Reihenfolge)
    train_hybrid = pd.concat([train_feat.reset_index(drop=True), bert_train], axis=1)
    test_hybrid = pd.concat([test_feat.reset_index(drop=True), bert_test], axis=1)

    # -------- 4) LightGBM auf kombinierten Features --------
    # Spalten vorbereiten
    drop_cols = ["url", "label"]
    feature_cols = [c for c in train_hybrid.columns if c not in drop_cols]

    X_train = train_hybrid[feature_cols]
    y_train = train_hybrid["label"]

    X_test = test_hybrid[feature_cols]
    y_test = test_hybrid["label"]

    print(f"[+] Anzahl Features (inkl. BERT): {X_train.shape[1]}")

    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
    )
    lgb.fit(X_train, y_train)

    metrics = evaluate(lgb, X_test, y_test)

    print("\n===== Hybrid (LightGBM + BERT) =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_path = RESULTS_DIR / "hybrid_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[+] Hybrid-Metriken gespeichert in: {out_path}")


if __name__ == "__main__":
    main()
