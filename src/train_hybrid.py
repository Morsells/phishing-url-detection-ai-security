from pathlib import Path
import json
import random
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from lightgbm import LGBMClassifier
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


# -----------------------------
# Config
# -----------------------------
SEED = 42
MAX_LEN = 64
BERT_BATCH_SIZE = 64

# CPU-friendly sampling sizes
N_TRAIN = 20000
N_TEST = 50000

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

# Root directory that contains run_* subfolders
BERT_ROOT_DIR = BASE_DIR / "models" / "bert"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def has_hf_weights(model_dir: Path) -> bool:
    """Return True if the directory contains a HF-compatible model weight file."""
    return any(
        (model_dir / fname).exists()
        for fname in ["pytorch_model.bin", "model.safetensors"]
    )


def find_latest_bert_run_dir(bert_root: Path) -> Path:
    """
    Select the newest models/bert/run_* directory that actually contains weights.
    Only use bert_root itself if it contains weights.
    """
    if not bert_root.exists():
        raise FileNotFoundError(f"Missing BERT directory: {bert_root}")

    # If root itself is a valid model directory (has weights), use it.
    if has_hf_weights(bert_root):
        return bert_root

    # Otherwise, look for run_* subdirs that contain weights
    run_dirs = [p for p in bert_root.glob("run_*") if p.is_dir() and has_hf_weights(p)]
    if not run_dirs:
        # Provide helpful debug info
        existing_runs = [p.name for p in bert_root.glob("run_*") if p.is_dir()]
        raise FileNotFoundError(
            f"No valid run_* model directories with weights found under {bert_root}.\n"
            f"Found run dirs: {existing_runs}\n"
            "Run `python src/train_bert.py` first (it should create models/bert/run_*/pytorch_model.bin)."
        )

    # run_YYYYMMDD_HHMMSS sorts lexicographically == chronologically
    run_dirs_sorted = sorted(run_dirs, key=lambda p: p.name)
    return run_dirs_sorted[-1]


# -------- BERT Dataset nur für Inferenz --------
class URLInferDataset(Dataset):
    def __init__(self, urls: List[str], tokenizer, max_len: int = 64):
        self.urls = [str(u) for u in urls]
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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def get_bert_features(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_len: int = 64,
) -> pd.DataFrame:
    dataset = URLInferDataset(df["url"].astype(str).tolist(), tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_logits: List[float] = []
    all_probas: List[float] = []

    use_fp16 = (device == "cuda")
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        # Non-deprecated AMP API (only enabled on CUDA)
        with torch.amp.autocast("cuda", enabled=use_fp16):
            out = model(input_ids, attention_mask=mask)
            logits = out.logits  # (B, 2)
            prob_1 = torch.softmax(logits, dim=1)[:, 1]

        all_logits.extend(logits[:, 1].float().cpu().tolist())
        all_probas.extend(prob_1.float().cpu().tolist())

    return pd.DataFrame({"bert_logit": all_logits, "bert_proba": all_probas})


def evaluate(model, X_test, y_test) -> Dict[str, Any]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[tn, fp],[fn,tp]]
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid model (LightGBM + BERT features).")
    parser.add_argument(
        "--bert-dir",
        type=str,
        default=None,
        help="Path to a saved BERT model directory. If omitted, the latest models/bert/run_* with weights is used.",
    )
    args = parser.parse_args()

    set_global_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # -------- 1) Features laden --------
    train_feat_path = PROCESSED_DIR / "urls_train_features.csv"
    test_feat_path = PROCESSED_DIR / "urls_test_features.csv"

    if not train_feat_path.exists() or not test_feat_path.exists():
        raise FileNotFoundError("Missing feature tables. Run `python src/features.py` first.")

    train_feat = pd.read_csv(train_feat_path)
    test_feat = pd.read_csv(test_feat_path)

    required_cols = {"url", "label"}
    if not required_cols.issubset(train_feat.columns) or not required_cols.issubset(test_feat.columns):
        raise ValueError("Feature CSVs must contain at least 'url' and 'label' columns.")

    # CPU-friendly sampling
    if len(train_feat) > N_TRAIN:
        train_feat = train_feat.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    if len(test_feat) > N_TEST:
        test_feat = test_feat.sample(n=N_TEST, random_state=SEED).reset_index(drop=True)

    print(f"[+] Hybrid Train-Size: {len(train_feat)}")
    print(f"[+] Hybrid Test-Size : {len(test_feat)}")

    # -------- 2) BERT + Tokenizer laden --------
    if args.bert_dir is not None:
        bert_dir = Path(args.bert_dir)
        if not bert_dir.exists():
            raise FileNotFoundError(f"--bert-dir not found: {bert_dir}")
        if not has_hf_weights(bert_dir):
            raise FileNotFoundError(
                f"--bert-dir does not contain weights (pytorch_model.bin/model.safetensors): {bert_dir}"
            )
    else:
        bert_dir = find_latest_bert_run_dir(BERT_ROOT_DIR)

    print(f"[+] Using BERT model dir: {bert_dir}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
    bert_model = DistilBertForSequenceClassification.from_pretrained(bert_dir)
    bert_model.to(device)

    # -------- 3) BERT-Features berechnen --------
    print("[+] Berechne BERT-Features für Train...")
    bert_train = get_bert_features(
        train_feat, tokenizer, bert_model, device, batch_size=BERT_BATCH_SIZE, max_len=MAX_LEN
    )
    print("[+] Berechne BERT-Features für Test...")
    bert_test = get_bert_features(
        test_feat, tokenizer, bert_model, device, batch_size=BERT_BATCH_SIZE, max_len=MAX_LEN
    )

    train_hybrid = pd.concat([train_feat.reset_index(drop=True), bert_train], axis=1)
    test_hybrid = pd.concat([test_feat.reset_index(drop=True), bert_test], axis=1)

    # -------- 4) LightGBM auf kombinierten Features --------
    drop_cols = ["url", "label"]
    feature_cols = [c for c in train_hybrid.columns if c not in drop_cols]

    X_train = train_hybrid[feature_cols]
    y_train = train_hybrid["label"].astype(int)

    X_test = test_hybrid[feature_cols]
    y_test = test_hybrid["label"].astype(int)

    print(f"[+] Anzahl Features (inkl. BERT): {X_train.shape[1]}")

    lgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        random_state=SEED,
    )
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_train, y_train)

    metrics = evaluate(lgb, X_test, y_test)

    print("\n===== Hybrid (LightGBM + BERT) =====")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"mcc:       {metrics['mcc']:.4f}")
    print(f"roc_auc:   {metrics['roc_auc']:.4f}")
    print(f"pr_auc:    {metrics['pr_auc']:.4f}")
    print(f"confusion_matrix [[tn, fp],[fn,tp]]: {metrics['confusion_matrix']}")

    payload = {
        "meta": {
            "seed": SEED,
            "device": device,
            "max_len": MAX_LEN,
            "bert_batch_size": BERT_BATCH_SIZE,
            "n_train_sampled": int(len(train_feat)),
            "n_test_sampled": int(len(test_feat)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "bert_model_dir": str(bert_dir),
            "lgbm_params": lgb_params,
            "feature_count": int(X_train.shape[1]),
            "feature_cols": feature_cols,
            "note": "Hybrid model: LightGBM on engineered URL features + BERT-derived features (logit/proba).",
        },
        "metrics": metrics,
    }

    out_path = RESULTS_DIR / "hybrid_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[+] Hybrid-Metriken gespeichert in: {out_path}")


if __name__ == "__main__":
    main()
