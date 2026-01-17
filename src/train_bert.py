from pathlib import Path
import json
import random
from typing import Dict, Any, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
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
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
SEED = 42
MAX_LEN = 64
N_TRAIN = 20000          # adjust for experiments, e.g. 50000
TRAIN_BS = 16
TEST_BS = 64
LR = 2e-5
EPOCHS = 1

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

# Save into a fresh run directory to avoid Windows file locking issues
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
BERT_DIR = BASE_DIR / "models" / "bert" / f"run_{run_id}"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism (best-effort; may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class URLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 64):
        self.urls = df["url"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]

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
            "label": torch.tensor(label, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, dataloader, device: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model.eval()

    preds: List[int] = []
    probs: List[float] = []
    labels: List[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        out = model(input_ids, attention_mask=mask)
        logits = out.logits
        prob_1 = torch.softmax(logits, dim=1)[:, 1]

        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        probs.extend(prob_1.cpu().tolist())
        labels.extend(y.cpu().tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    roc = roc_auc_score(labels, probs)
    pr = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds).tolist()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }

    extras = {
        "confusion_matrix": cm,  # [[tn, fp],[fn, tp]]
    }

    return metrics, extras


def main():
    set_global_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")  # fp16 only on CUDA
    print(f"[+] Using device: {device}")

    train_path = PROCESSED_DIR / "urls_train.csv"
    test_path = PROCESSED_DIR / "urls_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed split files. Run `python src/data_loading.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Train subset for faster runs
    if len(train_df) > N_TRAIN:
        train_df = train_df.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

    print(f"[+] Train-Samples für BERT: {len(train_df)}")
    print(f"[+] Test-Samples:           {len(test_df)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(device)

    train_ds = URLDataset(train_df, tokenizer, max_len=MAX_LEN)
    test_ds = URLDataset(test_df, tokenizer, max_len=MAX_LEN)

    # Reproducible shuffling
    g = torch.Generator()
    g.manual_seed(SEED)

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True, generator=g)
    test_dl = DataLoader(test_ds, batch_size=TEST_BS, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)

    # Use the non-deprecated AMP API (only enabled on CUDA)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        loop = tqdm(train_dl, total=len(train_dl))

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                out = model(input_ids, attention_mask=mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"Loss: {loss.item():.4f}")

    metrics, extras = evaluate(model, test_dl, device)

    print("\n===== BERT Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"confusion_matrix [[tn, fp],[fn,tp]]: {extras['confusion_matrix']}")

    # Save model + tokenizer
    # Workaround for Windows safetensors I/O errors: save as PyTorch .bin instead of .safetensors
    model.save_pretrained(BERT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(BERT_DIR)

    payload = {
        "meta": {
            "seed": SEED,
            "device": device,
            "fp16": bool(use_fp16),
            "max_len": MAX_LEN,
            "n_train_sampled": int(len(train_df)),
            "n_test": int(len(test_df)),
            "positive_rate_train": float(train_df["label"].mean()),
            "positive_rate_test": float(test_df["label"].mean()),
            "train_batch_size": TRAIN_BS,
            "test_batch_size": TEST_BS,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "base_model": "distilbert-base-uncased",
            "bert_dir": str(BERT_DIR),
            "safe_serialization": False,
            "note": "Fine-tuned DistilBERT on URL strings for phishing classification",
        },
        "metrics": {
            **metrics,
            **extras,
        },
    }

    out_path = RESULTS_DIR / "bert_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[+] Saved BERT model to {BERT_DIR}")
    print(f"[+] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
