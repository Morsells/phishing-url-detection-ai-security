from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Any, Dict, List, Tuple
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
# Konfiguration
# -----------------------------
SEED = 42
MAX_LEN = 64

# CPU-freundliche Defaults (für Experimente z.B. N_TRAIN=50000 / EPOCHS=2)
N_TRAIN = 20000
TRAIN_BS = 16
TEST_BS = 64
LR = 2e-5
EPOCHS = 1

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

# Jede Ausführung bekommt einen eigenen Run-Ordner (vermeidet Konflikte/Locks unter Windows)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
BERT_DIR = BASE_DIR / "models" / "bert" / f"run_{run_id}"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Setzt Seeds für reproduzierbare Läufe (best effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinismus (kann minimal langsamer sein, aber gut für Reproduzierbarkeit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class URLDataset(Dataset):
    """Torch-Dataset: URL-String -> Tokenizer -> Tensoren + Label."""

    def __init__(self, df: pd.DataFrame, tokenizer: DistilBertTokenizerFast, max_len: int = 64) -> None:
        self.urls: List[str] = df["url"].astype(str).tolist()
        self.labels: List[int] = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def _safe_auc_scores(y_true: List[int], y_score: List[float]) -> Tuple[float | None, float | None]:
    """
    ROC-AUC/PR-AUC können fehlschlagen, wenn nur eine Klasse vorkommt.
    Dann speichern wir None statt Crash.
    """
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = None

    try:
        pr = float(average_precision_score(y_true, y_score))
    except Exception:
        pr = None

    return roc, pr


def evaluate(model: DistilBertForSequenceClassification, dataloader: DataLoader, device: str) -> Tuple[Dict[str, float | None], Dict[str, Any]]:
    """Evaluation auf dem Testset: Metriken + Confusion Matrix."""
    model.eval()

    preds: List[int] = []
    probs: List[float] = []
    labels: List[int] = []

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            out = model(input_ids=input_ids, attention_mask=mask)
            logits = out.logits
            prob_1 = torch.softmax(logits, dim=1)[:, 1]

            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            probs.extend(prob_1.cpu().tolist())
            labels.extend(y.cpu().tolist())

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    mcc = float(matthews_corrcoef(labels, preds))
    roc, pr = _safe_auc_scores(labels, probs)
    cm = confusion_matrix(labels, preds).tolist()

    metrics: Dict[str, float | None] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc,
        "pr_auc": pr,
    }

    extras: Dict[str, Any] = {
        "confusion_matrix": cm,  # [[tn, fp],[fn, tp]]
    }

    return metrics, extras


def main() -> None:
    set_global_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")  # fp16 nur auf CUDA sinnvoll
    print(f"[+] Using device: {device}")

    train_path = PROCESSED_DIR / "urls_train.csv"
    test_path = PROCESSED_DIR / "urls_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Fehlende processed Splits. Bitte zuerst ausführen: python src/data_loading.py"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Teilmenge fürs Training (schneller / CPU-freundlich)
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

    # Reproduzierbares Shuffling
    g = torch.Generator()
    g.manual_seed(SEED)

    # Windows-freundliche Loader-Defaults: num_workers=0
    train_dl = DataLoader(
        train_ds,
        batch_size=TRAIN_BS,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=TEST_BS,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    # AMP/GradScaler: aktiv nur bei CUDA
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        loop = tqdm(train_dl, total=len(train_dl))

        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                out = model(input_ids=input_ids, attention_mask=mask, labels=y)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"Loss: {loss.item():.4f}")

    metrics, extras = evaluate(model, test_dl, device)

    print("\n===== BERT Metrics =====")
    for k, v in metrics.items():
        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")
    print(f"confusion_matrix [[tn, fp],[fn,tp]]: {extras['confusion_matrix']}")

    # Modell + Tokenizer speichern
    # Hinweis (Windows): safe_serialization=False speichert als PyTorch .bin statt .safetensors
    model.save_pretrained(BERT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(BERT_DIR)

    payload: Dict[str, Any] = {
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
            "note": "DistilBERT Fine-Tuning auf URL-Strings (Phishing vs. Benign).",
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
