from pathlib import Path
import json
import random
import argparse
from typing import Dict, Any, List, Optional

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
# Konfiguration
# -----------------------------
SEED = 42
MAX_LEN = 64
BERT_BATCH_SIZE = 64

# CPU-freundliche Sample-Größen
N_TRAIN = 20000
N_TEST = 50000

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

# Root-Verzeichnis, das run_* Unterordner enthält
BERT_ROOT_DIR = BASE_DIR / "models" / "bert"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Setzt Seeds für Reproduzierbarkeit (so gut es in PyTorch/Windows möglich ist)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinismus: kann minimal langsamer sein, aber reproduzierbarer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def has_hf_weights(model_dir: Path) -> bool:
    """Gibt True zurück, wenn ein Ordner HF-kompatible Gewichte enthält."""
    return any((model_dir / fname).exists() for fname in ["pytorch_model.bin", "model.safetensors"])


def find_latest_bert_run_dir(bert_root: Path) -> Path:
    """
    Wählt das neueste models/bert/run_*-Verzeichnis aus, das tatsächlich Gewichte enthält.
    bert_root wird nur dann direkt verwendet, wenn dort ebenfalls Gewichte liegen.
    """
    if not bert_root.exists():
        raise FileNotFoundError(f"Fehlendes BERT-Verzeichnis: {bert_root}")

    # Falls das Root-Verzeichnis selbst ein gültiges Modellverzeichnis ist, nutze es direkt
    if has_hf_weights(bert_root):
        return bert_root

    # Sonst: run_* Unterordner suchen, die Gewichte enthalten
    run_dirs = [p for p in bert_root.glob("run_*") if p.is_dir() and has_hf_weights(p)]
    if not run_dirs:
        existing_runs = [p.name for p in bert_root.glob("run_*") if p.is_dir()]
        raise FileNotFoundError(
            f"Keine gültigen run_* Modellordner (mit Gewichten) unter {bert_root} gefunden.\n"
            f"Gefundene run_* Ordner: {existing_runs}\n"
            "Bitte zuerst `python src/train_bert.py` ausführen "
            "(sollte models/bert/run_*/pytorch_model.bin erzeugen)."
        )

    # run_YYYYMMDD_HHMMSS sortiert lexikografisch == chronologisch
    run_dirs_sorted = sorted(run_dirs, key=lambda p: p.name)
    return run_dirs_sorted[-1]


# -------- BERT Dataset nur für Inferenz --------
class URLInferDataset(Dataset):
    def __init__(self, urls: List[str], tokenizer: DistilBertTokenizerFast, max_len: int = 64) -> None:
        self.urls = [str(u) for u in urls]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
    tokenizer: DistilBertTokenizerFast,
    model: DistilBertForSequenceClassification,
    device: str,
    batch_size: int = 64,
    max_len: int = 64,
) -> pd.DataFrame:
    """Berechnet BERT-Logit und BERT-Wahrscheinlichkeit (Klasse 1) für jede URL."""
    dataset = URLInferDataset(df["url"].astype(str).tolist(), tokenizer, max_len=max_len)

    # Windows: num_workers=0 ist meist am stabilsten
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_logits: List[float] = []
    all_probas: List[float] = []

    use_fp16 = (device == "cuda")

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        # AMP nur auf CUDA aktivieren
        with torch.amp.autocast("cuda", enabled=use_fp16):
            out = model(input_ids, attention_mask=mask)
            logits = out.logits  # (B, 2)
            prob_1 = torch.softmax(logits, dim=1)[:, 1]

        all_logits.extend(logits[:, 1].float().cpu().tolist())
        all_probas.extend(prob_1.float().cpu().tolist())

    return pd.DataFrame({"bert_logit": all_logits, "bert_proba": all_probas})


def evaluate(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Berechnet Metriken; ROC/PR-AUC sind robust gegen Edge-Cases abgesichert."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Robustheit: AUC kann crashen, falls im y_test nur eine Klasse vorkommt
    try:
        roc_auc = float(roc_auc_score(y_test, proba))
    except Exception:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_test, proba))
    except Exception:
        pr_auc = None

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[tn, fp],[fn,tp]]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trainiert das Hybrid-Modell (LightGBM + BERT-abgeleitete Features)."
    )
    parser.add_argument(
        "--bert-dir",
        type=str,
        default=None,
        help=(
            "Pfad zu einem gespeicherten BERT-Modellordner. "
            "Wenn nicht gesetzt, wird automatisch der neueste models/bert/run_* Ordner mit Gewichten verwendet."
        ),
    )
    args = parser.parse_args()

    set_global_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Verwende Device: {device}")

    # -------- 1) Engineered Features laden --------
    train_feat_path = PROCESSED_DIR / "urls_train_features.csv"
    test_feat_path = PROCESSED_DIR / "urls_test_features.csv"

    if not train_feat_path.exists() or not test_feat_path.exists():
        raise FileNotFoundError("Feature-Tabellen fehlen. Bitte zuerst `python src/features.py` ausführen.")

    train_feat = pd.read_csv(train_feat_path)
    test_feat = pd.read_csv(test_feat_path)

    required_cols = {"url", "label"}
    if not required_cols.issubset(train_feat.columns) or not required_cols.issubset(test_feat.columns):
        raise ValueError("Feature-CSV muss mindestens die Spalten 'url' und 'label' enthalten.")

    # CPU-freundliches Sampling
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
            raise FileNotFoundError(f"--bert-dir nicht gefunden: {bert_dir}")
        if not has_hf_weights(bert_dir):
            raise FileNotFoundError(
                f"--bert-dir enthält keine Gewichte (pytorch_model.bin/model.safetensors): {bert_dir}"
            )
    else:
        bert_dir = find_latest_bert_run_dir(BERT_ROOT_DIR)

    print(f"[+] Verwende BERT-Modelldir: {bert_dir}")

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
    print(f"roc_auc:   {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'}")
    print(f"pr_auc:    {metrics['pr_auc'] if metrics['pr_auc'] is not None else 'n/a'}")
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
            # kompatibel zu collect_results.py (unterstützt bert_dir ODER bert_model_dir)
            "bert_dir": str(bert_dir),
            "bert_model_dir": str(bert_dir),
            "lgbm_params": lgb_params,
            "feature_count": int(X_train.shape[1]),
            # NICHT die komplette Liste speichern → JSON bleibt klein und “abgabe-tauglich”
            "feature_cols_head": feature_cols[:30],
            "note": "Hybrid: LightGBM auf engineered URL-Features + BERT-Signale (Logit/Probability).",
        },
        "metrics": metrics,
    }

    out_path = RESULTS_DIR / "hybrid_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[+] Hybrid-Metriken gespeichert in: {out_path}")


if __name__ == "__main__":
    main()
