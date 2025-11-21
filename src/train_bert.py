from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
BERT_DIR = BASE_DIR / "models" / "bert"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)


class URLDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        # max_len 64 statt 128 -> schneller
        self.urls = df["url"].astype(str).tolist()
        self.labels = df["label"].tolist()
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
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()
    preds, probs, labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            out = model(input_ids, attention_mask=mask)
            logits = out.logits
            prob = torch.softmax(logits, dim=1)[:, 1]

            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            probs.extend(prob.cpu().tolist())
            labels.extend(label.cpu().tolist())

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    train_df = pd.read_csv(PROCESSED_DIR / "urls_train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "urls_test.csv")

    # *** WICHTIG: Train-Set verkleinern ***
    #N_TRAIN = 20000 #CPU
    N_TRAIN = 100000 #GPU
   
    if len(train_df) > N_TRAIN:
        train_df = train_df.sample(n=N_TRAIN, random_state=42).reset_index(drop=True)
    print(f"[+] Train-Samples für BERT: {len(train_df)}")
    print(f"[+] Test-Samples (voll):      {len(test_df)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(device)

    train_ds = URLDataset(train_df, tokenizer, max_len=64)
    test_ds = URLDataset(test_df, tokenizer, max_len=64)

    #train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True) # GPU
    test_dl = DataLoader(test_ds, batch_size=64)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    #EPOCHS = 1  # auf CPU erstmal nur 1 Epoche
    EPOCHS = 2 # auf GPU 2 Epochen
    model.train()

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        loop = tqdm(train_dl, total=len(train_dl))

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids, attention_mask=mask, labels=labels)
            loss = out.loss

            loss.backward()
            optimizer.step()

            loop.set_description(f"Loss: {loss.item():.4f}")

    metrics = evaluate(model, test_dl, device)

    print("\n===== BERT Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    model.save_pretrained(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)

    out_path = RESULTS_DIR / "bert_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[+] Saved BERT model to {BERT_DIR}")
    print(f"[+] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
