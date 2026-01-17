# Phishing-URL Detection (ML, Char-Ngrams, DistilBERT, Hybrid)

Dieses Projekt erkennt **Phishing-URLs** mittels:
1) **handgebauter URL-Features + klassische ML-Modelle** (RandomForest, XGBoost, LightGBM)
2) **Character n-gram TF-IDF Baseline** (Bi-/Trigrams + Logistic Regression)
3) **DistilBERT** (URL als Textsequenz)
4) **Hybrid-Modell** (LightGBM auf handgebauten Features + BERT-Score/Logit)

---

## Quickstart (One-Command)

### 1) Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

2) Pipeline laufen lassen
Alles (inkl. BERT + Hybrid):

python src/run_all.py
Schnell / CPU-freundlich (ohne BERT + Hybrid):

python src/run_all.py --skip-bert
Nur klassische Modelle (engineered features + ML):

python src/run_all.py --only-classical
Hybrid laufen lassen, aber BERT-Training überspringen (nur wenn models/bert/ bereits existiert):

python src/run_all.py --skip-bert --include-hybrid
Output
Nach dem Lauf findest du:

Einzelmetriken: results/metrics/*.json

Vergleichstabelle:

results/metrics/summary_results.csv

results/metrics/summary_results.md

Repository-Struktur
.
├── data/
│   ├── raw/                 # Rohdaten (GitHub CSV, OpenPhish Feed Snapshot)
│   └── processed/           # Splits + Feature-Tabellen (CSV)
├── models/
│   └── bert/                # Fine-tuned DistilBERT (Tokenizer + model.safetensors)
├── results/
│   └── metrics/             # JSON-Metriken + Summary (CSV/MD)
├── scripts/
│   └── download_datasets.py # optional: Download der Rohdaten
├── src/
│   ├── data_loading.py      # erstellt Train/Test-Splits (CSV)
│   ├── features.py          # baut handgebaute URL-Features (CSV)
│   ├── train_ml.py          # trainiert klassische ML-Modelle
│   ├── train_ngrams.py      # TF-IDF char bi/tri-gram baseline
│   ├── train_bert.py        # fine-tuned DistilBERT (optional, rechenintensiv)
│   ├── train_hybrid.py      # Hybrid: BERT-Features + LightGBM
│   ├── collect_results.py   # fasst alle Result-JSONs zu einer Tabelle zusammen
│   └── run_all.py           # One-Command Pipeline
└── notebooks/               # Exploration, Vergleich, Error Analysis
Daten
Für Reproduzierbarkeit enthält dieses ZIP bereits einen Snapshot der Daten unter data/raw/.
Du kannst daher alle Schritte ohne erneuten Download ausführen.

Optionaler Download für frische Daten:

python scripts/download_datasets.py
GitHub-Dataset: data/raw/phishing_site_urls_github.csv

OpenPhish Feed Snapshot: data/raw/openphish_feed.txt

Hinweis: OpenPhish ist ein Live-Feed; bei erneutem Download können sich Inhalte ändern → Ergebnisse können abweichen.

Manuelle Ausführung (Schritt für Schritt)
Wenn du nicht run_all.py nutzen willst:

Splits erzeugen:

python src/data_loading.py
Handgebaute Features:

python src/features.py
Klassische ML-Modelle:

python src/train_ml.py
Char-Ngram Baseline:

python src/train_ngrams.py
DistilBERT (optional):

python src/train_bert.py
Hybrid (optional, benötigt BERT-Modell unter models/bert/):

python src/train_hybrid.py
Ergebnis-Zusammenfassung:

python src/collect_results.py
Metriken
Die Scripts speichern konsistent folgende Metriken:

Accuracy

Precision

Recall

F1

MCC (Matthews Correlation Coefficient)

ROC-AUC

PR-AUC

Confusion Matrix ([[tn, fp],[fn, tp]])

Result-Dateien:

results/metrics/ml_results.json

results/metrics/ngram_results.json

results/metrics/bert_results.json

results/metrics/hybrid_results.json

Reproduzierbarkeit (Hinweise)
In den Trainingsscripts werden Seeds gesetzt (z. B. SEED=42) und im JSON unter meta dokumentiert.

Für identische Reproduktion verwende die im ZIP enthaltenen Dateien unter data/raw/ und die erzeugten Splits/Features unter data/processed/.

Bei erneutem Download der Rohdaten können sich Ergebnisse ändern (insbesondere OpenPhish).

Troubleshooting
Fehlende Pakete

ModuleNotFoundError: transformers → pip install transformers

ModuleNotFoundError: lightgbm → pip install lightgbm

ModuleNotFoundError: xgboost → pip install xgboost

BERT läuft nur auf CPU

Das ist in Ordnung; Standardkonfiguration nutzt max_len=64 und EPOCHS=1.

Hybrid schlägt fehl

Stelle sicher, dass models/bert/ existiert (durch python src/train_bert.py oder mitgeliefertes Modell).

Autor
Moritz Bauer