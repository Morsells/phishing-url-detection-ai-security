# Phishing URL Detection – Abschlussprojekt (AI Security)

Dieses Projekt implementiert und vergleicht mehrere Ansätze zur Erkennung von Phishing-URLs:

- **Klassische ML-Modelle** auf handgefertigten URL-Features (RandomForest, XGBoost, LightGBM)
- **Char n-gram Baseline** (TF-IDF 2–3 Grams + Logistic Regression)
- **DistilBERT** Fine-Tuning auf URL-Strings
- **Hybrid-Modell** (LightGBM auf engineered Features + BERT-derived Features)

## Hinweis zur Abgabegröße (Models-Ordner)

Der Ordner `models/` ist **absichtlich nicht** im ZIP enthalten (zu groß).  
Beim Ausführen des Projekts wird das BERT-Modell automatisch unter folgendem Pfad erzeugt:

`models/bert/run_YYYYMMDD_HHMMSS/`

Das Hybrid-Modell lädt automatisch den neuesten BERT-Run oder einen explizit angegebenen `--bert-dir`.

---

## Projektstruktur

```text
AbschlussProjekt/
  data/
    raw/                # Eingabedaten (GitHub CSV + openphish_feed.txt)
    processed/          # erzeugte Splits und Feature-Tabellen (auto-generiert)
  results/
    metrics/            # JSON-Metriken + Summary (CSV/MD)
  models/
    bert/
      run_*/            # gespeicherte BERT Runs (auto-generiert)
  src/
    data_loading.py
    features.py
    train_ml.py
    train_ngrams.py
    train_bert.py
    train_hybrid.py
    collect_results.py
    run_all.py
  notebooks/
    *.ipynb
  requirements.txt
  README.md

Voraussetzungen

Empfohlen: Python 3.11 (oder 3.12)
(Python 3.13 kann bei ML-Paketen auf Windows Installationsprobleme verursachen.)

Windows / Linux / macOS

Optional: GPU (CUDA) für schnelleres BERT-Training

Installation (Conda empfohlen)
1) Conda-Environment erstellen
conda create -n phishing-env python=3.11 -y
conda activate phishing-env

2) Abhängigkeiten installieren
pip install -r requirements.txt

Wichtig: Richtigen Python-Interpreter nutzen (Windows)

Unter Windows kann es vorkommen, dass python auf den System-Python (z.B. Python313) zeigt, obwohl ein Conda-Environment aktiviert wurde.

Check: Welcher Interpreter wird benutzt?
python -c "import sys; print(sys.executable)"

Erwartet (Beispiel):

...\miniconda3\envs\phishing-env\python.exe

Robuste Variante (immer korrekt): explizit env-Python verwenden
C:\Users\Moritz\miniconda3\envs\phishing-env\python.exe --version


Alle Kommandos unten funktionieren entsprechend auch so:

C:\Users\Moritz\miniconda3\envs\phishing-env\python.exe src/run_all.py

Daten

Lege folgende Dateien in data/raw/ ab:

phishing_site_urls_github.csv (GitHub Dataset; Spalten URL, Label mit good/bad)

openphish_feed.txt (OpenPhish Feed; pro Zeile eine URL)

Pipeline ausführen (End-to-End)

Die komplette Pipeline erzeugt Splits, Features, trainiert alle Modelle und sammelt Ergebnisse.

python src/run_all.py

Optionen

BERT überspringen (schneller):

python src/run_all.py --skip-bert


Nur klassische Modelle (ohne n-grams/BERT):

python src/run_all.py --only-classical


Falls BERT auf einem System fehlschlägt, aber die Pipeline weiterlaufen soll:

python src/run_all.py --allow-bert-fail

Einzelne Schritte (manuell)
1) Train/Test Split erzeugen
python src/data_loading.py

2) Features bauen (engineered)
python src/features.py

3) Klassische ML-Modelle
python src/train_ml.py

4) Char n-gram Baseline
python src/train_ngrams.py

5) DistilBERT Fine-Tuning
python src/train_bert.py


BERT-Modelle werden gespeichert unter:

models/bert/run_YYYYMMDD_HHMMSS/

6) Hybrid (LGBM + BERT Features)
python src/train_hybrid.py

7) Ergebnisse zusammenfassen
python src/collect_results.py


Outputs:

results/metrics/summary_results.csv

results/metrics/summary_results.md

Ergebnisse / Outputs

Metriken werden als JSON gespeichert unter:

results/metrics/ml_results.json

results/metrics/ngram_results.json

results/metrics/bert_results.json

results/metrics/hybrid_results.json

Die übersichtliche Summary ist in:

results/metrics/summary_results.md

Reproduzierbarkeit / Methodik

Train/Test-Leakage: Duplikate werden vor dem Split entfernt und URL-Overlap geprüft (sollte 0 sein).

Models im ZIP: models/ ist nicht enthalten; wird durch train_bert.py reproduzierbar erzeugt.

Sampling: BERT/Hybrid nutzen standardmäßig N_TRAIN=20000 (CPU-freundlich). Für bessere Ergebnisse:

N_TRAIN erhöhen (z.B. 50000)

und/oder EPOCHS erhöhen (z.B. 2–3)

Troubleshooting
ModuleNotFoundError: No module named 'tabulate'

Installiere tabulate in deinem Conda-Environment:

conda activate phishing-env
python -m pip install tabulate


oder per conda:

conda install -c conda-forge tabulate

Notebook lädt falschen BERT-Ordner

Wenn du im Notebook models/bert/ lädst, kommt ein Fehler, weil dort keine Gewichte liegen.
Nutze stattdessen einen Run-Ordner models/bert/run_*/ oder lese den Pfad aus:

results/metrics/bert_results.json (meta.bert_dir)

Autor

Moritz Bauer