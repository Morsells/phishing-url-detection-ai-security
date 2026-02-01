from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "metrics"

IN_FILES = {
    "ML": RESULTS_DIR / "ml_results.json",
    "NGRAM": RESULTS_DIR / "ngram_results.json",
    "BERT": RESULTS_DIR / "bert_results.json",
    "HYBRID": RESULTS_DIR / "hybrid_results.json",
}

OUT_CSV = RESULTS_DIR / "summary_results.csv"
OUT_MD = RESULTS_DIR / "summary_results.md"

METRIC_ORDER = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
]

EXTRA_COLS = [
    "model",
    "source_file",
    "n_train",
    "n_test",
    "pos_rate_train",
    "pos_rate_test",
    "device",
    "bert_dir",
]


def _read_json(path: Path) -> Any:
    """Liest eine JSON-Datei sicher ein."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    """Konvertiert Werte robust zu float (oder None)."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _get_meta(payload: Any) -> Dict[str, Any]:
    """Extrahiert meta-Daten, falls vorhanden."""
    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
        return payload["meta"]
    return {}


def _extract_common_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vereinheitlicht häufig genutzte Meta-Felder über alle Skripte hinweg.
    Unterstützt verschiedene Schlüsselvarianten aus den Ergebnissen.
    """
    return {
        "n_train": meta.get("n_train_sampled") or meta.get("n_train") or meta.get("train_size"),
        "n_test": meta.get("n_test_sampled") or meta.get("n_test") or meta.get("test_size"),
        "pos_rate_train": meta.get("positive_rate_train") or meta.get("pos_rate_train"),
        "pos_rate_test": meta.get("positive_rate_test") or meta.get("pos_rate_test"),
        "device": meta.get("device"),
        "bert_dir": meta.get("bert_dir") or meta.get("bert_model_dir"),
    }


def _normalize_record(
    model_name: str,
    metrics: Dict[str, Any],
    source_file: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalisiert ein Ergebnis in eine einheitliche Tabellenzeile:
    - Basisfelder (model, source_file)
    - Meta-Felder (optional)
    - Metriken (float oder None)
    """
    row: Dict[str, Any] = {
        "model": model_name,
        "source_file": source_file,
        "n_train": None,
        "n_test": None,
        "pos_rate_train": None,
        "pos_rate_test": None,
        "device": None,
        "bert_dir": None,
    }

    if meta:
        row.update(_extract_common_meta(meta))

    for m in METRIC_ORDER:
        row[m] = _safe_float(metrics.get(m))

    return row


def _extract_ml(payload: Any, source_file: str) -> List[Dict[str, Any]]:
    """
    Extrahiert Metriken für klassische ML-Modelle.

    Unterstützte Formate:
    - Format A (alt): [ { "model": "...", "accuracy": ..., ... }, ... ]
    - Format B (neu): { "meta": {...}, "results": [ { "model": "...", "accuracy": ..., ... }, ... ] }
    """
    rows: List[Dict[str, Any]] = []

    # Format A: Liste von Ergebnissen
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model", "ML_Model"))
            rows.append(_normalize_record(name, item, source_file, meta=None))
        return rows

    # Format B: Dict mit meta + results
    if isinstance(payload, dict):
        meta = _get_meta(payload)
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("model", "ML_Model"))
                rows.append(_normalize_record(name, item, source_file, meta=meta))
            return rows

    return rows


def _extract_single(payload: Any, default_name: str, source_file: str) -> List[Dict[str, Any]]:
    """
    Extrahiert Metriken für EIN einzelnes Modell (z.B. NGRAM, BERT, HYBRID).

    Unterstützte Formate:
    - Format A (empfohlen):
      { "meta": {...}, "metrics": { "accuracy": ..., "f1": ..., ... } }

    - Format B (Fallback / Legacy):
      { "accuracy": ..., "f1": ..., ... }
      (Meta kann optional vorhanden sein oder fehlen.)
    """
    if not isinstance(payload, dict):
        return []

    meta = _get_meta(payload)

    # Standardformat: Metriken unter "metrics"
    metrics_obj = payload.get("metrics")
    if isinstance(metrics_obj, dict):
        name = str(meta.get("model_name") or default_name)
        return [_normalize_record(name, metrics_obj, source_file, meta=meta)]

    # Fallback: payload selbst enthält die Metriken
    return [_normalize_record(default_name, payload, source_file, meta=meta)]


def _to_markdown_fallback(df: pd.DataFrame) -> str:
    """
    Erzeugt Markdown. Falls 'tabulate' fehlt, wird als CSV-Text zurückgefallen.
    """
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return df.to_csv(index=False)
    except Exception:
        # Letzter Fallback, falls irgendwas schiefgeht
        return df.to_csv(index=False)


def main() -> None:
    rows: List[Dict[str, Any]] = []

    for key, path in IN_FILES.items():
        if not path.exists():
            print(f"[!] Datei fehlt: {path} (überspringe)")
            continue

        payload = _read_json(path)
        source_file = path.name

        if key == "ML":
            rows.extend(_extract_ml(payload, source_file))
        elif key == "NGRAM":
            rows.extend(_extract_single(payload, default_name="CharTFIDF(2-3)+LogReg", source_file=source_file))
        elif key == "BERT":
            rows.extend(_extract_single(payload, default_name="DistilBERT", source_file=source_file))
        elif key == "HYBRID":
            rows.extend(_extract_single(payload, default_name="Hybrid(LGBM+BERT)", source_file=source_file))

    if not rows:
        raise RuntimeError("Keine Ergebnisdateien gefunden oder keine Metriken konnten geparst werden.")

    df = pd.DataFrame(rows)

    # Spaltenreihenfolge (nur vorhandene Spalten nehmen, um KeyError zu vermeiden)
    ordered_cols = EXTRA_COLS + METRIC_ORDER
    df = df[[c for c in ordered_cols if c in df.columns]]

    # Sortierung nach F1 absteigend
    if "f1" in df.columns:
        df = df.sort_values(by="f1", ascending=False, na_position="last").reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_CSV, index=False)

    md = _to_markdown_fallback(df)
    OUT_MD.write_text(md, encoding="utf-8")

    print(f"[+] Gespeichert: {OUT_CSV}")
    print(f"[+] Gespeichert: {OUT_MD}")
    print("\n=== Summary (sortiert nach F1 absteigend) ===")
    print(md)


if __name__ == "__main__":
    main()
