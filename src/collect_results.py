from pathlib import Path
import json
from typing import Any, Dict, List

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _get_meta(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        meta = payload.get("meta")
        if isinstance(meta, dict):
            return meta
    return {}


def _extract_common_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes commonly used meta fields across scripts.
    """
    return {
        "n_train": meta.get("n_train_sampled") or meta.get("n_train") or meta.get("train_size"),
        "n_test": meta.get("n_test_sampled") or meta.get("n_test") or meta.get("test_size"),
        "pos_rate_train": meta.get("positive_rate_train"),
        "pos_rate_test": meta.get("positive_rate_test"),
        "device": meta.get("device"),
        "bert_dir": meta.get("bert_dir") or meta.get("bert_model_dir"),
    }


def _normalize_record(
    model_name: str,
    metrics: Dict[str, Any],
    source_file: str,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
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
        cm = _extract_common_meta(meta)
        row.update(cm)

    # metrics
    for m in METRIC_ORDER:
        row[m] = _safe_float(metrics.get(m, None))

    return row


def _extract_ml(payload: Any, source_file: str) -> List[Dict[str, Any]]:
    """
    Handles:
      - old format: [ {model, accuracy, ...}, ... ]
      - new format: { meta: {...}, results: [ {model, accuracy,..., params}, ... ] }
    """
    rows: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        for item in payload:
            name = str(item.get("model", "ML_Model"))
            rows.append(_normalize_record(name, item, source_file, meta=None))
        return rows

    if isinstance(payload, dict):
        meta = _get_meta(payload)
        if "results" in payload and isinstance(payload["results"], list):
            for item in payload["results"]:
                name = str(item.get("model", "ML_Model"))
                rows.append(_normalize_record(name, item, source_file, meta=meta))
            return rows

    return rows


def _extract_single(payload: Any, default_name: str, source_file: str) -> List[Dict[str, Any]]:
    """
    Handles:
      - {meta: {...}, metrics: {...}}
      - {accuracy: ..., f1: ...}  (fallback)
    """
    if not isinstance(payload, dict):
        return []

    meta = _get_meta(payload)

    if "metrics" in payload and isinstance(payload["metrics"], dict):
        metrics = payload["metrics"]
        name = str(meta.get("model_name", default_name)) if isinstance(meta, dict) else default_name
        return [_normalize_record(name, metrics, source_file, meta=meta)]

    # fallback: assume dict itself are metrics
    return [_normalize_record(default_name, payload, source_file, meta=meta)]


def main() -> None:
    rows: List[Dict[str, Any]] = []

    for key, path in IN_FILES.items():
        if not path.exists():
            print(f"[!] Missing: {path} (skipping)")
            continue

        payload = _read_json(path)
        source_file = path.name

        if key == "ML":
            rows.extend(_extract_ml(payload, source_file))
        elif key == "NGRAM":
            rows.extend(
                _extract_single(
                    payload,
                    default_name="CharTFIDF(2-3)+LogReg",
                    source_file=source_file,
                )
            )
        elif key == "BERT":
            rows.extend(
                _extract_single(
                    payload,
                    default_name="DistilBERT",
                    source_file=source_file,
                )
            )
        elif key == "HYBRID":
            rows.extend(
                _extract_single(
                    payload,
                    default_name="Hybrid(LGBM+BERT)",
                    source_file=source_file,
                )
            )

    if not rows:
        raise RuntimeError("No result files found or no metrics could be parsed.")

    df = pd.DataFrame(rows)

    # Order columns (only keep those that exist to avoid KeyError)
    ordered_cols = EXTRA_COLS + METRIC_ORDER
    df = df[[c for c in ordered_cols if c in df.columns]]

    # Sort by F1 (desc) as default
    if "f1" in df.columns:
        df = df.sort_values(by="f1", ascending=False, na_position="last").reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Markdown output with robust fallback (no tabulate dependency)
    try:
        md = df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        md = df.to_csv(index=False)

    OUT_MD.write_text(md, encoding="utf-8")

    print(f"[+] Wrote: {OUT_CSV}")
    print(f"[+] Wrote: {OUT_MD}")
    print("\n=== Summary (sorted by f1 desc) ===")
    print(md)


if __name__ == "__main__":
    main()
