from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_BERT_DIR = BASE_DIR / "models" / "bert"


def _ensure_paths() -> None:
    """Prüft Basis-Pfade, damit Fehler früh und verständlich auftreten."""
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"src/ Ordner nicht gefunden: {SRC_DIR}")
    if not (BASE_DIR / "results").exists():
        # results/ wird in den Skripten bei Bedarf erstellt; kein harter Fehler.
        pass


def _find_bert_runs(bert_root: Path) -> list[Path]:
    """Findet BERT-Run-Ordner (z. B. models/bert/run_YYYYMMDD_HHMMSS)."""
    if not bert_root.exists():
        return []
    runs = [p for p in bert_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name)  # Namenssortierung reicht bei Timestamp-Pattern
    return runs


def _has_any_bert_run() -> bool:
    return len(_find_bert_runs(MODELS_BERT_DIR)) > 0


def run_step(script_name: str, allow_fail: bool = False) -> int:
    """
    Führt einen einzelnen Pipeline-Schritt aus (ein Python-Skript in src/).

    allow_fail=True: Pipeline läuft weiter, auch wenn der Schritt fehlschlägt.
    Sinnvoll für optionale/zeitintensive Schritte (z. B. BERT), die je nach System
    Probleme machen können.
    """
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Skript nicht gefunden: {script_path}")

    print(f"\n===== Running: {script_name} =====")
    cmd = [sys.executable, str(script_path)]

    # stdout/stderr nicht „verschlucken“ – direkt im Terminal anzeigen lassen
    completed = subprocess.run(cmd, cwd=str(BASE_DIR))
    rc = completed.returncode

    if rc != 0:
        msg = f"Schritt fehlgeschlagen: {script_name} (exit code {rc})"
        if allow_fail:
            print(f"[!] {msg} – mache weiter (allow_fail=True).")
            return rc
        raise RuntimeError(msg)

    return rc


def main() -> None:
    _ensure_paths()

    parser = argparse.ArgumentParser(
        description="Führt die komplette Phishing-URL-Pipeline end-to-end aus."
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Überspringt DistilBERT-Fine-Tuning (und damit standardmäßig auch Hybrid).",
    )
    parser.add_argument(
        "--only-classical",
        action="store_true",
        help="Nur data_loading, features und klassische ML-Modelle (kein n-gram, kein BERT).",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Versucht Hybrid auszuführen, falls ein gespeicherter BERT-Run vorhanden ist.",
    )
    parser.add_argument(
        "--allow-bert-fail",
        action="store_true",
        help="Bricht nicht ab, wenn train_bert.py oder train_hybrid.py fehlschlägt.",
    )

    args = parser.parse_args()

    # Transparenz: Welcher Interpreter wird verwendet?
    print(f"[*] Python Interpreter: {sys.executable}")
    print(f"[*] Project Base Dir:   {BASE_DIR}")

    # Immer notwendige Schritte
    run_step("data_loading.py")
    run_step("features.py")
    run_step("train_ml.py")

    if args.only_classical:
        run_step("collect_results.py")
        print("\n[+] Fertig (nur klassische Modelle).")
        return

    # N-gram Baseline
    run_step("train_ngrams.py")

    # BERT-Training optional
    if not args.skip_bert:
        run_step("train_bert.py", allow_fail=args.allow_bert_fail)

    # Hybrid-Logik:
    # - Wenn BERT nicht übersprungen wurde: Hybrid läuft, falls irgendein Run existiert.
    # - Wenn BERT übersprungen wurde: Hybrid läuft nur, wenn --include-hybrid gesetzt ist UND ein Run existiert.
    want_hybrid = (not args.skip_bert) or args.include_hybrid

    if want_hybrid:
        if _has_any_bert_run():
            run_step("train_hybrid.py", allow_fail=args.allow_bert_fail)
        else:
            print("[!] Hybrid übersprungen: Kein gespeicherter BERT-Run unter models/bert/ gefunden.")
            print("    Tipp: Entweder ohne --skip-bert ausführen oder einen run_* Ordner bereitstellen.")

    # Ergebnisse zusammenfassen
    run_step("collect_results.py")
    print("\n[+] Fertig.")


if __name__ == "__main__":
    main()
