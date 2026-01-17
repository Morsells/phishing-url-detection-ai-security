from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"


def run_step(script_name: str, allow_fail: bool = False) -> None:
    """
    Run a single pipeline step (a Python script in src/).

    If allow_fail=True, the pipeline will continue even if the step exits non-zero.
    This is useful for optional/heavy steps like BERT training that may fail on
    certain systems (e.g., Windows safetensors save issues).
    """
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n===== Running: {script_name} =====")
    cmd: List[str] = [sys.executable, str(script_path)]
    completed = subprocess.run(cmd, cwd=str(BASE_DIR))

    if completed.returncode != 0:
        if allow_fail:
            print(f"[!] Step failed but continuing: {script_name} (exit code {completed.returncode})")
            return
        raise RuntimeError(f"Step failed: {script_name} (exit code {completed.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full phishing-URL detection pipeline end-to-end."
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip DistilBERT fine-tuning and hybrid model.",
    )
    parser.add_argument(
        "--only-classical",
        action="store_true",
        help="Run only data loading, feature engineering, and classical ML models (no n-grams, no BERT).",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help=(
            "Run hybrid model (requires BERT model). If BERT is skipped but a saved model exists, "
            "hybrid can still run."
        ),
    )
    parser.add_argument(
        "--allow-bert-fail",
        action="store_true",
        help=(
            "Do not abort the pipeline if train_bert.py fails. Useful on systems where saving the "
            "BERT model can fail (e.g., Windows safetensors I/O error)."
        ),
    )

    args = parser.parse_args()

    # Always required steps
    run_step("data_loading.py")
    run_step("features.py")
    run_step("train_ml.py")

    if args.only_classical:
        run_step("collect_results.py")
        print("\n[+] Done (only classical).")
        return

    # N-gram baseline
    run_step("train_ngrams.py")

    # BERT / Hybrid
    if not args.skip_bert:
        run_step("train_bert.py", allow_fail=args.allow_bert_fail)

        # If BERT failed and we're not allowing failure, we would have already raised.
        # If BERT failed but we're allowing failure, only run hybrid if a saved model exists.
        bert_dir = BASE_DIR / "models" / "bert"
        bert_exists = bert_dir.exists() and any(bert_dir.iterdir())

        if bert_exists:
            run_step("train_hybrid.py", allow_fail=args.allow_bert_fail)
        else:
            print("[!] Skipping hybrid: no saved BERT model found under models/bert/.")

    else:
        # If user skipped BERT training but still wants hybrid and a model exists
        if args.include_hybrid:
            bert_dir = BASE_DIR / "models" / "bert"
            if not bert_dir.exists() or not any(bert_dir.iterdir()):
                raise FileNotFoundError(
                    "Hybrid requested but no saved BERT model found under models/bert/. "
                    "Run without --skip-bert or provide a saved model."
                )
            run_step("train_hybrid.py")

    # Always collect final summary
    run_step("collect_results.py")
    print("\n[+] Done.")


if __name__ == "__main__":
    main()
