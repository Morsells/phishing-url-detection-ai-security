from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_FILE = "phishing_site_urls_github.csv"
OPENPHISH_FILE = "openphish_feed.txt"


def _require_file(path: Path) -> None:
    """Bricht mit verständlicher Fehlermeldung ab, falls eine Datei fehlt."""
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}\n"
            f"Bitte lege die Datei in: {RAW_DIR}"
        )


def _clean_url_series(s: pd.Series) -> pd.Series:
    """
    Säubert URL-Strings minimal aber zuverlässig:
    - String-Konvertierung
    - Whitespace entfernen
    - leere/NaN-artige Einträge entfernen (später via dropna)
    """
    s = s.astype(str).str.strip()
    # manche CSVs enthalten "nan" als Text
    s = s.replace({"": None, "nan": None, "None": None})
    return s


def load_github_dataset() -> pd.DataFrame:
    """
    Lädt den GitHub-Datensatz und normalisiert Spalten:
    - 'URL'   -> 'url'
    - 'Label' -> 'label' (0=good, 1=bad)

    WICHTIG:
    - Entfernt Duplikate (url) VOR dem Split, um Train/Test-Leakage zu vermeiden.
    """
    csv_path = RAW_DIR / GITHUB_FILE
    _require_file(csv_path)
    print(f"[+] Lade GitHub-Datensatz: {csv_path}")

    # Speicherfreundlicher: nur relevante Spalten lesen
    df = pd.read_csv(
        csv_path,
        usecols=["URL", "Label"],
        dtype={"URL": "string", "Label": "string"},
        low_memory=False,
    )

    df = df.rename(columns={"URL": "url", "Label": "label"})

    # URLs säubern
    df["url"] = _clean_url_series(df["url"])
    df = df.dropna(subset=["url"]).reset_index(drop=True)

    # Labels säubern und mappen
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    label_map = {"good": 0, "bad": 1}
    df["label"] = df["label"].map(label_map)

    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    after = len(df)

    print(f"    Zeilen vor Label-Filter: {before}, nach Filter: {after}")
    print("    Label-Verteilung:")
    print(df["label"].value_counts())

    # Duplikate entfernen (sehr wichtig gegen Leakage)
    before_dups = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    after_dups = len(df)
    print(f"    Duplikate entfernt: {before_dups - after_dups} (von {before_dups} auf {after_dups})")

    df["source"] = "github"
    return df


def load_openphish_dataset() -> pd.DataFrame:
    """
    Lädt openphish_feed.txt (jede Zeile eine Phishing-URL).
    Label = 1 (immer phishing).
    """
    txt_path = RAW_DIR / OPENPHISH_FILE
    _require_file(txt_path)
    print(f"[+] Lade OpenPhish-Feed: {txt_path}")

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    urls = [ln.strip() for ln in lines if ln.strip()]

    df = pd.DataFrame({"url": urls})
    df["url"] = _clean_url_series(df["url"])
    df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    df["label"] = 1
    df["source"] = "openphish"

    print(f"    Anzahl URLs im OpenPhish-Feed (nach Clean/Dedup): {len(df)}")
    return df


def make_splits(
    github_df: pd.DataFrame,
    openphish_df: pd.DataFrame,
    train_frac: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/Test-Split:
    - Train:  train_frac der GitHub-Daten (stratifiziert)
    - Test:   Rest der GitHub-Daten + OpenPhish-URLs

    Zusätzlich:
    - Entfernt OpenPhish-URLs, die im GitHub-Train oder GitHub-Test bereits vorkommen,
      damit die Test-Erweiterung keine Overlap-/Doppelzählungs-Effekte erzeugt.
    """
    print("[+] Erzeuge Train/Test-Split aus GitHub-Datensatz")

    g_train, g_test = train_test_split(
        github_df,
        train_size=train_frac,
        random_state=random_state,
        stratify=github_df["label"],
    )

    g_train = g_train.reset_index(drop=True)
    g_test = g_test.reset_index(drop=True)

    print(f"    GitHub Train: {len(g_train)}, GitHub Test: {len(g_test)}")

    # Sanity-Check: nach Dedup sollte Overlap 0 sein
    overlap = set(g_train["url"]).intersection(set(g_test["url"]))
    print(f"    URL-Overlap Train/Test (GitHub): {len(overlap)}")

    # OpenPhish-URLs entfernen, die bereits in GitHub vorkommen (Train/Test)
    github_all_urls = set(github_df["url"])
    before_op = len(openphish_df)
    openphish_df = openphish_df[~openphish_df["url"].isin(github_all_urls)].reset_index(drop=True)
    removed_op = before_op - len(openphish_df)
    print(f"    OpenPhish-Overlap mit GitHub entfernt: {removed_op}")

    # Test enthält GitHub-Test + OpenPhish (keine Doppel-URLs)
    test = pd.concat([g_test, openphish_df], ignore_index=True)
    before_test_dedup = len(test)
    test = test.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    print(f"    Test gesamt (GitHub + OpenPhish): {len(test)} (Dedup entfernt {before_test_dedup - len(test)})")

    return g_train, test


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_path = PROCESSED_DIR / "urls_train.csv"
    test_path = PROCESSED_DIR / "urls_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"[+] Train gespeichert: {train_path}  ({len(train_df)} Zeilen)")
    print(f"[+] Test gespeichert:  {test_path}  ({len(test_df)} Zeilen)")


def main() -> None:
    print("[*] BASE_DIR:", BASE_DIR)

    github_df = load_github_dataset()
    openphish_df = load_openphish_dataset()

    train_df, test_df = make_splits(github_df, openphish_df)
    save_splits(train_df, test_df)

    print("[*] Fertig.")


if __name__ == "__main__":
    main()
