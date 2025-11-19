from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_github_dataset() -> pd.DataFrame:
    """
    Lädt phishing_site_urls_github.csv und normalisiert Spalten:
    - 'URL'  -> 'url'
    - 'Label' -> 'label' (0=good, 1=bad)
    """
    csv_path = RAW_DIR / "phishing_site_urls_github.csv"
    print(f"[+] Lade GitHub-Datensatz: {csv_path}")

    df = pd.read_csv(csv_path)

    # Erwartete Spalten: 'URL', 'Label'
    df = df.rename(columns={"URL": "url", "Label": "label"})

    # Strings normalisieren
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Auf 0/1 mappen
    label_map = {"good": 0, "bad": 1}
    df["label"] = df["label"].map(label_map)

    # Evtl. Zeilen mit unbekanntem Label entfernen
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    after = len(df)

    print(f"    Zeilen vor Filter: {before}, nach Filter: {after}")
    print(df["label"].value_counts())

    df["source"] = "github"

    return df


def load_openphish_dataset() -> pd.DataFrame:
    """
    Lädt openphish_feed.txt (jede Zeile eine Phishing-URL).
    Label = 1 (immer phishing).
    """
    txt_path = RAW_DIR / "openphish_feed.txt"
    print(f"[+] Lade OpenPhish-Feed: {txt_path}")

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    urls = [ln.strip() for ln in lines if ln.strip()]

    df = pd.DataFrame({"url": urls})
    df["label"] = 1  # alles phishing
    df["source"] = "openphish"

    print(f"    Anzahl URLs im OpenPhish-Feed: {len(df)}")
    return df



def make_splits(
    github_df: pd.DataFrame,
    openphish_df: pd.DataFrame,
    train_frac: float = 0.8,
    random_state: int = 42,
):
    """
    Train/Test-Split:
    - Train:   train_frac der GitHub-Daten (stratifiziert)
    - Test:    Rest der GitHub-Daten + ALLE OpenPhish-URLs
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

    # Test enthält GitHub-Test + alle OpenPhish-Phishing-URLs
    test = pd.concat([g_test, openphish_df], ignore_index=True)
    print(f"    Test gesamt (GitHub + OpenPhish): {len(test)}")

    return g_train, test


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_path = PROCESSED_DIR / "urls_train.csv"
    test_path = PROCESSED_DIR / "urls_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Train gespeichert: {train_path}  ({len(train_df)} Zeilen)")
    print(f"[+] Test gespeichert:  {test_path}  ({len(test_df)} Zeilen)")


def main():
    print("[*] BASE_DIR:", BASE_DIR)

    github_df = load_github_dataset()
    openphish_df = load_openphish_dataset()

    train_df, test_df = make_splits(github_df, openphish_df)
    save_splits(train_df, test_df)

    print("[*] Fertig.")


if __name__ == "__main__":
    main()
