import os
import pathlib
import csv
import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 1) GitHub-Dataset mit URLs + Label (phishing_site_urls.csv)
#    Repo spiegelt ein bekanntes Phishing-URL-Dataset wider. 
GITHUB_URL_DATASET = (
    "https://raw.githubusercontent.com/ArunBalajiR/"
    "Phishing-URL-Detection/master/dataset/phishing_site_urls.csv"
)
GITHUB_TARGET_FILE = RAW_DIR / "phishing_site_urls_github.csv"

# 2) OpenPhish Community Feed (nur Phishing-URLs, Textliste) 
OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"
OPENPHISH_TARGET_FILE = RAW_DIR / "openphish_feed.txt"


def download_file(url: str, target_path: pathlib.Path, text_mode: bool = True) -> None:
    print(f"[+] Lade herunter: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    if text_mode:
        target_path.write_text(resp.text, encoding="utf-8", newline="")
    else:
        target_path.write_bytes(resp.content)

    print(f"[+] Gespeichert unter: {target_path}")


def quick_check_github_csv(path: pathlib.Path) -> None:
    print(f"[+] Quick-Check CSV: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"    Header: {header}")
        # einige Zeilen anzeigen
        for i, row in enumerate(reader):
            print(f"    Zeile {i+1}: {row}")
            if i >= 4:  # nur 5 Beispielzeilen
                break


def quick_check_openphish(path: pathlib.Path) -> None:
    print(f"[+] Quick-Check OpenPhish: {path}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"    Anzahl Zeilen: {len(lines)}")
    print("    Erste 5 URLs:")
    for line in lines[:5]:
        print("    ", line)


def main():
    print("[*] RAW_DIR:", RAW_DIR)

    # 1) GitHub-Phishing-URL-Datensatz
    try:
        download_file(GITHUB_URL_DATASET, GITHUB_TARGET_FILE, text_mode=True)
        quick_check_github_csv(GITHUB_TARGET_FILE)
    except Exception as e:
        print("[!] Fehler beim GitHub-Datensatz:", e)

    # 2) OpenPhish Community Feed
    try:
        download_file(OPENPHISH_FEED_URL, OPENPHISH_TARGET_FILE, text_mode=True)
        quick_check_openphish(OPENPHISH_TARGET_FILE)
    except Exception as e:
        print("[!] Fehler beim OpenPhish-Feed:", e)

    print("[*] Fertig.")


if __name__ == "__main__":
    main()
