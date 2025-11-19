from pathlib import Path
from urllib.parse import urlsplit
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "bank",
    "paypal",
    "confirm",
    "free",
]


def parse_url(url: str):
    """Robustes Parsen, auch wenn 'http' fehlt."""
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        parts = urlsplit(url)
        return parts
    except Exception:
        return urlsplit("http://invalid")


def has_ip_in_domain(domain: str) -> int:
    # sehr einfacher IPv4-Check
    ip_pattern = re.compile(
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )
    return int(bool(ip_pattern.match(domain)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    urls = df["url"].astype(str)

    url_len = urls.str.len()
    num_digits = urls.str.count(r"\d")
    num_letters = urls.str.count(r"[A-Za-z]")

    num_slash = urls.str.count("/")
    num_dot = urls.str.count(r"\.")
    num_dash = urls.str.count("-")
    num_at = urls.str.count("@")
    num_qmark = urls.str.count(r"\?")
    num_equal = urls.str.count("=")
    num_percent = urls.str.count("%")

    # Domain-bezogene Features
    parts = urls.apply(parse_url)
    domains = parts.apply(lambda p: p.netloc.lower())
    paths = parts.apply(lambda p: p.path.lower())

    num_subdirs = paths.str.count("/")
    num_subdomains = domains.apply(lambda d: max(0, d.count(".")))  # grob

    has_ip = domains.apply(has_ip_in_domain)

    # Keyword-Features
    feat = pd.DataFrame(
        {
            "url": urls,
            "label": df["label"].values,
            "url_len": url_len,
            "num_digits": num_digits,
            "num_letters": num_letters,
            "num_slash": num_slash,
            "num_dot": num_dot,
            "num_dash": num_dash,
            "num_at": num_at,
            "num_qmark": num_qmark,
            "num_equal": num_equal,
            "num_percent": num_percent,
            "num_subdirs": num_subdirs,
            "num_subdomains": num_subdomains,
            "has_ip": has_ip,
        }
    )

    for kw in SUSPICIOUS_KEYWORDS:
        feat[f"kw_{kw}"] = paths.str.contains(kw, case=False, regex=False).astype(int)

    return feat


def main():
    train_path = PROCESSED_DIR / "urls_train.csv"
    test_path = PROCESSED_DIR / "urls_test.csv"

    print(f"[+] Lade: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"[+] Lade: {test_path}")
    test_df = pd.read_csv(test_path)

    print("[+] Baue Features für Train...")
    train_feat = build_features(train_df)
    print("[+] Baue Features für Test...")
    test_feat = build_features(test_df)

    train_out = PROCESSED_DIR / "urls_train_features.csv"
    test_out = PROCESSED_DIR / "urls_test_features.csv"

    train_feat.to_csv(train_out, index=False)
    test_feat.to_csv(test_out, index=False)

    print(f"[+] Train-Features gespeichert: {train_out} ({len(train_feat)} Zeilen)")
    print(f"[+] Test-Features gespeichert:  {test_out} ({len(test_feat)} Zeilen)")


if __name__ == "__main__":
    main()
