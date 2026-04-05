"""
src/utils/data_loader.py
------------------------
Handles downloading, loading, and combining multiple public datasets
for phishing/spam detection. Supports Enron, SpamAssassin, CEAS-08,
Nazario phishing corpus, and OpenPhish URL feeds.
"""

import os
import zipfile
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Dataset registry: URL, sha256, local filename
# ─────────────────────────────────────────────
DATASETS = {
    "enron_spam": {
        "description": "Enron Spam Dataset (6 corpora, ~30k emails)",
        "url": "https://spamassassin.apache.org/old/publiccorpus/",
        "kaggle": "wanderfj/enron-spam",
        "local": "data/raw/enron_spam.csv",
        "label_col": "Spam/Ham",
        "text_col": "Message",
        "label_map": {"ham": 0, "spam": 1},
        "size": "~30k samples",
    },
    "spamassassin": {
        "description": "SpamAssassin Public Corpus (easy/hard ham + spam)",
        "url": "https://spamassassin.apache.org/old/publiccorpus/",
        "local": "data/raw/spamassassin.csv",
        "label_col": "label",
        "text_col": "text",
        "label_map": {"ham": 0, "spam": 1},
        "size": "~6k samples",
    },
    "phishing_emails": {
        "description": "Nazario Phishing Email Corpus + CEAS-08",
        "kaggle": "naserabdullahalam/phishing-email-dataset",
        "local": "data/raw/phishing_emails.csv",
        "label_col": "Email Type",
        "text_col": "Email Text",
        "label_map": {"Safe Email": 0, "Phishing Email": 1},
        "size": "~18k samples",
    },
    "ling_spam": {
        "description": "Ling-Spam Dataset (linguistics mailing list)",
        "kaggle": "mandygu/lingspam-dataset",
        "local": "data/raw/ling_spam.csv",
        "label_col": "label",
        "text_col": "text",
        "label_map": {0: 0, 1: 1},
        "size": "~2.9k samples",
    },
}


class DatasetLoader:
    """
    Production-grade data loader for multi-source email security datasets.

    Usage:
        loader = DatasetLoader(data_dir="data/raw")
        df = loader.load_combined(sample_n=50000)
        X_train, X_test, y_train, y_test = loader.split(df)
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def load_combined(
        self,
        sample_n: Optional[int] = None,
        include_headers: bool = True,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Load and concatenate all available datasets into a single DataFrame.

        Returns columns: ['text', 'label', 'source', 'subject', 'sender']
        label: 0 = legitimate, 1 = phishing/spam
        """
        frames = []

        for name, cfg in DATASETS.items():
            path = Path(cfg["local"])
            if not path.exists():
                logger.warning(
                    f"[{name}] not found at {path}. "
                    f"Download from Kaggle: '{cfg.get('kaggle', 'N/A')}'"
                )
                continue

            try:
                df = self._load_single(name, cfg, path)
                df["source"] = name
                frames.append(df)
                logger.info(f"[{name}] loaded {len(df):,} samples")
            except Exception as exc:
                logger.error(f"[{name}] failed to load: {exc}")

        if not frames:
            logger.warning("No datasets found. Generating synthetic demo data.")
            return self._generate_demo_data(n=5000, seed=seed)

        combined = pd.concat(frames, ignore_index=True)
        combined = self._clean_combined(combined)

        if sample_n and sample_n < len(combined):
            combined = combined.sample(n=sample_n, random_state=seed)

        logger.info(
            f"Combined dataset: {len(combined):,} samples | "
            f"Spam/Phishing: {combined['label'].sum():,} "
            f"({combined['label'].mean()*100:.1f}%)"
        )
        return combined.reset_index(drop=True)

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple:
        """
        Stratified train/val/test split preserving class balance.

        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        X = df["text"]
        y = df["label"]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio, stratify=y_train_val, random_state=seed
        )

        logger.info(
            f"Split → train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _load_single(self, name: str, cfg: dict, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Normalise column names
        df = df.rename(
            columns={
                cfg["text_col"]: "text",
                cfg["label_col"]: "label",
            }
        )

        # Map label strings → 0/1
        label_map = cfg["label_map"]
        if isinstance(list(label_map.keys())[0], str):
            df["label"] = df["label"].map(label_map)
        else:
            df["label"] = df["label"].astype(int)

        # Keep only what we need
        cols = ["text", "label"] + [
            c for c in ["subject", "sender", "from", "to", "date"]
            if c in df.columns
        ]
        return df[cols].dropna(subset=["text", "label"])

    def _clean_combined(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        df = df[df["label"].isin([0, 1])]
        df = df.drop_duplicates(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 10]
        return df

    def _generate_demo_data(self, n: int = 5000, seed: int = 42) -> pd.DataFrame:
        """
        Fallback: generate synthetic but realistic demo emails.
        Used for testing when actual datasets are unavailable.
        """
        rng = np.random.default_rng(seed)

        legit_templates = [
            "Hi {name}, please find attached the {doc} for Q{q} review.",
            "Meeting reminder: {topic} scheduled for {day} at {time}.",
            "Your order #{num} has been shipped. Track at {url}",
            "Team update: {topic} — please review before {day}.",
            "Invoice #{num} for services rendered in {month}. Amount: ${amount}",
        ]
        phish_templates = [
            "URGENT: Verify your {bank} account NOW or it will be suspended! Click {url}",
            "Congratulations! You won ${amount}. Claim prize at {url} — expires {day}!",
            "Security Alert: Unusual login on your {service} account. Verify: {url}",
            "Your PayPal account is limited. Restore access: {url} — act within 24 hrs.",
            "Dear valued customer, confirm identity to avoid account closure: {url}",
            "IRS NOTICE: You owe ${amount} in taxes. Avoid arrest — pay now: {url}",
        ]

        names = ["John", "Sarah", "Mike", "Lisa", "David", "Anna"]
        banks = ["Chase", "Wells Fargo", "Bank of America", "Citibank"]
        services = ["Gmail", "Netflix", "Amazon", "PayPal", "Microsoft"]
        urls = ["bit.ly/3xK9", "secure-login.xyz/verify", "account-update.info/click"]

        records = []
        for i in range(n):
            is_phishing = rng.random() < 0.35  # ~35% phishing (realistic)
            templates = phish_templates if is_phishing else legit_templates
            tmpl = templates[rng.integers(len(templates))]
            text = tmpl.format(
                name=rng.choice(names),
                doc=rng.choice(["report", "contract", "proposal"]),
                q=rng.integers(1, 5),
                topic=rng.choice(["budget review", "product launch", "Q3 results"]),
                day=rng.choice(["Monday", "Tuesday", "Friday"]),
                time=f"{rng.integers(9,18)}:00",
                url=rng.choice(urls),
                num=rng.integers(10000, 99999),
                month=rng.choice(["January", "February", "March"]),
                amount=rng.integers(100, 9999),
                bank=rng.choice(banks),
                service=rng.choice(services),
            )
            records.append({"text": text, "label": int(is_phishing), "source": "demo"})

        return pd.DataFrame(records)


def get_dataset_info() -> pd.DataFrame:
    """Return a summary table of all supported datasets for documentation."""
    rows = []
    for name, cfg in DATASETS.items():
        rows.append({
            "Dataset": name,
            "Description": cfg["description"],
            "Size": cfg["size"],
            "Kaggle": cfg.get("kaggle", "—"),
        })
    return pd.DataFrame(rows)
