"""
src/preprocessing/text_cleaner.py
----------------------------------
Production-grade NLP preprocessing pipeline for email security.

Pipeline:
  1. HTML/CSS/JS stripping
  2. URL extraction and replacement
  3. Header parsing (From, Subject, Date)
  4. Unicode normalization
  5. Tokenization + stopword removal
  6. Lemmatization (spaCy) or stemming (NLTK fallback)
  7. Feature engineering hooks
"""

import re
import unicodedata
import email
import warnings
from typing import List, Optional, Dict, Tuple
from functools import lru_cache

import nltk
import numpy as np
from bs4 import BeautifulSoup
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

# Download required NLTK data (idempotent)
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# Phishing-specific keyword lexicons (rule engine integration)
# ──────────────────────────────────────────────────────────────
URGENCY_KEYWORDS = {
    "urgent", "immediately", "action required", "account suspended",
    "verify now", "limited time", "expires", "24 hours", "act now",
    "final notice", "last chance", "your account", "confirm identity",
}

FINANCIAL_LURE_KEYWORDS = {
    "winner", "congratulations", "prize", "lottery", "million dollars",
    "free gift", "claim now", "cash prize", "inheritance", "wire transfer",
    "investment opportunity", "double your money",
}

CREDENTIAL_HARVEST_KEYWORDS = {
    "click here", "login", "sign in", "verify your", "update your",
    "confirm your password", "reset your", "account details",
    "personal information", "social security",
}

ALL_PHISHING_KEYWORDS = (
    URGENCY_KEYWORDS | FINANCIAL_LURE_KEYWORDS | CREDENTIAL_HARVEST_KEYWORDS
)

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw",
    ".cc", ".top", ".click", ".link", ".info",
}

STOP_WORDS = set(stopwords.words("english"))
# Keep negations — critical for sentiment/intent detection
STOP_WORDS -= {"no", "not", "nor", "neither", "never", "nothing", "nowhere"}


class EmailPreprocessor:
    """
    Full preprocessing pipeline for raw email text.

    Example:
        preprocessor = EmailPreprocessor(method="lemmatize")
        clean_text = preprocessor.transform("Your account has been suspended...")
        features = preprocessor.extract_features("Your account...")
    """

    def __init__(
        self,
        method: str = "lemmatize",  # "lemmatize" | "stem" | "none"
        remove_stopwords: bool = True,
        min_token_len: int = 2,
        max_token_len: int = 30,
    ):
        self.method = method
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len

        self.lemmatizer = WordNetLemmatizer() if method == "lemmatize" else None
        self.stemmer = PorterStemmer() if method == "stem" else None

        self._url_pattern = re.compile(
            r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
            re.IGNORECASE,
        )
        self._email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I)
        self._html_tag_pattern = re.compile(r"<[^>]+>")
        self._whitespace_pattern = re.compile(r"\s+")
        self._non_alpha_pattern = re.compile(r"[^a-z0-9\s]")

    # ────────────────────────────────────────
    # Core transform
    # ────────────────────────────────────────

    def transform(self, text: str) -> str:
        """Full pipeline: raw text → clean, normalised string."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._parse_email_body(text)
        text = self._strip_html(text)
        text = self._replace_urls(text)
        text = self._replace_emails(text)
        text = self._normalise_unicode(text)
        text = text.lower()
        text = self._non_alpha_pattern.sub(" ", text)

        tokens = word_tokenize(text)
        tokens = self._filter_tokens(tokens)

        if self.method == "lemmatize":
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.method == "stem":
            tokens = [self.stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def transform_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """Batch transform with optional progress logging."""
        results = []
        n = len(texts)
        for i, text in enumerate(texts):
            results.append(self.transform(text))
            if show_progress and (i + 1) % 1000 == 0:
                logger.info(f"Preprocessed {i+1:,}/{n:,} emails")
        return results

    # ────────────────────────────────────────
    # Feature engineering
    # ────────────────────────────────────────

    def extract_features(self, raw_text: str) -> Dict[str, float]:
        """
        Extract handcrafted cybersecurity-relevant features from raw email.
        These complement ML embeddings with domain knowledge.

        Returns a dict of numeric features.
        """
        text_lower = raw_text.lower()
        urls = self._url_pattern.findall(raw_text)

        features = {
            # Length features
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()),
            "avg_word_length": np.mean([len(w) for w in raw_text.split()] or [0]),

            # URL features
            "url_count": len(urls),
            "has_url": int(bool(urls)),
            "suspicious_tld_count": sum(
                1 for url in urls
                if any(url.lower().endswith(tld) for tld in SUSPICIOUS_TLDS)
            ),
            "url_in_text_ratio": len(urls) / max(len(raw_text.split()), 1),

            # Phishing keyword features
            "urgency_keyword_count": sum(
                1 for kw in URGENCY_KEYWORDS if kw in text_lower
            ),
            "financial_lure_count": sum(
                1 for kw in FINANCIAL_LURE_KEYWORDS if kw in text_lower
            ),
            "credential_harvest_count": sum(
                1 for kw in CREDENTIAL_HARVEST_KEYWORDS if kw in text_lower
            ),
            "total_phishing_keywords": sum(
                1 for kw in ALL_PHISHING_KEYWORDS if kw in text_lower
            ),

            # Punctuation / formatting anomalies
            "exclamation_count": raw_text.count("!"),
            "question_count": raw_text.count("?"),
            "dollar_count": raw_text.count("$"),
            "capital_ratio": sum(1 for c in raw_text if c.isupper()) / max(len(raw_text), 1),
            "digit_ratio": sum(1 for c in raw_text if c.isdigit()) / max(len(raw_text), 1),

            # HTML features
            "has_html": int(bool(self._html_tag_pattern.search(raw_text))),
            "html_tag_count": len(self._html_tag_pattern.findall(raw_text)),

            # Email-specific
            "has_attachment_mention": int(
                bool(re.search(r"\battach(ment|ed)?\b", text_lower))
            ),
            "has_reply_to_mismatch": int("reply-to:" in text_lower),
        }

        return features

    def extract_header_features(self, raw_email: str) -> Dict[str, str]:
        """
        Parse email headers for forensic features.
        Returns: sender, reply_to, subject, x_mailer, spf_result
        """
        try:
            msg = email.message_from_string(raw_email)
            return {
                "from": msg.get("From", ""),
                "reply_to": msg.get("Reply-To", ""),
                "subject": msg.get("Subject", ""),
                "x_mailer": msg.get("X-Mailer", ""),
                "x_originating_ip": msg.get("X-Originating-IP", ""),
                "received_spf": msg.get("Received-SPF", ""),
                "authentication_results": msg.get("Authentication-Results", ""),
                "return_path": msg.get("Return-Path", ""),
            }
        except Exception:
            return {}

    # ────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────

    def _parse_email_body(self, text: str) -> str:
        """Extract body from RFC-2822 formatted emails."""
        if text.startswith(("From:", "Return-Path:", "Received:")):
            try:
                msg = email.message_from_string(text)
                parts = []
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() in ("text/plain", "text/html"):
                            try:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    parts.append(payload.decode("utf-8", errors="replace"))
                            except Exception:
                                pass
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        return payload.decode("utf-8", errors="replace")
                return " ".join(parts) if parts else text
            except Exception:
                return text
        return text

    def _strip_html(self, text: str) -> str:
        """Remove HTML/CSS/JS, extract visible text."""
        if "<" not in text:
            return text
        try:
            soup = BeautifulSoup(text, "lxml")
            for tag in soup(["script", "style", "head", "meta", "link"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            return self._html_tag_pattern.sub(" ", text)

    def _replace_urls(self, text: str) -> str:
        """Replace URLs with a token that preserves URL presence signal."""
        return self._url_pattern.sub(" URL_TOKEN ", text)

    def _replace_emails(self, text: str) -> str:
        return self._email_pattern.sub(" EMAIL_TOKEN ", text)

    def _normalise_unicode(self, text: str) -> str:
        """Normalise unicode to ASCII — handles obfuscation like 'ρаyраl'."""
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        filtered = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if len(tok) < self.min_token_len or len(tok) > self.max_token_len:
                continue
            if self.remove_stopwords and tok in STOP_WORDS:
                continue
            filtered.append(tok)
        return filtered


# ────────────────────────────────────────────────────────────────────
# Rule-based engine (pre-ML filter for high-confidence cases)
# ────────────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Lightweight keyword + heuristic rule engine.
    Used as a fast pre-filter and as a hybrid signal alongside ML.

    Returns a confidence score and matched rules for explainability.
    """

    RULES = [
        # (name, pattern_or_fn, weight, description)
        ("urgent_keywords", URGENCY_KEYWORDS, 0.25, "Urgency manipulation language"),
        ("financial_lure", FINANCIAL_LURE_KEYWORDS, 0.20, "Financial enticement"),
        ("credential_harvest", CREDENTIAL_HARVEST_KEYWORDS, 0.20, "Credential harvesting language"),
    ]

    def evaluate(self, text: str) -> Dict:
        """
        Return rule-based analysis of an email.

        Returns:
            score: float [0,1] — higher = more suspicious
            triggered_rules: list of triggered rule names
            explanations: human-readable explanations
        """
        text_lower = text.lower()
        total_score = 0.0
        triggered = []
        explanations = []

        for name, keyword_set, weight, description in self.RULES:
            matched = [kw for kw in keyword_set if kw in text_lower]
            if matched:
                # Diminishing returns for many hits in same category
                category_score = weight * min(len(matched), 3) / 3
                total_score += category_score
                triggered.append(name)
                explanations.append(
                    f"{description}: found '{', '.join(matched[:3])}'"
                )

        # Structural heuristics
        url_count = len(re.findall(r"https?://", text, re.I))
        if url_count > 3:
            total_score += 0.15
            triggered.append("many_urls")
            explanations.append(f"Multiple URLs detected ({url_count})")

        if text.count("!") > 5:
            total_score += 0.10
            triggered.append("excessive_exclamations")
            explanations.append(f"Excessive exclamation marks ({text.count('!')})")

        cap_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if cap_ratio > 0.25:
            total_score += 0.10
            triggered.append("high_caps_ratio")
            explanations.append(f"Unusual capitalisation ({cap_ratio:.0%})")

        return {
            "rule_score": min(total_score, 1.0),
            "triggered_rules": triggered,
            "explanations": explanations,
            "flagged": total_score > 0.3,
        }
