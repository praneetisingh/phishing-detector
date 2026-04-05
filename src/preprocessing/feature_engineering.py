"""
src/preprocessing/feature_engineering.py
------------------------------------------
Feature engineering pipeline: TF-IDF, n-grams, character features,
handcrafted cybersecurity features, and optional BERT embeddings.

Supports two modes:
  - "tfidf"     : TF-IDF + handcrafted features (fast, interpretable)
  - "bert"      : distilBERT sentence embeddings + handcrafted features
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from pathlib import Path
from typing import List, Optional, Union, Tuple
from loguru import logger

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.preprocessing.text_cleaner import EmailPreprocessor


# ─────────────────────────────────────────────
# Custom sklearn transformers
# ─────────────────────────────────────────────

class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that extracts cybersecurity-specific
    numeric features from raw email text.

    Outputs a dense numpy array per email.
    """

    def __init__(self):
        self.preprocessor = EmailPreprocessor()
        self.scaler = StandardScaler()
        self.feature_names_: List[str] = []

    def fit(self, X: List[str], y=None):
        features = self._extract_all(X)
        self.feature_names_ = list(features.columns)
        self.scaler.fit(features.values)
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        features = self._extract_all(X)
        return self.scaler.transform(features.values)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_

    def _extract_all(self, X: List[str]) -> pd.DataFrame:
        rows = [self.preprocessor.extract_features(text) for text in X]
        df = pd.DataFrame(rows).fillna(0.0)
        return df


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense (needed for some estimators)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return X


# ─────────────────────────────────────────────
# Main feature engineering pipeline
# ─────────────────────────────────────────────

class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    Modes:
        "tfidf"  — word/char TF-IDF + handcrafted features
        "bert"   — distilBERT embeddings + handcrafted features (GPU optional)

    Example:
        pipeline = FeaturePipeline(mode="tfidf")
        pipeline.fit(X_train)
        X_train_feats = pipeline.transform(X_train)
        X_test_feats  = pipeline.transform(X_test)
    """

    def __init__(
        self,
        mode: str = "tfidf",
        max_word_features: int = 50_000,
        max_char_features: int = 30_000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple = (1, 3),
        char_ngram_range: Tuple = (2, 4),
        include_handcrafted: bool = True,
        bert_model: str = "distilbert-base-uncased",
        cache_dir: str = "models/saved",
    ):
        self.mode = mode
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.char_ngram_range = char_ngram_range
        self.include_handcrafted = include_handcrafted
        self.bert_model = bert_model
        self.cache_dir = Path(cache_dir)

        self.preprocessor = EmailPreprocessor(method="lemmatize")
        self._pipeline = None
        self._bert_encoder = None
        self._handcrafted_extractor = None
        self.is_fitted = False

    # ──────────────────────────────────────────
    # Fit / transform
    # ──────────────────────────────────────────

    def fit(self, X: List[str], y=None) -> "FeaturePipeline":
        logger.info(f"Fitting FeaturePipeline (mode={self.mode}) on {len(X):,} samples")

        # Step 1: preprocess text
        X_clean = self.preprocessor.transform_batch(X)

        if self.mode == "tfidf":
            self._build_tfidf_pipeline()
            self._pipeline.fit(X_clean)
        elif self.mode == "bert":
            self._init_bert()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'tfidf' or 'bert'.")

        if self.include_handcrafted:
            self._handcrafted_extractor = HandcraftedFeatureExtractor()
            self._handcrafted_extractor.fit(X)  # raw text for heuristics

        self.is_fitted = True
        logger.info("FeaturePipeline fitted successfully")
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X_clean = self.preprocessor.transform_batch(X, show_progress=False)

        if self.mode == "tfidf":
            text_features = self._pipeline.transform(X_clean).toarray()
        elif self.mode == "bert":
            text_features = self._bert_encode(X_clean)

        if self.include_handcrafted:
            hc_features = self._handcrafted_extractor.transform(X)
            return np.hstack([text_features, hc_features])

        return text_features

    def fit_transform(self, X: List[str], y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    # ──────────────────────────────────────────
    # TF-IDF pipeline
    # ──────────────────────────────────────────

    def _build_tfidf_pipeline(self):
        """
        Combined word + character n-gram TF-IDF.

        Word n-grams: capture phrase patterns ("verify your account")
        Char n-grams: capture obfuscation ("ph1shing", "s3cure")
        """
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            max_features=self.max_word_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,       # log(1 + tf) — reduces dominance of common terms
            strip_accents="unicode",
            token_pattern=r"\b[a-zA-Z0-9_]{2,}\b",
        )

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",      # word boundary-aware character n-grams
            ngram_range=self.char_ngram_range,
            max_features=self.max_char_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
        )

        self._pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word_tfidf", word_vectorizer),
                ("char_tfidf", char_vectorizer),
            ])),
        ])

    # ──────────────────────────────────────────
    # BERT encoding
    # ──────────────────────────────────────────

    def _init_bert(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
            self._bert_model = AutoModel.from_pretrained(self.bert_model)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._bert_model.to(self._device)
            self._bert_model.eval()
            logger.info(f"BERT model loaded: {self.bert_model} on {self._device}")
        except ImportError:
            raise ImportError("transformers + torch required for BERT mode.")

    def _bert_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output = self._bert_model(**encoded)
                # CLS token embedding as sentence representation
                embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        path = path or str(self.cache_dir / "feature_pipeline.joblib")
        joblib.dump(self, path)
        logger.info(f"FeaturePipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        pipeline = joblib.load(path)
        logger.info(f"FeaturePipeline loaded from {path}")
        return pipeline

    def get_feature_names(self) -> List[str]:
        """Return all feature names for SHAP/LIME explainability."""
        names = []
        if self.mode == "tfidf" and self._pipeline:
            fu = self._pipeline.named_steps["features"]
            for name, vec in fu.transformer_list:
                names.extend([f"{name}__{f}" for f in vec.get_feature_names_out()])
        if self.include_handcrafted and self._handcrafted_extractor:
            names.extend(self._handcrafted_extractor.get_feature_names_out())
        return names


# ─────────────────────────────────────────────
# Class imbalance handling
# ─────────────────────────────────────────────

def handle_imbalance(X: np.ndarray, y: np.ndarray, strategy: str = "smote") -> Tuple:
    """
    Handle class imbalance using SMOTE or class weighting.

    Args:
        strategy: "smote" | "borderline_smote" | "class_weight" | "none"
    Returns:
        X_resampled, y_resampled (or original if strategy="class_weight")
    """
    from collections import Counter
    logger.info(f"Class distribution before: {Counter(y)}")

    if strategy == "none":
        return X, y

    if strategy == "class_weight":
        # Return original — caller uses class_weight='balanced' in model
        logger.info("Using class_weight='balanced' strategy (no resampling)")
        return X, y

    try:
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE

        if strategy == "smote":
            resampler = SMOTE(random_state=42, k_neighbors=5)
        elif strategy == "borderline_smote":
            resampler = BorderlineSMOTE(random_state=42, kind="borderline-1")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        X_res, y_res = resampler.fit_resample(X, y)
        logger.info(f"Class distribution after {strategy}: {Counter(y_res)}")
        return X_res, y_res

    except ImportError:
        logger.warning("imbalanced-learn not installed. Skipping resampling.")
        return X, y
