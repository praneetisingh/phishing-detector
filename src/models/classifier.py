"""
src/models/classifier.py
--------------------------
Model zoo: Logistic Regression, Naive Bayes, Random Forest, SVM,
LSTM (optional), and BERT fine-tune (optional).

Each model is wrapped in a unified interface:
  - fit(X_train, y_train)
  - predict(X) → np.ndarray
  - predict_proba(X) → np.ndarray[:,2]
  - score_report(X_test, y_test) → dict

ModelComparison orchestrates training and evaluation of all models.
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────
# Base wrapper
# ─────────────────────────────────────────────

class BaseEmailClassifier:
    """Unified interface wrapping sklearn estimators."""

    def __init__(self, name: str, estimator, save_dir: str = "models/saved"):
        self.name = name
        self.estimator = estimator
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.is_fitted = False
        self.train_time: float = 0.0

    def fit(self, X, y) -> "BaseEmailClassifier":
        logger.info(f"[{self.name}] Training on {len(y):,} samples...")
        t0 = time.perf_counter()
        self.estimator.fit(X, y)
        self.train_time = time.perf_counter() - t0
        self.is_fitted = True
        logger.info(f"[{self.name}] Trained in {self.train_time:.2f}s")
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        # Fallback for calibrated SVM
        return self.estimator.predict_proba(X)

    def score_report(self, X_test, y_test) -> Dict:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        return {
            "model": self.name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred),
            "train_time_s": round(self.train_time, 3),
        }

    def cross_validate(self, X, y, cv: int = 5) -> Dict:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.estimator, X, y, cv=skf, scoring="f1", n_jobs=-1
        )
        return {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}

    def save(self, filename: Optional[str] = None):
        filename = filename or f"{self.name.lower().replace(' ', '_')}.joblib"
        path = self.save_dir / filename
        joblib.dump(self, path)
        logger.info(f"[{self.name}] Saved to {path}")
        return str(path)

    @classmethod
    def load(cls, path: str) -> "BaseEmailClassifier":
        return joblib.load(path)


# ─────────────────────────────────────────────
# Individual model factories
# ─────────────────────────────────────────────

def build_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> BaseEmailClassifier:
    """
    Logistic Regression — strong baseline with L2 regularisation.
    Highly interpretable via coefficients. Excellent F1 on text.
    """
    estimator = LogisticRegression(
        C=C,
        solver="saga",           # handles large sparse TF-IDF well
        max_iter=max_iter,
        class_weight="balanced", # handles imbalanced classes
        random_state=42,
        n_jobs=-1,
    )
    return BaseEmailClassifier("Logistic Regression", estimator)


def build_naive_bayes(alpha: float = 0.1) -> BaseEmailClassifier:
    """
    Complement Naive Bayes — better than MultinomialNB for imbalanced text.
    Very fast, excellent for sparse TF-IDF features.
    Limitation: assumes feature independence (violated in practice).
    """
    estimator = ComplementNB(alpha=alpha)
    return BaseEmailClassifier("Complement Naive Bayes", estimator)


def build_random_forest(
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
) -> BaseEmailClassifier:
    """
    Random Forest — robust to noise, handles non-linear patterns.
    Provides feature_importances_ for explainability.
    Slower than LR but captures interaction effects.
    """
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    return BaseEmailClassifier("Random Forest", estimator)


def build_linear_svm(C: float = 1.0) -> BaseEmailClassifier:
    """
    LinearSVC with Platt scaling calibration for probability output.
    Typically best accuracy on high-dimensional sparse TF-IDF.
    Limitation: requires calibration for probabilities.
    """
    svc = LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=42)
    estimator = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
    return BaseEmailClassifier("Linear SVM", estimator)


def build_gradient_boosting() -> BaseEmailClassifier:
    """
    GradientBoostingClassifier — strong learner, excellent on tabular
    features. Best used with handcrafted features, not raw TF-IDF.
    """
    estimator = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    return BaseEmailClassifier("Gradient Boosting", estimator)


# ─────────────────────────────────────────────
# LSTM Model (PyTorch)
# ─────────────────────────────────────────────

class LSTMEmailClassifier:
    """
    Bidirectional LSTM with attention for sequential email modelling.
    Captures word order — something TF-IDF misses.

    Requires: torch, training with raw preprocessed token sequences.
    """

    def __init__(
        self,
        vocab_size: int = 30_000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 256,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required for LSTM model.")

        import torch
        import torch.nn as nn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        class _BiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.dropout = nn.Dropout(dropout)
                self.lstm = nn.LSTM(
                    embed_dim, hidden_dim,
                    num_layers=num_layers,
                    bidirectional=True,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                # Attention layer
                self.attention = nn.Linear(hidden_dim * 2, 1)
                self.fc = nn.Linear(hidden_dim * 2, 2)

            def forward(self, x):
                emb = self.dropout(self.embedding(x))
                out, _ = self.lstm(emb)              # (B, T, 2H)
                attn = torch.softmax(self.attention(out), dim=1)
                context = (out * attn).sum(dim=1)    # (B, 2H) weighted sum
                return self.fc(context)              # (B, 2)

        self.model = _BiLSTM().to(self.device)
        self.name = "BiLSTM with Attention"

    def predict_proba_from_tokens(self, token_ids) -> np.ndarray:
        """Forward pass — use after tokenization."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(token_ids, dtype=torch.long).to(self.device)
            logits = self.model(x)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba


# ─────────────────────────────────────────────
# BERT Fine-tune (HuggingFace)
# ─────────────────────────────────────────────

class BERTEmailClassifier:
    """
    Fine-tuned DistilBERT for email classification.
    State-of-the-art accuracy on nuanced phishing detection.
    Requires GPU for practical training.

    Training handled externally via HuggingFace Trainer API.
    This class provides inference only.
    """

    def __init__(self, model_path: str = "distilbert-base-uncased"):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("transformers required for BERT classifier.")

        from transformers import pipeline as hf_pipeline
        self.name = "DistilBERT"
        self._pipeline = hf_pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0,  # GPU if available, else CPU
            return_all_scores=True,
        )

    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        results = self._pipeline(texts, batch_size=batch_size, truncation=True, max_length=256)
        proba = []
        for result in results:
            scores = {r["label"]: r["score"] for r in result}
            # Assumes labels: LABEL_0 = ham, LABEL_1 = phishing
            proba.append([scores.get("LABEL_0", 0.5), scores.get("LABEL_1", 0.5)])
        return np.array(proba)


# ─────────────────────────────────────────────
# Hybrid Classifier (Rule Engine + ML)
# ─────────────────────────────────────────────

class HybridClassifier:
    """
    Combines rule-based engine confidence with ML model probability.

    Final score = (1 - alpha) * ml_proba + alpha * rule_score

    The rule engine acts as:
    1. A fast pre-filter for obvious cases
    2. A confidence booster for borderline ML predictions
    3. An explainability layer for security analysts
    """

    def __init__(
        self,
        ml_model: BaseEmailClassifier,
        alpha: float = 0.2,
        threshold: float = 0.5,
    ):
        from src.preprocessing.text_cleaner import RuleEngine
        self.ml_model = ml_model
        self.rule_engine = RuleEngine()
        self.alpha = alpha        # weight of rule engine
        self.threshold = threshold

    def predict_with_explanation(
        self,
        texts: List[str],
        features: np.ndarray,
    ) -> List[Dict]:
        """
        Returns per-email prediction with:
          - final_label: 0 (legitimate) or 1 (phishing/spam)
          - confidence: float [0,1]
          - ml_score: raw ML probability
          - rule_score: rule engine score
          - hybrid_score: blended score
          - triggered_rules: list of matched rules
          - explanations: human-readable reasons
        """
        ml_probas = self.ml_model.predict_proba(features)[:, 1]

        results = []
        for text, ml_proba in zip(texts, ml_probas):
            rule_result = self.rule_engine.evaluate(text)
            rule_score = rule_result["rule_score"]

            hybrid_score = (1 - self.alpha) * ml_proba + self.alpha * rule_score
            final_label = int(hybrid_score >= self.threshold)

            results.append({
                "final_label": final_label,
                "confidence": round(hybrid_score, 4),
                "ml_score": round(float(ml_proba), 4),
                "rule_score": round(rule_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "triggered_rules": rule_result["triggered_rules"],
                "explanations": rule_result["explanations"],
                "verdict": "PHISHING/SPAM" if final_label else "LEGITIMATE",
                "risk_level": _risk_level(hybrid_score),
            })

        return results


def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "SAFE"


# ─────────────────────────────────────────────
# Model comparison orchestrator
# ─────────────────────────────────────────────

class ModelComparison:
    """
    Train and evaluate all models, return a comparison DataFrame.

    Usage:
        comp = ModelComparison()
        results_df = comp.run(X_train, X_test, y_train, y_test)
        comp.print_summary()
    """

    def __init__(self, save_dir: str = "models/saved"):
        self.save_dir = save_dir
        self.models = {
            "logistic_regression": build_logistic_regression(),
            "naive_bayes": build_naive_bayes(),
            "random_forest": build_random_forest(),
            "linear_svm": build_linear_svm(),
            "gradient_boosting": build_gradient_boosting(),
        }
        self.results: List[Dict] = []
        self.best_model: Optional[BaseEmailClassifier] = None

    def run(
        self,
        X_train, X_test, y_train, y_test,
        metric: str = "f1",
    ) -> pd.DataFrame:
        logger.info(f"Running comparison across {len(self.models)} models")
        self.results = []

        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                report = model.score_report(X_test, y_test)
                self.results.append(report)
                logger.info(
                    f"[{name}] F1={report['f1']:.4f} "
                    f"ROC-AUC={report['roc_auc']:.4f} "
                    f"Recall={report['recall']:.4f}"
                )
                model.save()
            except Exception as exc:
                logger.error(f"[{name}] failed: {exc}")

        df = pd.DataFrame(self.results)
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)

        best_name = df.iloc[0]["model"]
        self.best_model = next(
            m for m in self.models.values() if m.name == best_name
        )
        logger.info(f"Best model by {metric}: {best_name}")
        return df

    def print_summary(self):
        if not self.results:
            print("No results yet. Call run() first.")
            return
        df = pd.DataFrame(self.results)[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s"]
        ]
        df = df.sort_values("f1", ascending=False)
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(df.to_string(index=False, float_format="{:.4f}".format))
        print("=" * 80)

        # Security context note
        print("\n⚠  CYBERSECURITY NOTE:")
        print("  In email security, RECALL is often more critical than PRECISION.")
        print("  A missed phishing email (FN) costs more than a false alarm (FP).")
        print(f"  Best Recall model: {pd.DataFrame(self.results).sort_values('recall').iloc[-1]['model']}")
        print(f"  Best F1 model:     {pd.DataFrame(self.results).sort_values('f1').iloc[-1]['model']}")
