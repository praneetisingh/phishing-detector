"""
src/explainability/explainer.py
---------------------------------
Model explainability using SHAP and LIME.
Critical for security analysts who need to understand WHY an email
was flagged — not just that it was.

Supports:
  - SHAP values for tree and linear models
  - LIME for any black-box model
  - Feature importance plots
  - Per-email explanation with top contributing features
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

warnings.filterwarnings("ignore")

PLOT_DIR = Path("reports/figures")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


class ModelExplainer:
    """
    Unified explainability interface for phishing detection models.

    Example:
        explainer = ModelExplainer(model, feature_names, X_train_sample)
        explanation = explainer.explain_instance(email_text, email_features)
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        X_background: np.ndarray,
        method: str = "auto",  # "shap" | "lime" | "auto"
    ):
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background
        self.method = method
        self._shap_explainer = None
        self._lime_explainer = None

    # ──────────────────────────────────────────
    # SHAP explainability
    # ──────────────────────────────────────────

    def _init_shap(self):
        if self._shap_explainer is not None:
            return
        try:
            import shap
            estimator = self.model.estimator

            # Choose the right SHAP explainer based on model type
            model_type = type(estimator).__name__

            if "Forest" in model_type or "Boosting" in model_type or "Tree" in model_type:
                self._shap_explainer = shap.TreeExplainer(estimator)
                logger.info("Initialised SHAP TreeExplainer")

            elif "Logistic" in model_type or "SVC" in model_type or "LinearSVC" in model_type:
                # Linear models: use LinearExplainer
                # For calibrated classifiers, extract underlying estimator
                if hasattr(estimator, "calibrated_classifiers_"):
                    inner = estimator.calibrated_classifiers_[0].base_estimator
                else:
                    inner = estimator
                if hasattr(inner, "coef_"):
                    self._shap_explainer = shap.LinearExplainer(
                        inner, self.X_background, feature_perturbation="interventional"
                    )
                else:
                    self._shap_explainer = shap.KernelExplainer(
                        lambda x: estimator.predict_proba(x)[:, 1],
                        shap.sample(self.X_background, 100),
                    )
                logger.info("Initialised SHAP LinearExplainer")

            else:
                # Fallback: KernelSHAP (model-agnostic, slower)
                self._shap_explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1],
                    shap.sample(self.X_background, 100),
                )
                logger.info("Initialised SHAP KernelExplainer (model-agnostic)")

        except ImportError:
            logger.warning("SHAP not installed. pip install shap")

    def compute_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        self._init_shap()
        if self._shap_explainer is None:
            return None
        try:
            import shap
            shap_values = self._shap_explainer.shap_values(X)
            # For binary classifiers, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return shap_values
        except Exception as exc:
            logger.error(f"SHAP computation failed: {exc}")
            return None

    def plot_shap_summary(
        self,
        X: np.ndarray,
        max_features: int = 20,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Global feature importance via SHAP beeswarm plot."""
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return None

        try:
            import shap
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names,
                max_display=max_features,
                show=False,
                plot_type="dot",
            )
            plt.title("SHAP Feature Importance — Global", fontweight="bold")
            plt.tight_layout()

            if save:
                path = PLOT_DIR / "shap_summary.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved: {path}")
            return fig
        except Exception as exc:
            logger.error(f"SHAP plot failed: {exc}")
            return None

    def plot_shap_waterfall(
        self,
        X_single: np.ndarray,
        instance_idx: int = 0,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Single-instance explanation: waterfall plot."""
        shap_values = self.compute_shap_values(X_single[instance_idx:instance_idx+1])
        if shap_values is None:
            return None

        try:
            import shap
            vals = shap_values[0]
            # Sort by absolute SHAP value
            sorted_idx = np.argsort(np.abs(vals))[::-1][:20]
            top_features = [self.feature_names[i] for i in sorted_idx]
            top_values = vals[sorted_idx]

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ["#d73027" if v > 0 else "#4575b4" for v in top_values]
            ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title(
                f"SHAP Waterfall — Instance {instance_idx}\n"
                "Red = pushes toward PHISHING | Blue = pushes toward LEGITIMATE",
                fontweight="bold"
            )
            plt.tight_layout()

            if save:
                path = PLOT_DIR / f"shap_waterfall_{instance_idx}.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
            return fig
        except Exception as exc:
            logger.error(f"SHAP waterfall failed: {exc}")
            return None

    # ──────────────────────────────────────────
    # LIME explainability
    # ──────────────────────────────────────────

    def _init_lime(self):
        if self._lime_explainer is not None:
            return
        try:
            from lime.lime_text import LimeTextExplainer
            self._lime_explainer = LimeTextExplainer(
                class_names=["Legitimate", "Phishing/Spam"],
                random_state=42,
            )
            logger.info("Initialised LIME TextExplainer")
        except ImportError:
            logger.warning("LIME not installed. pip install lime")

    def explain_instance_lime(
        self,
        raw_text: str,
        predict_fn,
        num_features: int = 15,
        num_samples: int = 1000,
    ) -> Optional[Dict]:
        """
        Generate LIME explanation for a single email.

        Args:
            raw_text: The email text (raw, not preprocessed)
            predict_fn: Function(texts) → proba array (N, 2)
            num_features: Number of top features to explain

        Returns:
            dict with feature_weights, html_explanation, text_explanation
        """
        self._init_lime()
        if self._lime_explainer is None:
            return None

        try:
            exp = self._lime_explainer.explain_instance(
                raw_text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            feature_weights = exp.as_list()
            positive = [(f, w) for f, w in feature_weights if w > 0]
            negative = [(f, w) for f, w in feature_weights if w <= 0]

            return {
                "feature_weights": feature_weights,
                "positive_features": positive,    # push toward phishing
                "negative_features": negative,    # push toward legitimate
                "predicted_class": exp.predict_proba.argmax(),
                "probability": exp.predict_proba[1],
                "html": exp.as_html(),
            }
        except Exception as exc:
            logger.error(f"LIME explanation failed: {exc}")
            return None

    def plot_lime_explanation(
        self,
        explanation: Dict,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        if not explanation:
            return None

        feature_weights = explanation["feature_weights"]
        words = [fw[0] for fw in feature_weights]
        weights = [fw[1] for fw in feature_weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d73027" if w > 0 else "#4575b4" for w in weights]
        ax.barh(words[::-1], weights[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Feature Weight")
        ax.set_title(
            f"LIME Explanation — Phishing Probability: {explanation['probability']:.1%}\n"
            "Red = evidence of PHISHING | Blue = evidence of LEGITIMATE",
            fontweight="bold",
        )
        plt.tight_layout()

        if save:
            path = PLOT_DIR / "lime_explanation.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # Feature importance (for tree/linear models)
    # ──────────────────────────────────────────

    def plot_feature_importance(
        self,
        top_n: int = 30,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot top-N most important features.
        Works for RandomForest (feature_importances_) and
        LogisticRegression (coef_).
        """
        estimator = self.model.estimator
        importances = None

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            importance_type = "Feature Importance (Gini)"

        elif hasattr(estimator, "coef_"):
            # LR: absolute coefficients for phishing class
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importances = np.abs(coef)
            importance_type = "|Coefficient| (Logistic Regression)"

        elif hasattr(estimator, "calibrated_classifiers_"):
            # CalibratedClassifierCV wrapping LinearSVC
            coefs = [
                cc.base_estimator.coef_[0]
                for cc in estimator.calibrated_classifiers_
                if hasattr(cc.base_estimator, "coef_")
            ]
            if coefs:
                importances = np.abs(np.mean(coefs, axis=0))
                importance_type = "|Coefficient| (LinearSVC)"

        if importances is None:
            logger.warning("Model doesn't support direct feature importance")
            return None

        # Top-N features
        n = min(top_n, len(importances))
        top_idx = np.argsort(importances)[::-1][:n]
        top_features = [self.feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]

        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.3)))
        colors = plt.cm.RdYlGn_r(top_importances / top_importances.max())
        ax.barh(top_features[::-1], top_importances[::-1], color=colors[::-1])
        ax.set_xlabel(importance_type)
        ax.set_title(f"Top {n} Features — {self.model.name}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        if save:
            path = PLOT_DIR / f"feature_importance_{self.model.name.lower().replace(' ', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def get_top_phishing_features(self, top_n: int = 20) -> pd.DataFrame:
        """Return DataFrame of top features driving phishing predictions."""
        estimator = self.model.estimator
        importances = None

        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                importances = coef[0]  # signed — positive = phishing
            else:
                importances = coef
        elif hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_

        if importances is None:
            return pd.DataFrame()

        n = min(top_n, len(importances))
        top_idx = np.argsort(importances)[::-1][:n]

        return pd.DataFrame({
            "feature": [self.feature_names[i] for i in top_idx],
            "importance": importances[top_idx],
            "direction": ["phishing" if v > 0 else "legitimate" for v in importances[top_idx]],
        })
