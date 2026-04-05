"""
src/evaluation/evaluator.py
-----------------------------
Comprehensive evaluation suite for email security classifiers.

Includes:
  - All standard classification metrics
  - Confusion matrix with cost analysis
  - ROC and PR curves
  - Per-threshold analysis for security tuning
  - Model calibration plots
  - Why each metric matters in cybersecurity
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


PLOT_DIR = Path("reports/figures")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


class SecurityEvaluator:
    """
    Production-grade evaluator tailored for email security use cases.

    Key philosophy:
      - FN (missed phishing) > FP (false alarm) in most enterprise settings
      - Threshold tuning is critical: default 0.5 is rarely optimal
      - Calibration matters for SIEM integrations (need real probabilities)
    """

    def __init__(self, threshold: float = 0.5, cost_fn: float = 10.0, cost_fp: float = 1.0):
        """
        Args:
            threshold: Classification threshold (tune to balance FP/FN)
            cost_fn:   Business cost of a missed phishing email (False Negative)
            cost_fp:   Business cost of blocking a legitimate email (False Positive)
        """
        self.threshold = threshold
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    # ──────────────────────────────────────────
    # Core evaluation
    # ──────────────────────────────────────────

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        verbose: bool = True,
    ) -> Dict:
        """
        Full evaluation suite.

        Returns dict with all metrics + cost analysis.
        """
        y_pred = (y_proba >= self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "model": model_name,
            "threshold": self.threshold,

            # Standard metrics
            "accuracy":   accuracy_score(y_true, y_pred),
            "precision":  precision_score(y_true, y_pred, zero_division=0),
            "recall":     recall_score(y_true, y_pred, zero_division=0),
            "f1":         f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":    roc_auc_score(y_true, y_proba),
            "avg_precision": average_precision_score(y_true, y_proba),  # AUPRC

            # Confusion matrix raw
            "true_negatives":  int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives":  int(tp),

            # Derived rates
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "specificity":   tn / (tn + fp) if (tn + fp) > 0 else 0,
            "miss_rate":     fn / (fn + tp) if (fn + tp) > 0 else 0,

            # Calibration
            "brier_score": brier_score_loss(y_true, y_proba),

            # Business cost
            "total_cost":  fn * self.cost_fn + fp * self.cost_fp,
        }

        if verbose:
            self._print_report(metrics, y_true, y_pred)

        return metrics

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        objective: str = "f1",   # "f1" | "recall" | "cost"
    ) -> Tuple[float, pd.DataFrame]:
        """
        Sweep thresholds from 0.1 to 0.9 to find the optimal operating point.

        Returns: (optimal_threshold, threshold_df)
        """
        thresholds = np.arange(0.05, 0.96, 0.025)
        rows = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rows.append({
                "threshold": round(t, 3),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall":    recall_score(y_true, y_pred, zero_division=0),
                "f1":        f1_score(y_true, y_pred, zero_division=0),
                "fpr":       fp / (fp + tn) if (fp + tn) > 0 else 0,
                "fnr":       fn / (fn + tp) if (fn + tp) > 0 else 0,
                "cost":      fn * self.cost_fn + fp * self.cost_fp,
            })

        df = pd.DataFrame(rows)

        if objective == "cost":
            optimal_t = df.loc[df["cost"].idxmin(), "threshold"]
        elif objective == "recall":
            # Max recall while keeping precision > 0.5
            filtered = df[df["precision"] >= 0.5]
            optimal_t = filtered.loc[filtered["recall"].idxmax(), "threshold"] if len(filtered) else 0.3
        else:  # f1
            optimal_t = df.loc[df["f1"].idxmax(), "threshold"]

        logger.info(f"Optimal threshold ({objective}): {optimal_t}")
        return float(optimal_t), df

    # ──────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True,
    ) -> plt.Figure:
        y_pred = (y_proba >= self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing/Spam"],
            yticklabels=["Legitimate", "Phishing/Spam"],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

        tn, fp, fn, tp = cm.ravel()
        ax.text(
            0.5, -0.18,
            f"FPR: {fp/(fp+tn):.2%}  |  FNR: {fn/(fn+tp):.2%}  |  "
            f"Business Cost: {fn*self.cost_fn + fp*self.cost_fp:.0f}",
            ha="center", transform=ax.transAxes, fontsize=10, color="gray"
        )

        plt.tight_layout()
        if save:
            path = PLOT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        return fig

    def plot_roc_pr_curves(
        self,
        results: List[Tuple[str, np.ndarray, np.ndarray]],
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot ROC and PR curves for multiple models side by side.

        Args:
            results: list of (model_name, y_true, y_proba) tuples
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = plt.cm.Set2(np.linspace(0, 0.8, len(results)))

        for (name, y_true, y_proba), color in zip(results, colors):
            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            ax1.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

            # PR
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            ax2.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2)

        # ROC formatting
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve", fontweight="bold")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.annotate(
            "⚠  Low FPR critical in\nenterprise deployments",
            xy=(0.05, 0.05), fontsize=8, color="gray"
        )

        # PR formatting
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve\n(better for imbalanced data)", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.annotate(
            "Higher area = better\n(random baseline ≈ class prevalence)",
            xy=(0.4, 0.05), fontsize=8, color="gray"
        )

        plt.suptitle("Model Comparison — ROC & PR Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = PLOT_DIR / "roc_pr_curves.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        return fig

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True,
    ) -> plt.Figure:
        _, df = self.find_optimal_threshold(y_true, y_proba, objective="f1")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(df["threshold"], df["precision"], label="Precision", color="steelblue", lw=2)
        ax1.plot(df["threshold"], df["recall"],    label="Recall",    color="coral", lw=2)
        ax1.plot(df["threshold"], df["f1"],        label="F1",        color="green", lw=2.5)
        ax1.axvline(self.threshold, color="gray", linestyle="--", alpha=0.6, label=f"Current ({self.threshold})")
        ax1.set_xlabel("Decision Threshold")
        ax1.set_ylabel("Score")
        ax1.set_title(f"Threshold vs Metrics — {model_name}", fontweight="bold")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(df["threshold"], df["cost"], color="darkred", lw=2)
        optimal_t = df.loc[df["cost"].idxmin(), "threshold"]
        ax2.axvline(optimal_t, color="green", linestyle="--", label=f"Min cost @ {optimal_t}")
        ax2.set_xlabel("Decision Threshold")
        ax2.set_ylabel(f"Business Cost (FN×{self.cost_fn} + FP×{self.cost_fp})")
        ax2.set_title("Business Cost vs Threshold", fontweight="bold")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save:
            path = PLOT_DIR / f"threshold_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def plot_model_comparison_bar(self, results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(results_df))
        width = 0.15
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(metrics)))

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(
                x + i * width,
                results_df[metric],
                width,
                label=metric.upper(),
                color=color,
                alpha=0.85,
            )
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.003,
                    f"{height:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45
                )

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(results_df["model"], rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save:
            path = PLOT_DIR / "model_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _print_report(self, metrics: Dict, y_true, y_pred):
        print(f"\n{'='*60}")
        print(f"  EVALUATION REPORT: {metrics['model']}")
        print(f"{'='*60}")
        print(f"  Threshold: {metrics['threshold']}")
        print(f"\n  STANDARD METRICS")
        print(f"  {'Accuracy':20s}: {metrics['accuracy']:.4f}")
        print(f"  {'Precision':20s}: {metrics['precision']:.4f}  ← When flagged, how often correct?")
        print(f"  {'Recall':20s}: {metrics['recall']:.4f}  ← What % of threats caught?  ⚠")
        print(f"  {'F1 Score':20s}: {metrics['f1']:.4f}")
        print(f"  {'ROC-AUC':20s}: {metrics['roc_auc']:.4f}")
        print(f"  {'Avg Precision':20s}: {metrics['avg_precision']:.4f}  (AUPRC)")
        print(f"\n  SECURITY METRICS")
        print(f"  {'FPR':20s}: {metrics['false_positive_rate']:.4f}  ← Legitimate emails blocked")
        print(f"  {'FNR (Miss Rate)':20s}: {metrics['false_negative_rate']:.4f}  ← Threats missed  ⚠")
        print(f"\n  CONFUSION MATRIX")
        print(f"  TP: {metrics['true_positives']:6d}  FP: {metrics['false_positives']:6d}")
        print(f"  FN: {metrics['false_negatives']:6d}  TN: {metrics['true_negatives']:6d}")
        print(f"\n  BUSINESS COST")
        print(f"  Total: {metrics['total_cost']:.0f}  (FN×{self.cost_fn} + FP×{self.cost_fp})")
        print(f"{'='*60}\n")
        print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing/Spam"]))
