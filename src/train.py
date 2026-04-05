"""
src/train.py
-------------
End-to-end training pipeline.

Steps:
  1. Load and combine datasets
  2. Preprocess text
  3. Feature engineering (TF-IDF + handcrafted)
  4. Handle class imbalance
  5. Train and compare all models
  6. Evaluate best model
  7. Save artifacts

Run: python src/train.py [--mode tfidf|bert] [--sample 50000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import DatasetLoader
from src.preprocessing.feature_engineering import FeaturePipeline, handle_imbalance
from src.models.classifier import ModelComparison, HybridClassifier
from src.evaluation.evaluator import SecurityEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train phishing detection models")
    parser.add_argument("--mode", default="tfidf", choices=["tfidf", "bert"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--imbalance", default="class_weight",
                        choices=["smote", "class_weight", "none"])
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Classification threshold (lower = higher recall)")
    parser.add_argument("--save-dir", default="models/saved")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PHISHING DETECTION TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load data ──
    loader = DatasetLoader()
    df = loader.load_combined(sample_n=args.sample)

    logger.info(f"Dataset: {len(df):,} emails")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")

    X_train, X_val, X_test, y_train, y_val, y_test = loader.split(df)

    # ── 2. Feature engineering ──
    logger.info(f"Building feature pipeline (mode={args.mode})...")
    feature_pipeline = FeaturePipeline(
        mode=args.mode,
        include_handcrafted=True,
        cache_dir=args.save_dir,
    )

    X_train_feats = feature_pipeline.fit_transform(X_train.tolist())
    X_val_feats   = feature_pipeline.transform(X_val.tolist())
    X_test_feats  = feature_pipeline.transform(X_test.tolist())

    logger.info(f"Feature matrix shape: {X_train_feats.shape}")

    # ── 3. Handle imbalance ──
    if args.imbalance not in ("none", "class_weight"):
        X_train_feats, y_train_arr = handle_imbalance(
            X_train_feats, y_train.values, strategy=args.imbalance
        )
    else:
        y_train_arr = y_train.values

    # ── 4. Train all models ──
    comparison = ModelComparison(save_dir=args.save_dir)
    results_df = comparison.run(
        X_train_feats, X_test_feats,
        y_train_arr, y_test.values,
    )
    comparison.print_summary()

    # ── 5. Save feature pipeline ──
    feature_pipeline.save()

    # ── 6. Detailed evaluation of best model ──
    best = comparison.best_model
    evaluator = SecurityEvaluator(threshold=args.threshold)

    y_proba = best.predict_proba(X_test_feats)[:, 1]
    metrics = evaluator.evaluate(y_proba=y_proba, y_true=y_test.values, model_name=best.name)

    # Threshold optimisation
    opt_t, threshold_df = evaluator.find_optimal_threshold(
        y_test.values, y_proba, objective="f1"
    )
    logger.info(f"Optimal F1 threshold: {opt_t}")

    # ── 7. Save plots ──
    logger.info("Generating evaluation plots...")
    evaluator.plot_confusion_matrix(y_test.values, y_proba, best.name)
    evaluator.plot_threshold_analysis(y_test.values, y_proba, best.name)

    all_results = []
    for name, model in comparison.models.items():
        if model.is_fitted:
            proba = model.predict_proba(X_test_feats)[:, 1]
            all_results.append((model.name, y_test.values, proba))
    evaluator.plot_roc_pr_curves(all_results)
    evaluator.plot_model_comparison_bar(results_df)

    # ── 8. Summary ──
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best model: {best.name}")
    logger.info(f"F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f} | FNR: {metrics['false_negative_rate']:.4f}")
    logger.info(f"Models saved to: {args.save_dir}/")
    logger.info(f"Plots saved to: reports/figures/")
    logger.info("=" * 60)

    print("\n✅ Next steps:")
    print("  1. Start API:      uvicorn api.main:app --reload")
    print("  2. Start frontend: streamlit run app/streamlit_app.py")
    print("  3. View docs:      http://localhost:8000/docs")


if __name__ == "__main__":
    main()
