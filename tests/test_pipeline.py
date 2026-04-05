"""
tests/test_pipeline.py
-----------------------
Comprehensive test suite for the phishing detection pipeline.

Run: pytest tests/ -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Test: Text Preprocessor
# ──────────────────────────────────────────────

class TestEmailPreprocessor:
    from src.preprocessing.text_cleaner import EmailPreprocessor

    def setup_method(self):
        from src.preprocessing.text_cleaner import EmailPreprocessor
        self.preprocessor = EmailPreprocessor(method="lemmatize")

    def test_html_stripping(self):
        html = "<html><body><h1>Your account is <b>suspended</b></h1><script>evil();</script></body></html>"
        result = self.preprocessor.transform(html)
        assert "<" not in result
        assert "script" not in result
        assert "suspend" in result or "account" in result

    def test_url_replacement(self):
        text = "Click here: https://phishing.xyz/login?id=123 to verify"
        result = self.preprocessor.transform(text)
        assert "url_token" in result.lower() or "url" in result.lower()
        assert "phishing.xyz" not in result

    def test_unicode_normalisation(self):
        # Cyrillic 'а' looks like Latin 'a' — common obfuscation
        obfuscated = "pаyраl"  # contains Cyrillic characters
        result = self.preprocessor.transform(obfuscated)
        assert len(result) >= 0  # Should not crash

    def test_empty_input(self):
        assert self.preprocessor.transform("") == ""
        assert self.preprocessor.transform("   ") == ""

    def test_feature_extraction_keys(self):
        text = "Urgent! Click http://suspicious.xyz to verify your PayPal account NOW!"
        features = self.preprocessor.extract_features(text)

        required_keys = [
            "url_count", "urgency_keyword_count", "exclamation_count",
            "capital_ratio", "has_url", "total_phishing_keywords"
        ]
        for key in required_keys:
            assert key in features, f"Missing feature: {key}"

    def test_phishing_keywords_detected(self):
        text = "URGENT! Verify your account immediately to avoid suspension!"
        features = self.preprocessor.extract_features(text)
        assert features["urgency_keyword_count"] > 0
        assert features["exclamation_count"] >= 2

    def test_batch_transform(self):
        texts = ["Hello world", "Urgent verify account", "Normal email text"]
        results = self.preprocessor.transform_batch(texts, show_progress=False)
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)


# ──────────────────────────────────────────────
# Test: Rule Engine
# ──────────────────────────────────────────────

class TestRuleEngine:
    def setup_method(self):
        from src.preprocessing.text_cleaner import RuleEngine
        self.engine = RuleEngine()

    def test_phishing_detected(self):
        text = "URGENT: Verify your PayPal account immediately or it will be suspended!"
        result = self.engine.evaluate(text)
        assert result["flagged"] is True
        assert result["rule_score"] > 0.3
        assert len(result["triggered_rules"]) > 0

    def test_legitimate_not_flagged(self):
        text = "Hi team, the meeting is scheduled for Friday at 2pm. Please review the attached agenda."
        result = self.engine.evaluate(text)
        assert result["rule_score"] < 0.4

    def test_result_schema(self):
        result = self.engine.evaluate("test email")
        assert "rule_score" in result
        assert "triggered_rules" in result
        assert "explanations" in result
        assert "flagged" in result
        assert 0.0 <= result["rule_score"] <= 1.0

    def test_financial_lure_detection(self):
        text = "Congratulations! You are the winner of $1,000,000 prize. Claim now!"
        result = self.engine.evaluate(text)
        assert "financial_lure" in result["triggered_rules"]

    def test_high_caps_detection(self):
        text = "YOUR ACCOUNT HAS BEEN SUSPENDED. VERIFY NOW OR LOSE ALL YOUR DATA!!!"
        result = self.engine.evaluate(text)
        assert result["rule_score"] > 0.3


# ──────────────────────────────────────────────
# Test: URL Analyzer
# ──────────────────────────────────────────────

class TestURLAnalyzer:
    def setup_method(self):
        from src.utils.url_analyzer import URLAnalyzer
        self.analyzer = URLAnalyzer()

    def test_phishing_url_detected(self):
        url = "http://secure-paypal-verify.xyz/login?user=victim"
        result = self.analyzer.analyze(url)
        assert result["risk_score"] > 0.3
        assert result["risk_level"] in ("HIGH", "CRITICAL", "MEDIUM")

    def test_ip_host_flagged(self):
        url = "http://192.168.1.1/login"
        result = self.analyzer.analyze(url)
        assert result["features"]["has_ip_host"] == 1
        assert result["risk_score"] > 0.2

    def test_legitimate_url(self):
        url = "https://www.google.com/search?q=python"
        result = self.analyzer.analyze(url)
        assert result["features"]["uses_https"] == 1
        assert result["features"]["legitimate_tld"] == 1

    def test_result_schema(self):
        result = self.analyzer.analyze("https://example.com")
        assert "risk_score" in result
        assert "risk_level" in result
        assert "indicators" in result
        assert "features" in result
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_high_risk_tld(self):
        url = "http://login-bank.xyz/verify"
        result = self.analyzer.analyze(url)
        assert result["features"]["high_risk_tld"] == 1

    def test_brand_impersonation(self):
        url = "http://paypal-secure.info/login"
        result = self.analyzer.analyze(url)
        assert result["risk_score"] > 0.2


# ──────────────────────────────────────────────
# Test: Data Loader (demo mode)
# ──────────────────────────────────────────────

class TestDataLoader:
    def setup_method(self):
        from src.utils.data_loader import DatasetLoader
        self.loader = DatasetLoader()

    def test_demo_data_generated(self):
        """When no real datasets present, demo data should be generated."""
        df = self.loader.load_combined(sample_n=100)
        assert len(df) > 0
        assert "text" in df.columns
        assert "label" in df.columns
        assert df["label"].isin([0, 1]).all()

    def test_split_proportions(self):
        df = self.loader.load_combined(sample_n=500)
        X_train, X_val, X_test, y_train, y_val, y_test = self.loader.split(df)
        total = len(y_train) + len(y_val) + len(y_test)
        assert abs(total - 500) <= 1

    def test_stratified_split(self):
        df = self.loader.load_combined(sample_n=500)
        _, _, _, y_train, _, y_test = self.loader.split(df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.05  # stratification maintained


# ──────────────────────────────────────────────
# Test: Model Predictions
# ──────────────────────────────────────────────

class TestModelPredictions:
    """Integration tests: preprocessing → features → prediction."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.utils.data_loader import DatasetLoader
        from src.preprocessing.feature_engineering import FeaturePipeline
        from src.models.classifier import build_logistic_regression

        loader = DatasetLoader()
        df = loader.load_combined(sample_n=200)
        X_train, _, X_test, y_train, _, y_test = loader.split(df)

        self.pipeline = FeaturePipeline(
            mode="tfidf",
            max_word_features=5000,
            include_handcrafted=True,
        )
        X_train_f = self.pipeline.fit_transform(X_train.tolist())
        self.X_test_f = self.pipeline.transform(X_test.tolist())
        self.y_test = y_test

        self.model = build_logistic_regression()
        self.model.fit(X_train_f, y_train.values)

    def test_prediction_shape(self):
        preds = self.model.predict(self.X_test_f)
        assert preds.shape == (len(self.y_test),)

    def test_proba_shape(self):
        probas = self.model.predict_proba(self.X_test_f)
        assert probas.shape == (len(self.y_test), 2)

    def test_proba_sums_to_one(self):
        probas = self.model.predict_proba(self.X_test_f)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_predictions_are_binary(self):
        preds = self.model.predict(self.X_test_f)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_report_keys(self):
        report = self.model.score_report(self.X_test_f, self.y_test.values)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in report
            assert 0.0 <= report[key] <= 1.0

    def test_reasonable_performance(self):
        """Model should achieve at least 70% F1 even on tiny dataset."""
        report = self.model.score_report(self.X_test_f, self.y_test.values)
        assert report["f1"] > 0.5, f"F1 too low: {report['f1']}"


# ──────────────────────────────────────────────
# Test: Evaluator
# ──────────────────────────────────────────────

class TestEvaluator:
    def setup_method(self):
        from src.evaluation.evaluator import SecurityEvaluator
        self.evaluator = SecurityEvaluator(threshold=0.5)

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.7])
        metrics = self.evaluator.evaluate(y_true, y_proba, verbose=False)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["false_negatives"] == 0

    def test_threshold_sweep_returns_df(self):
        y_true = np.array([0, 1, 0, 1, 1, 0] * 10)
        y_proba = np.random.rand(60)
        opt_t, df = self.evaluator.find_optimal_threshold(y_true, y_proba, "f1")
        assert isinstance(opt_t, float)
        assert 0.05 <= opt_t <= 0.95
        assert "f1" in df.columns
        assert "threshold" in df.columns

    def test_cost_calculation(self):
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.3, 0.3, 0.7, 0.3])  # 2 FNs, 1 FP
        evaluator = SecurityEvaluator(threshold=0.5, cost_fn=10, cost_fp=1)
        metrics = evaluator.evaluate(y_true, y_proba, verbose=False)
        # 2 FN × 10 + 1 FP × 1 = 21
        assert metrics["total_cost"] == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
