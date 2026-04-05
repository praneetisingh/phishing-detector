# Intelligent Suspicious Email Detection System

**Production-grade phishing and spam classifier — Hybrid ML + Rule Engine**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest-blueviolet.svg)](tests/)

---

## Problem Statement

Phishing and spam emails are the entry point for the majority of enterprise cyberattacks. Traditional rule-based filters are bypassed easily by modern adversarial techniques. This project implements a hybrid detection system that combines a trained ML classifier with a deterministic rule engine, so neither approach's weaknesses dominate.

**Real-world use cases this system addresses:**

| Use Case | How the system helps |
|----------|---------------------|
| Enterprise email gateway | Pre-screen inbound email before delivery |
| SOC analyst triage | Prioritise queue with per-email confidence scores and triggered rules |
| Business email compromise (BEC) | Detect credential-harvesting and impersonation patterns |
| Security awareness tooling | Show analysts and employees *why* an email was flagged |

---

## How It Works

The system has two parallel scoring paths that are blended at inference time.

**ML path** — The raw email text is cleaned (HTML stripped, URLs replaced with a token, unicode normalised, lemmatised), then transformed into a feature matrix combining word and character-level TF-IDF with 25 handcrafted cybersecurity-specific numeric features (URL count, capital ratio, urgency keyword count, exclamation count, etc.). A trained classifier produces a probability score.

**Rule engine path** — The same raw text is scanned against three keyword lexicons (urgency language, financial lures, credential-harvesting phrases) plus structural heuristics (URL count, capitalisation ratio, exclamation density). This produces a rule score between 0 and 1.

**Hybrid scorer** — The two scores are blended with a configurable weight `alpha`:

```
hybrid_score = (1 - alpha) * ml_score + alpha * rule_score
```

The default `alpha` is `0.2`, giving the ML model 80% of the weight while the rule engine acts as a fast secondary signal for obvious cases. The final label is `1` (phishing/spam) when `hybrid_score >= threshold` (default `0.5`, tunable at training time).

Every prediction returns the hybrid score, the raw ML score, the raw rule score, the triggered rule names, and human-readable explanations — so analysts know exactly what drove the decision.

---

## Architecture

```
Raw email (subject + body + sender)
            │
            ▼
    ┌───────────────────────────────────────────┐
    │           Preprocessing pipeline          │
    │  HTML strip → URL tokenise → unicode norm │
    │  Lemmatise → stopword removal             │
    └───────────────────────────────────────────┘
            │                       │
            ▼                       ▼
  ┌──────────────────┐    ┌───────────────────┐
  │  Feature matrix  │    │   Rule engine     │
  │  TF-IDF (word +  │    │  Urgency keywords │
  │  char n-gram) +  │    │  Financial lures  │
  │  25 handcrafted  │    │  Credential terms │
  │  features        │    │  + heuristics     │
  └──────────────────┘    └───────────────────┘
            │                       │
            ▼                       ▼
       ML classifier          Rule score [0,1]
       probability [0,1]
            │                       │
            └──────────┬────────────┘
                       ▼
              Hybrid scorer
       (1-α)×ml_score + α×rule_score
                       │
                       ▼
          Prediction + confidence +
          risk level + explanations
```

---

## Datasets

No datasets are bundled. The loader expects CSV files downloaded manually from the sources below. If no files are present, the system generates synthetic demo data so the pipeline runs end-to-end without them.

| Dataset | Approximate size | Source | Labels |
|---------|-----------------|--------|--------|
| Enron Spam | ~30,000 emails | [Kaggle — wanderfj/enron-spam](https://www.kaggle.com/datasets/wanderfj/enron-spam) | spam / ham |
| SpamAssassin Public Corpus | ~6,000 emails | [spamassassin.apache.org/old/publiccorpus](https://spamassassin.apache.org/old/publiccorpus/) | spam / ham |
| Phishing Email Dataset | ~18,000 emails | [Kaggle — naserabdullahalam/phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) | phishing / safe |
| Ling-Spam | ~2,900 emails | [Kaggle — mandygu/lingspam-dataset](https://www.kaggle.com/datasets/mandygu/lingspam-dataset) | spam / ham |

Place downloaded CSVs at the paths listed in `src/utils/data_loader.py` (`data/raw/enron_spam.csv`, etc.) before running training.

---

## Models

Five scikit-learn classifiers are trained and compared automatically. Two deep learning options are included but require separate setup.

**Scikit-learn (trained by default):**

| Model | Class used | Notes |
|-------|-----------|-------|
| Logistic Regression | `LogisticRegression(solver="saga", class_weight="balanced")` | Strong baseline; most interpretable via coefficients |
| Complement Naive Bayes | `ComplementNB(alpha=0.1)` | Fastest inference; better than MultinomialNB on imbalanced data |
| Random Forest | `RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample")` | Best native feature importance for SHAP |
| Linear SVM | `LinearSVC` wrapped in `CalibratedClassifierCV` | Typically highest accuracy on sparse TF-IDF |
| Gradient Boosting | `GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)` | Works well on the handcrafted numeric features |

**Deep learning (optional, not trained by default):**

| Model | File | Requirement |
|-------|------|-------------|
| Bidirectional LSTM with attention | `src/models/classifier.py — LSTMEmailClassifier` | PyTorch |
| DistilBERT fine-tune (inference only) | `src/models/classifier.py — BERTEmailClassifier` | `transformers`, GPU recommended |

`ModelComparison` in `classifier.py` trains all five sklearn models, scores each on the held-out test set, prints a comparison table, and saves the best model by F1 to `models/saved/`.

---

## Project Structure

```
phishing-detector/
├── data/
│   ├── raw/                         # Place downloaded dataset CSVs here
│   └── processed/                   # Reserved for intermediate outputs
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py          # EmailPreprocessor + RuleEngine
│   │   └── feature_engineering.py  # FeaturePipeline (TF-IDF + handcrafted)
│   ├── models/
│   │   └── classifier.py            # All models + HybridClassifier + ModelComparison
│   ├── evaluation/
│   │   └── evaluator.py             # SecurityEvaluator: metrics, plots, threshold sweep
│   ├── explainability/
│   │   └── explainer.py             # ModelExplainer: SHAP + LIME
│   ├── utils/
│   │   ├── data_loader.py           # DatasetLoader: multi-source load + train/val/test split
│   │   └── url_analyzer.py          # URLAnalyzer + EmailHeaderAnalyzer
│   └── train.py                     # End-to-end training script (CLI)
├── api/
│   └── main.py                      # FastAPI app
├── app/
│   └── streamlit_app.py             # Streamlit frontend
├── models/
│   └── saved/                       # Trained .joblib artifacts (written by train.py)
├── tests/
│   └── test_pipeline.py             # 30 pytest tests across 6 test classes
├── reports/
│   └── figures/                     # Evaluation plots (written by evaluator.py)
├── logs/
│   └── predictions.jsonl            # Append-only prediction audit log
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download datasets (optional)

```bash
pip install kaggle

kaggle datasets download -d wanderfj/enron-spam -p data/raw --unzip
kaggle datasets download -d naserabdullahalam/phishing-email-dataset -p data/raw --unzip
kaggle datasets download -d mandygu/lingspam-dataset -p data/raw --unzip

# Rename to match expected filenames
mv data/raw/emails.csv data/raw/enron_spam.csv
mv data/raw/Phishing_Email.csv data/raw/phishing_emails.csv
```

If you skip this step, `train.py` falls back to synthetic demo data automatically.

### 3. Train

```bash
# Train all five models, select best by F1
python src/train.py

# Options
python src/train.py --imbalance smote          # SMOTE oversampling instead of class weighting
python src/train.py --sample 50000             # Cap dataset at 50k emails
python src/train.py --threshold 0.35           # Lower threshold → higher recall
python src/train.py --mode bert                # Use DistilBERT embeddings (GPU recommended)
```

Trained models are saved to `models/saved/`. Evaluation plots are saved to `reports/figures/`.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### 5. Start the frontend

```bash
streamlit run app/streamlit_app.py
```

Frontend at `http://localhost:8501`. The frontend calls the API; start the API first. If the API is offline, the frontend falls back to a client-side rule-engine-only demo mode automatically.

### 6. Docker

```bash
docker-compose up --build
```

Starts the API on port 8000 and the Streamlit frontend on port 8501.

---

## API Reference

All endpoints are documented at `/docs` (Swagger UI) once the server is running.

### `POST /predict`

Classify a single email.

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "subject": "URGENT: Verify your PayPal account",
    "body": "Your account has been suspended. Verify at http://paypal-secure.xyz/login",
    "sender": "security@paypal-alert.info",
    "include_explanation": True
})

result = response.json()
# result["verdict"]            → "PHISHING/SPAM" or "LEGITIMATE"
# result["final_label"]        → 1 or 0
# result["confidence"]         → float 0–1  (hybrid score)
# result["ml_score"]           → float 0–1  (ML model probability)
# result["rule_score"]         → float 0–1  (rule engine score)
# result["risk_level"]         → "SAFE" / "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
# result["triggered_rules"]    → e.g. ["urgency_keywords", "credential_harvest"]
# result["explanations"]       → list of human-readable strings
# result["processing_time_ms"] → float
```

### `POST /predict/batch`

Classify up to 100 emails in one request.

```python
response = requests.post("http://localhost:8000/predict/batch", json={
    "emails": [
        {"subject": "...", "body": "...", "sender": "..."},
        {"subject": "...", "body": "...", "sender": "..."},
    ]
})
# response.json() → { total_processed, phishing_detected, legitimate, results[] }
```

### `GET /health`

Returns API status and whether the ML model is loaded.

### `GET /model/info`

Returns the loaded model name, request count, phishing detection count, and average latency.

### `GET /metrics`

Prometheus-compatible plain-text metrics: `email_requests_total`, `email_phishing_detected_total`, `email_avg_latency_ms`, `model_loaded`.

### `POST /simulate/realtime`

Returns a prediction for a randomly selected internal sample email. Used by the Streamlit live simulation page.

---

## Streamlit Frontend Pages

| Page | What it does |
|------|-------------|
| Analyze Email | Paste subject, body, sender — get verdict, confidence gauge, score breakdown, triggered rules, explanations |
| Batch Analysis | Upload a CSV with `subject`, `body`, `sender` columns — analyze up to 100 rows at once |
| Live Simulation | Runs 10 sequential calls to `/simulate/realtime` and streams results as they arrive |
| Model Dashboard | Bar chart and table comparing accuracy, precision, recall, F1, ROC-AUC across all five models |

---

## Evaluation

`SecurityEvaluator` in `src/evaluation/evaluator.py` is configurable with a `cost_fn` (cost of a missed phishing email) and `cost_fp` (cost of a false alarm), so threshold optimisation can minimise business cost rather than a pure ML metric.

**Why each metric matters in email security:**

| Metric | What it measures | Why it matters here |
|--------|-----------------|---------------------|
| Recall | Fraction of phishing emails caught | A missed phishing email is a potential breach — this is the primary security metric |
| Precision | Fraction of flagged emails that are actually malicious | Low precision causes alert fatigue and blocks legitimate mail |
| FPR | Fraction of legitimate emails incorrectly blocked | High FPR erodes user trust and disrupts business communication |
| ROC-AUC | Discrimination quality across all thresholds | Threshold-independent; useful for comparing models before deployment tuning |
| Brier score | Calibration of predicted probabilities | Matters when the raw confidence score is passed to a SIEM or ticketing system |

`find_optimal_threshold()` sweeps thresholds from 0.05 to 0.95 and returns the value that maximises F1, maximises recall subject to precision ≥ 0.5, or minimises business cost — selectable via the `objective` argument.

---

## Explainability

`ModelExplainer` in `src/explainability/explainer.py` supports:

- **SHAP** — auto-selects `TreeExplainer` for Random Forest and Gradient Boosting, `LinearExplainer` for Logistic Regression and SVM, `KernelExplainer` as a model-agnostic fallback. Produces global beeswarm plots and per-instance waterfall plots.
- **LIME** — `LimeTextExplainer` wraps any model via a `predict_fn` callback and returns per-word weights showing which tokens push toward phishing and which push toward legitimate.
- **Feature importance** — reads `feature_importances_` (tree models) or `coef_` (linear models) and plots the top-N features.

---

## URL and Header Analysis

**`URLAnalyzer`** (`src/utils/url_analyzer.py`) scores URLs on: IP address as host, high-risk TLDs, domain entropy, subdomain depth, hyphen count, brand impersonation / typosquatting, URL shortener detection, and HTTP vs HTTPS.

**`EmailHeaderAnalyzer`** scores parsed headers on: SPF result, DKIM signature validity, DMARC policy result, Reply-To domain mismatch vs From domain, and Return-Path domain mismatch vs From domain.

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=html
pytest tests/test_pipeline.py::TestURLAnalyzer -v
```

30 tests across 6 classes: `TestEmailPreprocessor`, `TestRuleEngine`, `TestURLAnalyzer`, `TestDataLoader`, `TestModelPredictions`, `TestEvaluator`.

---

## Security Considerations

- **Rate limiting** — not included. Add nginx rate limiting or `slowapi` before any public exposure.
- **API authentication** — not included. Add Bearer token middleware before production use.
- **Input validation** — all API inputs are validated and length-capped by Pydantic schemas (`body` max 50,000 chars).
- **Prediction logging** — every prediction is appended to `logs/predictions.jsonl` in the background. Monitor this file for distribution drift.
- **Model poisoning** — retrain periodically on fresh labeled data; do not blindly retrain on user-flagged emails without human review.
- **Adversarial robustness** — character-level TF-IDF (`char_wb` analyzer, 2–4 grams) provides partial resistance to obfuscation. The unicode normalisation step in `EmailPreprocessor` handles lookalike character attacks (e.g. Cyrillic characters substituted for Latin).

---

## Roadmap

- [ ] Jupyter notebooks for EDA, preprocessing experiments, modeling, and explainability
- [ ] BERT fine-tuning training script (currently inference-only wrapper exists)
- [ ] Real-time Kafka consumer integration
- [ ] Grafana dashboard consuming `/metrics`
- [ ] Active learning loop: analyst feedback updates training set
- [ ] VirusTotal API integration in `URLAnalyzer` for live domain reputation

---

## Resume Bullet Points

```
• Built a production-grade phishing and spam detection system using a hybrid ML + rule engine
  architecture (scikit-learn, FastAPI, Streamlit, Docker)
• Engineered an NLP preprocessing pipeline (HTML stripping, unicode normalisation,
  lemmatisation) and combined word/character TF-IDF with 25 cybersecurity-specific features
• Trained and compared five classifiers (Logistic Regression, Complement Naive Bayes,
  Random Forest, Linear SVM, Gradient Boosting) with automated threshold optimisation
  targeting recall-first security objectives
• Implemented SHAP and LIME explainability so security analysts can audit every prediction
• Built a REST API (FastAPI) with single and batch prediction endpoints, Prometheus metrics,
  and append-only prediction audit logging
• Added phishing URL detection (TLD risk, domain entropy, brand impersonation) and email
  header forensics (SPF/DKIM/DMARC) as standalone modules
• Containerised the full system with Docker Compose (API + Streamlit frontend)
• Wrote 30 pytest tests across 6 test classes covering preprocessing, rule engine,
  URL analysis, model predictions, and evaluation metrics
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgements

- [Enron Spam Dataset](http://www.aueb.gr/users/ion/data/enron-spam/)
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [SHAP](https://github.com/slundberg/shap) and [LIME](https://github.com/marcotcr/lime)
