"""
api/main.py
-----------
Production FastAPI REST API for the Intelligent Email Detection System.

Endpoints:
  POST /predict          — single email classification
  POST /predict/batch    — batch classification (up to 100 emails)
  GET  /health           — health check
  GET  /model/info       — model metadata
  GET  /metrics          — Prometheus-compatible metrics

Security note: Add API key authentication before production deployment.
"""

import os
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from loguru import logger

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_cleaner import EmailPreprocessor, RuleEngine
from src.preprocessing.feature_engineering import FeaturePipeline
from src.models.classifier import BaseEmailClassifier, HybridClassifier, _risk_level


# ──────────────────────────────────────────────
# Global model state
# ──────────────────────────────────────────────

MODEL_DIR = Path("models/saved")

class AppState:
    feature_pipeline: Optional[FeaturePipeline] = None
    ml_model: Optional[BaseEmailClassifier] = None
    hybrid_classifier: Optional[HybridClassifier] = None
    preprocessor: Optional[EmailPreprocessor] = None
    rule_engine: Optional[RuleEngine] = None
    model_metadata: Dict = {}
    request_count: int = 0
    phishing_detected: int = 0
    total_latency_ms: float = 0.0

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Loading models...")
    try:
        app_state.preprocessor = EmailPreprocessor()
        app_state.rule_engine = RuleEngine()

        fp_path = MODEL_DIR / "feature_pipeline.joblib"
        model_path = MODEL_DIR / "logistic_regression.joblib"

        if fp_path.exists() and model_path.exists():
            app_state.feature_pipeline = FeaturePipeline.load(str(fp_path))
            app_state.ml_model = BaseEmailClassifier.load(str(model_path))
            app_state.hybrid_classifier = HybridClassifier(
                ml_model=app_state.ml_model, alpha=0.2
            )
            app_state.model_metadata = {
                "name": app_state.ml_model.name,
                "loaded_at": datetime.utcnow().isoformat(),
                "model_path": str(model_path),
            }
            logger.info(f"Models loaded: {app_state.ml_model.name}")
        else:
            logger.warning(
                f"Model files not found at {MODEL_DIR}. "
                "Run training pipeline first: python src/train.py"
            )

    except Exception as exc:
        logger.error(f"Model load failed: {exc}")

    yield  # Application runs here

    logger.info("Shutting down...")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title="Intelligent Suspicious Email Detection API",
    description=(
        "Production-grade phishing and spam detection system. "
        "Combines ML models with rule-based analysis for explainable predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────

class EmailInput(BaseModel):
    subject:    str = Field("", max_length=500, description="Email subject line")
    body:       str = Field(..., min_length=1, max_length=50000, description="Email body text")
    sender:     str = Field("", max_length=200, description="Sender address")
    reply_to:   str = Field("", max_length=200, description="Reply-To header")
    include_explanation: bool = Field(True, description="Include rule/feature explanations")

    @validator("body")
    def body_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Email body cannot be empty")
        return v


class BatchEmailInput(BaseModel):
    emails: List[EmailInput] = Field(..., min_items=1, max_items=100)


class PredictionResult(BaseModel):
    request_id:     str
    verdict:        str        # "LEGITIMATE" | "PHISHING/SPAM"
    final_label:    int        # 0 = legitimate, 1 = phishing/spam
    confidence:     float      # [0.0, 1.0]
    risk_level:     str        # "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    ml_score:       float
    rule_score:     float
    hybrid_score:   float
    triggered_rules:    List[str]
    explanations:       List[str]
    processing_time_ms: float
    model_used:         str
    timestamp:          str


class HealthResponse(BaseModel):
    status:     str
    model_loaded: bool
    uptime_s:   float
    version:    str


# ──────────────────────────────────────────────
# Request logging middleware
# ──────────────────────────────────────────────

_start_time = time.time()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    app_state.request_count += 1
    app_state.total_latency_ms += duration_ms

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


# ──────────────────────────────────────────────
# Helper: run prediction
# ──────────────────────────────────────────────

def _run_prediction(email: EmailInput) -> Dict:
    if app_state.feature_pipeline is None or app_state.hybrid_classifier is None:
        # Fallback: rule-engine only if model not loaded
        combined_text = f"{email.subject} {email.body}"
        rule_result = app_state.rule_engine.evaluate(combined_text)
        score = rule_result["rule_score"]
        return {
            "final_label": int(score >= 0.5),
            "confidence": round(score, 4),
            "ml_score": 0.0,
            "rule_score": round(score, 4),
            "hybrid_score": round(score, 4),
            "triggered_rules": rule_result["triggered_rules"],
            "explanations": rule_result["explanations"] + ["⚠ ML model not loaded — rule-engine only"],
            "verdict": "PHISHING/SPAM" if score >= 0.5 else "LEGITIMATE",
            "risk_level": _risk_level(score),
        }

    combined_text = f"{email.subject} {email.body} {email.sender}"
    features = app_state.feature_pipeline.transform([combined_text])

    results = app_state.hybrid_classifier.predict_with_explanation(
        texts=[combined_text],
        features=features,
    )
    return results[0]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "model_loaded": app_state.ml_model is not None,
        "uptime_s": round(time.time() - _start_time, 1),
        "version": "1.0.0",
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    return {
        "model_metadata": app_state.model_metadata,
        "request_count": app_state.request_count,
        "phishing_detected": app_state.phishing_detected,
        "avg_latency_ms": round(
            app_state.total_latency_ms / max(app_state.request_count, 1), 2
        ),
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus-compatible text metrics."""
    lines = [
        f'email_requests_total {app_state.request_count}',
        f'email_phishing_detected_total {app_state.phishing_detected}',
        f'email_avg_latency_ms {app_state.total_latency_ms / max(app_state.request_count, 1):.2f}',
        f'model_loaded {int(app_state.ml_model is not None)}',
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(
    email: EmailInput,
    background_tasks: BackgroundTasks,
):
    """
    Classify a single email as legitimate or phishing/spam.

    Returns confidence score, risk level, and explainability data.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    try:
        result = _run_prediction(email)
    except Exception as exc:
        logger.error(f"[{request_id}] Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    processing_ms = (time.perf_counter() - t0) * 1000

    if result["final_label"] == 1:
        app_state.phishing_detected += 1

    # Log to background (non-blocking)
    background_tasks.add_task(
        _log_prediction, request_id, email, result, processing_ms
    )

    return PredictionResult(
        request_id=request_id,
        verdict=result["verdict"],
        final_label=result["final_label"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        ml_score=result["ml_score"],
        rule_score=result["rule_score"],
        hybrid_score=result["hybrid_score"],
        triggered_rules=result["triggered_rules"],
        explanations=result["explanations"] if email.include_explanation else [],
        processing_time_ms=round(processing_ms, 2),
        model_used=app_state.model_metadata.get("name", "rule-engine"),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(batch: BatchEmailInput):
    """
    Classify up to 100 emails in a single request.
    Returns list of predictions with the same schema as /predict.
    """
    t0 = time.perf_counter()
    results = []

    for email in batch.emails:
        try:
            result = _run_prediction(email)
            results.append(result)
        except Exception as exc:
            results.append({
                "error": str(exc),
                "verdict": "ERROR",
                "final_label": -1,
                "confidence": 0.0,
            })

    total_ms = (time.perf_counter() - t0) * 1000
    phishing_count = sum(1 for r in results if r.get("final_label") == 1)
    app_state.phishing_detected += phishing_count

    return {
        "total_processed": len(results),
        "phishing_detected": phishing_count,
        "legitimate": len(results) - phishing_count,
        "processing_time_ms": round(total_ms, 2),
        "results": results,
    }


@app.post("/simulate/realtime", tags=["Demo"])
async def simulate_realtime():
    """
    Demo endpoint: generate a random phishing or legitimate email
    and return its prediction. Used for frontend demos.
    """
    import random
    phishing_samples = [
        {
            "subject": "URGENT: Your bank account will be suspended",
            "body": "Dear Customer, We have detected suspicious activity on your account. "
                    "Please click here to verify your identity immediately or your account "
                    "will be suspended within 24 hours. Act NOW: http://secure-verify.xyz/login",
            "sender": "security@bank-alerts.info",
        },
        {
            "subject": "You won $500,000 in the Microsoft Lottery!",
            "body": "Congratulations! You have been selected as the winner of our annual "
                    "Microsoft Lottery. To claim your $500,000 prize, click this link and "
                    "provide your bank details: http://microsoft-lottery.tk/claim",
            "sender": "noreply@microsoft-prizes.ga",
        },
        {
            "subject": "IRS: Immediate Action Required — Tax Evasion",
            "body": "You have outstanding taxes. Failure to pay within 48 hours will result "
                    "in arrest. Call 1-800-TAX-SCAM or pay at http://irs-payments.xyz",
            "sender": "irs-notice@taxalert.pw",
        },
    ]
    legit_samples = [
        {
            "subject": "Team meeting — Thursday 3 PM",
            "body": "Hi everyone, just a reminder that our weekly sync is on Thursday at 3 PM. "
                    "Please review the Q3 performance report before then. The doc is on SharePoint.",
            "sender": "manager@company.com",
        },
        {
            "subject": "Your order #78291 has shipped",
            "body": "Your Amazon order has been dispatched and will arrive by Friday. "
                    "You can track your package at amazon.com/orders using your order ID.",
            "sender": "shipping@amazon.com",
        },
    ]

    sample = random.choice(phishing_samples + legit_samples)
    email = EmailInput(**sample)
    return await predict_single(email, BackgroundTasks())


# ──────────────────────────────────────────────
# Background tasks
# ──────────────────────────────────────────────

LOG_FILE = Path("logs/predictions.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)

def _log_prediction(request_id: str, email: EmailInput, result: Dict, latency_ms: float):
    """Append prediction to JSONL log for monitoring."""
    record = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "risk_level": result.get("risk_level"),
        "latency_ms": round(latency_ms, 2),
        "triggered_rules": result.get("triggered_rules", []),
        "sender": email.sender[:50] if email.sender else "",
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
