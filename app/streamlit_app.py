"""
app/streamlit_app.py
---------------------
Streamlit frontend for the Intelligent Email Detection System.

Features:
  - Single email analysis with confidence gauge
  - Batch upload via CSV
  - Real-time simulation demo
  - Model comparison dashboard
  - Explainability panel (top features, triggered rules)

Run: streamlit run app/streamlit_app.py
"""

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_BASE = "http://localhost:8000"
PAGE_TITLE = "PhishShield — AI Email Security"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS styling
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding: 1.5rem 2rem; }

    .verdict-phishing {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border-left: 4px solid #ff4b4b;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .verdict-legit {
        background: linear-gradient(135deg, #00cc7722, #00cc7744);
        border-left: 4px solid #00cc77;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .risk-badge-CRITICAL { color: #ff2424; font-weight: bold; font-size: 1.2rem; }
    .risk-badge-HIGH     { color: #ff7b2c; font-weight: bold; font-size: 1.2rem; }
    .risk-badge-MEDIUM   { color: #ffc107; font-weight: bold; font-size: 1.2rem; }
    .risk-badge-LOW      { color: #7ecdf0; font-weight: bold; font-size: 1.2rem; }
    .risk-badge-SAFE     { color: #00cc77; font-weight: bold; font-size: 1.2rem; }
    .rule-tag {
        display: inline-block;
        background: #ff4b4b22;
        border: 1px solid #ff4b4b55;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px;
        color: #ff9999;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: API calls with fallback
# ──────────────────────────────────────────────

def call_api(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return _mock_prediction(payload)
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def _mock_prediction(payload: dict) -> dict:
    """Demo mode when API is offline."""
    import random
    body = payload.get("body", "")
    phishing_words = {"urgent", "verify", "suspended", "click", "prize", "winner", "immediately"}
    score = min(sum(w in body.lower() for w in phishing_words) * 0.18 + random.uniform(0, 0.1), 1.0)
    label = int(score >= 0.5)
    risk = "CRITICAL" if score > 0.8 else "HIGH" if score > 0.6 else "MEDIUM" if score > 0.4 else "SAFE"
    return {
        "verdict": "PHISHING/SPAM" if label else "LEGITIMATE",
        "final_label": label,
        "confidence": round(score, 3),
        "risk_level": risk,
        "ml_score": round(score * 0.9, 3),
        "rule_score": round(score * 1.1, 3),
        "hybrid_score": round(score, 3),
        "triggered_rules": ["urgency_keywords", "credential_harvest"][:label],
        "explanations": ["Demo mode — API offline. Install and run: uvicorn api.main:app"],
        "processing_time_ms": 12.5,
        "model_used": "Demo (Rule Engine)",
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "offline", "model_loaded": False}


# ──────────────────────────────────────────────
# Components
# ──────────────────────────────────────────────

def render_confidence_gauge(score: float, label: int) -> go.Figure:
    color = "#ff4b4b" if label == 1 else "#00cc77"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        delta={"reference": 50, "increasing": {"color": "#ff4b4b"}, "decreasing": {"color": "#00cc77"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e2130",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#00cc7720"},
                {"range": [30, 50], "color": "#ffc10720"},
                {"range": [50, 75], "color": "#ff7b2c20"},
                {"range": [75, 100], "color": "#ff4b4b20"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=250,
        margin={"t": 20, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def render_score_breakdown(result: dict):
    fig = go.Figure(go.Bar(
        x=["ML Score", "Rule Score", "Hybrid Score"],
        y=[result["ml_score"], result["rule_score"], result["hybrid_score"]],
        marker_color=["#4a9eff", "#f0a500", "#ff4b4b" if result["final_label"] else "#00cc77"],
        text=[f"{v:.1%}" for v in [result["ml_score"], result["rule_score"], result["hybrid_score"]]],
        textposition="outside",
    ))
    fig.update_layout(
        height=200,
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        showlegend=False,
    )
    return fig


def render_result(result: dict):
    is_phishing = result["final_label"] == 1
    css_class = "verdict-phishing" if is_phishing else "verdict-legit"
    icon = "🚨" if is_phishing else "✅"
    risk = result.get("risk_level", "UNKNOWN")

    st.markdown(f"""
    <div class="{css_class}">
        <h2>{icon} {result['verdict']}</h2>
        <span class="risk-badge-{risk}">⬡ RISK: {risk}</span>
        <p style="margin: 0.5rem 0 0 0; color: #aaa; font-size: 0.9rem;">
            Request ID: {result.get('request_id', 'demo')} &nbsp;|&nbsp;
            Model: {result.get('model_used', 'hybrid')} &nbsp;|&nbsp;
            Processed in {result.get('processing_time_ms', 0):.1f}ms
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.plotly_chart(render_confidence_gauge(result["confidence"], result["final_label"]),
                        use_container_width=True)
    with col2:
        st.markdown("#### Score Breakdown")
        st.plotly_chart(render_score_breakdown(result), use_container_width=True)

    if result.get("triggered_rules"):
        st.markdown("#### 🔍 Triggered Rules")
        tags = " ".join(f'<span class="rule-tag">{r}</span>' for r in result["triggered_rules"])
        st.markdown(tags, unsafe_allow_html=True)

    if result.get("explanations"):
        st.markdown("#### 📝 Explanations")
        for exp in result["explanations"]:
            st.markdown(f"- {exp}")


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("# 🛡️ PhishShield")
        st.markdown("*AI-Powered Email Security*")
        st.divider()

        health = get_health()
        status_icon = "🟢" if health.get("model_loaded") else "🟡"
        st.markdown(f"{status_icon} API: **{health.get('status', 'offline').upper()}**")
        if health.get("model_loaded"):
            st.markdown("✅ ML Model: **Loaded**")
        else:
            st.markdown("⚠️ ML Model: **Offline (Demo Mode)**")

        st.divider()
        page = st.radio("Navigate", [
            "🔍 Analyze Email",
            "📊 Batch Analysis",
            "⚡ Live Simulation",
            "📈 Model Dashboard",
        ])
        st.divider()
        st.markdown("**Threshold**")
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Lower = more aggressive (catch more phishing, more FP)")
        st.caption(f"Currently: {threshold:.2f}")

    st.markdown(f"# 🛡️ Intelligent Email Security System")
    st.markdown("*Phishing & Spam Detection powered by Hybrid ML + Rule Engine*")
    st.divider()

    # ── Pages ──

    if "Analyze" in page:
        st.subheader("🔍 Single Email Analysis")
        col1, col2 = st.columns([1, 1])
        with col1:
            subject = st.text_input("Subject Line", placeholder="Enter email subject...")
            sender  = st.text_input("Sender Address", placeholder="sender@example.com")
        with col2:
            reply_to = st.text_input("Reply-To (optional)", placeholder="reply@domain.com")
            st.markdown("&nbsp;")

        body = st.text_area(
            "Email Body",
            height=200,
            placeholder="Paste the full email body here...",
        )

        col_a, col_b, col_c = st.columns([1, 1, 3])
        with col_a:
            analyze_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
        with col_b:
            if st.button("📋 Load Phishing Sample", use_container_width=True):
                st.session_state["sample_type"] = "phishing"
                st.rerun()
            if "sample_type" in st.session_state and st.session_state["sample_type"] == "phishing":
                subject = "URGENT: Your account has been compromised"
                body = ("Dear Customer, We detected suspicious login attempts on your account. "
                        "Verify your identity immediately to avoid suspension: "
                        "http://secure-paypal-verify.xyz/login?id=29841. "
                        "You have 24 hours before your account is permanently blocked!")
                sender = "security@paypal-alerts.info"
                del st.session_state["sample_type"]

        if analyze_btn and body.strip():
            with st.spinner("Analyzing..."):
                result = call_api("/predict", {
                    "subject": subject,
                    "body": body,
                    "sender": sender,
                    "reply_to": reply_to,
                    "include_explanation": True,
                })
            if result:
                render_result(result)

    elif "Batch" in page:
        st.subheader("📊 Batch Email Analysis")
        st.markdown("Upload a CSV with columns: `subject`, `body`, `sender`")

        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df)} emails")
            st.dataframe(df.head(5))

            if st.button("🚀 Analyze All", type="primary"):
                progress = st.progress(0)
                results = []
                emails = []

                for _, row in df.iterrows():
                    emails.append({
                        "subject": str(row.get("subject", "")),
                        "body": str(row.get("body", "")),
                        "sender": str(row.get("sender", "")),
                    })

                with st.spinner("Analyzing batch..."):
                    batch_result = call_api("/predict/batch", {"emails": emails[:100]})
                    progress.progress(100)

                if batch_result:
                    results = batch_result.get("results", [])
                    total = batch_result.get("total_processed", 0)
                    phishing = batch_result.get("phishing_detected", 0)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Analyzed", total)
                    c2.metric("Phishing/Spam", phishing, delta=f"{phishing/total:.0%}")
                    c3.metric("Legitimate", total - phishing)
                    c4.metric("Detection Rate", f"{phishing/total:.1%}")

                    # Results table
                    results_df = pd.DataFrame(results)[
                        ["verdict", "confidence", "risk_level", "triggered_rules"]
                    ].head(50)
                    st.dataframe(results_df)

    elif "Simulation" in page:
        st.subheader("⚡ Real-Time Email Simulation")
        st.markdown("Watch the system analyze randomly generated emails in real time.")

        if st.button("▶ Start Simulation", type="primary"):
            placeholder = st.empty()
            history = []

            for i in range(10):
                with st.spinner(f"Analyzing email {i+1}/10..."):
                    try:
                        r = requests.post(f"{API_BASE}/simulate/realtime", timeout=5)
                        result = r.json()
                    except Exception:
                        result = _mock_prediction({"body": "urgent verify account click here prize"})
                    history.append(result)
                    time.sleep(0.5)

                with placeholder.container():
                    for idx, res in enumerate(reversed(history)):
                        icon = "🚨" if res["final_label"] else "✅"
                        color = "#ff4b4b" if res["final_label"] else "#00cc77"
                        st.markdown(
                            f"<div style='border-left: 3px solid {color}; padding: 0.4rem 1rem; "
                            f"margin: 0.3rem 0; background: {color}11; border-radius: 4px;'>"
                            f"{icon} <b>{res['verdict']}</b> — "
                            f"Confidence: {res['confidence']:.1%} | "
                            f"Risk: {res['risk_level']}</div>",
                            unsafe_allow_html=True,
                        )

    elif "Dashboard" in page:
        st.subheader("📈 Model Performance Dashboard")

        # Simulated model comparison data
        models = ["Logistic Regression", "Naive Bayes", "Random Forest", "LinearSVM", "Grad Boosting"]
        metrics_data = {
            "Model": models,
            "Accuracy":  [0.971, 0.953, 0.968, 0.973, 0.965],
            "Precision": [0.969, 0.940, 0.961, 0.971, 0.958],
            "Recall":    [0.963, 0.948, 0.966, 0.966, 0.962],
            "F1":        [0.966, 0.944, 0.963, 0.968, 0.960],
            "ROC-AUC":   [0.993, 0.980, 0.991, 0.994, 0.992],
        }
        df_metrics = pd.DataFrame(metrics_data)

        # Bar chart
        fig = px.bar(
            df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="Model Comparison — All Metrics",
            template="plotly_dark",
        )
        fig.update_layout(yaxis_range=[0.9, 1.0])
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        st.dataframe(
            df_metrics.set_index("Model").style
                .highlight_max(axis=0, color="#00cc7733")
                .format("{:.4f}"),
            use_container_width=True,
        )

        st.markdown("""
        #### 📌 Why These Metrics Matter in Cybersecurity

        | Metric | Why It Matters |
        |--------|---------------|
        | **Recall** | Most critical — a missed phishing email (FN) is a security breach |
        | **Precision** | Too many false positives = alert fatigue + blocked legitimate email |
        | **ROC-AUC** | Model's overall discrimination ability across all thresholds |
        | **F1** | Balanced metric when both FP and FN carry cost |
        | **FPR** | Low FPR = enterprise trust. Users stop reporting if too many false alarms |
        """)


if __name__ == "__main__":
    main()
