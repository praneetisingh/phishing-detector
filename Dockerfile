FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="Intelligent Suspicious Email Detection System"

WORKDIR /app

# Install system deps (for lxml, scikit-learn)
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','stopwords','wordnet','omw-1.4']]"

COPY . .

# Create required directories
RUN mkdir -p models/saved logs reports/figures data/raw data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
