# Intelligent Detection & Pattern Discovery — Hackathon 🛡️

[![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)

## Overview
A comprehensive fraud detection system combining **LightGBM**, **Graph Analytics**, **Deep Learning (Autoencoder)**, and **LLM-powered reasoning**.

### Key Features
- **Hybrid Detection Engine**: Combines LightGBM (tabular) + LSTM Autoencoder (sequence) + Graph (network).
- **Deep Learning**: Sequential anomaly detection with attention mechanism and GPU acceleration.
- **Explainable AI**: SHAP values for feature importance and natural language explanations.
- **Interactive UI**: polished Streamlit dashboard for analysts.
- **Real-time API**: FastAPI endpoint for streaming transaction scoring.

---

## 🚀 Quick Start

### 1. Installation

Create a virtual environment and install dependencies:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Web Interface (UI)

Start the interactive dashboard:
```bash
python -m streamlit run webapp/app.py
```

### 3. Run the API (Streaming)

Start the real-time scoring engine:
```bash
python api/streaming.py
# or
python -m uvicorn api.streaming:app --reload --port 8001
```

### 4. Troubleshooting

- **`uvicorn` not recognized**: Try running with `python -m uvicorn ...` or add your Python Scripts folder to PATH.
- **`python-multipart` error**: If you see an upload error, ensure the package is installed: `pip install python-multipart`.
- **Port 8001 in use**: If the API fails to bind, kill the existing process or check if another instance is running.

### 5. Render Deployment

Render web services must bind to the injected `$PORT` on `0.0.0.0`.

Deploy the API with the included `render.yaml`, or use these manual settings:

```bash
Build Command: pip install -r requirements.txt
Start Command: python -m uvicorn api.app:app --host 0.0.0.0 --port $PORT
Health Check Path: /health
```

Deploy the Streamlit dashboard as a separate Render web service with:

```bash
Build Command: pip install -r requirements.txt
Start Command: python -m streamlit run webapp/app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

If you created the service manually in Render, update the Start Command in the Render dashboard. Existing manual services do not automatically adopt changes from `render.yaml`.

---

## 🧠 System Architecture

### Layer 1: Hybrid Detection Engine
- **LightGBM**: Trains on rolling features (1d/7d/30d) generates fraud probability.
- **LSTM Autoencoder**: Detects sequential anomalies (e.g., unusual transaction ordering).
- **Hybrid Score**: `0.6 * LightGBM + 0.4 * Autoencoder`

- **Graph Network**: Identifies connected components and fraud rings.
- **SHAP**: Explains *why* a model flagged a transaction.
- **LLM Analyst**: Generates natural language insights using GenAI.

### Layer 3: Performance & Evaluation
The hybrid model achieves robust detection rates on the FraudTrain dataset:
- **ROC AUC**: 0.8230 (Time-based split)
- **PR AUC**: 0.0881
- **Precision@TopK**: High precision for top-ranked alerts ensuring efficient analyst time.

---

## 🔧 Advanced Features

### Deep Learning & GPU
The system automatically detects CUDA-enabled GPUs for 5x faster inference.
- **Model Persistence**: Models are saved/loaded from `models/saved_models/`.
- **Attention Mechanism**: The Autoencoder uses Bahdanau attention to focus on critical transactions.

### LLM Analyst Setup
To enable the "🤖 LLM Analyst" tab:
1. Get a Google Gemini API Key from [ai.google.dev](https://ai.google.dev/)
2. Set environment variable:
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your-api-key-here"
   ```
   Or create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

---

## 📂 Project Structure

- `webapp/` - Streamlit dashboard application
- `api/` - FastAPI streaming endpoints
- `models/` - Machine Learning & Deep Learning models
  - `autoencoder.py` - LSTM Autoencoder implementation
  - `attention.py` - Attention mechanism
- `pipeline/` - Data processing and training scripts
  - `run_pipeline.py` - Main training pipeline
  - `tune_autoencoder.py` - Hyperparameter tuning script
- `data/` - Dataset storage (artifacts)
- `notebooks/` - Jupyter notebooks for exploration

---

## 📊 Notebooks & Training

To reproduce the pipeline or retrain models:

1. **Run Data Pipeline**:
   ```bash
   python pipeline/run_pipeline.py --mode sample --sample_n 200000
   ```
   Generates `data/enriched_sample.csv` and trains LightGBM.

2. **Hyperparameter Tuning**:
   ```bash
   python pipeline/tune_autoencoder.py
   ```
   Finds optimal parameters for the Autoencoder.

3. **Explore Notebooks**:
   Open `notebooks/intelligent_detection.ipynb` for step-by-step feature engineering and evaluation.

---

## 🔍 API Documentation

**Endpoint**: `POST /score`
```json
{
  "cc_num": "1234567890",
  "merchant": "fraud_test",
  "amt": 150.00,
  "lat": 38.0,
  "long": -77.0,
  "city_pop": 50000,
  "unix_time": 1609459200
}
```

**Response**:
```json
{
  "transaction_id": "1234567890",
  "ml_probability": 0.85,
  "anomaly_score": 0.72,
  "hybrid_risk": 0.798,
  "risk_level": "HIGH"
}
```

---

## Notes
- **Action Logging**: Alert decisions in the UI are persisted to `webapp/actions.db`.
- **Demo Mode**: The LLM Analyst works in logical demo mode if no API key is provided.
- **Performance**: For large datasets (>1M rows), ensure you have 16GB+ RAM.
