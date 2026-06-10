# Notebook README — Intelligent Detection Pipeline 🧭

This folder contains step-by-step instructions and runnable code snippets to reproduce the Intelligent Detection & Pattern Discovery pipeline.

Quick steps:
1. Run the data pipeline script (sample run):
   ```bash
   python pipeline/run_pipeline.py --mode sample --sample_n 200000
   ```
2. Inspect artifacts:
   - `data/enriched_sample.csv` or `data/enriched_full.csv`
   - `models/lgb_rolling_graph_{sample|full}.joblib`
   - `models/{sample|full}_metrics.json`
3. Open `notebooks/intelligent_detection.ipynb` (if present) or use these code snippets to build features and train.

Key cells to include in a notebook:
- Load and preview `fraudTrain.csv` (class balance, timestamp plot)
- Run `build_features(df)` (rolling windows + graph features)
- Train with time-aware split and compute ROC PR curves
- Save model + artifacts and record metrics

Production notes:
- Prefer storing rolling aggregates in a feature store (daily batch + micro-batch updates)
- Maintain incremental graph updates (append-only edges, nightly embedding job)
- Retraining cadence: weekly or triggered by drift alerts
- Monitoring: PR AUC and alert volume, plus feature-level KS tests and cardinality churn
