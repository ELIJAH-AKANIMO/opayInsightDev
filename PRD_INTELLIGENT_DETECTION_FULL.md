# PRD: Intelligent Detection & Pattern Discovery (Detailed)

## Goal
Produce a production-ready fraud detection pipeline that combines transactional features, rolling per-entity aggregates, and graph-derived signals to detect coordinated fraud patterns and minimize false positives.

## MVP Deliverables
- Enrichment pipeline that computes per-card and per-merchant rolling features (1/7/30 day windows) and simple graph features (degree, connected component size).
- Model training job using time-aware splits and forward-chaining validation.
- Evaluation report with ROC/PR curves and operating thresholds tuned for desired precision.
- Notebook and reproducible pipeline script `pipeline/run_pipeline.py`.

## Architecture
- Data ingestion: stream transactions into a staging area; batch daily snapshots go into the feature store.
- Feature store: store daily rolling aggregates per entity and allow fast lookup by transaction timestamp.
- Graph service: incremental updates to a graph DB / nightly job that produces embeddings and degree/component features.
- Model serving: score in near-real-time using precomputed rolling features + online micro-batch aggregates; fallback to batch scores if upstream unavailable.

## Retraining & Monitoring
- Retrain weekly or when PR AUC drops by >5% vs baseline for 3 consecutive runs.
- Monitor: feature distribution drift (KS tests), target drift, alert volume, latency, and top features importance.
- Alerting: Slack/email and runbook for data ops.

## Security & Compliance
- PII minimization: only store hashed identifiers and non-PII aggregates in feature store.
- Access controls: role-based access to models and data.
- Audit logs: all model changes and scoring requests logged.

## Next Steps
1. Add nightly Node2Vec embeddings and test their lift.
2. Build incremental feature service for real-time scoring.
3. Deploy a lightweight scoring API + batch re-score job and establish monitoring dashboards.

