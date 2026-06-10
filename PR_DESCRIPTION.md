PR: Add tests, CI, graph structural features, SHAP analysis, and pipeline improvements

Summary:
- Adds structural graph features to `pipeline/run_pipeline.py` (pagerank, eigenvector centrality, clustering, core number, neighbor-fraud).
- Adds unit tests (`tests/test_build_features.py`, `tests/test_train_time_split.py`) and a GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- Adds SHAP analysis script (`scripts/shap_analysis.py`) and graph struct computation script (`scripts/compute_graph_struct.py`).
- Adds top-level `README.md` with CI badge placeholder and `PRD_INTELLIGENT_DETECTION_FULL.md` already present.

Notes for reviewer:
- Tests pass locally (2 tests).
- The CI badge uses a placeholder URL; replace `<owner>/<repo>` after pushing.
- Node2Vec attempted, but `gensim` failed to build on Windows due to C-extension compilation issues; structural graph features improved metrics and are included.

Files changed (high level):
- pipeline/run_pipeline.py (feature engineering + structural graph persistence)
- scripts/compute_graph_struct.py
- scripts/train_graph_struct_model.py
- scripts/shap_analysis.py
- tests/*
- .github/workflows/ci.yml
- README.md, PR_DESCRIPTION.md

Suggested PR body:
- Describe the lift from structural features (PR AUC improved to ~0.8787) and include SHAP top features screenshot (`models/shap_top_features.png`).
- Request review focused on: production readiness for feature generation, acceptable retraining cadence, and any PII/PII-mitigation concerns.

Additional changes in this branch:
- Polished Streamlit UI with header, metrics cards, ranked-prob chart, top-alerts styling, and per-row action workflow (select, mark reviewed, apply action, log actions to a local SQLite DB at `webapp/actions.db`).
- Added `scripts/make_demo_gif.py` to generate a small placeholder demo GIF at `webapp/demo.gif` and updated `README_UI.md` to show the GIF and how to regenerate it.
- Added helper scripts and venv creation scripts for easier local runs.

Testing & Local steps:
- Create venv and install: `.\scripts\create_venv.ps1` (Windows) or `./scripts/create_venv.sh` (Unix).
- Run the API: `python -m uvicorn api.app:app --reload --port 8000` (optional).
- Run UI: `python -m streamlit run webapp/app.py`.
- Use the UI to score sample data, select alerts, apply actions, and download the actions log.
