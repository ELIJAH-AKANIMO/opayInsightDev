from fastapi.testclient import TestClient
import json
import time
from api.app import app
from pathlib import Path

client = TestClient(app)

# ensure a lightweight model is available for tests when app startup hasn't loaded a real model
class DummyModel:
    def predict_proba(self, X):
        import numpy as _np
        n = len(X)
        probs = _np.full(n, 0.1)
        return _np.vstack([1-probs, probs]).T

# inject dummy model for tests
try:
    from api import app as api_app
    api_app.MODEL = DummyModel()
    api_app.SCALER = None
    api_app.MODEL_FEATURES = None
except Exception:
    pass

# use sample row from data/enriched_sample.csv if available
SAMPLE_PATH = Path('data/enriched_sample.csv')

def _load_sample_row():
    if SAMPLE_PATH.exists():
        import pandas as pd
        df = pd.read_csv(SAMPLE_PATH)
        row = df.iloc[0].to_dict()
        # remove is_fraud label if present
        row.pop('is_fraud', None)
        return row
    # fallback synthetic minimal transaction
    return {'amt': 10.0, 'merchant': 'test_merch', 'card_id': '0000', 'cc_num': '0000', 'time': 0}


def test_post_single_tx_basic():
    tx = _load_sample_row()
    start = time.perf_counter()
    r = client.post('/score/tx', json=tx)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'result' in body
    assert body['n'] == 1
    assert 0.0 <= body['result']['proba'] <= 1.0
    # basic latency guard (should be quick on small models)
    assert elapsed < 5.0


def test_post_single_tx_with_explain():
    tx = _load_sample_row()
    r = client.post('/score/tx?explain=true', json=tx)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'shap' in body or 'shap_error' in body
