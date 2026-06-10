import time
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# inject a dummy model to keep tests deterministic
class DummyModel:
    def predict_proba(self, X):
        import numpy as _np
        n = len(X)
        probs = _np.full(n, 0.2)
        return _np.vstack([1-probs, probs]).T

try:
    from api import app as api_app
    api_app.MODEL = DummyModel()
except Exception:
    pass


def test_latency_single_tx():
    tx = {'amt': 5.0, 'merchant': 'bench', 'card_id': '000', 'cc_num': '000'}
    n = 20
    times = []
    for _ in range(n):
        start = time.perf_counter()
        r = client.post('/score/tx', json=tx)
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        times.append(elapsed)
    avg = sum(times)/len(times)
    # ensure average latency stays under a tighter target to validate fast-path trimming
    assert avg < 0.25, f"avg latency too high: {avg}"