from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# inject dummy model to ensure app runs
class DummyModel:
    def predict_proba(self, X):
        import numpy as _np
        n = len(X)
        probs = _np.full(n, 0.3)
        return _np.vstack([1-probs, probs]).T

try:
    from api import app as api_app
    api_app.MODEL = DummyModel()
except Exception:
    pass


def test_generate_sar_endpoint():
    txs = [
        {'orig_index': 1, 'amt': 10.0, 'merchant': 'a'},
        {'orig_index': 2, 'amt': 20.0, 'merchant': 'b'},
    ]
    actions = [{'orig_index': 1, 'action': 'Investigate', 'reviewer': 'alice', 'reason': 'suspicious', 'ticket_id': 'T123'}]
    r = client.post('/sar', json={'transactions': txs, 'actions': actions})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'csv_len' in body and 'csv_str' in body
    csv = body['csv_str']
    assert 'orig_index' in csv
    assert 'ticket_id' in csv
