from fastapi.testclient import TestClient
from api.app import app
from webapp import actions_db
from pathlib import Path

client = TestClient(app)

def test_post_and_list_actions(tmp_path):
    # isolate DB
    orig_db = actions_db.DB_PATH
    actions_db.DB_PATH = Path(tmp_path) / 'actions_api_test.db'
    try:
        payload = {'orig_index': '10', 'action': 'Investigate', 'reviewer': 'bob', 'reason': 'suspicious', 'ticket_id': 'T999'}
        r = client.post('/actions', json=payload)
        assert r.status_code == 200
        j = r.json()
        assert j.get('status') == 'ok'

        r2 = client.get('/actions')
        assert r2.status_code == 200
        j2 = r2.json()
        assert j2['n'] >= 1
        entries = j2['actions']
        found = [e for e in entries if e['orig_index'] == '10' and e['action'] == 'Investigate']
        assert len(found) == 1
    finally:
        if actions_db.DB_PATH.exists():
            actions_db.DB_PATH.unlink()
        actions_db.DB_PATH = orig_db
