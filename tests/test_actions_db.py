import tempfile
from pathlib import Path
import os
from webapp import actions_db


def test_log_and_export(tmp_path):
    # point DB_PATH to a temp file
    orig_db = actions_db.DB_PATH
    actions_db.DB_PATH = Path(tmp_path) / 'actions_test.db'
    try:
        actions_db.init_db()
        actions_db.log_action('5', 'Mark reviewed', reviewer='alice', reason='test', ticket_id='T123')
        acts = actions_db.get_actions()
        assert len(acts) == 1
        a = acts[0]
        assert a['orig_index'] == '5'
        assert a['action'] == 'Mark reviewed'
        assert a['reviewer'] == 'alice'
        csv_bytes = actions_db.export_csv_bytes()
        csv_text = csv_bytes.decode('utf-8')
        assert 'T123' in csv_text
    finally:
        # cleanup
        if actions_db.DB_PATH.exists():
            os.remove(actions_db.DB_PATH)
        actions_db.DB_PATH = orig_db
