import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import io
import csv

DB_PATH = Path('data') / 'graphdb.db'

def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _conn()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            orig_index TEXT NOT NULL,
            action TEXT NOT NULL,
            reviewer TEXT,
            reason TEXT,
            ticket_id TEXT
        )
    ''')
    # New Table for SARs (Automated Stage 9)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            orig_index TEXT NOT NULL,
            content TEXT NOT NULL
        )
    ''')
    # New Table for Authority Batches
    cur.execute('''
        CREATE TABLE IF NOT EXISTS authority_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            record_count INTEGER,
            data_json TEXT
        )
    ''')
    # Note: 'transactions' table is created dynamically via pandas.to_sql in ingest_transactions
    conn.commit()
    conn.close()

def ingest_transactions(df: Any):
    """Sync a dataframe into the transactions table."""
    init_db()
    conn = _conn()
    # Replace existing transactions with the new batch
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()

def get_transactions() -> Any:
    """Fetch all transactions from SQLite."""
    import pandas as pd
    init_db()
    conn = _conn()
    try:
        df = pd.read_sql('SELECT * FROM transactions', conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def log_sar(report_id: str, orig_index: str, content: str):
    """Log a SAR report to the database."""
    from datetime import datetime
    init_db()
    conn = _conn()
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute('INSERT INTO sars (report_id, ts, orig_index, content) VALUES (?, ?, ?, ?)',
                (report_id, ts, str(orig_index), content))
    conn.commit()
    conn.close()

def log_authority_report(risk_level: str, df: Any):
    """Log a batch report to the database."""
    from datetime import datetime
    import json
    init_db()
    conn = _conn()
    cur = conn.cursor()
    
    ts = datetime.utcnow().isoformat()
    report_id = f"AUTH-{int(datetime.utcnow().timestamp())}"
    count = len(df)
    # Serialize DF to JSON
    data_json = df.to_json(orient='records')
    
    cur.execute('INSERT INTO authority_reports (report_id, ts, risk_level, record_count, data_json) VALUES (?, ?, ?, ?, ?)',
                (report_id, ts, risk_level, count, data_json))
    conn.commit()
    conn.close()
    return report_id

def get_actions() -> List[Dict[str, Any]]:
    init_db()
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT id, ts, orig_index, action, reviewer, reason, ticket_id FROM actions ORDER BY id ASC')
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def export_csv_bytes() -> bytes:
    rows = get_actions()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id', 'ts', 'orig_index', 'action', 'reviewer', 'reason', 'ticket_id'])
    for r in rows:
        writer.writerow([r['id'], r['ts'], r['orig_index'], r['action'], r.get('reviewer'), r.get('reason'), r.get('ticket_id')])
    return buf.getvalue().encode('utf-8')

def get_reviewed_orig_indices() -> List[str]:
    # distinct orig_index values where action = 'Mark reviewed'
    init_db()
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT orig_index FROM actions WHERE action = 'Mark reviewed'")
    rows = cur.fetchall()
    conn.close()
    return [r['orig_index'] for r in rows]