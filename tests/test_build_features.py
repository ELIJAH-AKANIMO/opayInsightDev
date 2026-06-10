import os
from pathlib import Path
import pandas as pd
from pipeline.run_pipeline import build_features


def test_build_features_creates_graph_structures(tmp_path):
    # small synthetic dataset covering 14 days across two cards and two merchants
    data = []
    start = 1609459200  # 2021-01-01
    for i in range(14):
        data.append({
            'unix_time': start + i * 86400,
            'amt': 10 + i,
            'cc_num': 'card1' if i % 2 == 0 else 'card2',
            'merchant': 'merchA' if i % 3 == 0 else 'merchB',
            'is_fraud': 1 if i % 7 == 0 else 0,
        })
    df = pd.DataFrame(data)

    out = build_features(df)

    # check rolling and graph structural columns exist
    assert 'card_7d_tx_count' in out.columns
    assert 'card_30d_tx_count' in out.columns
    assert 'node_degree_card' in out.columns
    assert 'node_pagerank_card' in out.columns
    assert 'node_neighbor_fraud_card' in out.columns

    # check the models/graph_struct_dicts.pkl was created
    p = Path('models/graph_struct_dicts.pkl')
    assert p.exists()

    # cleanup
    try:
        p.unlink()
    except Exception:
        pass
