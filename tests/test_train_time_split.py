import os
import json
import pandas as pd
from pipeline.run_pipeline import train_time_split


def test_train_time_split_creates_metrics(tmp_path):
    # create a small synthetic dataset with numeric features and labels
    start = 1609459200
    rows = []
    for i in range(200):
        rows.append({
            'unix_time': start + i * 3600,
            'amt': float(i % 50 + 1),
            'cc_num': f'card{i%10}',
            'merchant': f'merch{i%20}',
            'is_fraud': 1 if (i % 33 == 0) else 0,
        })
    df = pd.DataFrame(rows)

    metrics = train_time_split(df, out_prefix='testunit')

    assert isinstance(metrics, dict)
    assert set(['precision','recall','f1','roc_auc','pr_auc']).issubset(set(metrics.keys()))

    # metrics file should exist
    assert os.path.exists('models/testunit_metrics.json')

    # cleanup artifact
    try:
        os.remove('models/testunit_metrics.json')
        os.remove('models/lgb_rolling_graph_testunit.joblib')
        os.remove('models/scaler_rolling_testunit.joblib')
    except Exception:
        pass
