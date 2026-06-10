from api import app as api_app
import pandas as pd


def test_align_and_prepare_single_maps_graph_features():
    # inject a small graph dicts set
    # ensure we don't have an enforced minimal feature set for this test
    api_app.MIN_FEATURE_SET = None
    api_app.GRAPH_DICTS = {
        'pr': {'0000': 0.12, 'm1': 0.01},
        'evc': {'0000': 0.3},
        'clust': {'0000': 0.0},
        'core': {'0000': 2},
        'neighbor_fraud': {'0000': 0.5}
    }

    df = pd.DataFrame([{'cc_num': '0000', 'merchant': 'm1', 'amt': 5.0}])
    X, df_enr = api_app.align_and_prepare_single(df)

    # expect node_pr_card and node_pr_merch to exist
    assert 'node_pr_card' in df_enr.columns
    assert 'node_pr_merch' in df_enr.columns
    # features should be present in returned X (numeric)
    assert 'node_pr_card' in X.columns
    assert X['node_pr_card'].iloc[0] == 0.12
    assert 'node_evc_card' in X.columns
    assert X['node_evc_card'].iloc[0] == 0.3
