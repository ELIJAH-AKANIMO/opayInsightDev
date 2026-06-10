from api import app as api_app
import pandas as pd


def test_align_single_minimal_features_enforced():
    # set a minimal feature set explicitly
    api_app.MIN_FEATURE_SET = ['amt', 'node_pr_card', 'node_neighbor_fraud_card']
    api_app.GRAPH_DICTS = {
        'pr': {'0000': 0.05},
        'neighbor_fraud': {'0000': 0.7}
    }
    df = pd.DataFrame([{'cc_num': '0000', 'merchant': 'm1', 'amt': 42.0}])
    X, df_enr = api_app.align_and_prepare_single(df)

    # X should contain exactly the minimal features (order preserved)
    assert list(X.columns) == api_app.MIN_FEATURE_SET
    assert X['amt'].iloc[0] == 42.0
    assert X['node_pr_card'].iloc[0] == 0.05
    assert X['node_neighbor_fraud_card'].iloc[0] == 0.7
