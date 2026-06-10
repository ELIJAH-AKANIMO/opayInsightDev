from api import app as api_app
import pandas as pd


def test_trim_toggle_off_disables_restriction():
    # Prepare graph dicts with multiple features
    api_app.GRAPH_DICTS = {
        'pr': {'0000': 0.1},
        'evc': {'0000': 0.2},
        'neighbor_fraud': {'0000': 0.3}
    }
    # force a MIN_FEATURE_SET that would normally restrict
    api_app.MIN_FEATURE_SET = ['amt', 'node_pr_card']

    # disable trimming
    api_app.MIN_FEATURE_TRIM_ENABLED = False

    df = pd.DataFrame([{'cc_num': '0000', 'merchant': 'm1', 'amt': 7.0}])
    X, df_enr = api_app.align_and_prepare_single(df)

    # when trimming disabled, we should see evc feature included from graph mapping
    assert 'node_evc_card' in X.columns


def test_trim_query_param_overrides():
    api_app.GRAPH_DICTS = {'pr': {'0000': 0.11}, 'evc': {'0000': 0.22}}
    api_app.MIN_FEATURE_SET = ['amt', 'node_pr_card']
    api_app.MIN_FEATURE_TRIM_ENABLED = True

    df = pd.DataFrame([{'cc_num': '0000', 'merchant': 'm1', 'amt': 3.0}])
    # trim True -> restricted
    X_trim, _ = api_app.align_and_prepare_single(df, trim=True)
    assert list(X_trim.columns) == api_app.MIN_FEATURE_SET

    # trim False -> unrestricted (contains evc)
    X_no_trim, _ = api_app.align_and_prepare_single(df, trim=False)
    assert 'node_evc_card' in X_no_trim.columns
