"""Run the Intelligent Detection pipeline (rolling + graph features and LightGBM training).

Usage:
  python pipeline/run_pipeline.py --mode sample --sample_n 200000
  python pipeline/run_pipeline.py --mode full

Outputs:
 - enriched CSV: data/enriched_sample.csv or data/enriched_full.csv
 - model: models/lgb_rolling_graph_{sample|full}.joblib
 - scaler: models/scaler_rolling_{sample|full}.joblib
 - metrics: models/{sample|full}_metrics.json
"""
import argparse
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
import joblib


def sanitize_columns(cols):
    new = []
    seen = {}
    for c in cols:
        nc = re.sub(r'[^0-9a-zA-Z_]', '_', str(c))
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}_{seen[nc]}"
        else:
            seen[nc] = 0
        new.append(nc)
    return new


def build_features(df):
    df = df.copy()
    df['trans_dt'] = pd.to_datetime(df['unix_time'], unit='s')
    df['date'] = df['trans_dt'].dt.floor('D')

    # per-card daily aggregates
    card_daily = df.groupby(['cc_num','date']).agg(card_tx_count_day=('amt','count'), card_amt_sum_day=('amt','sum')).reset_index().sort_values(['cc_num','date'])
    card_daily['card_7d_tx_count'] = card_daily.groupby('cc_num')['card_tx_count_day'].rolling(window=7, min_periods=1).sum().reset_index(level=0,drop=True)
    card_daily['card_30d_tx_count'] = card_daily.groupby('cc_num')['card_tx_count_day'].rolling(window=30, min_periods=1).sum().reset_index(level=0,drop=True)
    card_daily['card_7d_amt_sum'] = card_daily.groupby('cc_num')['card_amt_sum_day'].rolling(window=7, min_periods=1).sum().reset_index(level=0,drop=True)

    merch_daily = df.groupby(['merchant','date']).agg(merch_tx_count_day=('amt','count'), merch_amt_sum_day=('amt','sum')).reset_index().sort_values(['merchant','date'])
    merch_daily['merch_7d_tx_count'] = merch_daily.groupby('merchant')['merch_tx_count_day'].rolling(window=7, min_periods=1).sum().reset_index(level=0,drop=True)
    merch_daily['merch_30d_tx_count'] = merch_daily.groupby('merchant')['merch_tx_count_day'].rolling(window=30, min_periods=1).sum().reset_index(level=0,drop=True)

    df = df.merge(card_daily[['cc_num','date','card_7d_tx_count','card_30d_tx_count','card_7d_amt_sum']], on=['cc_num','date'], how='left')
    df = df.merge(merch_daily[['merchant','date','merch_7d_tx_count','merch_30d_tx_count','merch_amt_sum_day']], on=['merchant','date'], how='left')
    df.fillna(0, inplace=True)

    # graph features (bipartite + structural)
    edges = df[['cc_num','merchant']].drop_duplicates().values.tolist()
    G = nx.Graph()
    G.add_edges_from(edges)

    # basic graph features
    df['node_degree_card'] = df['cc_num'].map(lambda x: G.degree(x) if x in G else 0)
    df['node_degree_merchant'] = df['merchant'].map(lambda x: G.degree(x) if x in G else 0)

    # connected component sizes
    comp_sizes = {}
    for comp in nx.connected_components(G):
        size = len(comp)
        for n in comp:
            comp_sizes[n] = size
    df['node_comp_size'] = df['cc_num'].map(lambda x: comp_sizes.get(x,0))

    # structural graph measures (may be costly on very large graphs)
    pr = nx.pagerank(G, alpha=0.85)
    try:
        evc = nx.eigenvector_centrality_numpy(G)
    except Exception:
        evc = nx.eigenvector_centrality(G, max_iter=1000)
    clust = nx.clustering(G)
    core = nx.core_number(G)

    # node-level fraud rates (card or merchant)
    card_fraud = df.groupby('cc_num')['is_fraud'].mean().to_dict()
    merch_fraud = df.groupby('merchant')['is_fraud'].mean().to_dict()
    node_fraud = {}
    for node in G.nodes():
        if node in card_fraud:
            node_fraud[node] = card_fraud[node]
        elif node in merch_fraud:
            node_fraud[node] = merch_fraud[node]
        else:
            node_fraud[node] = 0.0

    # neighbor fraud rate = mean fraud rate of neighbors
    neighbor_fraud = {}
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if not nbrs:
            neighbor_fraud[node] = 0.0
        else:
            neighbor_fraud[node] = float(np.mean([node_fraud.get(n,0.0) for n in nbrs]))

    # map structural features back to transactions
    df['node_pagerank_card'] = df['cc_num'].map(lambda x: pr.get(x,0.0))
    df['node_pagerank_merch'] = df['merchant'].map(lambda x: pr.get(x,0.0))
    df['node_eig_card'] = df['cc_num'].map(lambda x: evc.get(x,0.0))
    df['node_eig_merch'] = df['merchant'].map(lambda x: evc.get(x,0.0))
    df['node_clust_card'] = df['cc_num'].map(lambda x: clust.get(x,0.0))
    df['node_clust_merch'] = df['merchant'].map(lambda x: clust.get(x,0.0))
    df['node_core_card'] = df['cc_num'].map(lambda x: core.get(x,0))
    df['node_core_merch'] = df['merchant'].map(lambda x: core.get(x,0))
    df['node_neighbor_fraud_card'] = df['cc_num'].map(lambda x: neighbor_fraud.get(x,0.0))
    df['node_neighbor_fraud_merch'] = df['merchant'].map(lambda x: neighbor_fraud.get(x,0.0))

    # persist node-level dicts for reuse (models/graph_struct_dicts.pkl)
    Path('models').mkdir(exist_ok=True)
    joblib.dump({'pr': pr, 'evc': evc, 'clust': clust, 'core': core, 'neighbor_fraud': neighbor_fraud}, 'models/graph_struct_dicts.pkl')

    return df


def train_time_split(df, out_prefix='sample'):
    df_sorted = df.sort_values('unix_time').reset_index(drop=True)
    idx = int(len(df_sorted) * 0.7)
    split_time = df_sorted.loc[idx, 'unix_time']
    train_time = df_sorted[df_sorted['unix_time'] <= split_time].copy()
    val_time = df_sorted[df_sorted['unix_time'] > split_time].copy()

    # drop identifiers
    drop_cols = ['first','last','street','city','state','zip','dob','trans_num','trans_date_trans_time']
    for c in drop_cols:
        if c in train_time.columns: train_time.drop(columns=[c], inplace=True)
        if c in val_time.columns: val_time.drop(columns=[c], inplace=True)

    X_train = train_time.select_dtypes(include=["number"]).drop(columns=['is_fraud']).fillna(0)
    y_train = train_time['is_fraud'].astype(int)
    X_val = val_time.select_dtypes(include=["number"]).drop(columns=['is_fraud']).fillna(0)
    y_val = val_time['is_fraud'].astype(int)

    X_train.columns = sanitize_columns(X_train.columns)
    X_val.columns = sanitize_columns(X_val.columns)

    scaler = StandardScaler()
    amt_cols = [c for c in X_train.columns if 'amt' in c or c=='amt']
    if amt_cols:
        X_train[amt_cols] = scaler.fit_transform(X_train[amt_cols])
        X_val[amt_cols] = scaler.transform(X_val[amt_cols])

    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    y_val_proba = clf.predict_proba(X_val)[:,1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    pr_auc = average_precision_score(y_val, y_val_proba)

    Path('models').mkdir(exist_ok=True)
    joblib.dump(clf, f'models/lgb_rolling_graph_{out_prefix}.joblib')
    joblib.dump(scaler, f'models/scaler_rolling_{out_prefix}.joblib')

    metrics = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}
    with open(f'models/{out_prefix}_metrics.json','w') as f:
        json.dump(metrics, f)

    print('Saved model and metrics for', out_prefix)
    print(metrics)

    # --- Train Deep Learning Autoencoder ---
    print("\n[Deep Learning] Training LSTM Autoencoder...")
    try:
        import sys
        import os
        # Add project root to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from models import AnomalyDetector
        # Filter for deep learning (sequence data)
        # We use the same train/val split but need to ensure it's suitable for the DL class
        # The DL class handles its own preprocessing/sequences
        
        # Initialize detector
        dl_model = AnomalyDetector(sequence_length=10, hidden_dim=64, num_layers=2)
        
        # Train on normal transactions ideally, or all train data
        # Unsupervised, so we can use X_train equivalent from df (need raw data with meta)
        
        # Re-using the time-split dataframes for consistency
        dl_train_df = train_time.copy()
        
        # Train
        dl_model.train(dl_train_df, epochs=5, save_best=True)
        
        # Save
        dl_save_path = f'models/saved_models/autoencoder_{out_prefix}.pth'
        dl_model.save_model(dl_save_path)
        print(f"[Deep Learning] Model saved to {dl_save_path}")
        
    except Exception as e:
        print(f"[Deep Learning] Training failed: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sample', 'full'], default='sample')
    parser.add_argument('--sample_n', type=int, default=200000)
    args = parser.parse_args()

    p_train = Path('creditcard/dataset1/fraudTrain.csv')
    assert p_train.exists(), 'fraudTrain.csv not found at creditcard/dataset1'
    df = pd.read_csv(p_train)

    if args.mode == 'sample':
        df = df.sample(n=min(args.sample_n, len(df)), random_state=42)
        out_prefix = 'sample'
    else:
        out_prefix = 'full'

    print('Building features on', len(df), 'rows')
    df_enr = build_features(df)
    Path('data').mkdir(exist_ok=True)
    df_enr.to_csv(f'data/enriched_{out_prefix}.csv', index=False)
    print('Saved data/enriched_{0}.csv'.format(out_prefix))

    metrics = train_time_split(df_enr, out_prefix=out_prefix)

    print('Pipeline finished for mode:', args.mode)

if __name__ == '__main__':
    main()